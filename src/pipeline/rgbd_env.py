"""Closed-loop RGB-D adapter. Only the underlying simulator owns physical IDs.

Perception runs while physics is paused at planning boundaries and after release.
The controller receives tracked geometry, FK and two binary tactile signals.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from perception.acquisition import capture, load_frame
from perception.association import ObjectTracker, fuse_views
from perception.bridge import fit_known_box, to_scene_graph
from perception.local_masks import predict_color_masks
from perception.reconstruction import proposals, public
from rgbd_hssg import build_scene_graph, graph_delta, save_graph, save_json

from .types import Pose, TaskRequest, check


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
    matrix[:3, 3] = pose.position
    return matrix


def matrix_pose(matrix):
    return Pose(tuple(matrix[:3, 3]), tuple(Rotation.from_matrix(matrix[:3, :3]).as_quat()))


class RGBDEnvironment:
    def __init__(self, simulator, target_spec, *, output, method='color', inference_command=None):
        check(method in ('color', 'command'), 'unknown RGB-D inference method')
        check(method != 'command' or isinstance(inference_command, list) and inference_command and
              all(isinstance(arg, str) for arg in inference_command),
              'command inference requires a JSON array of arguments')
        self.simulator = simulator
        self.output = Path(output)
        self.method, self.inference_command = method, inference_command
        self.target_spec = target_spec
        height = target_spec.size_m[2] if target_spec.size_m else .03
        floor = simulator.config['placement_trays']['floor_thickness']
        destination = Pose((*simulator.config['destination'][:2], floor+height/2))
        buffer = Pose((*simulator.config['buffer_position'][:2], floor+height/2))
        self.task = TaskRequest('unresolved:target', destination, buffer, target_spec)
        # Known object-size vocabulary is an experimental prior, never per-object GT.
        self.box_sizes = (simulator.config['shelf_scene']['object_sizes'] if simulator.shelf_scene else
                          [(simulator.config['cube_size'],)*3])
        self.snapshot = 0
        self.tracker = ObjectTracker()
        self.objects = ()
        self.graph = None
        self.last_graph = None
        self.last_capture_time = None
        self.active = None
        self.held_id = None
        self.grasp_transform = None
        self.contact_z = None
        self.contact_ticks = 0
        self.pick_ticks = 0
        self.release_ticks = 0
        self.verified = False
        self.events = []
        self.regions = []
        self.jaw_bodies = (simulator.fixed_jaw_body, simulator.moving_jaw_body)
        base = simulator.model.body('base').id
        self.robot_bodies = {base}
        for body in range(simulator.model.nbody):
            parent = body
            while parent and parent != base:
                parent = int(simulator.model.body_parentid[parent])
            if parent == base:
                self.robot_bodies.add(body)

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def reset(self, seed=0):
        self.simulator.reset(seed, privileged=False)
        self.snapshot = 0
        self.tracker = ObjectTracker()
        self.objects, self.graph, self.last_graph = (), None, None
        self.last_capture_time = self.active = self.held_id = self.grasp_transform = self.contact_z = None
        self.contact_ticks = self.release_ticks = self.pick_ticks = 0
        self.verified, self.events = False, []
        return self.refresh()

    def _touch(self):
        touched = [False, False]
        for index, contact in enumerate(self.data.contact):
            bodies = {int(self.model.geom_bodyid[int(geom)]) for geom in contact.geom}
            if not bodies - self.robot_bodies:
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] > .1:
                for j, body in enumerate(self.jaw_bodies):
                    touched[j] |= body in bodies
        return tuple(bool(value) for value in touched)

    def observe(self):
        raw = self.simulator.observe(privileged=False)
        objects = list(self.objects)
        if self.held_id is not None and self.grasp_transform is not None:
            pose = matrix_pose(pose_matrix(raw.ee_pose) @ self.grasp_transform)
            objects = [replace(obj, pose=pose) if obj.object_id == self.held_id else obj for obj in objects]
        return replace(raw, source='rgbd', ground_truth=None, contacts=(),
                       held_object_id=self.held_id, perceived_objects=tuple(objects),
                       finger_contacts=self._touch())

    def refresh(self):
        observation = self.observe()
        if self.last_capture_time == observation.timestamp:
            return observation
        directory = self.output/f'snapshot_{self.snapshot:03d}'
        framepath = directory/'inputs/frame_000'
        meta = capture(self.simulator, framepath)
        self.regions = meta['regions']
        if self.events:
            meta['action_events'] = self.events
            save_json(meta, framepath/'capture.json')
        meta, frames = load_frame(framepath)
        prediction = directory/'predictions/frame_000'
        if self.method == 'color':
            predict_color_masks(framepath, prediction)
        else:
            argv = [arg.format(inputs=str((directory/'inputs').resolve()),
                               output=str((directory/'predictions').resolve()))
                    for arg in self.inference_command]
            subprocess.run(argv, check=True)
        objects = fuse_views({name: proposals(name, frame, meta, prediction) for name, frame in frames.items()})
        rejected = 0
        for obj in objects:
            try:
                obj['box_model_fit'] = fit_known_box(obj, self.box_sizes)
                fit = obj['box_model_fit']
                floor = next((r.get('floor_top_m', 0.) for r in meta['regions'] if all(
                    abs(fit['position_m'][i]-r['center'][i]) <= r['size'][i]/2 for i in (0, 1))), 0.)
                top = fit['position_m'][2]+fit['size_m'][2]/2
                max_top = self.config['shelf_scene']['floor_thickness'] + self.config['shelf_scene']['max_stack_height']*max(
                    size[2] for size in self.box_sizes) + .015
                check(fit['position_m'][2]-fit['size_m'][2]/2 >= floor-.004 and top < max_top,
                      'box candidate outside configured support/stack height range')
            except ValueError:
                obj['control_geometry_invalid'] = True
                rejected += 1
        objects = [obj for obj in objects if not obj.get('control_geometry_invalid')]
        # RGB highlights can split one box into disjoint color masks. Reject
        # duplicate fitted top surfaces without merging different stack levels.
        distinct = []
        for obj in sorted(objects, key=lambda o: o['points'], reverse=True):
            fit = obj['box_model_fit']
            pos, size = np.array(fit['position_m']), np.array(fit['size_m'])
            duplicate = any(np.linalg.norm(pos[:2]-np.array(other['box_model_fit']['position_m'])[:2]) < min(size[:2])*.4 and
                abs(pos[2]+size[2]/2-other['box_model_fit']['position_m'][2]-other['box_model_fit']['size_m'][2]/2) < .006
                for other in distinct)
            if not duplicate:
                distinct.append(obj)
        save_json(dict(method=self.method, geometry_prior='configured upright square boxes',
                       raw_candidates=len(objects)+rejected, invalid_geometry=rejected,
                       duplicate_surfaces=len(objects)-len(distinct), current_candidates=len(distinct)),
                  directory/'perception_report.json')
        objects = distinct
        tracks = self.tracker.update(objects, meta['episode_id'], meta['timestamp'], self.events, meta['regions'])
        raw_graph = build_scene_graph(tracks, meta)
        save_graph(raw_graph, directory/'scene_graph.json')
        save_json(graph_delta(self.last_graph, raw_graph), directory/'graph_delta.json')
        save_json(dict(episode=self.tracker.episode, timestamp=self.tracker.timestamp, next_id=self.tracker.next_id,
                       tracks=[public(t) for t in tracks]), directory/'tracker_state.json')
        held = next((obj for obj in observation.perceived_objects if obj.object_id == self.held_id), None)
        self.graph = to_scene_graph(raw_graph, held_object=held)
        self.objects = self.graph.nodes
        self.last_graph, self.last_capture_time = raw_graph, observation.timestamp
        self.snapshot += 1
        print(f'RGB-D snapshot {self.snapshot}: {len(self.objects)} current objects, held={self.held_id}', flush=True)
        return self.observe()

    def generate_scene_graph(self, observation):
        check(self.graph is not None, 'RGB-D scene has not been observed')
        check(observation.timestamp == self.last_capture_time, 'refresh RGB-D before planning')
        held = next((obj for obj in observation.perceived_objects if obj.object_id == self.held_id), None)
        return to_scene_graph(self.last_graph, held_object=held)

    def _begin(self, goal):
        if self.active is not None and self.active.subgoal_id == goal.subgoal_id:
            return
        self.active = goal
        self.contact_z = None
        self.contact_ticks = self.release_ticks = self.pick_ticks = 0
        self.verified = False

    def step(self, command):
        self.simulator.step(command, privileged=False)
        observation = self.observe()
        if self.active is None:
            return observation
        if self.active.primitive == 'PICK':
            self.pick_ticks += 1
            self.contact_ticks = self.contact_ticks+1 if all(observation.finger_contacts) else 0
            if self.contact_ticks >= 3 and self.held_id is None:
                obj = next((obj for obj in observation.perceived_objects if obj.object_id == self.active.object_id), None)
                check(obj is not None, 'gripper contacted an unobserved target; cannot establish tracked grasp')
                self.held_id = self.active.object_id
                self.contact_z = observation.ee_pose.position[2]
                self.grasp_transform = np.linalg.solve(pose_matrix(observation.ee_pose),
                                                       pose_matrix(obj.pose))
            elif self.held_id is None and observation.robot_state[5] > .8 and self.pick_ticks % 20 == 0:
                # A missed approach may push a box. Reobserve before retries;
                # the expert then replans from the measured, not commanded pose.
                self.refresh()
        else:
            if observation.robot_state[5] < .25:
                if self.held_id is not None:
                    self.objects = observation.perceived_objects
                    self.held_id = None
                self.release_ticks += 1
                if self.release_ticks == 12 or self.release_ticks > 12 and self.release_ticks % 15 == 0:
                    goal = self.active
                    region = next((r['id'] for r in self.regions if all(
                        abs(goal.place_pose.position[i]-r['center'][i]) <= r['size'][i]/2 for i in (0, 1))), None)
                    if region is not None:
                        self.events = [dict(event_id=goal.subgoal_id, episode_id=observation.episode_id,
                            source_instance_id=goal.object_id, destination_region=region,
                            expected_position_m=list(goal.place_pose.position),
                            started_at=float(self.tracker.timestamp), completed_at=observation.timestamp)]
                    observation = self.refresh()
                    placed = next((obj for obj in observation.perceived_objects if obj.object_id == goal.object_id), None)
                    self.verified = placed is not None and np.linalg.norm(
                        np.array(placed.pose.position)-goal.place_pose.position) < .012
            else:
                self.release_ticks = 0
        return self.observe()

    def subgoal_complete(self, subgoal):
        self._begin(subgoal)
        observation = self.observe()
        if subgoal.primitive == 'PICK':
            return bool(self.held_id == subgoal.object_id and self.contact_z is not None and
                        all(observation.finger_contacts) and observation.ee_pose.position[2] > self.contact_z+.012)
        return bool(self.verified)

    def task_complete(self, task):
        observation = self.observe()
        obj = next((obj for obj in observation.perceived_objects if obj.object_id == task.target_object_id), None)
        return bool(obj is not None and self.held_id is None and observation.robot_state[5] < .25 and
                    np.linalg.norm(np.array(obj.pose.position)-task.destination.position) < .012)
