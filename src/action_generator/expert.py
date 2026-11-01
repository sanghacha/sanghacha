"""Feedback PICK/PLACE expert for the simulated SO-101; no state teleportation.

Uses the public observation/subgoal contract and a private kinematic robot model.
This is a scripted integration baseline and demonstration teacher, not ACT.
"""

from dataclasses import dataclass, replace
import json

import numpy as np

from pipeline.mock_data import ROOT
from pipeline.types import ActionChunk, check
from .kinematics import SO101Kinematics, tilted_rotation, top_rotation


@dataclass
class Waypoint:
    position: np.ndarray
    rotation: np.ndarray
    gripper: float
    mode: str = 'move'


class SO101Expert:
    def __init__(self, *, grasp_height_offset=.001):
        self.kinematics = SO101Kinematics()
        config = json.loads((ROOT/'configs/envs/so101.json').read_text(encoding='utf-8'))
        self.max_delta = np.array(config['max_delta'])*.75
        self.key = None
        self.last_time = None
        self.previous_primitive = None
        self.path = []
        self.index = 0
        self.stage_ticks = 0
        self.close_ticks = 0
        self.attempt = 0
        self._q_target = None
        check(np.isfinite(grasp_height_offset) and abs(grasp_height_offset) <= .01, 'invalid grasp height offset')
        self.grasp_height_offset = float(grasp_height_offset)
        self.recovering = False
        self.recoveries = 0

    def _object(self, observation, object_id):
        objects = observation.perceived_objects if observation.source == 'rgbd' else observation.ground_truth
        check(objects is not None, 'SO-101 expert requires observed object geometry')
        found = next((obj for obj in objects if obj.object_id == object_id), None)
        check(found is not None, f'expert cannot localize {object_id}')
        return found

    def _line(self, start, end, rotation, gripper, *, spacing=.004):
        n = max(1, int(np.ceil(np.linalg.norm(end-start)/spacing)))
        self.path.extend(Waypoint(start+(end-start)*t, rotation, gripper)
                         for t in np.linspace(1/n, 1., n))

    def _begin(self, observation, subgoal):
        k = self.kinematics
        current, actual_rotation = k.forward(observation.robot_state)
        obj = self._object(observation, subgoal.object_id)
        self.path = []
        self.index = self.stage_ticks = self.close_ticks = 0
        self._q_target = None
        p = np.array(obj.pose.position)
        self.geometry_position = p.copy()
        rotation = tilted_rotation(p, np.deg2rad(35.), flip=self.attempt % 2 == 0)
        if subgoal.primitive == 'PICK':
            check(observation.held_object_id in (None, subgoal.object_id), 'another object is already held')
            # PLACE already retracts before it fully opens. A second blind
            # retraction can exceed the vertical wrist limit near the base.
            # The gripper site is offset from the fixed jaw's actual inner surface.
            yaw = np.arctan2(rotation[1, 2], rotation[0, 2])
            quat = np.array(obj.pose.quaternion_xyzw)
            object_yaw = np.arctan2(2*(quat[3]*quat[2]+quat[0]*quat[1]),
                                    1-2*(quat[1]**2+quat[2]**2))
            vertical = p[0] < .21 and p[2] < .04
            if vertical:
                rotation = top_rotation(object_yaw)
                yaw = object_yaw
            width = abs(np.cos(yaw-object_yaw))*obj.size[0]+abs(np.sin(yaw-object_yaw))*obj.size[1]
            grasp = p+rotation[:, 2]*(.0201-width/2)
            grasp[2] += self.grasp_height_offset
            opening = .75 if vertical else .7
            pre = grasp-rotation[:, 0]*(.035 if vertical else .065)
            if observation.source == 'rgbd':
                # Allow for measured-pose error, and open before any retry
                # motion so a closed jaw does not sweep the displaced box.
                opening = .70 if vertical else .60
                self.path.append(Waypoint(current.copy(), actual_rotation, opening, 'open'))
                if not vertical and p[2] < .04:
                    high = pre.copy()
                    high[2] = max(pre[2]+.045, current[2])
                    self.path.append(Waypoint(high, rotation, opening))
                    self._line(high, pre, rotation, opening)
            self.path.append(Waypoint(pre, rotation, opening))
            self._line(pre, grasp, rotation, opening)
            self.path.append(Waypoint(grasp, rotation, 1., 'close'))
            lift = grasp+np.array([0., 0., .05])
            self._line(grasp, lift, rotation, 1.)
            self.path.append(Waypoint(lift, rotation, 1., 'hold'))
        else:
            check(observation.held_object_id == subgoal.object_id, 'PLACE requires the requested object to be held')
            relative = actual_rotation.T@(p-current)
            destination = np.array(subgoal.place_pose.position)
            reference = tilted_rotation(p, np.deg2rad(35.))
            flip = np.dot(actual_rotation[:, 2], reference[:, 2]) < 0
            vertical = actual_rotation[2, 0] < -.98
            rotation = (top_rotation(np.arctan2(actual_rotation[1, 2], actual_rotation[0, 2])) if vertical else
                        tilted_rotation(destination, np.deg2rad(35.), flip=flip))
            place = destination-rotation@relative
            # Finish lifting before traversing, even if PICK completed after only 1cm.
            high = current.copy()
            high[2] = max(.065 if vertical else .115, p[2]+(.025 if vertical else .04))
            self._line(current, high, actual_rotation, 1.)
            pre = place-rotation[:, 0]*(.04 if vertical else .065)
            self.path.append(Waypoint(pre, rotation, 1.))
            lower = place-rotation[:, 0]*.006
            self._line(pre, lower, rotation, 1.)
            self.path.append(Waypoint(lower.copy(), rotation, 1., 'align'))
            self.path.append(Waypoint(lower.copy(), rotation, .60, 'ungrip'))
            retreat = lower+np.array([0., 0., .035])
            self._line(lower, retreat, rotation, .60)
            self.path.append(Waypoint(retreat, rotation, 0., 'release'))

    def predict_action_chunk(self, observation, subgoal):
        check(observation.robot_id == 'so101' and observation.source in ('mujoco_gt', 'rgbd'),
              'scripted expert requires SO-101 geometry and contact feedback')
        key = (observation.episode_id, subgoal)
        continuous = self.last_time is not None and np.isclose(
            observation.timestamp-self.last_time, observation.control_dt, rtol=1e-5, atol=1e-8)
        if key != self.key or not continuous:
            self.previous_primitive = None if self.key is None or observation.episode_id != self.key[0] or not continuous else self.primitive
            self.key, self.primitive, self.attempt = key, subgoal.primitive, 0
            self.recovering, self.recoveries = False, 0
            self._begin(observation, subgoal)
        self.last_time = observation.timestamp
        if observation.source == 'rgbd' and subgoal.primitive == 'PICK' and observation.held_object_id is None:
            position = np.array(self._object(observation, subgoal.object_id).pose.position)
            if np.linalg.norm(position-self.geometry_position) > .005:
                self.attempt += 1
                check(self.attempt < 4, 'observed object kept moving during grasp approach')
                self._begin(observation, subgoal)
        if self.recovering:
            obj = self._object(observation, subgoal.object_id)
            if observation.held_object_id == subgoal.object_id and obj.pose.position[2] > self.recovery_z+.012:
                self.recovering = False
                self._begin(observation, subgoal)
        waypoint = self.path[self.index]
        k = self.kinematics
        if waypoint.mode == 'align':
            object_position = np.array(self._object(observation, subgoal.object_id).pose.position)
            correction = np.array(subgoal.place_pose.position)[:2]-object_position[:2]
            if np.linalg.norm(correction) > .002:
                waypoint.position[:2] += np.clip(correction*.5, -.0015, .0015)
                self._q_target = None
        if self._q_target is None:
            q, position_error, rotation_error = k.solve(waypoint.position, waypoint.rotation, observation.robot_state)
            # A second seed avoids a folded-elbow local solution.
            if position_error > .004 or rotation_error > .18:
                for roll in (0., -1.5, 1.5):
                    seed = observation.robot_state.copy()
                    seed[:5] = [-np.arctan2(waypoint.position[1], waypoint.position[0]-.0084), -.6, .8, .8, roll]
                    q2, ep2, er2 = k.solve(waypoint.position, waypoint.rotation, seed, orientation_weight=.04)
                    if ep2+.05*er2 < position_error+.05*rotation_error:
                        q, position_error, rotation_error = q2, ep2, er2
                    if position_error < .003 and rotation_error < .15:
                        break
            check(position_error < .005 and rotation_error < .25,
                  f'unreachable expert waypoint: position error={position_error:.4f}m, rotation error={rotation_error:.3f}')
            self._q_target = q
        self.stage_ticks += 1
        actual, _ = k.forward(observation.robot_state)
        reached = (np.linalg.norm(actual-waypoint.position) < .0065 or
                   np.max(np.abs(self._q_target[:5]-observation.robot_state[:5])) < .025)
        advance = reached
        if waypoint.mode == 'open':
            advance = observation.robot_state[5] <= waypoint.gripper+.025 and self.stage_ticks >= 3
        elif waypoint.mode == 'close':
            fingers = set()
            if observation.source == 'rgbd':
                fingers = {name for name, active in zip(('fixed', 'moving'), observation.finger_contacts) if active}
            for contact in (() if observation.source == 'rgbd' else observation.contacts):
                if subgoal.object_id not in (contact.entity_a, contact.entity_b) or contact.normal_force < .1:
                    continue
                for entity in (contact.entity_a, contact.entity_b):
                    if entity.startswith('fixed_jaw'):
                        fingers.add('fixed')
                    if entity.startswith('moving_jaw'):
                        fingers.add('moving')
            self.close_ticks = self.close_ticks+1 if len(fingers) == 2 else 0
            advance = self.close_ticks >= 6
            if self.stage_ticks > 35 and not advance:
                self.attempt += 1
                check(self.attempt < 3, 'expert failed to establish two-finger contact after 3 attempts')
                retry_goal = (replace(subgoal, primitive='PICK', place_pose=None,
                                      object_pose=self._object(observation, subgoal.object_id).pose)
                              if self.recovering else subgoal)
                self._begin(observation, retry_goal)
                command = observation.robot_state.copy()
                command[5] = max(0., command[5]-.075)
                return ActionChunk(subgoal.subgoal_id, command[None, :], observation.control_dt, observation.robot_id)
        elif waypoint.mode == 'align':
            advance = np.linalg.norm(correction) <= .002 and self.stage_ticks >= 5
        elif waypoint.mode == 'ungrip':
            advance = observation.robot_state[5] < .63 and self.stage_ticks >= 7
        elif waypoint.mode in ('release', 'hold'):
            advance = False  # Runtime's physical completion test decides when this subgoal ends.
            if waypoint.mode == 'release' and self.stage_ticks > 20 and observation.robot_state[5] < .25:
                obj = self._object(observation, subgoal.object_id)
                error = np.linalg.norm(np.array(obj.pose.position)-subgoal.place_pose.position)
                if error > .009:
                    self.recoveries += 1
                    check(self.recoveries <= 2, f'placement did not settle at its goal after two corrections ({error:.3f}m)')
                    self.recovering, self.recovery_z = True, obj.pose.position[2]
                    pick = replace(subgoal, primitive='PICK', object_pose=obj.pose, place_pose=None)
                    self._begin(observation, pick)
                    return ActionChunk(subgoal.subgoal_id, observation.robot_state[None, :],
                                       observation.control_dt, observation.robot_id)
        if advance and self.index < len(self.path)-1:
            self.index += 1
            self.stage_ticks = self.close_ticks = 0
            self._q_target = None
        check(self.stage_ticks <= 100, f'expert stalled in {subgoal.primitive} waypoint {self.index} ({waypoint.mode})')
        target = self._q_target if self._q_target is not None else observation.robot_state.astype(float).copy()
        target = target.copy()
        target[5] = waypoint.gripper
        delta = np.minimum(self.max_delta, observation.control_dt*np.array([1.35]*5+[2.4]))
        command = observation.robot_state+np.clip(target-observation.robot_state, -delta, delta)
        command = np.clip(command, k.lower+1e-6, k.upper-1e-6).astype(np.float32)
        return ActionChunk(subgoal.subgoal_id, command[None, :], observation.control_dt, observation.robot_id)
