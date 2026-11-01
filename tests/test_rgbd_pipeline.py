"""Appearance identity contracts and camera-to-action physical integration."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np

from pipeline.target import TargetResolver
from pipeline.types import Pose, SceneGraph, SceneObject, TargetSpec, TaskRequest
from planner import plan_next_subgoal

ROOT = Path(__file__).resolve().parents[1]
HAS_PERCEPTION = all(importlib.util.find_spec(name) is not None for name in ('open3d', 'scipy', 'networkx'))


def scene(*objects, timestamp=0., held=None, episode='test'):
    return SceneGraph(tuple(objects), (), {'workspace': None, 'shelf_2': 'workspace', 'output_tray': 'workspace'},
                      held, timestamp, episode, source='rgbd')


def blue(key='instance_0001', region='shelf_2', z=.021):
    return SceneObject(key, Pose((.24, 0., z)), (.03, .03, .03), parent_id=region,
                       color_rgb=(.15, .35, .95))


class TargetResolverTests(unittest.TestCase):
    def task(self):
        return TaskRequest('unresolved:target', Pose((.08, .18, .021)), Pose((.08, -.18, .021)),
                           TargetSpec((.15, .35, .95), region='shelf_2', category='cube'))

    def test_resolves_appearance_and_keeps_identity_after_region_change(self):
        resolver = TargetResolver()
        task = resolver.resolve(scene(blue()), self.task())
        self.assertEqual(task.target_object_id, 'instance_0001')
        # A same-color distractor at the original shelf must not replace the bound ID.
        task = resolver.resolve(scene(blue(region='output_tray'), blue('instance_0002'), timestamp=1), task)
        self.assertEqual(task.target_object_id, 'instance_0001')
        task = resolver.resolve(scene(blue('instance_0099'), episode='next'), task)
        self.assertEqual(task.target_object_id, 'instance_0099')

    def test_ambiguous_target_stops_and_absence_does_not_create_a_pose(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous target'):
            TargetResolver().resolve(scene(blue(), blue('instance_0002')), self.task())
        top = replace(blue('top', z=.077), color_rgb=(.9, .1, .1))
        graph = scene(top)
        task = TargetResolver().resolve(graph, self.task())
        self.assertEqual(task.target_object_id, 'unresolved:target')
        goal = plan_next_subgoal(graph, task, np.zeros(6, np.float32))
        self.assertEqual((goal.primitive, goal.object_id), ('PICK', 'top'))

    def test_invalid_visual_request_is_rejected(self):
        for rgb in ((0., 0., 0.), (1.2, .1, .1), (float('nan'), .1, .1)):
            with self.assertRaises(ValueError):
                TargetSpec(rgb)


@unittest.skipUnless(HAS_PERCEPTION, 'Install the perception extra in Python 3.11')
class RGBDBridgeTests(unittest.TestCase):
    def test_tactile_contact_cannot_assign_a_missing_track(self):
        from types import SimpleNamespace
        from pipeline.rgbd_env import RGBDEnvironment
        from pipeline.types import Observation, Subgoal
        env = object.__new__(RGBDEnvironment)
        env.simulator = SimpleNamespace(step=lambda *args, **kwargs: None)
        env.active = Subgoal('test', 'PICK', 'lost', Pose((.2, 0., .02)), Pose((.2, 0., .1)))
        env.pick_ticks, env.contact_ticks, env.held_id = 0, 2, None
        env.observe = lambda: Observation(np.zeros((8, 8, 3), np.uint8), np.zeros(6, np.float32),
            source='rgbd', finger_contacts=(True, True), ee_pose=Pose((.2, 0., .1)))
        with self.assertRaisesRegex(ValueError, 'unobserved target'):
            env.step(np.zeros(6, np.float32))
        self.assertIsNone(env.held_id)

    def test_history_target_is_excluded_and_raw_obb_evidence_is_preserved(self):
        from rgbd_hssg import build_scene_graph
        from perception.bridge import to_scene_graph
        tracks = [dict(instance_id='old_target', visible_surface_centroid_m=[.24, 0., .021],
                       obb=dict(center_m=[.24, 0., .021], extent_m=[.03]*3, rotation=np.eye(3).tolist()),
                       pose_kind='partial_surface_OBB', status='history', last_seen=0., mean_rgb=[.15, .35, .95])]
        raw = build_scene_graph(tracks, dict(episode_id='test', timestamp=1.,
            regions=[dict(id='shelf_2', center=[.24, 0.], size=[.17, .13])]))
        graph = to_scene_graph(raw)
        self.assertEqual(graph.nodes, ())
        self.assertIn('old_target', raw)

    def test_rgbd_sample_roundtrip_retains_target_and_tactile_fields(self):
        from pipeline.sample_data import Sample, load_sample, save_sample
        from pipeline.types import Observation
        graph = scene(blue())
        task = TargetResolverTests().task()
        goal = plan_next_subgoal(graph, replace(task, target_object_id='instance_0001'), np.zeros(6, np.float32))
        obs = Observation(np.zeros((8, 8, 3), np.uint8), np.zeros(6, np.float32),
                          source='rgbd', episode_id='test', perceived_objects=graph.nodes,
                          finger_contacts=(True, False))
        folder = ROOT/'outputs/test_rgbd'/uuid4().hex
        save_sample(folder, Sample(obs, graph, task, goal))
        sample = load_sample(folder)
        self.assertEqual(sample.task.target_spec, task.target_spec)
        self.assertEqual(sample.observation.perceived_objects, graph.nodes)
        self.assertEqual(sample.observation.finger_contacts, (True, False))
        self.assertIsNone(sample.observation.ground_truth)


@unittest.skipUnless(HAS_PERCEPTION and (ROOT/'assets/so101/upstream/so101.xml').exists(),
                     'Install perception and SO-101 assets')
class RGBDPhysicsTests(unittest.TestCase):
    def test_camera_to_retrieval_uses_tracks_and_passes_independent_physical_check(self):
        from action_generator.expert import SO101Expert
        from pipeline.mujoco_env import MujocoEnvironment
        from pipeline.rgbd_env import RGBDEnvironment
        from pipeline.runner import run_pipeline
        for case, seed, spec_file in [('accessible', 300, 'blue_box.json'), ('blocked', 42, 'blue_box.json'),
                                     ('shelves', 42, 'shelf_2_yellow_green_box.json')]:
            with self.subTest(case=case, seed=seed):
                simulator = MujocoEnvironment(case)
                spec = TargetSpec(**json.loads((ROOT/'configs/targets'/spec_file).read_text()))
                env = RGBDEnvironment(simulator, spec, output=ROOT/'outputs/test_rgbd'/uuid4().hex)
                def check_privacy(before, command, after, goal):
                    for observation in (before, after):
                        self.assertEqual(observation.source, 'rgbd')
                        self.assertIsNone(observation.ground_truth)
                        self.assertEqual(observation.contacts, ())
                        self.assertTrue(all(obj.object_id.startswith('instance_') for obj in observation.perceived_objects))
                    self.assertTrue(goal.object_id.startswith('instance_'))
                try:
                    goals = run_pipeline(env, env.task, seed=seed, generate_graph=env.generate_scene_graph,
                        refresh_observation=env.refresh, predict_action=SO101Expert().predict_action_chunk,
                        on_step=check_privacy)
                    self.assertEqual(len(goals), 2 if case == 'accessible' else 4)
                    # Oracle is read only AFTER execution; it never supplies an action or ID.
                    self.assertTrue(simulator.task_complete(simulator.task))
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
