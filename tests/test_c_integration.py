from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from action_generator import predict_action_chunk
from action_generator.expert import SO101Expert
from action_generator.generator import _load_expert
from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.runner import run_pipeline
from pipeline.types import Pose


@unittest.skipUnless((ROOT/'assets/so101/upstream/so101.xml').exists(), 'run scripts/setup_assets.py first')
class ModuleCIntegrationTests(unittest.TestCase):
    def test_default_module_c_executes_physical_blocker_removal_and_retrieval(self):
        env = MujocoEnvironment('blocked', render=False)
        self.addCleanup(env.close)
        _load_expert.cache_clear()
        completions = []
        def observe(before, command, after, subgoal):
            self.assertEqual(command.dtype, np.float32)
            if env.subgoal_complete(subgoal):
                completions.append((subgoal.primitive, subgoal.object_id, after.held_object_id))
                if subgoal.primitive == 'PICK':
                    self.assertGreater(len([c for c in after.contacts if subgoal.object_id in (c.entity_a, c.entity_b)]), 1)
        with patch.dict('os.environ', {'HSSG_C_BACKEND': 'expert'}):
            goals = run_pipeline(env, env.task, seed=0, on_step=observe)
        self.assertEqual([(g.primitive, g.object_id) for g in goals], [('PICK', 'A'), ('PLACE', 'A'), ('PICK', 'B'), ('PLACE', 'B')])
        self.assertEqual(completions, [('PICK', 'A', 'A'), ('PLACE', 'A', None), ('PICK', 'B', 'B'), ('PLACE', 'B', None)])
        self.assertTrue(env.task_complete(env.task))
        self.assertTrue(env.stopped)

    def test_default_shelf_scene_retrieves_target_on_multiple_seeds(self):
        for seed in (0, 2, 7, 42):
            with self.subTest(seed=seed):
                env = MujocoEnvironment('shelves', render=False)
                try:
                    policy = SO101Expert()
                    goals = run_pipeline(env, env.task, seed=seed, predict_action=policy.predict_action_chunk)
                    self.assertGreaterEqual(len(goals), 4)
                    if seed == 2:
                        self.assertEqual(len(goals), 6)  # two blockers, two buffer slots, placement correction
                    self.assertTrue(env.task_complete(env.task))
                    self.assertEqual(goals[-1].object_id, env.task.target_object_id)
                finally:
                    env.close()

    def test_kinematics_never_edits_live_scene_and_unknown_target_fails(self):
        env = MujocoEnvironment('accessible', render=False)
        self.addCleanup(env.close)
        observation = env.reset(0)
        subgoal = plan_next_subgoal(generate_scene_graph(observation), env.task, observation.robot_state)
        policy = SO101Expert()
        state = env.data.qpos.copy()
        policy.predict_action_chunk(observation, subgoal)
        np.testing.assert_array_equal(env.data.qpos, state)
        with self.assertRaisesRegex(ValueError, 'cannot localize'):
            policy.predict_action_chunk(observation, replace(subgoal, object_id='absent', subgoal_id='invalid'))

    def test_unreachable_goal_raises_without_advancing_physics(self):
        env = MujocoEnvironment('accessible', render=False)
        self.addCleanup(env.close)
        observation = env.reset(0)
        subgoal = plan_next_subgoal(generate_scene_graph(observation), env.task, observation.robot_state)
        unreachable = Pose((.9, 0., .02))
        objects = tuple(replace(obj, pose=unreachable) if obj.object_id == subgoal.object_id else obj
                        for obj in observation.ground_truth)
        bad_observation = replace(observation, ground_truth=objects)
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            SO101Expert().predict_action_chunk(bad_observation, replace(subgoal, object_pose=unreachable))
        self.assertEqual(env.observe().timestamp, 0.)


if __name__ == '__main__':
    unittest.main()
