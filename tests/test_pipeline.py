from dataclasses import replace
import unittest

import numpy as np

from action_generator import predict_action_chunk
from hssg_generator import generate_scene_graph
from pipeline.environment import MockEnvironment
from pipeline.mock_data import make_mock_data
from pipeline.runner import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_complete_sequence_replans_after_each_action_and_resets_between_runs(self):
        env = MockEnvironment()
        try:
            for n in (1, 4, 20):
                env.config["n_action_steps"] = n
                completed = run_pipeline(env, env.task, seed=n)
                self.assertEqual(tuple((g.primitive, g.object_id) for g in completed), make_mock_data().expected_order)
                self.assertTrue(env.task_complete(env.task))
                self.assertTrue(env.stopped)
        finally:
            env.close()

    def test_accessible_cases_skip_rearrangement_and_missing_target_stops(self):
        for case in ("accessible", "after_move", "missing_target"):
            env = MockEnvironment(case)
            try:
                if case == "missing_target":
                    with self.assertRaisesRegex(ValueError, "unknown target"):
                        run_pipeline(env, env.task)
                else:
                    self.assertEqual(len(run_pipeline(env, env.task)), 2)
                self.assertTrue(env.stopped)
            finally:
                env.close()

    def test_chunk_exhaustion_is_not_subgoal_success(self):
        def no_motion(observation, goal):
            return replace(predict_action_chunk(observation, goal), commands=observation.robot_state[None, :])

        env = MockEnvironment()
        env.config["max_steps_per_subgoal"] = 3
        try:
            with self.assertRaisesRegex(ValueError, "step budget"):
                run_pipeline(env, env.task, predict_action=no_motion)
            self.assertTrue(env.stopped)
            self.assertAlmostEqual(env.observe().timestamp, 3 / 30)
        finally:
            env.close()

    def test_stale_graph_and_action_metadata_are_rejected_before_movement(self):
        def stale_graph(observation):
            return replace(generate_scene_graph(observation), episode_id="other")

        mutations = (
            lambda chunk: replace(chunk, subgoal_id="old-goal"),
            lambda chunk: replace(chunk, robot_id="other-robot"),
            lambda chunk: replace(chunk, command_dt=.1),
            lambda chunk: replace(chunk, commands=np.full((1, 6), 100., dtype=np.float32)),
        )
        for mutation in mutations:
            def bad_action(observation, goal):
                return mutation(predict_action_chunk(observation, goal))

            env = MockEnvironment()
            try:
                with self.assertRaises(ValueError):
                    run_pipeline(env, env.task, predict_action=bad_action)
                self.assertTrue(env.stopped)
                self.assertEqual(env.observe().timestamp, 0.)
            finally:
                env.close()
        env = MockEnvironment()
        try:
            with self.assertRaises(ValueError):
                run_pipeline(env, env.task, generate_graph=stale_graph)
            self.assertTrue(env.stopped)
        finally:
            env.close()

    def test_subgoal_budget_stops_after_first_pick(self):
        env = MockEnvironment()
        env.config["max_subgoals"] = 1
        try:
            with self.assertRaisesRegex(ValueError, "count budget"):
                run_pipeline(env, env.task)
            self.assertTrue(env.stopped)
            self.assertEqual(env.observe().held_object_id, "A")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
