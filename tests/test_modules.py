from dataclasses import replace
import unittest

import numpy as np

from action_generator import predict_action_chunk
from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.environment import MockEnvironment
from pipeline.mock_data import make_mock_data, make_policy_input


class ModuleTests(unittest.TestCase):
    def test_generator_matches_fixture_relations_and_groups(self):
        for case in ("blocked", "accessible", "after_move"):
            data = make_mock_data(case)
            graph = generate_scene_graph(data.observation)
            self.assertEqual(graph, data.graph)
        before = make_mock_data("blocked")
        after = make_mock_data("after_move")
        self.assertEqual({obj.object_id for obj in before.graph.nodes}, {obj.object_id for obj in after.graph.nodes})
        self.assertFalse(after.graph.edges)

    def test_generator_does_not_silently_use_hidden_gt(self):
        with self.assertRaises(ValueError):
            generate_scene_graph(replace(make_mock_data().observation, ground_truth=None))

    def test_planner_first_subgoal_and_unknown_target(self):
        for case in ("blocked", "accessible", "after_move"):
            data = make_mock_data(case)
            goal = plan_next_subgoal(data.graph, data.task, data.observation.robot_state)
            self.assertEqual((goal.primitive, goal.object_id), data.expected_order[0])
        data = make_mock_data("missing_target")
        with self.assertRaisesRegex(ValueError, "unknown object"):
            plan_next_subgoal(data.graph, data.task, data.observation.robot_state)

    def test_held_object_is_placed_and_occupied_buffer_is_rejected(self):
        data = make_mock_data()
        graph = replace(data.graph, held_object_id="A")
        goal = plan_next_subgoal(graph, data.task, data.observation.robot_state)
        self.assertEqual((goal.primitive, goal.object_id, goal.place_pose), ("PLACE", "A", data.task.buffer_pose))
        with self.assertRaises(ValueError):
            plan_next_subgoal(graph, replace(data.task, buffer_pose=graph.object("B").pose), data.observation.robot_state)

    def test_action_inputs_and_output_limits_for_pick_and_place(self):
        for primitive in ("PICK", "PLACE"):
            observation, goal = make_policy_input(primitive)
            self.assertEqual(observation.held_object_id, "A" if primitive == "PLACE" else None)
            chunk = predict_action_chunk(observation, goal)
            self.assertEqual(chunk.commands.shape, (16, 6))
            self.assertEqual(chunk.commands.dtype, np.float32)
            self.assertEqual(chunk.subgoal_id, goal.subgoal_id)
            previous = observation.robot_state
            for command in chunk.commands:
                self.assertTrue(np.all(np.abs(command - previous) <= 1 + 1e-6))
                previous = command

    def test_environment_rejects_invalid_commands(self):
        env = MockEnvironment()
        try:
            env.reset()
            for command in (np.full(6, 100., dtype=np.float32),
                            np.array([2., 0., 3., 0., 0., 0.], dtype=np.float32)):
                with self.assertRaises(ValueError):
                    env.step(command)
            self.assertEqual(env.observe().timestamp, 0.)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
