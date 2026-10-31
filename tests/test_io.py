from dataclasses import replace
import unittest

import numpy as np

from pipeline.mock_data import make_mock_data
from pipeline.types import ActionChunk, Edge, Pose, wxyz_to_xyzw


class InputOutputTests(unittest.TestCase):
    def test_observation_rejects_bad_dtype_shape_frame_and_nan(self):
        observation = make_mock_data().observation
        for changes in (
            {"rgb": observation.rgb.astype(np.float32)},
            {"rgb": np.zeros((3, 16, 16), dtype=np.uint8)},
            {"robot_state": np.full(6, np.nan, dtype=np.float32)},
            {"frame_id": "camera"},
            {"depth": np.zeros((2, 2), dtype=np.float32)},
            {"source": "rgbd"},  # GT를 노출한 채 RGB-D 관측이라고 표시할 수 없음.
        ):
            with self.assertRaises(ValueError):
                replace(observation, **changes)

    def test_quaternion_order_calibration_and_readonly_arrays(self):
        self.assertEqual(wxyz_to_xyzw((.5, .5, .5, .5)), (.5, .5, .5, .5))
        self.assertEqual(wxyz_to_xyzw((1., 0., 0., 0.)), (0., 0., 0., 1.))
        with self.assertRaises(ValueError):
            Pose((0., 0., 0.), (0., 0., 0., 0.))
        observation = make_mock_data().observation
        with self.assertRaises(ValueError):
            replace(observation, camera_to_robot_base=np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            observation.robot_state[0] = 1

    def test_graph_rejects_duplicate_ids_dangling_edges_and_cycles(self):
        graph = make_mock_data().graph
        for changes in (
            {"nodes": graph.nodes + (graph.nodes[0],)},
            {"edges": (Edge("unknown", "B", "supports"),)},
            {"groups": {"workspace": "bin", "bin": "workspace"}},
        ):
            with self.assertRaises(ValueError):
                replace(graph, **changes)

    def test_subgoal_place_contract_and_action_shape(self):
        from pipeline.mock_data import make_policy_input

        _, goal = make_policy_input()
        with self.assertRaises(ValueError):
            replace(goal, primitive="PLACE")
        for commands in (np.zeros(6, dtype=np.float32), np.zeros((0, 6), dtype=np.float32),
                         np.full((1, 6), np.inf, dtype=np.float32)):
            with self.assertRaises(ValueError):
                ActionChunk("goal", commands)


if __name__ == "__main__":
    unittest.main()
