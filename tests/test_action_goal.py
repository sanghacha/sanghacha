from dataclasses import replace
import unittest

import numpy as np

from action_generator.goal import GOAL_DIM, encode_goal, control_goal_error
from pipeline.types import Pose, Subgoal


class GoalEncodingTests(unittest.TestCase):
    def test_cartesian_feedback_uses_current_tcp_and_frozen_grasp_offset(self):
        pick = Subgoal('pick', 'PICK', 'object', Pose((.2, .1, .03)), Pose((.2, .1, .1)))
        np.testing.assert_allclose(control_goal_error(encode_goal(pick), Pose((.1, .1, .03))), [1., 0., 0.], atol=1e-7)
        place = replace(pick, primitive='PLACE', place_pose=Pose((.11, .02, .1)), object_pose=Pose((.11, 0., .1)))
        encoded = encode_goal(place, Pose((.1, 0., .1)))
        rotated = Pose((.1, 0., .1), (0., 0., np.sqrt(.5), np.sqrt(.5)))
        np.testing.assert_allclose(control_goal_error(encoded, rotated), [.1, .1, 0.], atol=1e-7)

    def test_ids_and_mock_ee_targets_are_not_network_features(self):
        goal = Subgoal('one', 'PICK', 'track_007', Pose((.2, .1, .03)), Pose((0., 0., .5)))
        renamed = replace(goal, subgoal_id='two', object_id='unrelated', target_ee_pose=Pose((1., 1., 1.)))
        np.testing.assert_array_equal(encode_goal(goal), encode_goal(renamed))
        self.assertEqual(encode_goal(goal).shape, (GOAL_DIM,))
        self.assertEqual(encode_goal(goal).dtype, np.float32)
        np.testing.assert_array_equal(encode_goal(goal)[9:], np.zeros(GOAL_DIM-9))

    def test_place_destination_and_equivalent_quaternions(self):
        goal = Subgoal('p', 'PLACE', 'object', Pose((.2, 0., .03)), Pose((0., 0., .1)), Pose((.1, .2, .03)))
        encoded = encode_goal(goal)
        np.testing.assert_array_equal(encoded[:2], [0., 1.])
        self.assertEqual(encoded[16], 1.)
        self.assertFalse(np.array_equal(encoded, encode_goal(replace(goal, place_pose=Pose((.3, .2, .03))))))
        equivalent = replace(goal, object_pose=Pose(goal.object_pose.position, (0., 0., 0., -1.)))
        np.testing.assert_array_equal(encoded, encode_goal(equivalent))

    def test_place_includes_initial_grasp_relative_to_measured_ee(self):
        goal = Subgoal('place', 'PLACE', 'object', Pose((.2, 0., .03)), Pose((0., 0., .1)), Pose((.1, .2, .03)))
        encoded = encode_goal(goal, Pose((.2, 0., .05)))
        np.testing.assert_allclose(encoded[17:20], [0., 0., -.02], atol=1e-7)
        np.testing.assert_array_equal(encoded[20:25], [0., 0., 0., 1., 1.])
        rotated = replace(goal, object_pose=Pose((1., 0., 0.)))
        encoded = encode_goal(rotated, Pose((0., 0., 0.), (0., 0., np.sqrt(.5), np.sqrt(.5))))
        np.testing.assert_allclose(encoded[17:20], [0., -1., 0.], atol=1e-7)
        np.testing.assert_allclose(encoded[20:24], [0., 0., -np.sqrt(.5), np.sqrt(.5)], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
