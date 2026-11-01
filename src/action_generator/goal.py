"""Versioned numerical goal shared by demonstration collection and ACT inference."""

import numpy as np

from pipeline.types import Pose, Subgoal

GOAL_VERSION = "so101-goal-v2"
GOAL_DIM = 25
GOAL_FIELDS = ("primitive_onehot[2]", "initial_object_xyz[3]", "initial_object_xyzw[4]",
               "place_xyz[3]", "place_xyzw[4]", "place_valid[1]",
               "object_in_ee_xyz[3]", "object_in_ee_xyzw[4]", "grasp_valid[1]")


def control_goal_error(goal, ee_pose: Pose) -> np.ndarray:
    """Observable Cartesian feedback feature, in 0.1m units; never an action."""
    current = np.array(ee_pose.position, dtype=float)
    place = goal[1] > .5
    if place:
        x, y, z, w = ee_pose.quaternion_xyzw
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        current += rotation @ goal[17:20]
    target = goal[9:12] if place else goal[2:5]
    return ((target-current)/.1).astype(np.float32)


def encode_goal(subgoal: Subgoal, ee_pose: Pose | None = None) -> np.ndarray:
    """Freeze object_pose at subgoal start; IDs and mock EE targets are not features.

    Position is robot_base/metres. Canonicalize quaternion signs because q and -q
    describe the same orientation. Missing PLACE pose is zero with valid=0.
    """
    def pose_values(pose):
        q = np.array(pose.quaternion_xyzw, dtype=np.float32)
        nonzero = np.flatnonzero(np.abs(q[::-1]) > 1e-7)
        if len(nonzero) and q[::-1][nonzero[0]] < 0:
            q = -q
        return [*pose.position, *q]

    place = pose_values(subgoal.place_pose) if subgoal.place_pose is not None else [0.]*7
    grasp = [0.]*8
    if subgoal.primitive == 'PLACE' and ee_pose is not None:
        x, y, z, w = ee_pose.quaternion_xyzw
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        relative_position = rotation.T@(np.array(subgoal.object_pose.position)-ee_pose.position)
        a, b = -np.array([x, y, z]), np.array(subgoal.object_pose.quaternion_xyzw[:3])
        bw = subgoal.object_pose.quaternion_xyzw[3]
        relative_quat = [*(w*b+bw*a+np.cross(a, b)), w*bw-np.dot(a, b)]
        grasp = [*pose_values(Pose(tuple(relative_position), tuple(relative_quat))), 1.]
    return np.array([float(subgoal.primitive == "PICK"), float(subgoal.primitive == "PLACE"),
                     *pose_values(subgoal.object_pose), *place,
                     float(subgoal.place_pose is not None), *grasp], dtype=np.float32)
