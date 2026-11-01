"""SO-101 kinematics on a private robot model; never edits the live environment."""

import mujoco
import numpy as np

from pipeline.mock_data import ROOT
from pipeline.types import Pose, check, wxyz_to_xyzw

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


class SO101Kinematics:
    def __init__(self):
        path = ROOT/'assets/so101/upstream/so101.xml'
        check(path.exists(), 'run scripts/setup_assets.py first')
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.dofs = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.raw_limits = np.array([self.model.actuator(j).ctrlrange for j in JOINTS])
        self.lower, self.upper = self.raw_limits[:, 0].copy(), self.raw_limits[:, 1].copy()
        self.lower[5], self.upper[5] = 0., 1.
        self.site = self.model.site('gripperframe').id

    def forward(self, joints):
        raw = np.array(joints, dtype=float)
        raw[5] = self.raw_limits[5, 1]-raw[5]*np.ptp(self.raw_limits[5])
        self.data.qpos[self.qadr] = raw
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, position, rotation, initial, *, max_iterations=150, orientation_weight=.08):
        q = np.array(initial, dtype=float)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        best, best_error = q.copy(), float('inf')
        desired_quat = np.empty(4)
        mujoco.mju_mat2Quat(desired_quat, np.asarray(rotation).ravel())
        last_cost = float('inf')
        for iteration in range(max_iterations):
            p, r = self.forward(q)
            ep = np.array(position)-p
            current_quat, inverse, difference, er = np.empty(4), np.empty(4), np.empty(4), np.empty(3)
            mujoco.mju_mat2Quat(current_quat, r.ravel())
            mujoco.mju_negQuat(inverse, current_quat)
            mujoco.mju_mulQuat(difference, desired_quat, inverse)
            mujoco.mju_quat2Vel(er, difference, 1.)
            error = np.r_[ep, orientation_weight*er]
            cost = np.linalg.norm(ep)+orientation_weight*np.linalg.norm(er)
            if cost < best_error:
                best, best_error = q.copy(), cost
            if iteration > 12 and abs(cost-last_cost) < 1e-8:
                break
            last_cost = cost
            if np.linalg.norm(ep) < .0003 and np.linalg.norm(er) < .008:
                break
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack((jp[:, self.dofs[:5]], orientation_weight*jr[:, self.dofs[:5]]))
            step = jac.T@np.linalg.solve(jac@jac.T + .00001*np.eye(6), error)
            q[:5] = np.clip(q[:5]+np.clip(step, -.15, .15), self.lower[:5]+1e-4, self.upper[:5]-1e-4)
        p, r = self.forward(best)
        return best, float(np.linalg.norm(p-position)), float(np.linalg.norm(r-rotation))

    def pose(self, joints):
        position, rotation = self.forward(joints)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        return Pose(tuple(position), wxyz_to_xyzw(tuple(quaternion)))


def top_rotation(yaw):
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.])
    down = np.array([0., 0., -1.])
    return np.column_stack((down, np.cross(closing, down), closing))


def tilted_rotation(position, tilt, flip=False):
    radial = np.array([position[0]-.0084, position[1], 0.])
    radial /= np.linalg.norm(radial)
    down = np.array([0., 0., -1.])
    approach = np.cos(tilt)*down+np.sin(tilt)*radial
    closing = np.cross(radial, down)*(-1. if flip else 1.)
    return np.column_stack((approach, np.cross(closing, approach), closing))
