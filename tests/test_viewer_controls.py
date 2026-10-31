import os
import unittest
from unittest.mock import patch

import glfw
import mujoco
import numpy as np

from pipeline.mock_data import ROOT
from pipeline.viewer import JointJogControl, SimulationViewer


class JogControlTests(unittest.TestCase):
    def make_control(self, initial=None):
        initial = np.zeros(6, dtype=np.float32) if initial is None else initial
        return JointJogControl(initial, [-2]*5+[0], [2]*5+[1], [.06]*5+[.1])

    def test_holding_moves_continuously_without_additional_key_events(self):
        state = np.zeros(6, dtype=np.float32)
        control = self.make_control()
        for _ in range(30):
            state = control.command(state, 1/30, joint=0, direction=1)
        self.assertAlmostEqual(state[0], np.deg2rad(60), places=5)
        stopped = control.command(state, 1/30, joint=0, direction=0)
        np.testing.assert_allclose(stopped[:5], state[:5])

    def test_blocked_joint_does_not_accumulate_a_far_away_goal(self):
        state = np.zeros(6, dtype=np.float32)
        control = self.make_control()
        for _ in range(300):
            command = control.command(state, 1/30, joint=0, direction=1)
            self.assertLessEqual(command[0], .048 + 1e-6)
        self.assertAlmostEqual(control.command(state, 1/30, direction=0)[0], 0.)

    def test_selection_release_home_gripper_and_bounds(self):
        state = np.zeros(6, dtype=np.float32)
        control = self.make_control()
        state = control.command(state, 1/30, 0, 1)
        second = control.command(state, 1/30, 1, 1)
        self.assertEqual(second[0], state[0])
        self.assertGreater(second[1], 0.)
        state = second
        for _ in range(100):
            previous = state.copy()
            state = control.command(state, 1/30, 5, 1)
            self.assertTrue(np.all(np.abs(state-previous) <= control.max_delta + 1e-6))
        self.assertLessEqual(state[5], 1.)
        self.assertGreater(state[5], .99)
        control.home()
        back = control.command(state, 1/30)
        self.assertLess(back[5], state[5])

    def test_window_key_poll_handles_hold_release_focus_and_short_taps(self):
        viewer = SimulationViewer.__new__(SimulationViewer)
        viewer.window, viewer._pending_tap = object(), 0
        pressed = {glfw.KEY_UP: glfw.PRESS, glfw.KEY_DOWN: glfw.RELEASE}
        with patch('pipeline.viewer.glfw.poll_events'), patch('pipeline.viewer.glfw.get_window_attrib', return_value=True), \
             patch('pipeline.viewer.glfw.get_key', side_effect=lambda window, key: pressed[key]):
            viewer._key(viewer.window, glfw.KEY_UP, 0, glfw.PRESS, 0)
            for _ in range(60):
                self.assertEqual(viewer.poll_direction(), 1)
            pressed[glfw.KEY_UP] = glfw.RELEASE
            self.assertEqual(viewer.poll_direction(), 0)
            viewer._key(viewer.window, glfw.KEY_UP, 0, glfw.PRESS, 0)
            viewer._key(viewer.window, glfw.KEY_UP, 0, glfw.RELEASE, 0)
            self.assertEqual(viewer.poll_direction(), 1)
            self.assertEqual(viewer.poll_direction(), 0)
            pressed[glfw.KEY_UP] = pressed[glfw.KEY_DOWN] = glfw.PRESS
            self.assertEqual(viewer.poll_direction(), 0)
        with patch('pipeline.viewer.glfw.poll_events'), patch('pipeline.viewer.glfw.get_window_attrib', return_value=False):
            viewer._pending_tap = 1
            self.assertEqual(viewer.poll_direction(), 0)

    @unittest.skipUnless((ROOT / 'assets/so101/upstream/so101.xml').exists(), 'SO-101 assets required')
    def test_real_simulator_tracks_hold_and_stops_after_release(self):
        from pipeline.mujoco_env import MujocoEnvironment

        env = MujocoEnvironment(render=False)
        try:
            observation = env.reset(42)
            initial = observation.robot_state.copy()
            control = JointJogControl(initial, env.lower, env.upper, env.config['max_delta'])
            for _ in range(30):
                command = control.command(observation.robot_state, observation.control_dt, 0, 1)
                observation = env.step(command)
            self.assertGreater(observation.robot_state[0]-initial[0], .5)
            released = observation.robot_state.copy()
            for _ in range(5):
                observation = env.step(control.command(observation.robot_state, observation.control_dt, 0, 0))
            self.assertLess(abs(observation.robot_state[0]-released[0]), .03)
        finally:
            env.close()


@unittest.skipUnless(os.environ.get('HSSG_TEST_GUI') == '1', 'set HSSG_TEST_GUI=1 for native GUI checks')
class NativeViewerTests(unittest.TestCase):
    def test_window_is_visible_after_offscreen_renderer_initialization(self):
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=64, width=64)
        viewer = None
        try:
            renderer.update_scene(data)
            renderer.render()
            viewer = SimulationViewer(model, data)
            glfw.poll_events()
            self.assertEqual(glfw.get_window_attrib(viewer.window, glfw.VISIBLE), glfw.TRUE)
            viewer.close()
            viewer = None
            # Closing the visible window must leave RGB-D rendering usable.
            renderer.update_scene(data)
            self.assertEqual(renderer.render().shape, (64, 64, 3))
        finally:
            if viewer is not None:
                viewer.close()
            renderer.close()


if __name__ == '__main__':
    unittest.main()
