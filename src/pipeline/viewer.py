"""시뮬레이션 창 안에서만 키 누름/해제를 처리하는 수동 관절 조작 뷰어."""

from pathlib import Path

import glfw
import mujoco
import numpy as np
from PIL import Image

from .types import check


class JointJogControl:
    def __init__(self, initial, lower, upper, max_delta, *, joint_speed=60., gripper_speed=.8):
        check(np.isfinite(joint_speed) and joint_speed > 0 and np.isfinite(gripper_speed) and gripper_speed > 0,
              "jog speeds must be positive")
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.max_delta = np.asarray(max_delta)*.8
        self.joint_speed, self.gripper_speed = np.deg2rad(joint_speed), gripper_speed
        self.reset(initial)

    def reset(self, initial):
        self.initial = np.array(initial, dtype=np.float32)
        self.target = self.initial.copy()
        self._moving_joint = None

    def home(self):
        self.target[:] = self.initial
        self._moving_joint = None

    def command(self, current, dt, joint=0, direction=0):
        check(0 <= joint < 6 and direction in (-1, 0, 1), "invalid jog input")
        check(np.isfinite(dt) and dt > 0, "invalid control dt")
        # 키 해제/관절 변경 시 미리 쌓인 목표를 버리고 현재 위치에서 유지한다.
        if self._moving_joint is not None and (direction == 0 or joint != self._moving_joint):
            self.target[self._moving_joint] = current[self._moving_joint]
            self._moving_joint = None
        if direction:
            if self._moving_joint is None:
                self.target[joint] = current[joint]
            speed = self.gripper_speed if joint == 5 else self.joint_speed
            self.target[joint] += direction*speed*dt
            # 충돌 등으로 움직이지 못할 때 목표 각도가 계속 누적되지 않도록 한다.
            self.target[joint] = np.clip(self.target[joint], current[joint]-self.max_delta[joint],
                                         current[joint]+self.max_delta[joint])
            self._moving_joint = joint
        self.target[:] = np.clip(self.target, self.lower+1e-6, self.upper-1e-6)
        command = current + np.clip(self.target-current, -self.max_delta, self.max_delta)
        return np.clip(command, self.lower+1e-6, self.upper-1e-6).astype(np.float32)


class SimulationViewer:
    """MuJoCo 렌더러 + GLFW 키 상태. OS 전역 키보드 훅은 사용하지 않는다."""

    def __init__(self, model, data, *, interactive=True, inspector=None):
        self.interactive = interactive
        self.inspector = inspector
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        # MuJoCo's offscreen renderer leaves GLFW's global VISIBLE hint disabled.
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(1600 if inspector else 1100, 960 if inspector else 800,
                                         "SO-101 pipeline inspector" if inspector else "SO-101 - hold Up/Down to move", None, None)
        if not self.window:
            raise RuntimeError("Cannot create MuJoCo OpenGL window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.model, self.data = model, data
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.cam.lookat[:] = [.17, 0, .12]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1., 135, -35
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.joint = 0
        self.reset_requested = self.home_requested = False
        self._pending_tap = 0
        self._mouse = None
        self._status = ("", "")
        glfw.set_key_callback(self.window, self._key)
        glfw.set_window_focus_callback(self.window, self._focus)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_scroll_callback(self.window, self._scroll)
        if inspector is not None:
            glfw.set_mouse_button_callback(self.window, self._click)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        inspector = getattr(self, 'inspector', None)
        if inspector is not None and key in (glfw.KEY_SPACE, glfw.KEY_N):
            inspector.handle_key('space' if key == glfw.KEY_SPACE else 'next')
            return
        if not getattr(self, 'interactive', True) and key not in (*range(glfw.KEY_1, glfw.KEY_6+1), glfw.KEY_ESCAPE):
            return
        if glfw.KEY_1 <= key <= glfw.KEY_6:
            self.joint = key-glfw.KEY_1
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self._pending_tap = 1 if key == glfw.KEY_UP else -1
        elif key == glfw.KEY_SPACE:
            self.home_requested = True
        elif key == glfw.KEY_R:
            self.reset_requested = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def _focus(self, window, focused):
        if not focused:
            self._pending_tap = 0

    def poll_direction(self):
        glfw.poll_events()
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            self._pending_tap = 0
            return 0
        up = glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS
        down = glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS
        direction = int(up) - int(down)
        pending, self._pending_tap = self._pending_tap, 0
        if up and down:
            return 0
        return direction or pending

    def _mouse_move(self, window, x, y):
        previous, self._mouse = self._mouse, (x, y)
        if previous is None:
            return
        if self.inspector is not None:
            w = glfw.get_window_size(window)[0]
            if x >= w*.625:
                return
        height = max(1, glfw.get_window_size(window)[1])
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, (x-previous[0])/height, (y-previous[1])/height, self.cam)

    def _click(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        width, height = glfw.get_window_size(window)
        left = width*.625
        if x >= left and height > 0:
            self.inspector.select_at((x-left)/(width-left)*600, y/height*960)

    def _markers(self, observation):
        """Decorations use the public observation/graph only, never body IDs."""
        inspector = self.inspector
        if inspector is None:
            return
        objects = observation.perceived_objects if observation.source == 'rgbd' else (observation.ground_truth or ())
        for obj in objects:
            if self.scene.ngeom >= self.scene.maxgeom:
                break
            selected = inspector.goal is not None and obj.object_id == inspector.goal.object_id
            target = inspector.task is not None and obj.object_id == inspector.task.target_object_id
            focused = selected or target or obj.object_id == inspector.selected
            related = obj.object_id in inspector.explanation.get('chain', [])
            geom = self.scene.geoms[self.scene.ngeom]
            color = np.array([1., .8, .25, 1.] if target else [.2, .95, .9, 1.] if focused else [.6, .7, .85, .7], dtype=np.float32)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.0035]*3),
                               np.array(obj.pose.position)+[0, 0, obj.size[2]/2+.006], np.eye(3).ravel(), color)
            from .inspector import short
            geom.label = short(obj.object_id) if focused or related else ''
            self.scene.ngeom += 1
        goal = inspector.goal
        if goal is not None and goal.place_pose is not None and self.scene.ngeom < self.scene.maxgeom:
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.006]*3),
                               np.array(goal.place_pose.position), np.eye(3).ravel(), np.array([.2, 1., .8, .8], np.float32))
            geom.label = 'PLACE goal'
            self.scene.ngeom += 1

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, self.cam)

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def sync(self, observation, control, joint_names, screenshot: Path | None = None):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        scene_viewport = mujoco.MjrRect(0, 0, int(width*.625), height) if self.inspector else viewport
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._markers(observation)
        mujoco.mjr_render(scene_viewport, self.scene, self.context)
        j = self.joint
        value, target, lower, upper = (observation.robot_state[j], control.target[j], control.lower[j], control.upper[j])
        if j < 5:
            value, target, lower, upper = np.rad2deg([value, target, lower, upper])
            unit = "deg"
        else:
            value, target, lower, upper = np.array([value, target, lower, upper])*100
            unit = "% closed"
        self._status = (f"Joint {j+1}: {joint_names[j]}\nCurrent / target\nRange\nControls\nScene\nCamera",
                        f"Hold UP / DOWN to move\n{value:+.1f} / {target:+.1f} {unit}\n{lower:.1f} .. {upper:.1f} {unit}\n1..6 select | SPACE home | ESC close\nR: new seeded layout\nLeft drag rotate | Right drag pan | Wheel zoom")
        if not self.interactive:
            self._status = (self._status[0],
                            f"Executing task\n{value:+.1f} / {target:+.1f} {unit}\n{lower:.1f} .. {upper:.1f} {unit}\n"
                            "1..6 inspect joint | ESC stop\nSee window title for current task\n"
                            "Left drag rotate | Right drag pan | Wheel zoom")
        if self.inspector is not None:
            goal = self.inspector.goal
            from .inspector import short
            self._status = (f'SO-101 | {self.inspector.controller.upper()}\nSubgoal\nJoint {j+1}\nCamera',
                            f'Pipeline inspector\n{goal.primitive}({short(goal.object_id)})' if goal else 'Pipeline inspector\nObserving')
            self._status = (self._status[0], self._status[1]+f'\n{value:+.1f} / {target:+.1f} {unit}\nDrag rotate/pan | Wheel zoom')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           scene_viewport, *self._status, self.context)
        if self.inspector is not None:
            if self.inspector.camera_image is not None:
                thumb_width = min(420, int(scene_viewport.width*.42))
                thumb_height = int(180*thumb_width/420)
                thumb = self.inspector.camera_image.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
                rect = mujoco.MjrRect(scene_viewport.width-thumb_width-12, height-thumb_height-12, thumb_width, thumb_height)
                mujoco.mjr_drawPixels(np.ascontiguousarray(np.flipud(thumb)).reshape(-1), None, rect, self.context)
            side = mujoco.MjrRect(scene_viewport.width, 0, width-scene_viewport.width, height)
            pixels = np.ascontiguousarray(np.flipud(self.inspector.render(side.width, height)))
            mujoco.mjr_drawPixels(pixels.reshape(-1), None, side, self.context)
        if screenshot is not None:
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, viewport, self.context)
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.flipud(pixels)).save(screenshot)
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None
