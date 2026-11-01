"""Collect physics-verified keyboard demonstrations; no hardware connection."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from uuid import uuid4

import glfw
import mujoco

from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.types import Pose, TaskRequest, check
from pipeline.viewer import JointJogControl, SimulationViewer
from .data import SegmentWriter


class RecordingViewer(SimulationViewer):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.paused = True

    def _key(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ENTER and action == glfw.PRESS:
            self.paused = not self.paused
        else:
            super()._key(window, key, scancode, action, mods)


def scene_group(env, seed):
    scene = {key: value for key, value in env.config.items()
             if key not in ('max_steps_per_subgoal', 'max_subgoals', 'n_action_steps', 'max_delta')}
    digest = sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()[:16]
    return f'{env.case}:{digest}:seed_{seed}'


def provenance(env, seed):
    return {'case': env.case, 'seed': seed, 'config': env.config,
            'model_source': json.loads((ROOT/'assets/so101/source.json').read_text(encoding='utf-8')),
            'mujoco_version': mujoco.__version__, 'collector': 'keyboard-joint-jog',
            'collection_id': uuid4().hex}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'training/data/demos')
    parser.add_argument('--case', choices=('accessible', 'blocked', 'after_move', 'shelves'), default='accessible')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', help='GT object ID to select a different training target; metadata only')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--max-steps', type=int, default=9000, help='per-subgoal control ticks, including idle time')
    parser.add_argument('--joint-speed', type=float, default=60.)
    parser.add_argument('--gripper-speed', type=float, default=.8)
    args = parser.parse_args()
    check(args.seed >= 0 and args.max_steps > 0, 'invalid seed or step budget')
    env = MujocoEnvironment(args.case, args.config) if args.config else MujocoEnvironment(args.case)
    viewer, writer = None, None
    try:
        seed = args.seed
        observation = env.reset(seed)
        task = env.task
        if args.target:
            check(args.target in env.object_definitions, 'unknown target ID')
            z = env.config['placement_trays']['floor_thickness']+env.object_definitions[args.target]['size'][2]/2
            def placement(pose):
                return Pose((*pose.position[:2], z), pose.quaternion_xyzw)
            task = TaskRequest(args.target, placement(task.destination), placement(task.buffer_pose))
        viewer = RecordingViewer(env.model, env.data)
        control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'],
                                  joint_speed=args.joint_speed, gripper_speed=args.gripper_speed)
        print('Collecting GT-conditioned MuJoCo demonstrations. No hardware connected.', flush=True)
        print('Enter: start/pause | 1..6: select joint | hold Up/Down: move | Space: home | R: discard/new seed | Esc: stop', flush=True)
        while viewer.is_running() and not env.task_complete(task):
            subgoal = plan_next_subgoal(generate_scene_graph(observation), task, observation.robot_state)
            color = env.object_definitions[subgoal.object_id]['color'][:3]
            print(f'{subgoal.primitive}({subgoal.object_id}) xyz={subgoal.object_pose.position} RGB={color}', flush=True)
            writer = SegmentWriter(args.output, observation, subgoal, group_id=scene_group(env, seed),
                                   lower=env.lower, upper=env.upper, max_delta=env.config['max_delta'],
                                   provenance=provenance(env, seed))
            confirmed = 0
            reset = False
            step = 0
            while step < args.max_steps:
                tick_start = time.perf_counter()
                direction = viewer.poll_direction()
                if not viewer.is_running():
                    break
                if viewer.reset_requested:
                    reset = True
                    viewer.reset_requested = False
                    break
                if viewer.home_requested:
                    control.home()
                    viewer.home_requested = False
                status = 'PAUSED - Enter starts' if viewer.paused else 'RECORDING - Enter pauses'
                glfw.set_window_title(viewer.window, f'{status} | {subgoal.primitive}({subgoal.object_id}) | seed {seed}')
                if viewer.paused:
                    control.command(observation.robot_state, observation.control_dt, viewer.joint, 0)
                    viewer.sync(observation, control, env.joints)
                    time.sleep(.02)
                    continue
                command = control.command(observation.robot_state, observation.control_dt, viewer.joint, direction)
                before = observation
                observation = env.step(command)
                writer.append(before, command)  # only label commands that actually executed
                step += 1
                viewer.sync(observation, control, env.joints)
                confirmed = confirmed+1 if env.subgoal_complete(subgoal) else 0
                if confirmed >= 3:
                    path = writer.finish(success=True, reason='physical_completion_confirmed_3_ticks')
                    writer = None
                    print('Saved successful segment:', path, flush=True)
                    break
                time.sleep(max(0., tick_start+observation.control_dt-time.perf_counter()))
            if writer is not None:
                path = writer.finish(success=False, reason='reset' if reset else 'interrupted_or_step_budget')
                writer = None
                print('Saved excluded failed segment:', path, flush=True)
                if not reset:
                    break
            if reset:
                seed += 1
                observation = env.reset(seed)
                control.reset(observation.robot_state)
                viewer.paused = True
            else:
                # Stop integrating the last held key across the subgoal boundary.
                control.reset(observation.robot_state)
        print('Task complete:', env.task_complete(task), flush=True)
    finally:
        try:
            if writer is not None and not writer.finished:
                writer.finish(success=False, reason='exception_or_keyboard_interrupt')
        finally:
            if viewer is not None:
                viewer.close()
            env.close()


if __name__ == '__main__':
    main()
