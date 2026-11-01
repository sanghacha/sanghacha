"""uv run python scripts/run_pipeline.py"""

import argparse
from importlib import import_module
import os
from pathlib import Path
import time

from pipeline.environment import MockEnvironment
from pipeline.runner import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B/C orchestration in mock or MuJoCo; no physical robot")
    parser.add_argument("--case", choices=("shelves", "blocked", "accessible", "after_move", "missing_target"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("mock", "mujoco"), default="mock")
    parser.add_argument("--hssg-module", default="hssg_generator")
    parser.add_argument("--planner-module", default="planner")
    parser.add_argument("--action-module", default="action_generator")
    parser.add_argument('--controller', choices=('auto', 'expert', 'act', 'hold'), default='auto')
    parser.add_argument('--checkpoint', type=Path, help='ACT checkpoint; also accepts HSSG_ACT_CHECKPOINT')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--gui', action='store_true', help='show the autonomous MuJoCo rollout')
    parser.add_argument('--physics-only', action='store_true', help='expert CPU checks without RGB rendering')
    parser.add_argument('--snapshot', type=Path, help='save the final GUI view or camera RGB as a PNG')
    args = parser.parse_args()
    args.case = args.case or ("shelves" if args.backend == "mujoco" else "blocked")
    if args.snapshot and (args.backend != 'mujoco' or (args.physics_only and not args.gui)):
        parser.error('--snapshot requires rendered MuJoCo RGB or --gui')
    if args.backend == "mujoco" and args.case == "missing_target":
        parser.error("missing_target is an offline/mock error fixture; use shelves, blocked, accessible or after_move in MuJoCo")
    if args.backend == "mock" and args.case == "shelves":
        parser.error("shelves is a MuJoCo scene; select --backend mujoco")
    if args.backend == "mujoco":
        from pipeline.mujoco_env import MujocoEnvironment
        os.environ['HSSG_C_BACKEND'] = args.controller
        if args.checkpoint:
            os.environ['HSSG_ACT_CHECKPOINT'] = str(args.checkpoint.resolve())
        if args.device:
            os.environ['HSSG_ACT_DEVICE'] = args.device
        env = MujocoEnvironment(args.case, args.config, render=not args.physics_only) if args.config else MujocoEnvironment(
            args.case, render=not args.physics_only)
        if args.action_module == 'action_generator':
            selected = args.controller if args.controller != 'auto' else ('act' if os.environ.get('HSSG_ACT_CHECKPOINT') else 'expert')
            print('Module C:', selected, '(simulation only)')
    else:
        env = MockEnvironment(args.case, args.config) if args.config else MockEnvironment(args.case)
    viewer = None
    clock = time.perf_counter()
    def display(before, command, observation, subgoal):
        nonlocal viewer, clock
        if not args.gui or args.backend != 'mujoco':
            return
        import glfw
        from pipeline.viewer import JointJogControl, SimulationViewer
        if viewer is None:
            viewer = SimulationViewer(env.model, env.data, interactive=False)
        viewer.poll_direction()
        if not viewer.is_running():
            raise RuntimeError('rollout interrupted by window close')
        control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'])
        control.target[:] = command
        glfw.set_window_title(viewer.window, f'SO-101 autonomous: {subgoal.primitive}({subgoal.object_id})')
        viewer.sync(observation, control, env.joints)
        time.sleep(max(0., observation.control_dt-(time.perf_counter()-clock)))
        clock = time.perf_counter()
    try:
        subgoals = run_pipeline(env, env.task, seed=args.seed,
                               generate_graph=import_module(args.hssg_module).generate_scene_graph,
                               plan_subgoal=import_module(args.planner_module).plan_next_subgoal,
                               predict_action=import_module(args.action_module).predict_action_chunk,
                               on_step=display if args.gui else None)
        for goal in subgoals:
            print(f"{goal.primitive}({goal.object_id})")
        if args.snapshot:
            observation = env.observe()
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if viewer is not None:
                from pipeline.viewer import JointJogControl
                control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'])
                viewer.sync(observation, control, env.joints, screenshot=args.snapshot)
            else:
                from PIL import Image
                Image.fromarray(observation.rgb).save(args.snapshot)
        print(f"{args.backend} succeeded: {len(subgoals)} subgoals")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"{args.backend} failed: {exc}")
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
