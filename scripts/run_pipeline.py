"""uv run python scripts/run_pipeline.py"""

import argparse
import json
from importlib import import_module
import os
from pathlib import Path
import time
from uuid import uuid4

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
    parser.add_argument('--act-execute-steps', type=int)
    parser.add_argument('--act-temporal-ensemble', action='store_true')
    parser.add_argument('--gui', action='store_true', help='show the autonomous MuJoCo rollout')
    parser.add_argument('--physics-only', action='store_true', help='expert CPU checks without RGB rendering')
    parser.add_argument('--snapshot', type=Path, help='save the final GUI view or camera RGB as a PNG')
    parser.add_argument('--perception', choices=('gt', 'rgbd-color', 'rgbd-command'), default='gt')
    parser.add_argument('--target-spec', type=Path, help='JSON observable target attributes for RGB-D')
    parser.add_argument('--perception-command', type=Path, help='JSON argv; placeholders {inputs}, {output}')
    parser.add_argument('--perception-output', type=Path)
    parser.add_argument('--inspect', action='store_true', help='show scene graph, planner reason and Subgoal alongside simulation')
    parser.add_argument('--inspect-paused', action='store_true', help='start paused at the first Subgoal')
    parser.add_argument('--inspect-no-hold', action='store_true', help='close inspector automatically after completion (for capture/tests)')
    args = parser.parse_args()
    args.case = args.case or ("shelves" if args.backend == "mujoco" else "blocked")
    if args.inspect:
        if args.backend != 'mujoco':
            parser.error('--inspect requires --backend mujoco')
        args.gui = True
    elif args.inspect_paused or args.inspect_no_hold:
        parser.error('--inspect-paused/--inspect-no-hold require --inspect')
    if args.perception != 'gt':
        if args.backend != 'mujoco' or args.physics_only or not args.target_spec:
            parser.error('RGB-D requires --backend mujoco, rendered RGB, and --target-spec')
        if args.hssg_module != 'hssg_generator':
            parser.error('RGB-D uses its graph adapter; omit --hssg-module')
        if args.perception == 'rgbd-command' and not args.perception_command:
            parser.error('rgbd-command requires --perception-command')
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
        if args.act_execute_steps is not None:
            os.environ['HSSG_ACT_EXECUTE_STEPS'] = str(args.act_execute_steps)
        if args.act_temporal_ensemble:
            os.environ['HSSG_ACT_TEMPORAL_ENSEMBLE'] = '1'
        env = MujocoEnvironment(args.case, args.config, render=not args.physics_only) if args.config else MujocoEnvironment(
            args.case, render=not args.physics_only)
        if args.perception != 'gt':
            from pipeline.rgbd_env import RGBDEnvironment
            from pipeline.types import TargetSpec
            try:
                spec = TargetSpec(**json.loads(args.target_spec.read_text(encoding='utf-8')))
                command = json.loads(args.perception_command.read_text(encoding='utf-8')) if args.perception_command else None
                output = args.perception_output or Path('outputs/rgbd_pipeline')/args.case/f'seed_{args.seed}_{uuid4().hex[:8]}'
                env = RGBDEnvironment(env, spec, output=output,
                                      method='color' if args.perception == 'rgbd-color' else 'command',
                                      inference_command=command)
            except Exception:
                env.close()
                raise
            print('Perception:', args.perception, '| output:', output)
        if args.action_module == 'action_generator':
            selected = args.controller if args.controller != 'auto' else ('act' if os.environ.get('HSSG_ACT_CHECKPOINT') else 'expert')
            print('Module C:', selected, '(simulation only)')
    else:
        env = MockEnvironment(args.case, args.config) if args.config else MockEnvironment(args.case)
    viewer = None
    inspector = None
    if args.inspect:
        from pipeline.inspector import PipelineInspector
        label = (args.controller if args.controller != 'auto' else
                 ('act' if os.environ.get('HSSG_ACT_CHECKPOINT') else 'expert'))
        inspector = PipelineInspector(controller=label if args.action_module == 'action_generator' else args.action_module,
                                      paused=args.inspect_paused)
        if args.perception != 'gt':
            env.on_scene = inspector.observe_scene
        print('Inspector: SPACE pause/resume | N run to next Subgoal | click graph node | ESC close', flush=True)
    clock = time.perf_counter()
    def show(observation, command=None, screenshot=None):
        nonlocal viewer
        import glfw
        from pipeline.viewer import JointJogControl, SimulationViewer
        if viewer is None:
            viewer = SimulationViewer(env.model, env.data, interactive=False, inspector=inspector)
        viewer.poll_direction()
        if not viewer.is_running():
            raise RuntimeError('rollout interrupted by window close')
        control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'])
        control.target[:] = command if command is not None else observation.robot_state
        goal = inspector.goal if inspector else None
        if goal:
            glfw.set_window_title(viewer.window, f'SO-101: {goal.primitive}({goal.object_id}) | pipeline inspector')
        viewer.sync(observation, control, env.joints, screenshot=screenshot)

    def wait_paused(observation):
        nonlocal clock
        waited = inspector is not None and inspector.paused
        while inspector is not None and inspector.paused and not inspector.finished and not inspector.error:
            show(observation)
            time.sleep(.025)
        if waited:
            clock = time.perf_counter()

    def event(kind, **payload):
        if inspector is None:
            return
        if kind == 'scene':
            payload['raw_graph'] = env.last_graph if args.perception != 'gt' else None
        if kind == 'planned':
            explain = getattr(import_module(args.planner_module), 'explain_subgoal', None)
            payload['explanation'] = explain(payload['graph'], payload['task'], payload['subgoal']) if explain else None
        inspector.on_event(kind, **payload)
        if (kind in ('scene', 'planned', 'completed', 'finished', 'failed') and inspector.observation is not None
                and (viewer is None or viewer.is_running())):
            show(inspector.observation)
        if kind == 'planned':
            wait_paused(payload['observation'])

    def inspect_end():
        if inspector is None or inspector.observation is None:
            return
        if viewer is not None and viewer.is_running():
            show(env.observe(), screenshot=args.snapshot)
        while not args.inspect_no_hold and viewer is not None and viewer.is_running():
            viewer.poll_direction()
            if viewer.is_running():
                show(env.observe())
            time.sleep(.03)

    def display(before, command, observation, subgoal):
        nonlocal viewer, clock
        if not args.gui or args.backend != 'mujoco':
            return
        if inspector is not None:
            inspector.on_step(before, command, observation, subgoal)
            show(observation, command)
            wait_paused(observation)
            time.sleep(max(0., observation.control_dt-(time.perf_counter()-clock)))
            clock = time.perf_counter()
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
                               generate_graph=(env.generate_scene_graph if args.perception != 'gt' else
                                               import_module(args.hssg_module).generate_scene_graph),
                               plan_subgoal=import_module(args.planner_module).plan_next_subgoal,
                               predict_action=import_module(args.action_module).predict_action_chunk,
                               on_step=display if args.gui else None,
                               refresh_observation=env.refresh if args.perception != 'gt' else None,
                               on_event=event if inspector is not None else None)
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
        inspect_end()
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"{args.backend} failed: {exc}")
        inspect_end()
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
