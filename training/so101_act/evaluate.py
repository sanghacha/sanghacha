"""Bounded held-out-seed MuJoCo rollouts; loss and chunk exhaustion are not success."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.runner import run_pipeline
from pipeline.types import Pose, TargetSpec, TaskRequest, check
from pipeline.viewer import JointJogControl, SimulationViewer
from .collect import scene_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('act', 'expert'), default='act')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--case', choices=('accessible', 'blocked', 'after_move', 'shelves'), default='accessible')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--target', help='select another simulation target for goal-conditioning evaluation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102])
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--physics-only', action='store_true', help='expert checks without RGB/OpenGL')
    parser.add_argument('--max-steps', type=int, default=600, help='control ticks per subgoal')
    parser.add_argument('--allow-training-seeds', action='store_true', help='debug only; not a held-out evaluation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--perception', choices=('gt', 'rgbd-color'), default='gt')
    parser.add_argument('--target-spec', type=Path)
    parser.add_argument('--execute-steps', type=int, help='replan after this many ACT chunk commands')
    parser.add_argument('--temporal-ensemble', action='store_true')
    args = parser.parse_args()
    check(args.max_steps > 0 and all(s >= 0 for s in args.seeds), 'invalid seed or rollout budget')
    if args.perception != 'gt' and (not args.target_spec or args.target or args.physics_only):
        parser.error('RGB-D evaluation requires --target-spec, RGB and appearance targets (omit --target)')
    spec = TargetSpec(**json.loads(args.target_spec.read_text(encoding='utf-8'))) if args.target_spec else None
    used_groups = set()
    if args.controller == 'act':
        if args.checkpoint is None or args.physics_only:
            parser.error('ACT evaluation requires --checkpoint and RGB rendering (no --physics-only)')
        import torch
        from .policy import ACTPolicy
        policy = ACTPolicy(args.checkpoint, device=args.device, execute_steps=args.execute_steps,
                           temporal_ensemble=args.temporal_ensemble)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        used_groups = {s['group_id'] for part in payload['split'].values() for s in part}
    else:
        from action_generator.expert import SO101Expert
        policy = SO101Expert()
    results = []
    for seed in args.seeds:
        env = MujocoEnvironment(args.case, args.config, render=not args.physics_only) if args.config else MujocoEnvironment(
            args.case, render=not args.physics_only)
        simulator = env
        if args.perception != 'gt':
            from pipeline.rgbd_env import RGBDEnvironment
            capture_output = ROOT/'outputs/act_rgbd_evaluation'/f'{args.case}_{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}'
            env = RGBDEnvironment(simulator, spec, output=capture_output)
        viewer = None
        start = time.perf_counter()
        ticks = 0
        last_goal = None
        privacy_ticks = 0
        dataset_scene = scene_group(env, seed) in used_groups
        try:
            check(args.allow_training_seeds or not dataset_scene,
                  'seed belongs to the training/validation dataset; choose an unseen seed')
            task = env.task
            if args.target:
                check(args.target in env.object_definitions, 'unknown evaluation target')
                z = env.config['placement_trays']['floor_thickness']+env.object_definitions[args.target]['size'][2]/2
                task = TaskRequest(args.target, Pose((*task.destination.position[:2], z)), Pose((*task.buffer_pose.position[:2], z)))
            env.config['max_steps_per_subgoal'] = args.max_steps
            if args.gui:
                viewer = SimulationViewer(env.model, env.data, interactive=False)
            def predict(observation, subgoal):
                nonlocal ticks, last_goal, privacy_ticks
                last_goal = subgoal
                if args.perception != 'gt':
                    check(observation.source == 'rgbd' and observation.ground_truth is None and not observation.contacts,
                          'RGB-D policy observation exposed privileged object state')
                    privacy_ticks += 1
                if viewer is not None:
                    viewer.poll_direction()
                    check(viewer.is_running(), 'evaluation interrupted by window close')
                    control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'])
                    viewer.sync(observation, control, env.joints)
                ticks += 1
                return policy.predict_action_chunk(observation, subgoal)
            completed = run_pipeline(env, task, seed=seed, predict_action=predict,
                                    **(dict(generate_graph=env.generate_scene_graph, refresh_observation=env.refresh)
                                       if args.perception != 'gt' else {}))
            oracle = bool(simulator.task_complete(simulator.task if args.perception != 'gt' else task))
            result = {'seed': seed, 'success': oracle, 'observational_complete': True,
                      'oracle_success': oracle, 'completed_subgoals': len(completed),
                      'subgoals': [{'primitive': g.primitive, 'object_id': g.object_id} for g in completed]}
        except (ValueError, RuntimeError) as exc:
            result = {'seed': seed, 'success': False, 'error': str(exc)}
            if last_goal is not None:
                observed = env.observe()
                result['failed_subgoal'] = dict(primitive=last_goal.primitive, object_id=last_goal.object_id,
                    object_position=list(last_goal.object_pose.position),
                    place_position=list(last_goal.place_pose.position) if last_goal.place_pose else None,
                    final_ee_position=list(observed.ee_pose.position), joint_state=observed.robot_state.tolist(),
                    finger_contacts=list(observed.finger_contacts), held_object_id=observed.held_object_id)
        finally:
            if viewer is not None:
                viewer.close()
            env.close()
        result.update(control_ticks=ticks, privacy_checked_ticks=privacy_ticks,
                      dataset_scene=dataset_scene, wall_seconds=time.perf_counter()-start)
        results.append(result)
        print(json.dumps(result), flush=True)
    output = args.output or ROOT/'training/runs'/f'evaluation-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'controller': args.controller,
              'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
              'checkpoint_sha256': sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              'case': args.case, 'target': args.target, 'allow_training_seeds': args.allow_training_seeds,
              'perception': args.perception, 'target_spec': str(args.target_spec) if args.target_spec else None,
              'execute_steps': args.execute_steps or (policy.execute_steps if args.controller == 'act' else None),
              'temporal_ensemble': bool(args.temporal_ensemble),
              'success_rate': sum(r['success'] for r in results)/len(results), 'results': results}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('report:', output)
    return 0 if all(r['success'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
