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
from pipeline.types import Pose, TaskRequest, check
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
    args = parser.parse_args()
    check(args.max_steps > 0 and all(s >= 0 for s in args.seeds), 'invalid seed or rollout budget')
    used_groups = set()
    if args.controller == 'act':
        if args.checkpoint is None or args.physics_only:
            parser.error('ACT evaluation requires --checkpoint and RGB rendering (no --physics-only)')
        import torch
        from .policy import ACTPolicy
        policy = ACTPolicy(args.checkpoint, device=args.device)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        used_groups = {s['group_id'] for part in payload['split'].values() for s in part}
    else:
        from action_generator.expert import SO101Expert
        policy = SO101Expert()
    results = []
    for seed in args.seeds:
        env = MujocoEnvironment(args.case, args.config, render=not args.physics_only) if args.config else MujocoEnvironment(
            args.case, render=not args.physics_only)
        viewer = None
        start = time.perf_counter()
        ticks = 0
        try:
            check(args.allow_training_seeds or scene_group(env, seed) not in used_groups,
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
                nonlocal ticks
                if viewer is not None:
                    viewer.poll_direction()
                    check(viewer.is_running(), 'evaluation interrupted by window close')
                    control = JointJogControl(observation.robot_state, env.lower, env.upper, env.config['max_delta'])
                    viewer.sync(observation, control, env.joints)
                ticks += 1
                return policy.predict_action_chunk(observation, subgoal)
            completed = run_pipeline(env, task, seed=seed, predict_action=predict)
            result = {'seed': seed, 'success': True, 'completed_subgoals': len(completed)}
        except (ValueError, RuntimeError) as exc:
            result = {'seed': seed, 'success': False, 'error': str(exc)}
        finally:
            if viewer is not None:
                viewer.close()
            env.close()
        result.update(control_ticks=ticks, wall_seconds=time.perf_counter()-start)
        results.append(result)
        print(json.dumps(result), flush=True)
    output = args.output or ROOT/'training/runs'/f'evaluation-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'controller': args.controller,
              'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
              'checkpoint_sha256': sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              'case': args.case, 'target': args.target, 'allow_training_seeds': args.allow_training_seeds,
              'success_rate': sum(r['success'] for r in results)/len(results), 'results': results}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('report:', output)
    return 0 if all(r['success'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
