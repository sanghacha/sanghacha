"""Automatically record physically successful scripted-expert demonstrations."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from action_generator.expert import SO101Expert
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.runner import run_pipeline
from pipeline.types import Pose, TaskRequest, check
from .collect import provenance, scene_group
from .data import SegmentWriter


def generate_episode(output, case, seed, *, target=None, config_path=None, max_steps=300, grasp_jitter=0.):
    env = MujocoEnvironment(case, config_path) if config_path else MujocoEnvironment(case)
    import numpy as np
    check(0 <= grasp_jitter <= .008, 'grasp jitter must be within 0..0.008m')
    offset = .001+np.random.default_rng(seed+10000).uniform(-grasp_jitter, grasp_jitter)
    policy = SO101Expert(grasp_height_offset=offset)
    writer = None
    saved = []
    ticks = 0
    result = {'case': case, 'seed': seed, 'target': target, 'success': False}
    try:
        env.config['max_steps_per_subgoal'] = max_steps
        task = env.task
        if target is not None:
            check(target in env.object_definitions, 'unknown teacher target')
            z = env.config['placement_trays']['floor_thickness']+env.object_definitions[target]['size'][2]/2
            task = TaskRequest(target, Pose((*task.destination.position[:2], z)), Pose((*task.buffer_pose.position[:2], z)))
        def predict(observation, subgoal):
            nonlocal writer
            if writer is None:
                origin = provenance(env, seed)
                origin['collector'] = 'scripted-expert-v1'
                origin['grasp_height_offset'] = offset
                writer = SegmentWriter(output, observation, subgoal, group_id=scene_group(env, seed),
                                       lower=env.lower, upper=env.upper, max_delta=env.config['max_delta'],
                                       provenance=origin)
            return policy.predict_action_chunk(observation, subgoal)
        def record(before, command, after, subgoal):
            nonlocal writer, ticks
            ticks += 1
            writer.append(before, command)
            if env.subgoal_complete(subgoal):
                saved.append(str(writer.finish(success=True, reason='expert_physical_subgoal_complete')))
                writer = None
        completed = run_pipeline(env, task, seed=seed, predict_action=predict, on_step=record)
        result.update(success=True, completed_subgoals=len(completed))
    except (ValueError, RuntimeError) as exc:
        result['error'] = str(exc)
    finally:
        try:
            if writer is not None and not writer.finished:
                writer.finish(success=False, reason=result.get('error', 'interrupted'))
        finally:
            env.close()
    return dict(result, control_ticks=ticks, saved_segments=saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'training/data/demos')
    parser.add_argument('--case', choices=('accessible', 'blocked', 'after_move', 'shelves'), default='accessible')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--target')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--grasp-jitter', type=float, default=0., help='vary grasp depth in metres for robust PLACE conditioning')
    args = parser.parse_args()
    check(args.max_steps > 0 and all(s >= 0 for s in args.seeds), 'invalid teacher rollout budget/seed')
    results = []
    for seed in args.seeds:
        result = generate_episode(args.output, args.case, seed, target=args.target,
                                  config_path=args.config, max_steps=args.max_steps, grasp_jitter=args.grasp_jitter)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output/f'generation-{uuid4().hex[:8]}.json'
    report.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print('generation report:', report)
    return 0 if all(r['success'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
