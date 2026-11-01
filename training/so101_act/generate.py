"""Automatically record physically successful scripted-expert demonstrations."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4
from hashlib import sha256

from action_generator.expert import SO101Expert
from action_generator.goal import encode_goal
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.runner import run_pipeline
from pipeline.types import Pose, TargetSpec, TaskRequest, check
from .collect import provenance, scene_group
from .data import SegmentWriter


def generate_episode(output, case, seed, *, target=None, config_path=None, max_steps=300, grasp_jitter=0.,
                     perception='gt', target_spec=None, teacher='observed', learner_checkpoint=None, learner_steps=0):
    check(teacher in ('observed', 'privileged'), 'unknown demonstration teacher')
    check(learner_steps >= 0 and (learner_steps == 0 or learner_checkpoint is not None), 'learner prefix requires a checkpoint')
    learner = None
    if learner_checkpoint is not None:
        from .policy import ACTPolicy
        learner = ACTPolicy(learner_checkpoint, device='cuda')
    simulator = MujocoEnvironment(case, config_path) if config_path else MujocoEnvironment(case)
    env = simulator
    if perception != 'gt':
        check(perception == 'rgbd-color' and target_spec is not None and target is None,
              'RGB-D collection requires target_spec and uses appearance instead of --target')
        from pipeline.rgbd_env import RGBDEnvironment
        env = RGBDEnvironment(simulator, target_spec,
                              output=Path(output)/'_perception'/f'{case}_{seed}_{uuid4().hex[:8]}')
    import numpy as np
    check(0 <= grasp_jitter <= .008, 'grasp jitter must be within 0..0.008m')
    offset = .001+np.random.default_rng(seed+10000).uniform(-grasp_jitter, grasp_jitter)
    policy = SO101Expert(grasp_height_offset=offset)
    writer = None
    saved = []
    ticks = 0
    oracle_objects = {}
    oracle_goal = None
    teacher_goal = None
    active_id, prefix_left, initial_goal, initial_ee = None, 0, None, None
    result = {'case': case, 'seed': seed, 'target': target, 'perception': perception, 'teacher': teacher, 'success': False}
    try:
        env.config['max_steps_per_subgoal'] = max_steps
        task = env.task
        if target is not None:
            check(target in env.object_definitions, 'unknown teacher target')
            z = env.config['placement_trays']['floor_thickness']+env.object_definitions[target]['size'][2]/2
            task = TaskRequest(target, Pose((*task.destination.position[:2], z)), Pose((*task.buffer_pose.position[:2], z)))
        def predict(observation, subgoal):
            nonlocal writer, oracle_goal, teacher_goal, active_id, prefix_left, initial_goal, initial_ee
            if active_id != subgoal.subgoal_id:
                active_id, prefix_left = subgoal.subgoal_id, learner_steps
                initial_goal, initial_ee = encode_goal(subgoal, observation.ee_pose), observation.ee_pose
                # Collection-only association: never enters the student's
                # graph/goal. The explicit privileged teacher may use it.
                if perception != 'gt':
                    if subgoal.primitive == 'PICK':
                        distances = [(float(np.linalg.norm(simulator.data.xpos[body]-subgoal.object_pose.position)), key)
                                     for key, body in simulator.object_bodies.items()]
                        distance, oracle_id = min(distances)
                        check(distance < .018, 'cannot verify teacher object against physical oracle')
                        oracle_objects[subgoal.object_id] = oracle_id
                    oracle_goal = replace(subgoal, object_id=oracle_objects[subgoal.object_id])
                else:
                    oracle_goal = subgoal
            if prefix_left > 0:
                prefix_left -= 1
                return learner.predict_action_chunk(observation, subgoal)
            if writer is None:
                origin = provenance(env, seed)
                origin.update(collector='scripted-expert-v1', grasp_height_offset=offset,
                              perception=perception, teacher=teacher, oracle_label_object_id=oracle_goal.object_id,
                              learner_prefix_steps=learner_steps,
                              learner_checkpoint_sha256=sha256(Path(learner_checkpoint).read_bytes()).hexdigest() if learner_checkpoint else None)
                if teacher == 'privileged':
                    oracle_object = next(obj for obj in simulator.observe().ground_truth
                                         if obj.object_id == oracle_goal.object_id)
                    teacher_goal = replace(oracle_goal, object_pose=oracle_object.pose)
                writer = SegmentWriter(output, observation, subgoal, group_id=scene_group(env, seed),
                                       lower=env.lower, upper=env.upper, max_delta=env.config['max_delta'],
                                       provenance=origin)
                # The condition is frozen at the original Subgoal boundary,
                # including PLACE's initial grasp transform. Prefix commands
                # are not falsely labeled as expert actions or saved as such.
                writer.goal = initial_goal.copy()
                writer.meta['initial_ee_pose'] = (dict(position=initial_ee.position, quaternion_xyzw=initial_ee.quaternion_xyzw)
                                                  if initial_ee else None)
            # Privileged teacher is explicit and collection-only. Stored RGB,
            # state and encoded goal remain the original RGB-D student input.
            return policy.predict_action_chunk(simulator.observe(), teacher_goal) if teacher == 'privileged' else policy.predict_action_chunk(observation, subgoal)
        def record(before, command, after, subgoal):
            nonlocal writer, ticks
            ticks += 1
            if writer is None:
                return
            writer.append(before, command)
            if env.subgoal_complete(subgoal):
                verified = bool(simulator.subgoal_complete(oracle_goal))
                path = writer.finish(success=verified, reason='expert_physical_subgoal_complete' if verified else 'oracle_rejected')
                if verified:
                    saved.append(str(path))
                writer = None
        completed = run_pipeline(env, task, seed=seed, predict_action=predict, on_step=record,
                                 **(dict(generate_graph=env.generate_scene_graph, refresh_observation=env.refresh)
                                    if perception != 'gt' else {}))
        verified = bool(simulator.task_complete(simulator.task if perception != 'gt' else task))
        result.update(success=verified, completed_subgoals=len(completed), oracle_success=verified)
        if not verified:
            result['error'] = 'physical oracle rejected completed task'
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
    parser.add_argument('--perception', choices=('gt', 'rgbd-color'), default='gt')
    parser.add_argument('--target-spec', type=Path)
    parser.add_argument('--teacher', choices=('observed', 'privileged'), default='observed')
    parser.add_argument('--learner-checkpoint', type=Path, help='collection only: learner prefix followed by teacher recovery')
    parser.add_argument('--learner-steps', type=int, default=0)
    args = parser.parse_args()
    check(args.max_steps > 0 and all(s >= 0 for s in args.seeds), 'invalid teacher rollout budget/seed')
    results = []
    if args.perception != 'gt' and not args.target_spec:
        parser.error('RGB-D collection requires --target-spec')
    spec = TargetSpec(**json.loads(args.target_spec.read_text(encoding='utf-8'))) if args.target_spec else None
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output/f'generation-{uuid4().hex[:8]}.json'
    for seed in args.seeds:
        result = generate_episode(args.output, args.case, seed, target=args.target,
                                  config_path=args.config, max_steps=args.max_steps, grasp_jitter=args.grasp_jitter,
                                  perception=args.perception, target_spec=spec, teacher=args.teacher,
                                  learner_checkpoint=args.learner_checkpoint, learner_steps=args.learner_steps)
        results.append(result)
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
        pending.replace(report)
        print(json.dumps(result), flush=True)
    print('generation report:', report)
    return 0 if all(r['success'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
