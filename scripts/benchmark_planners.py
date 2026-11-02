"""Prepare a frozen, outcome-blind RGB-D benchmark; run paired MuJoCo rollouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import mujoco
import numpy as np

from hssg_generator import generate_scene_graph
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.rgbd_env import RGBDEnvironment
from pipeline.runner import run_pipeline
from pipeline.target import TargetResolver
from pipeline.types import Pose, TargetSpec, TaskRequest, check
from planner.baselines import METHODS, plan_baseline

GROUPS = ('visible_free', 'visible_blocked', 'unlocalized_known_region', 'unlocalized_unknown_region')


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def digest(data):
    return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def prepare(args):
    check(not args.manifest.exists(), 'refusing to overwrite a frozen manifest')
    config = json.loads((ROOT/'configs/envs/so101.json').read_text(encoding='utf-8'))
    config['max_subgoals'] = args.max_subgoals
    config['max_steps_per_subgoal'] = args.max_steps
    config_path = args.output/'config.json'
    dump(config_path, config)
    cases, scan = [], []
    for seed in range(args.start_seed, args.start_seed+args.max_scan):
        group = GROUPS[len(cases) % len(GROUPS)]
        sim = MujocoEnvironment('shelves', config_path, render=False)
        env = RGBDEnvironment(sim, TargetSpec((.2, .5, .9)), output=args.output/'screening'/f'seed_{seed}')
        try:
            env.reset(seed)
            gt = sim.observe(privileged=True)
            gtgraph = generate_scene_graph(gt)
            blocked = {e.target for e in gtgraph.edges if e.relation == 'on_top_of'}
            candidates = []
            for obj in gt.ground_truth:
                definition = sim.object_definitions[obj.object_id]
                spec = TargetSpec(tuple(definition['color'][:3]),
                    region=definition['shelf_id'] if group == 'unlocalized_known_region' else None,
                    size_m=obj.size, color_tolerance=.08, ambiguity_margin=.015)
                try:
                    task = TargetResolver().resolve(env.graph, replace(env.task, target_spec=spec))
                except ValueError:
                    continue
                match = next((o for o in env.graph.nodes if o.object_id == task.target_object_id), None)
                localized = match is not None and np.linalg.norm(np.array(match.pose.position)-obj.pose.position) < .012
                eligible = (localized and obj.object_id not in blocked if group == 'visible_free' else
                            localized and obj.object_id in blocked if group == 'visible_blocked' else
                            match is None and obj.object_id in blocked)
                if eligible:
                    candidates.append((obj.object_id, spec, task.target_object_id))
            scan.append(dict(seed=seed, requested_group=group, eligible=len(candidates)))
            if not candidates:
                print(f'screen {seed}: no {group} target', flush=True)
                continue
            # Outcome blind deterministic selection; cycle shelf preference.
            preferred = f'shelf_{len(cases)%3+1}'
            candidates.sort(key=lambda row: (not row[0].startswith(preferred+'_'), row[0]))
            target, spec, track = candidates[0]
            cases.append(dict(case_id=f'scene_{len(cases):03d}', seed=seed, group=group,
                target_gt_id=target, target_spec=asdict(spec), initial_track=track,
                initial_state_sha256=sha256(sim.data.qpos.tobytes()).hexdigest(),
                initial_graph=asdict(env.graph), object_count=len(gt.ground_truth),
                initial_target_gt_position=list(sim.data.xpos[sim.object_bodies[target]])))
            print(f'prepared {len(cases)}/{args.count}: seed={seed} {group} target={target}', flush=True)
        finally:
            env.close()
        if len(cases) == args.count:
            break
    check(len(cases) == args.count, 'insufficient eligible scenes; manifest not written')
    dump(args.manifest, dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='rgbd-color-paired-planners-v1', config=config, config_sha256=digest(config),
        groups=GROUPS, selection='initial A localization + GT support labels, never rollout outcomes',
        target_gt_id_usage='screening labels and independent evaluation only; not passed to A/B/C',
        screening=scan, cases=cases))
    print('manifest:', args.manifest, flush=True)


def run_case(manifest, case, method, args):
    folder = args.output/args.controller/method/case['case_id']
    result_path = folder/'result.json'
    identity = dict(manifest_sha256=digest(manifest), method=method, controller=args.controller,
                    checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None)
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding='utf-8'))
        check(all(old.get(k) == v for k, v in identity.items()), 'existing result has different configuration')
        return old
    config_path = args.output/'config.json'
    if not config_path.exists():
        dump(config_path, manifest['config'])
    check(digest(json.loads(config_path.read_text())) == manifest['config_sha256'], 'config changed')
    sim = MujocoEnvironment('shelves', config_path, render=args.controller == 'act')
    env = RGBDEnvironment(sim, TargetSpec(**case['target_spec']), output=folder/'perception')
    if args.controller == 'expert':
        from action_generator.expert import SO101Expert
        policy = SO101Expert()
    else:
        sys.path.insert(0, str(ROOT/'training'))
        from so101_act.policy import ACTPolicy
        policy = ACTPolicy(args.checkpoint, device=args.device)
    started = time.perf_counter()
    trace, plan_inputs = [], []
    metrics = dict(control_ticks=0, privacy_checked_ticks=0, plan_seconds=0., perception_seconds=0.,
                   perception_refreshes=0, planning_calls=0, discovered_at_sim_seconds=None,
                   discovery_after_completed_removals=None, observational_complete=False)
    current_task, active = env.task, None
    stage = 'reset'
    initial_state = None
    original_refresh = env.refresh
    def refresh():
        t, n = time.perf_counter(), env.snapshot
        value = original_refresh()
        metrics['perception_seconds'] += time.perf_counter()-t
        metrics['perception_refreshes'] += env.snapshot-n
        return value
    env.refresh = refresh
    def privacy(observation):
        check(observation.source == 'rgbd' and observation.ground_truth is None and not observation.contacts,
              'privileged object state exposed')
        check(all(obj.object_id.startswith('instance_') for obj in observation.perceived_objects), 'simulator ID exposed')
    def oracle_match(goal):
        # Diagnostic after a decision only. Never feeds a policy or graph.
        p = np.array(goal.object_pose.position)
        key = min(sim.object_bodies, key=lambda k: np.linalg.norm(sim.data.xpos[sim.object_bodies[k]]-p))
        return key if np.linalg.norm(sim.data.xpos[sim.object_bodies[key]]-p) < .015 else None
    def plan(graph, task, state):
        nonlocal stage
        stage = 'planning'
        t = time.perf_counter()
        try:
            return plan_baseline(method, graph, task, state)
        finally:
            metrics['plan_seconds'] += time.perf_counter()-t
            metrics['planning_calls'] += 1
            plan_inputs.append(dict(graph=asdict(graph), task=asdict(task), robot_state=state.tolist()))
    def predict(obs, goal):
        nonlocal stage
        stage = 'control'
        privacy(obs)
        metrics['privacy_checked_ticks'] += 1
        return policy.predict_action_chunk(obs, goal)
    def step(before, command, obs, goal):
        privacy(obs)
        metrics['control_ticks'] += 1
    def event(kind, **data):
        nonlocal current_task, active, initial_state, stage
        if kind == 'scene':
            current_task = data['task']
            if initial_state is None:
                initial_state = sha256(sim.data.qpos.tobytes()).hexdigest()
                check(initial_state == case['initial_state_sha256'], 'paired initial physics state mismatch')
                check(digest(asdict(data['graph'])) == digest(case['initial_graph']), 'initial perception mismatch')
            if metrics['discovered_at_sim_seconds'] is None and any(
                    obj.object_id == current_task.target_object_id for obj in data['graph'].nodes):
                # Correct physical identity is also audited; binding alone can be wrong.
                obj = data['graph'].object(current_task.target_object_id)
                correct = np.linalg.norm(np.array(obj.pose.position)-sim.data.xpos[sim.object_bodies[case['target_gt_id']]]) < .015
                if correct:
                    metrics['discovered_at_sim_seconds'] = data['observation'].timestamp
                    metrics['discovery_after_completed_removals'] = sum(
                        row['event'] == 'completed' and row['primitive'] == 'PLACE' and not row['target'] for row in trace)
        elif kind == 'planned':
            active = data['subgoal']
            trace.append(dict(event=kind, timestamp=data['observation'].timestamp,
                primitive=active.primitive, object_id=active.object_id, goal=asdict(active),
                target=active.object_id == current_task.target_object_id, physical_id=oracle_match(active)))
        elif kind == 'completed':
            goal = data['subgoal']
            trace.append(dict(event=kind, timestamp=data['observation'].timestamp, primitive=goal.primitive,
                object_id=goal.object_id, target=goal.object_id == current_task.target_object_id, steps=data['steps']))
        elif kind == 'finished':
            metrics['observational_complete'] = True
    error, failed_stage = None, None
    try:
        run_pipeline(env, env.task, seed=case['seed'], generate_graph=env.generate_scene_graph,
                     refresh_observation=env.refresh, predict_action=predict, plan_subgoal=plan,
                     on_step=step, on_event=event)
    except (ValueError, RuntimeError) as exc:
        error, failed_stage = str(exc), stage
    finally:
        # Independent physical success uses the requested true object, even when
        # the appearance resolver accidentally bound a different track.
        z = manifest['config']['placement_trays']['floor_thickness']+case['target_spec']['size_m'][2]/2
        oracle_task = TaskRequest(case['target_gt_id'], Pose((*env.task.destination.position[:2], z)), env.task.buffer_pose)
        oracle = bool(sim.task_complete(oracle_task))
        final_time = float(sim.data.time-sim._start_time)
        failed_goal = asdict(active) if error and active else None
        binding_correct = None
        target_obj = next((o for o in env.observe().perceived_objects if o.object_id == current_task.target_object_id), None)
        if target_obj is not None:
            binding_correct = bool(np.linalg.norm(np.array(target_obj.pose.position)-sim.data.xpos[
                sim.object_bodies[case['target_gt_id']]]) < .015)
        completed = [row for row in trace if row['event'] == 'completed']
        result = dict(**identity, case_id=case['case_id'], seed=case['seed'], group=case['group'],
            success=oracle, error=error, failed_stage=failed_stage, failed_subgoal=failed_goal,
            final_binding_correct=binding_correct, simulation_seconds=final_time,
            wall_seconds=time.perf_counter()-started, initial_state_sha256=initial_state,
            completed_subgoals=len(completed), completed_picks=sum(row['primitive']=='PICK' for row in completed),
            completed_removals=sum(row['primitive']=='PLACE' and not row['target'] for row in completed),
            attempted_picks=sum(row['event']=='planned' and row['primitive']=='PICK' for row in trace),
            unique_picked_tracks=len({row['object_id'] for row in completed if row['primitive']=='PICK'}),
            trace=trace, **metrics)
        dump(folder/'plan_inputs.json', plan_inputs)
        dump(result_path, result)
        env.close()
    return result


def run(args):
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    check(args.controller != 'act' or args.checkpoint is not None, 'ACT needs --checkpoint')
    for case in manifest['cases']:
        if args.case_ids and case['case_id'] not in args.case_ids:
            continue
        for method in args.methods:
            row = run_case(manifest, case, method, args)
            print(json.dumps({key: row[key] for key in ('case_id','seed','group','method','controller','success',
                'completed_removals','attempted_picks','simulation_seconds','wall_seconds','error')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--manifest', type=Path, default=ROOT/'experiments/planner_benchmark/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/planner_benchmark/main')
    parser.add_argument('--count', type=int, default=52)
    parser.add_argument('--start-seed', type=int, default=3100)
    parser.add_argument('--max-scan', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=450)
    parser.add_argument('--max-subgoals', type=int, default=12)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--controller', choices=('expert', 'act'), default='expert')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='cuda')
    args = parser.parse_args()
    check(args.count > 0 and args.max_steps > 0 and args.max_subgoals > 0, 'invalid budget')
    (prepare if args.operation == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
