"""Paired RGB-D planning diagnostic with a common ideal removal executor.

No ACT or physical grasp is simulated. A chosen, unblocked observed object is
teleported to an external buffer, then the original cameras and perception run
again. All other poses and the parked robot are fixed. Oracle state is used by
the executor/evaluator only, never to select an action or ground the request.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import mujoco
import numpy as np

from hssg_generator import generate_scene_graph
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.rgbd_env import RGBDEnvironment
from pipeline.target import TargetResolver
from pipeline.types import TargetSpec
from planner.act_comparison import CONDITIONS, ComparisonPlanner
from benchmark_planners import GROUPS, digest, dump

SOURCES = ['scripts/benchmark_planner_search.py', 'src/planner/act_comparison.py',
           'src/planner/indexed.py', 'src/planner/baselines.py', 'src/planner/planner.py',
           'src/pipeline/rgbd_env.py', 'src/pipeline/mujoco_env.py',
           'src/pipeline/target.py', 'src/perception/acquisition.py',
           'src/perception/association.py', 'src/perception/bridge.py']


def source_hashes():
    return {p: sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def check_initial_graph(actual, expected):
    """Permit tiny GPU rasterization rounding, never identity/geometry changes."""
    actual=json.loads(json.dumps(asdict(actual)))
    assert {k:v for k,v in actual.items() if k!='nodes'}=={k:v for k,v in expected.items() if k!='nodes'}
    assert len(actual['nodes'])==len(expected['nodes'])
    max_error=0.
    for a,b in zip(actual['nodes'],expected['nodes']):
        for key in ('object_id','label','parent_id'):
            assert a[key]==b[key], f'initial graph identity changed: {key}'
        for x,y,tolerance in ((a['pose']['position'],b['pose']['position'],1e-5),
                              (a['pose']['quaternion_xyzw'],b['pose']['quaternion_xyzw'],1e-5),
                              (a['size'],b['size'],1e-5), (a['color_rgb'],b['color_rgb'],1e-4)):
            if x is None or y is None:
                assert x is y
            else:
                delta=float(np.max(np.abs(np.asarray(x)-y)))
                assert delta<=tolerance, f'initial graph numerical mismatch: {delta}'
                max_error=max(max_error,delta)
    return max_error


def nearest_physical(sim, position, removed):
    """Executor correspondence AFTER a planner has committed its public goal."""
    options = [(float(np.linalg.norm(sim.data.xpos[body]-position)), key)
               for key, body in sim.object_bodies.items() if key not in removed]
    distance, key = min(options)
    return key if distance < .015 else None


def blockers(graph, key, removed):
    return sorted(e.source for e in graph.edges
                  if e.relation == 'on_top_of' and e.target == key and e.source not in removed)


def freeze(args):
    assert args.count > 0 and args.count % 4 == 0
    assert not args.manifest.exists(), 'frozen manifest already exists'
    config = json.loads((ROOT/'configs/envs/so101.json').read_text(encoding='utf-8'))
    config_path = args.output/'config.json'
    dump(config_path, config)
    protocol = dict(schema='rgbd-ideal-removal-v1', count=args.count, start_seed=args.start_seed,
        max_scan=args.max_scan, max_picks=28, primary='non-target removals capped at 27 on failure',
        comparisons=['b_no_hidden vs a_greedy: all strata',
                     'c_full vs b_no_hidden: initially unlocalized strata'],
        secondary=['discovery removals', 'success within 1/2/3/5/8/12 removals',
                   'initial inferred-cue availability, no outcome-based subgroup definition',
                   'geometric base-object-buffer path proxy; not robot execution time'],
        selection='fixed strata; uniform RNG choice among eligible targets; no planner outcomes/cue screening',
        execution='oracle object correspondence and support check after decision; teleport outside scene',
        limits=['no ACT, IK, collision trajectory, placement congestion, grasp failures or object settling',
                'all remaining object poses and the parked robot stay fixed',
                'scene families are conditional strata, not the natural scene prevalence'],
        source_sha256=source_hashes(), created_utc=datetime.now(timezone.utc).isoformat())
    protocol_path = args.output/'protocol.json'
    assert not protocol_path.exists(), 'use a new directory for a new frozen protocol'
    dump(protocol_path, protocol)
    cases, screening = [], []
    for seed in range(args.start_seed, args.start_seed+args.max_scan):
        group = GROUPS[len(cases) % 4]
        sim = MujocoEnvironment('shelves', config_path, render=False)
        env = RGBDEnvironment(sim, TargetSpec((.2,.5,.9)), output=args.output/'screening'/f'seed_{seed}')
        try:
            env.reset(seed)
            oracle = sim.observe(privileged=True)
            true_graph = generate_scene_graph(oracle)
            pool = []
            for obj in oracle.ground_truth:
                definition = sim.object_definitions[obj.object_id]
                spec = TargetSpec(tuple(definition['color'][:3]),
                    region=definition['shelf_id'] if group == 'unlocalized_known_region' else None,
                    size_m=obj.size, color_tolerance=.08, ambiguity_margin=.015)
                try:
                    task = TargetResolver().resolve(env.graph, replace(env.task,target_spec=spec))
                except ValueError:
                    continue
                match = next((o for o in env.graph.nodes if o.object_id == task.target_object_id),None)
                localized = match is not None and np.linalg.norm(np.array(match.pose.position)-obj.pose.position)<.012
                blocked = bool(blockers(true_graph,obj.object_id,set()))
                eligible = (localized and not blocked if group == 'visible_free' else
                            localized and blocked if group == 'visible_blocked' else match is None and blocked)
                if eligible:
                    pool.append((obj.object_id,spec))
            screening.append(dict(seed=seed,group=group,eligible=len(pool)))
            if pool:
                pool.sort(key=lambda row:row[0])
                key,spec = pool[int(np.random.default_rng(seed+170003).integers(len(pool)))]
                request=TargetResolver().resolve(env.graph,replace(env.task,target_spec=spec))
                probe=ComparisonPlanner('c_full')
                probe.plan_next_subgoal(env.graph,request,env.observe().robot_state)
                cue_count=sum(r['hidden_count'] for r in probe.last_decision.get('candidates',[]))
                cases.append(dict(case_id=f'scene_{len(cases):03d}',seed=seed,group=group,
                    target_gt_id=key,target_spec=asdict(spec),initial_graph=asdict(env.graph),
                    initial_inferred_hidden_count=cue_count,
                    initial_state_sha256=sha256(sim.data.qpos.tobytes()).hexdigest()))
                print(f'freeze {len(cases)}/{args.count}: seed={seed} {group}',flush=True)
        finally:
            env.close()
        if len(cases)==args.count:
            break
    assert len(cases)==args.count, 'not enough eligible scenes; keep screening, do not change outcome-based selection'
    dump(args.manifest,dict(protocol=protocol,config=config,screening=screening,cases=cases))


def run_case(args, manifest, case, condition):
    folder = args.output/'runs'/condition/case['case_id']
    result_path = folder/'result.json'
    identity = dict(manifest_sha256=digest(manifest),source_sha256=source_hashes())
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['identity']==identity
        return result
    sim = MujocoEnvironment('shelves',args.output/'config.json',render=False)
    env = RGBDEnvironment(sim,TargetSpec(**case['target_spec']),output=folder/'perception')
    planner, resolver = ComparisonPlanner(condition), TargetResolver()
    removed, trace, inputs = set(), [], []
    success, discovery, error, path = False, None, None, 0.
    initial_graph_max_error=None
    started = time.perf_counter()
    try:
        env.reset(case['seed'])
        assert sha256(sim.data.qpos.tobytes()).hexdigest()==case['initial_state_sha256']
        initial_graph_max_error=check_initial_graph(env.graph,case['initial_graph'])
        true_graph = generate_scene_graph(sim.observe(privileged=True))
        request = env.task
        for step in range(manifest['protocol']['max_picks']):
            graph = env.graph
            observation = env.observe()
            assert observation.source=='rgbd' and observation.ground_truth is None and not observation.contacts
            assert all(o.object_id.startswith('instance_') for o in graph.nodes)
            request = resolver.resolve(graph,request)
            target = next((o for o in graph.nodes if o.object_id==request.target_object_id),None)
            correct_binding = target is not None and nearest_physical(sim,target.pose.position,removed)==case['target_gt_id']
            if correct_binding and discovery is None:
                discovery = len(removed)
            t = time.perf_counter_ns()
            goal = planner.plan_next_subgoal(graph,request,observation.robot_state)
            plan_ns = time.perf_counter_ns()-t
            assert goal.primitive=='PICK'
            inputs.append(dict(graph=asdict(graph),task=asdict(request),goal=asdict(goal),
                               decision=planner.last_decision,plan_ns=plan_ns))
            # Only the committed goal now crosses the executor boundary.
            key = nearest_physical(sim,goal.object_pose.position,removed)
            obstructed_by = blockers(true_graph,key,removed) if key else []
            row = dict(step=step,track=goal.object_id,physical_id=key,
                requested_target=goal.object_id==request.target_object_id,
                correct_binding=bool(correct_binding),blockers=obstructed_by)
            trace.append(row)
            if key is None:
                error='executor cannot associate observed pose within 15mm';break
            if obstructed_by:
                error='executor rejected a physically supported/blocked pick';break
            if goal.object_id==request.target_object_id and key!=case['target_gt_id']:
                error='appearance resolver selected the wrong physical target';break
            xyz = sim.data.xpos[sim.object_bodies[key]].copy()
            # Distance is explicitly a geometry proxy, never ACT time/energy.
            endpoint = request.destination.position if key==case['target_gt_id'] else request.buffer_pose.position
            path += float(np.linalg.norm(xyz)+np.linalg.norm(xyz-np.array(endpoint)))
            if key==case['target_gt_id'] and goal.object_id!=request.target_object_id:
                error='selected the physical target without recognizing the requested appearance';break
            if key==case['target_gt_id']:
                success=True;break
            removed.add(key)
            planner.completed(replace(goal,primitive='PLACE',place_pose=request.buffer_pose))
            address=sim.object_qadr[key]
            sim.data.qpos[address:address+3]=[-2.-len(removed)*.1,-2.,.05]
            sim.data.qvel[:]=0
            sim.data.time+=1.  # monotone perception revision; not physical time
            mujoco.mj_forward(sim.model,sim.data)
            env.refresh()
        else:
            error='ideal removal action budget exhausted'
    except (ValueError,RuntimeError) as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        env.close()
    result=dict(identity=identity,case_id=case['case_id'],seed=case['seed'],group=case['group'],
        condition=condition,success=success,error=error,non_target_removals=len(removed),
        capped_removals=len(removed) if success else 27,discovery_removals=discovery,
        path_proxy_m=path,initial_graph_max_error=initial_graph_max_error,
        wall_seconds=time.perf_counter()-started,trace=trace)
    dump(folder/'plan_inputs.json',inputs)
    dump(result_path,result)
    return result


def run(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    assert manifest['protocol']['source_sha256']==source_hashes(), 'source changed after protocol freeze'
    assert json.loads((args.output/'config.json').read_text())==manifest['config']
    for case in manifest['cases']:
        # Rotate ordering across paired scenes; no timings are primary endpoints.
        offset=int(case['case_id'].split('_')[1])%3
        for condition in CONDITIONS[offset:]+CONDITIONS[:offset]:
            result=run_case(args,manifest,case,condition)
            print(json.dumps({k:result[k] for k in ('case_id','condition','success','non_target_removals','error')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('freeze','run'))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/planner_search/confirmatory')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--count',type=int,default=80)
    parser.add_argument('--start-seed',type=int,default=8100)
    parser.add_argument('--max-scan',type=int,default=500)
    args=parser.parse_args()
    args.manifest=args.manifest or args.output/'manifest.json'
    (freeze if args.operation=='freeze' else run)(args)


if __name__=='__main__':
    main()
