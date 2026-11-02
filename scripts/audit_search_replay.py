"""Replay every recorded public decision and reject privileged identifiers."""
import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import re

import numpy as np

from benchmark_planner_search import ComparisonPlanner, CONDITIONS, digest, dump
from report_planner_benchmark import parse_graph
from pipeline.types import Pose, TargetSpec, TaskRequest


def audit(args):
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    if (args.input/'public_inputs.json.gz').exists():
        bundle=json.loads(gzip.decompress((args.input/'public_inputs.json.gz').read_bytes()))
        bundled_inputs={(r['case_id'],r['condition']):r['steps'] for r in bundle}
        bundled_rows={(r['case_id'],r['condition']):r for r in json.loads(
            gzip.decompress((args.input/'trials.json.gz').read_bytes()))}
    else:
        bundled_inputs=bundled_rows=None
    decisions=0;trials=0;max_initial_error=0.
    for case in manifest['cases']:
        for condition in CONDITIONS:
            folder=args.input/'runs'/condition/case['case_id']
            row=(bundled_rows[(case['case_id'],condition)] if bundled_rows is not None else
                 json.loads((folder/'result.json').read_text(encoding='utf-8')))
            inputs=(bundled_inputs[(case['case_id'],condition)] if bundled_inputs is not None else
                    json.loads((folder/'plan_inputs.json').read_text(encoding='utf-8')))
            assert row['identity']['manifest_sha256']==digest(manifest)
            assert row['identity']['source_sha256']==manifest['protocol']['source_sha256']
            assert not re.search(r'shelf_[123]_obj_\d+',json.dumps(inputs)), 'physical object ID in public input'
            planner=ComparisonPlanner(condition)
            for i,entry in enumerate(inputs):
                graph=parse_graph(entry['graph']);raw=entry['task']
                task=TaskRequest(raw['target_object_id'],Pose(**raw['destination']),Pose(**raw['buffer_pose']),
                                 TargetSpec(**raw['target_spec']) if raw['target_spec'] else None)
                assert graph.source=='rgbd' and graph.held_object_id is None
                assert all(o.object_id.startswith('instance_') for o in graph.nodes)
                # Production planners validate q but do not condition selection
                # on it: all choices use the fixed robot-base origin.
                goal=planner.plan_next_subgoal(graph,task,np.zeros(6,np.float32))
                assert digest(asdict(goal))==digest(entry['goal']), (case['case_id'],condition,i)
                assert digest(planner.last_decision)==digest(entry['decision'])
                assert goal.object_id in {o.object_id for o in graph.nodes}
                decisions+=1
                if i<len(inputs)-1:
                    planner.completed(replace(goal,primitive='PLACE',place_pose=task.buffer_pose))
            trials+=1
            max_initial_error=max(max_initial_error,row['initial_graph_max_error'])
    value=dict(manifest_sha256=digest(manifest),trials=trials,decisions=decisions,
        every_goal_and_explanation_reproduced=True,no_physical_object_ids_in_public_inputs=True,
        max_initial_graph_numeric_error=max_initial_error,
        q_note='zero q only for offline selection replay; current planners use q for validation, fixed base for distances')
    dump(args.input/'replay_audit.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    audit(parser.parse_args())
