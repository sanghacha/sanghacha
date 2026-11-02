"""Regression only: replay old real RGB-D planner inputs against the legacy B."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pipeline.mock_data import ROOT
from pipeline.types import Pose, TargetSpec, TaskRequest
from planner.indexed import IndexedPlanner, VARIANTS
from planner.planner import plan_next_subgoal
from report_planner_benchmark import parse_graph


def outcome(call):
    try: return asdict(call())
    except ValueError as exc: return dict(error=str(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'outputs/planner_benchmark/main/expert')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/hsg_indexed/replay_audit.json')
    args=parser.parse_args()
    episodes=calls=0;digests={}
    for path in sorted(args.inputs.glob('*/*/plan_inputs.json')):
        episodes+=1
        planners={v:IndexedPlanner(v) for v in VARIANTS}
        for value in json.loads(path.read_text(encoding='utf-8')):
            graph=parse_graph(value['graph']);t=value['task']
            pose=lambda p:Pose(tuple(p['position']),tuple(p['quaternion_xyzw']))
            task=TaskRequest(t['target_object_id'],pose(t['destination']),pose(t['buffer_pose']),
                             TargetSpec(**t['target_spec']) if t['target_spec'] else None)
            state=np.asarray(value['robot_state'],dtype=np.float32)
            expected=outcome(lambda:plan_next_subgoal(graph,task,state))
            for planner in planners.values():
                assert outcome(lambda:planner.plan_next_subgoal(graph,task,state))==expected,str(path)
            calls+=1
        digests[str(path.relative_to(args.inputs))]=sha256(path.read_bytes()).hexdigest()
    assert calls,'no saved planner inputs'
    report=dict(purpose='behavioral regression; old seeds are NOT a new performance evaluation',
                input_files=episodes,graph_calls=calls,variant_calls=calls*4,all_match_legacy=True,input_sha256=digests)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{episodes} saved input files, {calls} graph calls x 4 variants: all match legacy',flush=True)


if __name__=='__main__': main()
