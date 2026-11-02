"""Independent oracle audit. Never imported by the planner/executor benchmark."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import benchmark_planner_search as benchmark
from report_planner_benchmark import parse_graph


def ancestors(edges,target):
    above={}
    for upper,lower in edges:
        above.setdefault(lower,set()).add(upper)
    pending=list(above.get(target,()));result=set()
    while pending:
        key=pending.pop()
        assert key!=target,'support graph cycle'
        if key not in result:
            result.add(key);pending.extend(above.get(key,()))
    return result


def audit(args):
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    if (args.input/'layout_protocol.json').exists():
        import benchmark_planner_occlusion as stress
        stress.ShelfScene.sample_layout=stress.random_order
        hashes=stress.hashes()
    else:
        hashes=benchmark.source_hashes()
    assert hashes==manifest['protocol']['source_sha256']
    rows=[]
    for case in manifest['cases']:
        sim=benchmark.MujocoEnvironment('shelves',args.input/'config.json',render=False)
        try:
            sim.reset(case['seed'],privileged=False)
            assert sha256(sim.data.qpos.tobytes()).hexdigest()==case['initial_state_sha256']
            graph=benchmark.generate_scene_graph(sim.observe(privileged=True))
            edges=[(e.source,e.target) for e in graph.edges if e.relation=='on_top_of']
            minimal=ancestors(edges,case['target_gt_id'])
            initial=parse_graph(case['initial_graph'])
            observed={benchmark.nearest_physical(sim,o.pose.position,set()) for o in initial.nodes}
            observed.discard(None)
            rows.append(dict(case_id=case['case_id'],seed=case['seed'],group=case['group'],
                support_removal_lower_bound=len(minimal),necessary_blockers=sorted(minimal),
                requested_target_has_geometric_candidate=case['target_gt_id'] in observed,
                physically_unrepresented_objects=len(sim.object_bodies)-len(observed),
                inferred_hidden_count=case['initial_inferred_hidden_count']))
        finally:
            sim.close()
    value=dict(manifest_sha256=benchmark.digest(manifest),rows=rows,
        scope='independent oracle diagnostic; initial vertical-support lower bound under one-object top-clear removal',
        warning='unrepresented is not synonymous with zero visible camera pixels; 15mm control-pose association')
    benchmark.dump(args.input/'oracle_audit.json',value)
    hidden=[r for r in rows if r['group'].startswith('unlocalized')]
    print(json.dumps(dict(n=len(rows),hidden=len(hidden),
        hidden_with_cue=sum(r['inferred_hidden_count']>0 for r in hidden),
        hidden_with_geometric_target=sum(r['requested_target_has_geometric_candidate'] for r in hidden))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=benchmark.ROOT/'outputs/planner_search/confirmatory')
    audit(parser.parse_args())
