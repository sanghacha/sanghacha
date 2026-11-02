"""Frozen 2x2 planner-storage benchmark; synthetic graphs are NOT robot trials."""
import argparse
import gzip
import gc
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pipeline.mock_data import ROOT
from pipeline.types import Pose, SceneGraph, SceneObject, TargetSpec, TaskRequest
from planner.indexed import IndexedPlanner, VARIANTS
from planner.planner import plan_next_subgoal

PROTOCOL=dict(version='indexed-ablation-v2',region_counts=[3,24,192],objects_per_region=9,
    seeds=list(range(6300,6312)),frames=20,updates=['local','global'],queries=['hidden','visible'],
    noise_m=.00005,cell_size_m=.05,
    timing='serial; order offset=(frame+seed)%4; cyclic GC disabled during timed B calls; snapshot diff and summary maintenance included',
    controls='same policy, spatial hash, adjacency, object/region inputs and stream; only storage/incrementality differ')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,ensure_ascii=False).encode('utf-8')
    if path.suffix=='.gz': path.write_bytes(gzip.compress(data,mtime=0))
    else: path.write_bytes(data)


def load(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def stream(region_count,seed,update_kind,query_kind,frames):
    rng=np.random.default_rng(seed)
    nodes=[];groups={'workspace':None};regions={}
    for region in range(region_count):
        key=f'shelf_{region:04d}';groups[key]='workspace'
        cx,cy=.24+(region%8)*.24,-.18+(region//8)*.2
        regions[key]=dict(region_type='shelf',center=[cx,cy],size=[.17,.13],floor_top_m=.006)
        for col in range(6):
            missing=int(rng.integers(0,3))
            for level in range(2 if col<3 else 1):
                ident=f'track_{region:04d}_{col}_{level}'
                pose=Pose((cx+(col//2-1)*.05,cy+(-.0275 if col%2==0 else .0275),.020+(missing+level)*.028))
                nodes.append(SceneObject(ident,pose,(.028,)*3,parent_id=key))
    source_nodes=tuple(nodes)
    target=nodes[0].object_id if query_kind=='visible' else 'unresolved:target'
    request=TaskRequest(target,Pose((.08,.18,.020)),Pose((.08,-.18,.020)),TargetSpec((.9,.2,.1)))
    for tick in range(frames):
        started=perf_counter_ns()
        # Periodic no-op frames measure exact reuse without hiding scan costs.
        if tick and tick%4:
            selected=[int(rng.integers(region_count))] if update_kind=='local' else list(range(region_count))
            offsets={f'shelf_{r:04d}':rng.uniform(-PROTOCOL['noise_m'],PROTOCOL['noise_m'],2) for r in selected}
            nodes=[replace(obj,pose=Pose((source_nodes[i].pose.position[0]+offsets[obj.parent_id][0],
                        source_nodes[i].pose.position[1]+offsets[obj.parent_id][1],obj.pose.position[2])))
                   if obj.parent_id in offsets else obj for i,obj in enumerate(nodes)]
        graph=SceneGraph(tuple(nodes),(),groups,timestamp=float(tick),episode_id=f'indexed-{region_count}-{seed}-{update_kind}-{query_kind}',
                         source='rgbd',regions=regions)
        yield graph,request,perf_counter_ns()-started


def call(planner,graph,task,state,control_gc=False):
    enabled=gc.isenabled()
    if control_gc: gc.disable()
    try:
        try: return asdict(planner.plan_next_subgoal(graph,task,state))
        except ValueError as exc: return dict(error=str(exc))
    finally:
        if control_gc and enabled: gc.enable()


def benchmark(args):
    protocol_path=args.report_dir/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==PROTOCOL,'protocol differs; choose a new report directory'
    else:
        dump(protocol_path,PROTOCOL)  # Frozen before observing results.
    output=args.report_dir/'compute.json.gz'
    assert not output.exists(),'refusing to overwrite results; use another report directory'
    rows=[];state=np.zeros(6,np.float32)
    # Imports and a tiny run precede measurement, not counted as cold graph setup.
    tiny,task,_=next(stream(3,6099,'local','hidden',1))
    for variant in VARIANTS: call(IndexedPlanner(variant),tiny,task,state)
    for regions in PROTOCOL['region_counts']:
        for update_kind in PROTOCOL['updates']:
            for query_kind in PROTOCOL['queries']:
                for seed in PROTOCOL['seeds']:
                    planners={v:IndexedPlanner(v) for v in VARIANTS}
                    for frame,(graph,task,build_ns) in enumerate(stream(regions,seed,update_kind,query_kind,PROTOCOL['frames'])):
                        expected=None
                        if seed==PROTOCOL['seeds'][0] and frame==0 and update_kind=='local':
                            try: expected=asdict(plan_next_subgoal(graph,task,state))
                            except ValueError as exc: expected=dict(error=str(exc))
                        outcomes={}
                        offset=(frame+seed)%4
                        order=VARIANTS[offset:]+VARIANTS[:offset]
                        for variant in order:
                            planner=planners[variant]
                            outcome=call(planner,graph,task,state,control_gc=True)
                            outcomes[variant]=outcome
                            rows.append(dict(region_count=regions,object_count=len(graph.nodes),update_kind=update_kind,
                                query_kind=query_kind,seed=seed,frame=frame,variant=variant,
                                input_build_ns=build_ns,outcome=outcome,work=asdict(planner.stats),cache=planner.cache_counts()))
                        reference=outcomes[VARIANTS[0]]
                        assert all(value==reference for value in outcomes.values()),'variant plan mismatch'
                        assert expected is None or reference==expected,'legacy reference plan mismatch'
                    print(f'compute {regions*9} nodes, {update_kind}, {query_kind}, seed {seed}: all 4 plans equal',flush=True)
    # Separate allocation measurement; no tracemalloc overhead in timing above.
    memory=[]
    for regions in PROTOCOL['region_counts']:
        samples=list(stream(regions,PROTOCOL['seeds'][0],'local','hidden',PROTOCOL['frames']))
        for variant in VARIANTS:
            tracemalloc.start()
            planner=IndexedPlanner(variant)
            for graph,task,_ in samples: call(planner,graph,task,state)
            retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            memory.append(dict(object_count=regions*9,variant=variant,retained_bytes=retained,peak_bytes=peak))
            del planner
    sources=['src/planner/indexed.py','src/planner/planner.py','scripts/benchmark_indexed_planners.py']
    dump(output,dict(protocol=PROTOCOL,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},rows=rows,memory=memory,
        allocations_note='additional planner-owned Python allocations; input graphs prebuilt; not RSS; measured separately from timing'))
    print('saved',output,flush=True)


def physical(args):
    # Reuse the original immutable rollout harness, with only B replaced.
    import benchmark_planners as harness
    import action_generator.expert as expert
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    original_expert=expert.SO101Expert
    source_hash=sha256((ROOT/'src/planner/indexed.py').read_bytes()).hexdigest()
    for case in manifest['cases']:
        for variant in args.variants:
            folder=args.output/'expert'/variant/case['case_id']
            audit_path=folder/'indexed_audit.json'
            if audit_path.exists():
                old=json.loads(audit_path.read_text())
                assert old['indexed_source_sha256']==source_hash,'source changed; use new output path'
                continue
            assert not (folder/'result.json').exists(),'incomplete audit; preserve this output and use a new path'
            planner=IndexedPlanner(variant)
            works=[];command_hash=sha256();command_count=0
            class AuditedExpert(original_expert):
                def predict_action_chunk(self,observation,subgoal):
                    nonlocal command_count
                    chunk=super().predict_action_chunk(observation,subgoal)
                    command_hash.update(chunk.commands.tobytes())
                    command_count+=len(chunk.commands)
                    return chunk
            def plan(method,graph,task,state):
                assert method==variant
                try: return planner.plan_next_subgoal(graph,task,state)
                finally: works.append(asdict(planner.stats))
            harness.plan_baseline=plan
            expert.SO101Expert=AuditedExpert
            try:
                row=harness.run_case(manifest,case,variant,args)
            finally:
                expert.SO101Expert=original_expert
            dump(audit_path,dict(indexed_source_sha256=source_hash,command_sha256=command_hash.hexdigest(),
                command_count=command_count,works=works))
            print(json.dumps(dict(case_id=case['case_id'],variant=variant,success=row['success'],
                                  error=row['error'],commands=command_count)),flush=True)


def summarize(args):
    path=args.report_dir/'compute.json.gz'
    data=load(path if path.exists() else args.report_dir/'compute.json')
    protocol=data['protocol']
    grouped=defaultdict(list)
    for row in data['rows']:
        grouped[(row['object_count'],row['update_kind'],row['query_kind'],row['variant'])].append(row)
    summaries=[]
    for (n,update_kind,query_kind,variant),rows in grouped.items():
        warm=[r for r in rows if r['frame']>0]
        by_seed={seed:[r for r in warm if r['seed']==seed] for seed in protocol['seeds']}
        item=dict(object_count=n,update_kind=update_kind,query_kind=query_kind,variant=variant,
            cold_total_ms=statistics.median(r['work']['total_ns']/1e6 for r in rows if r['frame']==0),
            median_total_ms=statistics.median(r['work']['total_ns']/1e6 for r in warm),
            median_query_ms=statistics.median(r['work']['query_ns']/1e6 for r in warm),
            median_graph_plus_B_ms=statistics.median((r['input_build_ns']+r['work']['total_ns'])/1e6 for r in warm),
            median_snapshot_nodes_scanned=statistics.median(r['work']['snapshot_nodes_scanned'] for r in warm),
            median_summary_objects=statistics.median(r['work']['summary_objects_visited'] for r in warm),
            median_pair_tests=statistics.median(r['work']['spatial_pair_tests'] for r in warm),
            median_query_objects=statistics.median(r['work']['query_objects_visited'] for r in warm),
            seed_median_total_ms={str(seed):statistics.median(r['work']['total_ns']/1e6 for r in part) for seed,part in by_seed.items()})
        summaries.append(item)
    lookup={(r['object_count'],r['update_kind'],r['query_kind'],r['variant']):r for r in summaries}
    comparisons=[]
    for n in [v*9 for v in protocol['region_counts']]:
        for update_kind in protocol['updates']:
            for query_kind in protocol['queries']:
                a=lookup[(n,update_kind,query_kind,'flat_incremental')]
                b=lookup[(n,update_kind,query_kind,'hsg_incremental')]
                ratios=[a['seed_median_total_ms'][str(seed)]/b['seed_median_total_ms'][str(seed)] for seed in protocol['seeds']]
                bootstrap=np.median(np.random.default_rng(7123).choice(ratios,size=(10000,len(ratios))),axis=1)
                comparisons.append(dict(object_count=n,update_kind=update_kind,query_kind=query_kind,
                    flat_inc_over_hsg_inc_speedup_median=statistics.median(ratios),speedup_seed_range=[min(ratios),max(ratios)],
                    speedup_seed_bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist()))
    physical_rows=[]
    if args.manifest.exists():
        manifest=json.loads(args.manifest.read_text())
        for case in manifest['cases']:
            variants=[]
            for variant in VARIANTS:
                folder=args.output/'expert'/variant/case['case_id']
                row=json.loads((folder/'result.json').read_text())
                audit=json.loads((folder/'indexed_audit.json').read_text())
                assert row['initial_state_sha256']==case['initial_state_sha256']
                assert 'mismatch' not in (row['error'] or '')
                physical_rows.append(dict(result=row,audit=audit));variants.append((row,audit))
            assert len({audit['command_sha256'] for _,audit in variants})==1,'physical command sequence mismatch'
            assert len({row['success'] for row,_ in variants})==1,'physical outcome mismatch'
    summary=dict(protocol=protocol,compute=summaries,comparisons=comparisons,physical=physical_rows,
                 all_compute_plans_equal=True,all_physical_command_sequences_equal=True)
    dump(args.report_dir/'summary.json',summary)
    lines=['# Indexed HSG: 2 × 2 ablation','',
        'Four variants use the same policy, spatial hash and adjacency indices. HSG additionally caches regional totals and best candidates.',
        'Synthetic graph streams measure computation only. They do not represent robot reachability or RGB-D accuracy.',
        'All reported B times include full-snapshot diffing, index updates, summaries and querying. Initial setup is reported separately.',
        'Protocol: '+protocol['timing']+'. Reference counting remains enabled. GC outside the B call and RGB-D acquisition are not timed as B.',
        '', '| Nodes | Update | Query | Flat full ms | Flat inc ms | HSG full ms | HSG inc ms | Flat-inc / HSG-inc |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for comp in comparisons:
        n,u,q=comp['object_count'],comp['update_kind'],comp['query_kind']
        values=[lookup[(n,u,q,v)]['median_total_ms'] for v in VARIANTS]
        lines.append(f'| {n} | {u} | {q} | '+' | '.join(f'{v:.3f}' for v in values)+
                     f" | {comp['flat_inc_over_hsg_inc_speedup_median']:.2f}x |")
    lines+=['',f"Times are medians after frame 0. Ratios use per-seed medians ({len(protocol['seeds'])} seeds), not independent frame-level significance tests.",
            'Seed-level bootstrap 95% intervals and ranges are in summary.json. These are conditional timing results on this machine, not universal speed guarantees.',
            'Global updates and visible-target queries are included as controls; HSG need not improve these workloads.',
            '', '## Physical validation','']
    for variant in VARIANTS:
        part=[r['result'] for r in physical_rows if r['result']['method']==variant]
        lines.append(f"- {variant}: {sum(r['success'] for r in part)}/{len(part)} physical retrievals.")
    lines+=['','All four variants produced bitwise identical command sequences per physical scene. Failed scenes remain in the denominator.',
            'This validates behavioral equivalence, not a higher physical success rate.',
            '', '## Allocation measurements','',
            'Separate tracemalloc runs count additional planner-owned Python allocations, excluding prebuilt input graphs and native/GPU memory.',
            '', '| Nodes | Variant | Retained KiB | Peak KiB |','|---:|---|---:|---:|']
    for row in data['memory']:
        lines.append(f"| {row['object_count']} | {row['variant']} | {row['retained_bytes']/1024:.1f} | {row['peak_bytes']/1024:.1f} |")
    lines+=['','Full source hashes, per-frame work counts, cold-start times, input-build-plus-B times and seed ranges are in compute.json.gz / summary.json.',
            'Input construction in this synthetic experiment is not RGB-D inference time. No end-to-end robot speedup follows from B-only measurements.','']
    (args.report_dir/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    sizes=[v*9 for v in protocol['region_counts']]
    for update_kind,query_kind,label in [('local','hidden','Local update / hidden target'),
                                       ('global','hidden','Global update / hidden target'),
                                       ('local','visible','Local update / visible target')]:
        rows=[next(c for c in comparisons if c['object_count']==n and c['update_kind']==update_kind and c['query_kind']==query_kind) for n in sizes]
        y=[c['flat_inc_over_hsg_inc_speedup_median'] for c in rows]
        lo=[c['speedup_seed_bootstrap_95'][0] for c in rows];hi=[c['speedup_seed_bootstrap_95'][1] for c in rows]
        axes[0].plot(sizes,y,'o-',label=label)
        axes[0].fill_between(sizes,lo,hi,alpha=.12)
    axes[0].axhline(1,color='black',linestyle='--',linewidth=1)
    axes[0].set(xscale='log',xlabel='Observed objects (synthetic graphs)',ylabel='Flat incremental / HSG incremental',title='Speedup including snapshot diff + updates + query')
    axes[0].legend(fontsize=8)
    for variant in VARIANTS:
        y=[lookup[(n,'local','hidden',variant)]['median_total_ms'] for n in sizes]
        axes[1].plot(sizes,y,'o-',label=variant)
    axes[1].set(xscale='log',yscale='log',xlabel='Observed objects',ylabel='B call (ms)',title='Local updates, hidden target; same plans')
    axes[1].legend(fontsize=8)
    fig.savefig(args.report_dir/'results.png',dpi=170);plt.close(fig)
    print(json.dumps(comparisons,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('compute','physical','report'))
    parser.add_argument('--report-dir',type=Path,default=ROOT/'experiments/hsg_indexed')
    parser.add_argument('--manifest',type=Path,default=ROOT/'experiments/hsg_indexed/physical_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/hsg_indexed/physical')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--controller',choices=('expert',),default='expert')
    parser.add_argument('--checkpoint',type=Path,default=None)
    parser.add_argument('--seeds',type=int,nargs='+',help='compute only: freeze a new independent seed cohort')
    args=parser.parse_args()
    if args.seeds:
        assert args.operation=='compute' and len(set(args.seeds))==len(args.seeds)
        PROTOCOL['seeds']=args.seeds
    {'compute':benchmark,'physical':physical,'report':summarize}[args.operation](args)


if __name__=='__main__': main()
