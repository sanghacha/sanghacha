"""Fixed-ACT A/B/C comparison: robot-base Greedy, no-hidden HSG, full HSG.

Preparation is outcome-blind. Hidden-target strata require an observation-derived
hidden-space cue, so the ablated feature is actually present. No rollout result
is used to select a scene. This is a cue-available benchmark, not all scenes.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pipeline.mock_data import ROOT
from pipeline.target import TargetResolver
from pipeline.types import Pose, TargetSpec, TaskRequest
from planner.act_comparison import CONDITIONS, ComparisonPlanner
import benchmark_planners as harness
from report_planner_benchmark import parse_graph, failure


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def digest(value):
    return sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def initial_request(manifest,case):
    spec=TargetSpec(**case['target_spec'])
    z=manifest['config']['placement_trays']['floor_thickness']+spec.size_m[2]/2
    request=TaskRequest('unresolved:target',Pose((.08,.18,z)),Pose((.08,-.18,z)),spec)
    graph=parse_graph(case['initial_graph'])
    return graph,TargetResolver().resolve(graph,request)


def prepare(args):
    assert args.count%4==0 and args.count>0,'count must be positive and divisible by four'
    assert not args.manifest.exists(),'frozen manifest already exists'
    candidate_path=args.output/'candidate_manifest.json'
    if not candidate_path.exists():
        proposal=argparse.Namespace(manifest=candidate_path,output=args.output/'screening',count=args.count*4,
            start_seed=args.start_seed,max_scan=500,max_subgoals=12,max_steps=450)
        harness.prepare(proposal)
    candidates=json.loads(candidate_path.read_text(encoding='utf-8'))
    selected=[];screen=[];counts={key:0 for key in harness.GROUPS}
    for original in candidates['cases']:
        graph,request=initial_request(candidates,original)
        decisions={}
        for condition in CONDITIONS:
            planner=ComparisonPlanner(condition)
            goal=planner.plan_next_subgoal(graph,request,np.zeros(6,np.float32))
            decisions[condition]=dict(goal=asdict(goal),decision=planner.last_decision)
        hidden=original['group'].startswith('unlocalized')
        has_cue=any(row.get('hidden_count',0)>0 for row in decisions['c_full']['decision'].get('candidates',[]))
        eligible=not hidden or has_cue
        screen.append(dict(seed=original['seed'],group=original['group'],hidden_cue_exists=has_cue,
                           eligible=eligible,quota_already_filled=counts[original['group']]>=args.count//4))
        if eligible and counts[original['group']]<args.count//4:
            selected.append(dict(original,case_id=f'scene_{len(selected):03d}',initial_decisions=decisions))
            counts[original['group']]+=1
    assert len(selected)==args.count,f'insufficient eligible scenes: {counts}; preserve screening and prepare a new larger cohort'
    manifest=dict(candidates,protocol='fixed-act-abc-v1',cases=selected,candidate_screening=screen,
        scope='visible controls + unlocalized targets with available height-derived hidden-space cues',
        selection='initial observation and cue availability only; no ACT outcomes or method success used',
        conditions=list(CONDITIONS),robot_distance='Euclidean 3D distance to robot_base origin (0,0,0)',
        shared_rules=['target priority when locally clear','same local geometry eligibility',
            'same region hint','same completion-based placed-object history',
            'same placement policy without hidden-space scores','same ACT and Subgoal encoding'],
        ablation='disable computation/use of inferred hidden-space counts; visible poses and support relations remain')
    dump(args.manifest,manifest)
    print('frozen:',args.manifest,counts,flush=True)


def run(args):
    import torch
    sys.path.insert(0,str(ROOT/'training'))
    import so101_act.policy as policy_module
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    used={int(item['group_id'].rsplit('seed_',1)[1]) for part in payload['split'].values() for item in part}
    assert not used & {case['seed'] for case in manifest['cases']},'training/validation seed overlap'
    checkpoint_hash=sha256(args.checkpoint.read_bytes()).hexdigest()
    sources=['src/planner/act_comparison.py','src/planner/indexed.py','src/planner/baselines.py','src/planner/planner.py',
             'src/action_generator/goal.py','training/so101_act/policy.py',
             'src/pipeline/rgbd_env.py','src/pipeline/runner.py','scripts/benchmark_act_conditions.py']
    metadata=dict(protocol='fixed-act-abc-v1',manifest_sha256=digest(manifest),checkpoint_sha256=checkpoint_hash,
        checkpoint=str(args.checkpoint),controller='act',device=args.device,execute_steps=4,temporal_ensemble=False,
        dataset_seeds=sorted(used),evaluation_seeds=[case['seed'] for case in manifest['cases']],
        torch=torch.__version__,source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None)
    meta_path=args.output/'run_metadata.json'
    if meta_path.exists():
        assert json.loads(meta_path.read_text())==metadata,'configuration/source differs; use a new output path'
    else:
        dump(meta_path,metadata)
    original_runner=harness.run_pipeline
    original_policy=policy_module.ACTPolicy
    args.controller='act'
    for case in manifest['cases']:
        for condition in args.conditions:
            folder=args.output/'act'/condition/case['case_id']
            audit_path=folder/'condition_audit.json'
            if audit_path.exists():
                assert json.loads(audit_path.read_text())['checkpoint_sha256']==checkpoint_hash
                continue
            assert not (folder/'result.json').exists(),'incomplete audit; use a new output path'
            planner=ComparisonPlanner(condition)
            decisions=[];runtime={};commands=sha256()
            class AuditedACT(original_policy):
                def __init__(self,*a,**kw):
                    super().__init__(*a,**kw)
                    assert self.execute_steps==4 and not self.temporal_ensemble
                    runtime.update(execute_steps=self.execute_steps,temporal_ensemble=self.temporal_ensemble,
                                   model_config=asdict(self.config),goal_dim=25)
                def predict_action_chunk(self,observation,subgoal):
                    chunk=super().predict_action_chunk(observation,subgoal)
                    commands.update(chunk.commands.tobytes())
                    return chunk
            def plan(method,graph,request,state):
                assert method==condition
                goal=planner.plan_next_subgoal(graph,request,state)
                decisions.append(dict(timestamp=graph.timestamp,goal=asdict(goal),decision=planner.last_decision))
                return goal
            def runner(*a,**kw):
                event=kw.get('on_event')
                def observed(kind,**data):
                    if kind=='completed': planner.completed(data['subgoal'])
                    if event is not None: event(kind,**data)
                kw['on_event']=observed
                return original_runner(*a,**kw)
            torch.manual_seed(7199)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            harness.plan_baseline=plan
            harness.run_pipeline=runner
            policy_module.ACTPolicy=AuditedACT
            try:
                row=harness.run_case(manifest,case,condition,args)
            finally:
                harness.run_pipeline=original_runner
                policy_module.ACTPolicy=original_policy
            dump(audit_path,dict(condition=condition,checkpoint_sha256=checkpoint_hash,runtime=runtime,
                command_sha256=commands.hexdigest(),decisions=decisions,placed_tracks=sorted(planner._placed)))
            print(json.dumps({k:row[k] for k in ('case_id','method','success','completed_removals','attempted_picks','error')}),flush=True)


def report(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    metadata=json.loads((args.output/'run_metadata.json').read_text())
    rows=[];audits=[]
    for case in manifest['cases']:
        for condition in CONDITIONS:
            folder=args.output/'act'/condition/case['case_id']
            row=json.loads((folder/'result.json').read_text())
            audit=json.loads((folder/'condition_audit.json').read_text())
            assert row['checkpoint_sha256']==audit['checkpoint_sha256']==metadata['checkpoint_sha256']
            assert row['manifest_sha256']==digest(manifest)
            assert row['initial_state_sha256']==case['initial_state_sha256']
            assert 'mismatch' not in (row['error'] or '')
            rows.append(row);audits.append(audit)
    summaries={}
    for condition in CONDITIONS:
        part=[r for r in rows if r['method']==condition]
        counts={}
        for r in part: counts[failure(r)]=counts.get(failure(r),0)+1
        summaries[condition]=dict(n=len(part),successes=sum(r['success'] for r in part),
            failure_counts=counts,completed_removals=sum(r['completed_removals'] for r in part),
            attempted_picks=sum(r['attempted_picks'] for r in part))
    result=dict(protocol=manifest['protocol'],metadata=metadata,summary=summaries,rows=rows,audits=audits)
    dump(args.report_dir/'results.json',result)
    lines=['# 동일 ACT의 A/B/C 비교','',
        'A: robot_base 원점에서 가까운 관측 후보부터 치우는 Greedy.',
        'B: 동일 HSG planner에서 높이·가림 기반 숨은 공간 추정을 계산·사용하지 않음.',
        'C: 관측에서 추정한 숨은 공간 점수를 이용하는 HSG planner.','',
        f"장면 {len(manifest['cases'])}개 × 3조건. 같은 ACT 체크포인트·4-step 실행 cache·Subgoal 25D encoding을 고정했다.",
        '초기 물리 상태와 첫 RGB-D 그래프를 짝지었고, 성공은 실제 요청한 물체의 물리적 회수로 판정했다.',
        '숨은 목표 조건은 추정 단서가 존재하는 장면으로 한정했다. 전체 임의 장면의 평균 성공률로 일반화하지 않는다.','',
        '| 조건 | 실제 회수 성공 | 전체 완료 제거 수¹ | 전체 시도 PICK 수¹ |','|---|---:|---:|---:|']
    for key,value in summaries.items():
        lines.append(f"| {key} | {value['successes']}/{value['n']} | {value['completed_removals']} | {value['attempted_picks']} |")
    lines+=['','¹ 실패가 포함된 총합이며, 적을수록 우수하다는 뜻이 아니다. 성공률과 이력을 함께 읽어야 한다.',
            '', '## 실패/종료 유형','']
    for key,value in summaries.items(): lines.append(f"- {key}: {value['failure_counts']}")
    lines+=['','## 해석','',
        '- A 대 C는 표현뿐 아니라 의사결정 규칙도 달라지는 전체 방식 비교다.',
        '- B 대 C는 같은 planner에서 숨은 공간 추정 기능의 사용을 제거하는 ablation이다. 목표 pose를 유지하므로 모든 암묵적 높이 정보를 지운다는 의미는 아니다.',
        '- 정답 hidden object ID/pose는 어떤 조건에도 넣지 않았다. 과거 미관측 물체 pose를 현재 파지 대상으로 쓰지도 않는다.',
        '- B와 C 모두 단서가 0이어도 탐색을 지속한다. 배치에서는 두 조건 모두 숨은 공간 점수를 사용하지 않는다.',
        '- 현재 집기 후보 검사는 국소 적층 geometry 검사다. 충돌 없는 실제 접근이나 ACT 성공을 보장하지 않는다.',
        '- 같은 ACT를 사용해도 선택한 물체의 위치·자세에 따라 실행 난도가 달라진다. 제어 실패를 planner 논리 실패와 구분한다.',
        '- 소규모 cohort는 연결·조건 분리 확인용이다. 우수성을 주장하려면 새로 고정한 더 큰 cohort와 불확실성 분석이 필요하다.','']
    (args.report_dir/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','report'))
    parser.add_argument('--manifest',type=Path,default=ROOT/'experiments/act_conditions/pilot_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/act_conditions/pilot')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'experiments/act_conditions')
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--start-seed',type=int,default=7200)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'training/runs/act_rgbd_shelves_absolute/best.pt')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'report':report}[args.operation](args)


if __name__=='__main__': main()
