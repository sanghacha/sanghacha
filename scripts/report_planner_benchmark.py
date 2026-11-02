"""Validate paired results and report successes, censoring and common-input timing."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import itertools
import json
from math import comb, sqrt
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco
from pipeline.mock_data import ROOT
from pipeline.target import TargetResolver
from pipeline.types import Edge, Pose, SceneGraph, SceneObject, TargetSpec, TaskRequest
from planner.baselines import METHODS, plan_baseline
from planner.planner import _hidden_by_shelf, _on_top_of

GROUPS = ('visible_free','visible_blocked','unlocalized_known_region','unlocalized_unknown_region')
LABELS = dict(visible_free='목표 식별·즉시 집기', visible_blocked='목표 식별·방해물 있음',
              unlocalized_known_region='목표 미식별·선반 알려짐', unlocalized_unknown_region='목표 미식별·선반 모름')


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False),encoding='utf-8')


def digest(value):
    return sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def wilson(k,n):
    if not n:
        return None
    z=1.959963984540054
    p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def failure(row):
    error=row['error'] or ''
    if row['success']:
        return 'success'
    if row['observational_complete']:
        return 'observational_completion_failed_physical_audit'
    for needle, label in [('no hidden space left','planner_no_hidden_space'),
                          ('unreachable expert waypoint','controller_ik_unreachable'),
                          ('expert stalled','controller_stalled'),
                          ('cannot localize','active_track_lost'),
                          ('ambiguous target','target_ambiguous'),
                          ('subgoal step budget','execution_budget'),
                          ('subgoal count budget','planning_action_budget'),
                          ('outside workspace','invalid_subgoal_workspace'),
                          ('no free placement','no_placement_space')]:
        if needle in error:
            return label
    return error or 'physical_audit_failure'


def discovery_picks(row):
    """An occluder can reveal the goal after PICK, before its buffer PLACE."""
    discovered=row.get('discovered_at_sim_seconds')
    if discovered is None:
        return None
    return sum(event['event']=='completed' and event['primitive']=='PICK' and not event['target']
               and event['timestamp']<=discovered+1e-9 for event in row['trace'])


def paired(left,right):
    a={r['case_id']:r for r in left}
    b={r['case_id']:r for r in right}
    ids=sorted(a.keys() & b.keys())
    wins=sum(a[k]['success'] and not b[k]['success'] for k in ids)
    losses=sum(b[k]['success'] and not a[k]['success'] for k in ids)
    both=[k for k in ids if a[k]['success'] and b[k]['success']]
    n=wins+losses
    p=min(1., 2*sum(comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
    def trace(row):
        return [(r['event'],r['primitive'],r['object_id']) for r in row['trace']]
    return dict(n=len(ids),left_only_success=wins,right_only_success=losses,
        success_delta_percentage_points=100*(wins-losses)/len(ids) if ids else None,
        mcnemar_exact_two_sided_p=p, both_success_n=len(both),
        mean_removal_delta_both_success=statistics.mean(a[k]['completed_removals']-b[k]['completed_removals'] for k in both) if both else None,
        mean_simulation_seconds_delta_both_success=statistics.mean(a[k]['simulation_seconds']-b[k]['simulation_seconds'] for k in both) if both else None,
        identical_decision_and_completion_traces=sum(trace(a[k])==trace(b[k]) for k in ids))


def summarize(rows):
    result={}
    for method in METHODS:
        result[method]={}
        for group in ('all',)+GROUPS:
            subset=[r for r in rows if r['method']==method and (group=='all' or r['group']==group)]
            success=[r for r in subset if r['success']]
            discovered=[r for r in subset if r['discovery_after_completed_removals'] is not None]
            k,n=len(success),len(subset)
            result[method][group]=dict(n=n,successes=k,success_rate=k/n if n else None,
                success_rate_wilson_95=wilson(k,n), failure_counts=dict(Counter(failure(r) for r in subset)),
                mean_removals_success=statistics.mean(r['completed_removals'] for r in success) if success else None,
                mean_simulation_seconds_success=statistics.mean(r['simulation_seconds'] for r in success) if success else None,
                target_discovered_n=len(discovered),target_never_correctly_bound_n=n-len(discovered),
                mean_removals_until_discovery_among_discovered=statistics.mean(r['discovery_after_completed_removals'] for r in discovered) if discovered else None,
                mean_search_picks_until_discovery_among_discovered=statistics.mean(discovery_picks(r) for r in discovered) if discovered else None,
                attempted_picks_all=sum(r['attempted_picks'] for r in subset),
                completed_removals_all=sum(r['completed_removals'] for r in subset),
                total_wall_seconds=sum(r['wall_seconds'] for r in subset),
                total_plan_seconds=sum(r['plan_seconds'] for r in subset),
                total_perception_seconds=sum(r['perception_seconds'] for r in subset),
                false_observational_completion=sum(r['observational_complete'] and not r['success'] for r in subset))
    comparisons={a+'_minus_'+b:paired([r for r in rows if r['method']==a], [r for r in rows if r['method']==b])
                 for a,b in [('hsg','greedy'),('hsg','flat'),('flat','greedy')]}
    return dict(by_method=result,paired=comparisons)


def parse_graph(data):
    def pose(p): return Pose(tuple(p['position']),tuple(p['quaternion_xyzw']))
    return SceneGraph(tuple(SceneObject(o['object_id'],pose(o['pose']),tuple(o['size']),o['label'],o['parent_id'],
                        tuple(o['color_rgb']) if o['color_rgb'] is not None else None) for o in data['nodes']),
                      tuple(Edge(**e) for e in data['edges']),data['groups'],data['held_object_id'],
                      data['timestamp'],data['episode_id'],data['frame_id'],data['source'],data['regions'])


def timing(manifest,repeats):
    rows=[]
    state=np.asarray(manifest['config']['shelf_scene']['home'],dtype=np.float32)
    for case in manifest['cases']:
        graph=parse_graph(case['initial_graph'])
        spec=TargetSpec(**case['target_spec'])
        z=manifest['config']['placement_trays']['floor_thickness']+spec.size_m[2]/2
        task=TaskRequest('unresolved:target',Pose((.08,.18,z)),Pose((.08,-.18,z)),spec)
        task=TargetResolver().resolve(graph,task)
        samples={method:[] for method in METHODS}
        outcomes={}
        for trial in range(repeats+1):
            # Alternate order to avoid consistently rewarding one warm-cache method.
            methods=METHODS[trial%3:]+METHODS[:trial%3]
            for method in methods:
                start=time.perf_counter_ns()
                try:
                    goal=plan_baseline(method,graph,task,state)
                    outcome=dict(primitive=goal.primitive,object_id=goal.object_id)
                except ValueError as exc:
                    outcome=dict(error=str(exc))
                elapsed=(time.perf_counter_ns()-start)/1e6
                outcomes[method]=outcome
                if trial:
                    samples[method].append(elapsed)
        rows.append(dict(case_id=case['case_id'],group=case['group'],outcomes=outcomes,
                         hidden_counts_by_region={k:[n for _,n in stacks if n>0]
                             for k,stacks in _hidden_by_shelf(graph,_on_top_of(graph)).items()
                             if any(n>0 for _,n in stacks)},
                         median_milliseconds={m:statistics.median(v) for m,v in samples.items()}))
    aggregate={method:dict(median_case_milliseconds=statistics.median(r['median_milliseconds'][method] for r in rows),
        valid_initial_plans=sum('error' not in r['outcomes'][method] for r in rows)) for method in METHODS}
    unknown=[r for r in rows if r['group']=='unlocalized_unknown_region']
    diagnostic=dict(n=len(unknown),initial_flat_hsg_selection_disagreements=sum(
        r['outcomes']['flat']!=r['outcomes']['hsg'] for r in unknown),
        number_of_candidate_regions_histogram=dict(Counter(len(r['hidden_counts_by_region']) for r in unknown)))
    return dict(protocol='serial common initial inputs; first call warm-up; rotating method order; no A or C',
                repeats=repeats,robot_state='configured home; current planners only validate this argument',
                rows=rows,aggregate=aggregate,initial_hierarchy_diagnostic=diagnostic)


def audit_captures(output,controller,case_ids):
    from perception.acquisition import load_frame
    comparisons=exact_rgb=exact_depth=0
    max_rgb=max_changed=0
    for case_id in case_ids:
        captured=[]
        for method in METHODS:
            _,frames=load_frame(output/controller/method/case_id/'perception/snapshot_000/inputs/frame_000')
            captured.append(frames)
        for frames in captured[1:]:
            for key,reference in captured[0].items():
                comparisons+=1
                difference=np.abs(reference['rgb'].astype(np.int16)-frames[key]['rgb'].astype(np.int16))
                exact_rgb+=int(not difference.any())
                max_rgb=max(max_rgb,int(difference.max()))
                max_changed=max(max_changed,int(np.count_nonzero(difference.max(axis=2))))
                exact_depth+=int(np.array_equal(reference['depth'],frames[key]['depth']))
    return dict(scenes_checked=len(case_ids),methods_per_scene=len(METHODS),camera_frame_comparisons=comparisons,
        rgb_pixel_identical_frames=exact_rgb,depth_array_identical_frames=exact_depth,
        maximum_rgb_channel_absolute_difference=max_rgb,maximum_changed_rgb_pixels_per_frame=max_changed,
        primary_pairing_contract='strictly identical initial qpos and initial common SceneGraph; verified in each rollout')


def load_rows(manifest,output,controller,case_ids):
    rows=[]
    for case in manifest['cases']:
        if case['case_id'] not in case_ids:
            continue
        for method in METHODS:
            path=output/controller/method/case['case_id']/'result.json'
            if not path.exists():
                raise ValueError(f'missing result: {path}')
            row=json.loads(path.read_text(encoding='utf-8'))
            row['search_picks_before_discovery']=discovery_picks(row)
            assert row['manifest_sha256']==digest(manifest)
            assert row['method']==method and row['controller']==controller and row['seed']==case['seed']
            assert row['case_id']==case['case_id'] and row['group']==case['group']
            assert row['initial_state_sha256']==case['initial_state_sha256']
            assert 'mismatch' not in (row['error'] or ''), 'protocol mismatch cannot count as a policy failure'
            rows.append(row)
    assert len(rows)==len(case_ids)*len(METHODS), 'requested cases are missing from manifest'
    assert len({row['checkpoint_sha256'] for row in rows})==1, 'mixed checkpoints'
    return rows


def write_report(path,report):
    main=report['expert']['summary']
    lines=['# 플래너 비교 실험 결과','',
           '52개 고정 장면 × Greedy / Flat SG / HSG, 총 156회 MuJoCo 물리 실행. 각 조건은 13개 장면이다.',
           '같은 초기 qpos와 첫 RGB-D 그래프를 검증했고, 실제 목표의 접촉·위치·속도로 회수 성공을 별도 판정했다.',
           '', '| 조건 | Greedy | Flat SG | HSG |','|---|---:|---:|---:|']
    for group in ('all',)+GROUPS:
        cells=[]
        for method in METHODS:
            s=main['by_method'][method][group]
            cells.append(f"{s['successes']}/{s['n']} ({s['success_rate']:.1%})")
        lines.append('| '+('전체' if group=='all' else LABELS[group])+' | '+' | '.join(cells)+' |')
    hp=main['paired']['hsg_minus_flat']
    gp=main['paired']['hsg_minus_greedy']
    diagnostic=report['timing']['initial_hierarchy_diagnostic']
    lines+=['',f"HSG−Greedy 성공률 차이는 {gp['success_delta_percentage_points']:+.2f}%p이며, "
            f"paired McNemar 정확 검정 p={gp['mcnemar_exact_two_sided_p']:.3f}다. 향상을 입증한 결과로 해석하지 않는다.",
            f"Flat SG와 HSG는 {hp['identical_decision_and_completion_traces']}/{hp['n']} 장면에서 같은 행동 이력을 냈다.",
            f"선반 미지정·목표 미식별 초기 장면 {diagnostic['n']}개에서 두 방법의 선택이 갈라진 경우는 "
            f"{diagnostic['initial_flat_hsg_selection_disagreements']}개였다.",
            '따라서 이번 장면 집합은 계층 탐색의 추가 효과를 구별하기에 부족하다. 계층 구조 일반의 효과가 없다는 결론도 내리지 않는다.']
    audit=report['pairing_audit']['expert']
    lines+=['',f"초기 센서 감사: 비교한 {audit['camera_frame_comparisons']}개 카메라 프레임 중 "
            f"RGB 픽셀이 완전히 같은 것은 {audit['rgb_pixel_identical_frames']}개, 깊이 배열이 완전히 같은 것은 "
            f"{audit['depth_array_identical_frames']}개다. RGB 최대 채널 차이는 "
            f"{audit['maximum_rgb_channel_absolute_difference']}/255, 프레임당 차이 픽셀 최대 "
            f"{audit['maximum_changed_rgb_pixels_per_frame']}개였다.",
            'RGB가 바이트 단위로 모두 같았다고 주장하지 않는다. B가 받은 초기 그래프와 물리 상태는 엄격히 동일하다.']
    lines+=['','성공률의 Wilson 95% 구간:', '']
    for method in METHODS:
        s=main['by_method'][method]['all'];lo,hi=s['success_rate_wilson_95']
        lines.append(f"- {method}: {s['success_rate']:.1%}, 95% CI {lo:.1%}–{hi:.1%}")
    lines+=['','## 동일 장면 쌍 비교','',
        '| 비교 (왼쪽−오른쪽) | 왼쪽만 성공 | 오른쪽만 성공 | 둘 다 성공 | 제거 횟수 차이¹ | 수행 시간 차이¹ | 동일 행동 이력 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '해당 없음' if v is None else f'{v:+.2f}'
    for key,s in main['paired'].items():
        lines.append(f"| {key} | {s['left_only_success']} | {s['right_only_success']} | {s['both_success_n']} | "
                     f"{fmt(s['mean_removal_delta_both_success'])} | {fmt(s['mean_simulation_seconds_delta_both_success'])} s | "
                     f"{s['identical_decision_and_completion_traces']}/{s['n']} |")
    lines+=['','¹ 두 방법 모두 성공한 동일 장면만 비교한다. 실패한 실행을 0회 이동/짧은 시간의 효율 향상으로 취급하지 않는다.',
            'McNemar 정확 검정 및 장면별 원자료는 results.json에 있다. 작은 표본과 낮은 성공률을 고려해 탐색적 결과로 해석한다.',
            '', '## 목표 발견과 실패','',
            '발견은 계획 경계에서 목표 track이 실제 목표 위치에 올바르게 결합된 최초 시점이다. 실패/미발견은 0회로 채우지 않는다.',
            '아래 발견 수는 처음 미식별인 두 조건(총 26개)만 집계한다.','',
            '| 방법 | 올바른 목표 발견 | 미발견 | 발견 전 탐색 PICK 평균² | 관측 완료이나 실제 회수 실패 |',
            '|---|---:|---:|---:|---:|']
    for method in METHODS:
        rows=[r for r in report['expert']['rows'] if r['method']==method and r['group'].startswith('unlocalized')]
        found=[discovery_picks(r) for r in rows if discovery_picks(r) is not None]
        avg=f'{statistics.mean(found):.2f}' if found else '해당 없음'
        false=main['by_method'][method]['all']['false_observational_completion']
        lines.append(f'| {method} | {len(found)}/26 | {26-len(found)} | {avg} | {false} |')
    lines+=['','² 발견한 실행에 한정한 평균이며 성공률과 함께 읽어야 한다. 가림 물체를 집은 직후, PLACE 전에 목표가 드러날 수 있어 이 지표는 완료된 탐색 PICK 수를 센다.','',
            '| 실패/종료 유형 | Greedy | Flat SG | HSG |','|---|---:|---:|---:|']
    labels=sorted(set().union(*(main['by_method'][m]['all']['failure_counts'] for m in METHODS)))
    for label in labels:
        lines.append('| '+label+' | '+' | '.join(str(main['by_method'][m]['all']['failure_counts'].get(label,0)) for m in METHODS)+' |')
    lines+=['','## ACT 통합 확인 (별도 소규모 실험)','',
            '평가 시작 전에 정한 첫 네 장면(조건별 하나)에 동일 absolute ACT 체크포인트를 적용했다. 학습/검증 seed와 겹치지 않는다.',
            '총 12회이며 일반화 성공률의 정밀 추정으로 사용하지 않는다.','',
            '| 방법 | 실제 회수 성공 |','|---|---:|']
    for method in METHODS:
        s=report['act']['summary']['by_method'][method]['all']
        lines.append(f"| {method} | {s['successes']}/{s['n']} |")
    lines+=['','## 계산 시간','',
            '병렬 물리 실행의 wall time은 하드웨어 경합을 포함하므로 방법 간 속도 우열에 사용하지 않는다.',
            '아래는 전체 실행 후 별도로 측정한 직렬·공통 초기 그래프의 B 호출 시간이다. A 비용은 포함하지 않는다.',
            '실제 실행의 A 갱신 비용, B 비용, 수행 시간도 원자료에 보존했다.','',
            '| 방법 | 장면별 B 호출 중앙값의 중앙값 | 초기 계획 반환 가능 장면 |','|---|---:|---:|']
    for method,s in report['timing']['aggregate'].items():
        lines.append(f"| {method} | {s['median_case_milliseconds']:.3f} ms | {s['valid_initial_plans']}/52 |")
    lines+=['','실패 반환까지의 시간도 포함한다. 유효한 계획을 빠르게 반환했다는 의미와 구분해야 한다.',
            '', '## 해석과 제한','',
            '- HSG는 기존 코드를 수정하지 않은 방법이다. Flat SG는 목표 식별 후 같은 방해물 관계를 사용하고, 미식별 시 선반별 숨은 수 합계를 제외한다.',
            '- 따라서 Flat SG 대비 차이는 현재 선반 단위 탐색 휴리스틱의 효과이며, 그래프 자료구조 자체의 인과 효과를 입증하지 않는다.',
            '- Greedy에도 같은 영역 정보와 기하학적 집기 여유 검사를 제공한다. 전역 의존 관계를 따라 제거 대상을 정하지 않는다.',
            '- A는 실제 두 카메라 RGB-D의 색상 분할 + 알려진 직립 상자 크기 prior다. SAM/DINO나 일반 물체 인식 실험이 아니다.',
            '- 미식별 조건은 완전 가림만 의미하지 않는다. 색상 분할/식별 누락도 포함하며, GT 적층 장애가 있는 목표로 제한했다.',
            '- 초기 조건에 따른 층화 표본이다. 무작위 작업 전체의 평균 성공률을 대표하지 않는다. 결과를 보고 장면을 제외하거나 방법별로 다르게 선정하지 않았다.',
            '- 도달 불가와 제어 실패도 모두 유지했다. 따라서 이 실험만으로 계획 단독의 최적성이나 제어 오차가 없는 조건의 성능을 주장하지 않는다.',
            '- expert는 관측 geometry/FK/촉각만 사용한다. expert 실행의 불필요한 전면 카메라 매 tick 촬영은 끄고 A의 실제 두 카메라 촬영은 유지했다. ACT는 학습된 전면 RGB를 사용한다.',
            '- 물체는 27개, 기본 선반 배치 설정 하나다. 물체 수 증가에 따른 확장성, 메모리 효과, 갱신 제거 실험은 이번에 평가하지 않았다.',
            '- 모든 초기 공통 그래프를 읽는 현재 구현은 계층적 부분 그래프 확장을 통한 계산량 절감을 구현하지 않는다.',
            '', '## 다음 실험 설계에 필요한 변경','',
            '이번 결과는 성능·효율 향상의 증거로 쓰기보다, 구현과 평가 환경의 한계를 드러낸 첫 비교 실험으로 사용하는 것이 적절하다.',
            '',
            '1. 선반 전체 숨은 공간 합계와 개별 더미 깊이의 순위가 실제로 충돌하는 장면을 별도 조건으로 사전 설계한다. 목표 위치는 한 방법에 유리하게 고르지 않고 선반별로 균형화한다.',
            '2. 완전 가림과 색상/추적 실패를 분리한다. 목표 미식별인데 추정 숨은 공간이 0이면 즉시 종료하는 현재 가정도 별도 대체 탐색 규칙과 비교한다.',
            '3. 관측 오차를 통제한 계획 단독 평가와 전체 RGB-D·물리 평가를 병행하여 인식/제어 실패가 B의 차이를 가리지 않도록 한다.',
            '4. 계층을 이용한 계산량 절감을 주장하려면 부분 그래프 선택·가지치기를 구현하고, 물체/더미 수를 늘린 조건에서 검사 수와 A 포함 비용을 측정한다.',
            '',
            '위 변경 후에는 새 seed와 고정 프로토콜을 사용한다. 이번 52개 결과를 재선정하거나 개선 모델의 미사용 최종 평가로 재사용하지 않는다.',
            '', '재현 방법과 고정 프로토콜은 README.md, 실행별 원자료는 results.json 및 outputs/planner_benchmark/main/에 있다.','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def chart(path,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#64748b','#38bdf8','#14b8a6']
    data=report['expert']['summary']['by_method']
    for i,m in enumerate(METHODS):
        s=data[m]['all'];lo,hi=s['success_rate_wilson_95'];v=s['success_rate']
        axes[0].bar(i,v,color=colors[i],width=.6)
        axes[0].errorbar(i,v,yerr=[[v-lo],[hi-v]],color='black',capsize=5)
        axes[0].text(i,hi+.025,f"{s['successes']}/{s['n']}",ha='center')
    axes[0].set(xticks=range(3),xticklabels=['Greedy','Flat SG','HSG'],ylim=(0,1),title='Physical retrieval success (52 scenes)')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for i,m in enumerate(METHODS):
        values=[data[m][g]['success_rate'] for g in GROUPS]
        axes[1].bar(np.arange(4)+(i-1)*.24,values,width=.24,color=colors[i],label=m)
    axes[1].set(xticks=range(4),xticklabels=['Visible\nclear','Visible\nblocked','Unlocalized\nregion known','Unlocalized\nregion unknown'],
                ylim=(0,1),title='Success by initial condition (13 scenes each)')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend()
    fig.suptitle('RGB-D + identical scripted expert | 27 objects | Wilson 95% intervals on left')
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'experiments/planner_benchmark/manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/planner_benchmark/main')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'experiments/planner_benchmark')
    parser.add_argument('--timing-only',action='store_true')
    parser.add_argument('--repeats',type=int,default=21)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    timed_path=args.report_dir/'timing.json'
    if args.timing_only:
        timed=timing(manifest,args.repeats)
        timed['manifest_sha256']=digest(manifest)
        dump(timed_path,timed)
        print(json.dumps(timed['aggregate']),flush=True)
        return
    assert len(manifest['cases'])==52 and Counter(c['group'] for c in manifest['cases'])==Counter(dict.fromkeys(GROUPS,13)), \
        'this report template requires the frozen 52-scene protocol'
    expert=load_rows(manifest,args.output,'expert',{c['case_id'] for c in manifest['cases']})
    act=load_rows(manifest,args.output,'act',{'scene_000','scene_001','scene_002','scene_003'})
    provenance=json.loads((args.report_dir/'act_provenance.json').read_text(encoding='utf-8'))
    assert not provenance['all_52_benchmark_seed_overlap']
    assert all(row['checkpoint_sha256']==provenance['checkpoint_sha256'] for row in act)
    timed=json.loads(timed_path.read_text(encoding='utf-8'))
    assert timed['manifest_sha256']==digest(manifest)
    files=['src/planner/planner.py','src/planner/baselines.py','scripts/benchmark_planners.py',
           'src/pipeline/runner.py','src/pipeline/rgbd_env.py','src/action_generator/expert.py']
    report=dict(protocol='rgbd-color-paired-planners-v1',manifest_sha256=digest(manifest),
        environment=dict(python=platform.python_version(),platform=platform.platform(),mujoco=mujoco.__version__,numpy=np.__version__),
        source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        expert=dict(summary=summarize(expert),rows=expert),
        act=dict(summary=summarize(act),rows=act,provenance=provenance),timing=timed)
    report['pairing_audit']={controller:audit_captures(args.output,controller,{r['case_id'] for r in rows})
                             for controller,rows in [('expert',expert),('act',act)]}
    dump(args.report_dir/'results.json',report)
    write_report(args.report_dir/'RESULTS.md',report)
    chart(args.report_dir/'results.png',report)
    print(json.dumps({controller:report[controller]['summary']['paired'] for controller in ('expert','act')},indent=2))


if __name__=='__main__':
    main()
