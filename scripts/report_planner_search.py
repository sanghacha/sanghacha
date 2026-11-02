"""Report the frozen ideal-removal diagnostic; never call it ACT performance."""
import argparse
from collections import Counter
import gzip
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline.mock_data import ROOT
from planner.act_comparison import CONDITIONS
from benchmark_planners import GROUPS, digest, dump


def paired_delta(left,right,bootstrap_seed=9917):
    """Positive means right needs fewer removals. Seed/case is the unit."""
    assert set(left)==set(right) and left
    values=np.array([left[k]-right[k] for k in sorted(left)],dtype=float)
    wins=int(np.sum(values>0));losses=int(np.sum(values<0));n=wins+losses
    p=min(1.,2*sum(comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
    rng=np.random.default_rng(bootstrap_seed)
    draws=values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1)
    return dict(n=len(values),mean_saved=float(values.mean()),
                mean_saved_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                right_wins=wins,left_wins=losses,ties=int(np.sum(values==0)),
                exact_two_sided_sign_p=p)


def holm(pvalues):
    order=sorted(pvalues,key=pvalues.get);adjusted={};last=0.
    for i,key in enumerate(order):
        last=max(last,min(1.,pvalues[key]*(len(order)-i)))
        adjusted[key]=last
    return adjusted


def summarize(rows,manifest):
    expected={(case['case_id'],condition) for case in manifest['cases'] for condition in CONDITIONS}
    assert {(r['case_id'],r['condition']) for r in rows}==expected
    assert len(rows)==len(expected), 'missing/duplicate trials'
    assert all(r['identity']['manifest_sha256']==digest(manifest) for r in rows)
    assert all(r['identity']['source_sha256']==manifest['protocol']['source_sha256'] for r in rows)
    summary={};comparisons={}
    for group in ('all','hidden')+GROUPS:
        subset=[r for r in rows if group=='all' or
                (group=='hidden' and r['group'].startswith('unlocalized')) or r['group']==group]
        summary[group]={}
        for method in CONDITIONS:
            values=[r for r in subset if r['condition']==method]
            if not values: continue
            summary[group][method]=dict(n=len(values),successes=sum(r['success'] for r in values),
                discovered=sum(r['discovery_removals'] is not None for r in values),
                mean_capped_removals=float(np.mean([r['capped_removals'] for r in values])),
                mean_discovery_capped=float(np.mean([r['discovery_removals'] if r['discovery_removals'] is not None else 27 for r in values])),
                success_within_removal_budget={str(b):sum(r['success'] and r['non_target_removals']<=b for r in values)
                                              for b in (0,1,2,3,5,8,12)},
                errors=dict(Counter(r['error'] for r in values if r['error'])))
        for left,right in (('a_greedy','b_no_hidden'),('b_no_hidden','c_full'),('a_greedy','c_full')):
            a={r['case_id']:r['capped_removals'] for r in subset if r['condition']==left}
            b={r['case_id']:r['capped_removals'] for r in subset if r['condition']==right}
            if a:
                comparison=paired_delta(a,b)
                good_left={r['case_id']:r for r in subset if r['condition']==left and r['success']}
                good_right={r['case_id']:r for r in subset if r['condition']==right and r['success']}
                common=set(good_left)&set(good_right)
                comparison['both_success_n']=len(common)
                if common:
                    comparison['both_success_removal_difference']=paired_delta(
                        {k:good_left[k]['non_target_removals'] for k in common},
                        {k:good_right[k]['non_target_removals'] for k in common})
                comparisons[f'{group}:{left}->{right}']=comparison
    primary=['all:a_greedy->b_no_hidden','hidden:b_no_hidden->c_full']
    corrected=holm({k:comparisons[k]['exact_two_sided_sign_p'] for k in primary})
    for key in primary:
        comparisons[key]['primary_sign_p_holm_2']=corrected[key]
    hidden_rows=[r for r in rows if r['group'].startswith('unlocalized')]
    discovery={m:{r['case_id']:r['discovery_removals'] if r['discovery_removals'] is not None else 27
                  for r in hidden_rows if r['condition']==m} for m in CONDITIONS}
    return dict(summary=summary,comparisons=comparisons,
                exploratory_discovery_b_to_c=paired_delta(discovery['b_no_hidden'],discovery['c_full']),
                primary=primary,complete_cases=len(manifest['cases']))


def report(args):
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads((args.input/'runs'/m/c['case_id']/'result.json').read_text(encoding='utf-8'))
          for c in manifest['cases'] for m in CONDITIONS]
    result=summarize(rows,manifest)
    cue_cases={c['case_id'] for c in manifest['cases'] if c['group'].startswith('unlocalized')
               and c['initial_inferred_hidden_count']>0}
    if cue_cases:
        a={r['case_id']:r['capped_removals'] for r in rows if r['case_id'] in cue_cases and r['condition']=='b_no_hidden'}
        b={r['case_id']:r['capped_removals'] for r in rows if r['case_id'] in cue_cases and r['condition']=='c_full'}
        result['cue_available_exploratory']=paired_delta(a,b)
    oracle_path=args.input/'oracle_audit.json'
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text(encoding='utf-8'))
        assert oracle['manifest_sha256']==digest(manifest)
        index={r['case_id']:r for r in oracle['rows']}
        assert set(index)=={c['case_id'] for c in manifest['cases']}
        hidden=[r for r in oracle['rows'] if r['group'].startswith('unlocalized')]
        result['oracle_audit']=dict(hidden_n=len(hidden),hidden_cue_available=len(cue_cases),
            hidden_geometric_target_present=sum(r['requested_target_has_geometric_candidate'] for r in hidden),
            visible_blocked={m:dict(n=sum(r['group']=='visible_blocked' and r['condition']==m for r in rows),
                exact_lower_bound_successes=sum(r['group']=='visible_blocked' and r['condition']==m and r['success'] and
                    r['non_target_removals']==index[r['case_id']]['support_removal_lower_bound'] for r in rows)) for m in CONDITIONS})
        assert all(not r['success'] or r['non_target_removals']>=index[r['case_id']]['support_removal_lower_bound'] for r in rows)
    args.output.mkdir(parents=True,exist_ok=True)
    # Keep reviewable evidence in the repository; large camera captures remain
    # under outputs. Deterministic gzip avoids timestamps in exported data.
    for filename in ('manifest.json','protocol.json','layout_protocol.json','runtime_metadata.json',
                     'oracle_audit.json','replay_audit.json'):
        path=args.input/filename
        if path.exists(): (args.output/filename).write_bytes(path.read_bytes())
    (args.output/'trials.json.gz').write_bytes(gzip.compress(
        json.dumps(rows,ensure_ascii=False).encode('utf-8'),mtime=0))
    public_inputs=[dict(case_id=c['case_id'],condition=m,
        steps=json.loads((args.input/'runs'/m/c['case_id']/'plan_inputs.json').read_text(encoding='utf-8')))
        for c in manifest['cases'] for m in CONDITIONS]
    (args.output/'public_inputs.json.gz').write_bytes(gzip.compress(
        json.dumps(public_inputs,ensure_ascii=False).encode('utf-8'),mtime=0))
    family='적재 순서 무작위 가림 조건' if (args.input/'layout_protocol.json').exists() else '기본 적재 환경'
    result['family']=family
    dump(args.output/'results.json',result)
    lines=[f'# {family}: MuJoCo RGB-D + 이상적 제거 실행기의 계획 비교','',
        '**ACT·물리 파지·실제 배치 성공률 실험이 아니다.** 공통 실행기가 선택된 물체를 장면 밖으로 옮긴 뒤 같은 시뮬레이션 카메라와 기존 인식을 다시 실행한다.',
        '',f"새 장면 {len(manifest['cases'])}개 × A/B/C, 총 {len(rows)}회. 각 장면은 독립 seed이며 목표는 적격 후보에서 무작위 선택했다.",
        '실행 전에 표본 수·seed 시작점·비교·지표·소스를 고정했다. 숨은 공간 단서가 있거나 C가 이기는지는 선정 조건이 아니다.',
        '', '| 장면 조건 | n/방법 | A 비용 | B 비용 | C 비용 | A/B/C 이상적 회수 성공 |',
        '|---|---:|---:|---:|---:|---|']
    labels=dict(all='전체',hidden='목표 미식별 합계',visible_free='목표 식별·즉시 집기',
                visible_blocked='목표 식별·방해물',unlocalized_known_region='미식별·선반 힌트 있음',
                unlocalized_unknown_region='미식별·선반 힌트 없음')
    for group,methods in result['summary'].items():
        if not methods: continue
        counts='/'.join(str(methods[m]['successes']) for m in CONDITIONS)
        means=' | '.join(f"{methods[m]['mean_capped_removals']:.2f}" for m in CONDITIONS)
        lines.append(f"| {labels[group]} | {methods['a_greedy']['n']} | {means} | {counts} |")
    lines += ['', '비용은 목표 외 물체의 제거 횟수다. 실패에는 고정 벌점 27을 부여하며 성공 건만 골라 평균을 내지 않는다. 벌점은 실제로 관측한 제거 횟수가 아니다.',
        '', '## 실행 전 지정한 두 비교', '',
        '| 비교 | n | 평균 절감 [bootstrap 95%] | 우측 승/패/동률 | 양측 sign p, Holm 보정 |',
        '|---|---:|---|---|---:|']
    common_notes=[]
    for key in result['primary']:
        row=result['comparisons'][key];lo,hi=row['mean_saved_bootstrap_95']
        lines.append(f"| {key} | {row['n']} | {row['mean_saved']:.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"{row['right_wins']}/{row['left_wins']}/{row['ties']} | {row['primary_sign_p_holm_2']:.4g} |")
        if 'both_success_removal_difference' in row:
            common=row['both_success_removal_difference']
            common_notes.append(f"{key}: 두 방법 모두 성공한 {common['n']}개 장면에서의 실제 제거 횟수 차이는 평균 "
                                f"{common['mean_saved']:.2f}개다. 실패 제외에 따른 선택 편향이 있으므로 보조 지표로만 읽는다.")
    for note in common_notes:
        lines += ['',note]
    lines += ['', '## 처음 미식별이었던 목표의 발견 지표', '',
              '| 방법 | 올바른 목표 대응 / 표본 | 발견까지 평균 제거(미발견 27) |', '|---|---:|---:|']
    for m in CONDITIONS:
        r=result['summary']['hidden'][m]
        lines.append(f"| {m} | {r['discovered']} / {r['n']} | {r['mean_discovery_capped']:.2f} |")
    discovery=result['exploratory_discovery_b_to_c']
    lines += ['',f"C의 B 대비 발견 비용 차이(탐색적): {discovery['mean_saved']:.2f}개, "
              f"장면 bootstrap 95% 구간 {discovery['mean_saved_bootstrap_95']}. 회수 지표와 구분한다."]
    if cue_cases:
        row=result['cue_available_exploratory'];lo,hi=row['mean_saved_bootstrap_95']
        lines += ['',f"초기 관측에서 추정 단서가 있는 미식별 장면은 {len(cue_cases)}개다. 이 사전 정의한 하위 집합에서 "
                  f"C의 B 대비 평균 절감은 {row['mean_saved']:.2f}개 [{lo:.2f}, {hi:.2f}]다. 전체 미식별 표본을 대체하지 않는 탐색적 결과다."]
    if 'oracle_audit' in result:
        audit=result['oracle_audit']
        lines += ['',f"미식별 목표 {audit['hidden_n']}개 중 {audit['hidden_geometric_target_present']}개는 정답 위치 15mm 내에 "
                  '기하 후보가 이미 있었다. 이는 목표 속성 대응 실패와 완전 미관측을 구분해야 함을 뜻한다. 카메라 픽셀 가시성의 정답 판정은 아니다.',
                  '', '관측된 목표에 방해물이 있는 장면에서, 독립 정답 지지 그래프의 최소 제거 하한과 일치하면서 성공한 횟수:',
                  '', '| 방법 | 하한과 일치한 성공 / 표본 |','|---|---:|']
        for m,row in audit['visible_blocked'].items():
            lines.append(f"| {m} | {row['exact_lower_bound_successes']} / {row['n']} |")
    lines += ['', 'Bootstrap은 장면 단위 평균 차이의 구간이다. sign 검정은 동률을 제외한 승패에 대한 검정이며 평균 차이 검정과 같은 가설이 아니다.',
        '두 사전 지정 sign 검정에 Holm 보정을 적용했다. 나머지 조건별/예산별 결과는 탐색적 기술 통계다.',
        '', '## 실패와 한계', '',
        '- 실행기는 결정 이후에만 정답 위치로 관측 track을 대응하고 실제 지지 관계로 집기 가능성을 검사한다. 실패한 선택을 다른 물체로 바꾸지 않는다.',
        '- Planner에는 관측 track·RGB-D 추정 자세·공통 목표 속성만 전달한다. 실제 숨은 목표 ID/좌표는 전달하지 않는다.',
        '- 팔은 초기 자세에 고정된다. 물체 제거 후 나머지 물체는 정지하며 낙하·마찰·접촉·카메라를 가리는 팔 움직임은 모델링하지 않는다.',
        '- 외부 버퍼 용량이 무한하고 PLACE 정책을 실행하지 않는다. 이는 PICK 선택과 재관측의 비용만 분리한 진단이다.',
        '- A/C는 규칙과 표현을 함께 바꾼 비교다. B/C는 높이 기반 숨은 공간 단서 사용의 효과이며 그래프 구조만의 우월성은 아니다.',
        '- 네 조건을 균등하게 선정했으므로 전체 평균은 이 조건 혼합에 대한 결과다. 자연 환경 전체의 평균 성공률이 아니다.',
        '- 기존 ACT 0/4 및 기존 물리 실행 결과는 별도로 유지한다. 이 결과를 ACT 성공률 향상으로 합치지 않는다.',
        '', f'원자료: `{args.input.as_posix()}`. `runs/<method>/<scene>/plan_inputs.json`과 카메라 입력, 결정 이후 oracle 평가를 보존한다.', '']
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(result,args.output/'results.png')
    print(json.dumps({k:result['comparisons'][k] for k in result['primary']},indent=2),flush=True)


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={font.name for font in font_manager.fontManager.ttflist}
    korean_font=next((name for name in ('Malgun Gothic','NanumGothic','Noto Sans CJK KR') if name in available),None)
    if korean_font is None:
        raise RuntimeError('Install a Korean font: Malgun Gothic, NanumGothic or Noto Sans CJK KR')
    plt.rcParams.update({'font.family':korean_font,'axes.unicode_minus':False})
    methods=CONDITIONS;colors=['#8c919a','#dfab3e','#287eaa']
    labels=['A: 거리 우선 탐색','B: 숨은 공간 단서 제외','C: 제안 방법 (HSG)']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    groups=GROUPS;x=np.arange(4);width=.24
    for i,(method,color) in enumerate(zip(methods,colors)):
        axes[0].bar(x+(i-1)*width,[result['summary'][g][method]['mean_capped_removals'] for g in groups],
                    width,label=labels[i],color=color)
        values=result['summary']['hidden'][method]
        budgets=(0,1,2,3,5,8,12)
        axes[1].plot(budgets,[values['success_within_removal_budget'][str(b)]/values['n'] for b in budgets],
                     marker='o',color=color,label=labels[i])
    axes[0].set_xticks(x,['목표 식별\n방해물 없음','목표 식별\n방해물 있음','목표 미식별\n선반 힌트 있음','목표 미식별\n선반 힌트 없음'])
    axes[0].set_ylabel('평균 제거 비용 (실패 시 27)')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('허용된 비목표 물체 제거 횟수')
    axes[1].set_ylabel('이상적 회수 성공률\n(초기 목표 미식별 장면)')
    axes[1].set_ylim(0,1.03);axes[1].legend(fontsize=8)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"{result['family']}\nMuJoCo RGB-D 관측 · 이상적 물체 제거 · ACT·물리 파지 제외")
    fig.savefig(path,dpi=200);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/planner_search/confirmatory')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/planner_search')
    report(parser.parse_args())
