"""Keep independent task, computation and fixed-ACT evidence separate."""
import json
from pathlib import Path

from report_planner_search import holm, dump, ROOT


def main():
    root=ROOT/'experiments/planner_search'
    families={name:json.loads((root/name/'results.json').read_text(encoding='utf-8'))
              for name in ('ordinary','occlusion')}
    p={f'{name}:{key}':data['comparisons'][key]['exact_two_sided_sign_p']
       for name,data in families.items() for key in data['primary']}
    corrected=holm(p)
    dump(root/'evidence.json',dict(families=families,primary_sign_p_holm_4=corrected,
        warning='task data use ideal removal, not ACT; computation study is separate synthetic graph timing'))
    lines=['# 플래너 검증 결과 개요','',
        'MuJoCo에서 새로 렌더링한 RGB-D 장면 120개에서 A/B/C를 비교한 360개 완료 평가다. 물체 제거는 공통 이상적 실행기로 처리했다.',
        '**아래 회수율은 ACT·실제 물리 파지의 성공률이 아니다.** 기본 환경과 가림 스트레스 결과를 합산하지 않는다.',
        '', '| 집합 | 장면 조건 | n/방법 | A 비용 | B 비용 | C 비용 | A/B/C 이상적 회수 성공 |',
        '|---|---|---:|---:|---:|---:|---|']
    for name,data in families.items():
        for group,label in [('all','전체'),('visible_blocked','관측 목표·방해물'),('hidden','미식별 목표')]:
            methods=data['summary'][group]
            scores=' | '.join(f"{methods[m]['mean_capped_removals']:.2f}" for m in ('a_greedy','b_no_hidden','c_full'))
            counts='/'.join(str(methods[m]['successes']) for m in ('a_greedy','b_no_hidden','c_full'))
            lines.append(f"| {data['family']} | {label} | {methods['a_greedy']['n']} | {scores} | {counts} |")
    lines += ['', '비용은 비목표 제거 횟수이며 실패는 고정 벌점 27이다. 실제 제거 횟수와 실패 벌점을 혼동하지 않는다.',
        '', '## 사전 지정 비교 네 개 전체에 대한 다중 비교 보정', '',
        '| 집합·비교 | 평균 비용 절감 [bootstrap 95%] | 우측 승/패/동률 | sign p (Holm 4개) |',
        '|---|---|---|---:|']
    supported=[]
    for name,data in families.items():
        for key in data['primary']:
            row=data['comparisons'][key];lo,hi=row['mean_saved_bootstrap_95'];adjusted=corrected[f'{name}:{key}']
            lines.append(f"| {name}: {key} | {row['mean_saved']:.2f} [{lo:.2f}, {hi:.2f}] | "
                         f"{row['right_wins']}/{row['left_wins']}/{row['ties']} | {adjusted:.5g} |")
            if lo>0 and adjusted<.05: supported.append(f'{name}: {key}')
    lines += ['', '구간은 장면별 평균 차이의 bootstrap 구간, p는 동률을 제외한 승패의 양측 exact sign 검정이다. 서로 다른 통계 가설을 다룬다.',
        '',f"양의 평균 차이 구간과 네 비교 보정 후 승패 검정이 함께 지지한 범위: {', '.join(supported) if supported else '없음'}.",
        '', '## 해석할 때 구분할 기여', '',
        '1. **관계 기반 선택:** A와 B/C의 차이는 관측된 목표 위 방해물을 우선하는 규칙과 표현의 효과다. 강한 다른 휴리스틱보다 항상 우수함을 증명한 것은 아니다.',
        '2. **추정 단서 사용:** B/C는 같은 planner에서 높이 기반 숨은 공간 추정 사용만 다르다. 계층 자료구조 자체의 고유한 우월성과 구분한다.',
        '3. **계산 비용:** 기존 별도 통제 실험에서 같은 정책과 공간 인덱스·부분 갱신을 유지했을 때, 국소 변화·미식별 탐색의 216/1,728 객체에서 HSG 요약의 추가 이점이 있었다. 27 객체에서는 불확실했다.',
        '4. **실제 ACT 작업:** 기존 동일 ACT 파일럿은 A/B/C 모두 0/4였다. 이번 결과로 ACT 실행 안정성이 개선되었다고 주장하지 않는다.',
        '', '## 원자료와 그림', '',
        '- [기본 환경 전체 결과](ordinary/RESULTS.md) · [그래프](ordinary/results.png)',
        '- [가림 스트레스 전체 결과](occlusion/RESULTS.md) · [그래프](occlusion/results.png)',
        '- [MuJoCo RGB-D 결정 이력 예시](planning_trace.png): 기본 환경 seed 8101에서 A 9회, B/C 2회 제거. 사후 선택한 설명용 사례이며 별도 통계 증거가 아니다.',
        '- [기존 계산 비용 실험](../hsg_indexed/controlled/RESULTS.md)',
        '- [기존 동일 ACT 실행](../act_conditions/RESULTS.md)',
        '- [프로토콜·재현 명령·가정](README.md)',
        '', '새 통계/실행기 경계와 기존 planner에 대한 관련 회귀 테스트 31개를 통과했다.', '']
    (root/'OVERVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(corrected,indent=2))


if __name__=='__main__': main()
