# 그림 설명

## MuJoCo RGB-D 결정 이력 예시

파일: [planning_trace.png](planning_trace.png), [편집 가능한 SVG](planning_trace.svg).

같은 초기 장면(seed 8101)에서의 A/B/C 선택 이력이다. 흰 원은 해당 관측에서 다음 PICK으로 선택한 물체다.
각 열은 초기 상태, 비목표 물체 1개 제거 후, 목표를 집기 직전의 관측이다. 마지막 열까지 제거한 비목표 물체는 A 9개, B/C 2개다.
모든 방법은 같은 RGB-D 인식과 목표 속성 대응을 사용하며, 물체 제거는 이상적 실행기로 처리한다. 로봇 팔은 고정되어 있고 ACT나 물리적 파지는 실행하지 않는다.
이 예시는 평가 이후 선택한 설명용 사례이며, 평균 성능이나 통계적 유의성의 근거로 사용하지 않는다.

Recorded decision sequences from the same initial MuJoCo RGB-D scene (seed 8101). White circles indicate the next selected PICK.
Columns show the initial scene, the scene after one non-target removal, and the scene immediately before the target pick.
The Greedy baseline removes nine non-target objects; both relation-aware conditions remove two.
An ideal removal executor is used with a parked robot arm; ACT and physical grasp execution are excluded.
This post-evaluation example illustrates behavior and is not additional statistical evidence.

## 집합별 비교 그래프

파일: `ordinary/results.png`, `occlusion/results.png`.

왼쪽은 네 장면 조건에서의 평균 제거 비용이다. 목표 외 물체의 제거 횟수를 사용하며, 실패에는 고정 벌점 27을 부여한다.
오른쪽은 처음 미식별인 목표를 주어진 비목표 제거 예산 안에 회수한 비율이다.
두 그림 모두 MuJoCo에서 렌더링한 RGB-D와 동일한 이상적 제거 실행기를 사용한 계획 진단이며 ACT 또는 실제 파지의 성공률을 나타내지 않는다.
기본 생성기와 적재 순서를 무작위로 바꾼 가림 스트레스 집합은 별도로 평가한다. 표본 수와 paired 신뢰구간은 각 RESULTS.md에 제시한다.

Mean removal cost by initial-scene stratum (left) and ideal target retrieval within a non-target removal budget for initially unlocalized targets (right).
Failures receive a fixed cost of 27. Each condition uses the same RGB-D pipeline and ideal removal executor.
These are planning diagnostics, not ACT or physical grasp success rates. Ordinary and randomized-stack-order scene families are evaluated separately.
