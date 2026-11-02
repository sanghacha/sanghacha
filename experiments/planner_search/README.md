# 플래너의 효과를 실행 실패와 분리해서 검증

MuJoCo에서 렌더링한 RGB-D 인식과 현재 A/B/C의 PICK 선택을 유지하고, 공통 실행기만 이상적인 물체 제거로 바꾼 진단이다.
물체를 집는 ACT가 실패해서 모든 계획이 같은 실패로 끝나는 문제를 분리한다.
기존 [ACT 결과](../act_conditions/RESULTS.md)를 대체하거나 성공률을 합산하지 않는다.

완료 결과는 [OVERVIEW.md](OVERVIEW.md), 논문에 사용할 수 있는 문장은 [CONCLUSIONS.md](CONCLUSIONS.md)에 있다.

## 두 개의 독립된 환경 집합

| 집합 | 새 장면 | 변경한 환경 조건 | 유지한 조건 |
|---|---:|---|---|
| 기본 환경 | 80 | 기존 생성기 그대로 | 카메라·RGB-D·플래너·목표 속성 대응 |
| 가림 스트레스 | 40 | 기존 기둥 안에서 적재 순서만 균등 무작위 순열 | 물체 수·크기·색·기둥 좌표·카메라·인식·플래너 |

기본 생성기는 큰 물체부터 아래에 배치한다. 아래 물체의 가장자리가 관측될 수 있어, 숨은 공간 추론이 필요한 상황의 범위를 별도로 확인한다.
추가 조건은 큰 물체가 작은 물체 위에 올라갈 수 있게 허용하며, 변경은 초기 물리 안정화와 관측 전에 적용한다.
프로덕션 `ShelfScene`이나 플래너 코드는 수정하지 않았다. 두 집합은 결과를 합치지 않는다.

각 집합은 목표 식별·즉시 집기 / 목표 식별·방해물 / 미식별·선반 힌트 있음 / 미식별·힌트 없음의 네 조건을 균등하게 포함한다.
적격 목표 중 하나를 고정 RNG로 무작위 선택한다. C의 숨은 공간 단서 유무나 실행 승패로 표본을 고르지 않는다.

## 공통 실행기

1. 원래 RGB-D 인식·추적·`TargetResolver`를 실행한다.
2. A/B/C가 동일한 public `Subgoal(PICK)` 형식으로 관측 track을 선택한다.
3. **결정 후에만** 실행기가 정답 위치로 해당 track의 물리적 대응을 확인하고, 지지 관계상 위가 비었는지 검사한다.
4. 집을 수 있는 비목표 물체를 카메라 밖 외부 버퍼로 옮긴다. 나머지 물체와 로봇의 자세는 고정한다.
5. 같은 카메라를 다시 렌더링하고 동일한 인식·추적 과정을 실행한다.

정답 물체를 골라도 목표 속성으로 인식하지 못했다면 회수 성공으로 처리하지 않는다.
잘못된 물체 대응·막힌 파지·잘못된 목표 대응은 실패로 보존한다. 올바른 물체로 선택을 고쳐주지 않는다.

제외한 요소: ACT·IK·이동 중 충돌·파지 접촉·배치 공간 제약·낙하·마찰·팔 움직임에 의한 가림.
따라서 이 실험의 회수는 **이상적 제거 실행기에서의 회수**이며, 실제 로봇/ACT 성공률이 아니다.

## 지표와 검증

- 주 지표: 비목표 제거 횟수. 실패는 27로 상한 처리하고 실패 수를 함께 표시한다.
- 사전 지정 비교: 전체에서 B 대 A, 미식별 목표에서 C 대 B.
- 장면별 paired 비교, 장면 bootstrap 20,000회, 양측 exact sign 검정. 집합별 두 비교에는 Holm 보정을 적용한다.
- 보조 지표: 제한된 제거 예산 내 회수율, 올바른 목표 발견까지의 제거, 두 방법 모두 성공한 장면의 제거 차이.
- 독립 정답 감사: 초기 수직 지지 그래프의 필수 방해물 수와 관측 후보 존재 여부. Planner 입력에는 사용하지 않는다.
- 추가 가림 조건까지 선택적으로 주장하는 문제를 줄이기 위해 최종 개요에서 네 주 비교의 Holm 보정도 함께 제시한다.

초기 물리 상태 SHA256는 정확히 일치해야 한다. 초기 그래프의 track ID·부모·영역도 같아야 한다.
동일 OpenGL 입력에서도 픽셀 반올림 때문에 평균 RGB가 약 4e-6 달라진 개발 사례가 있어,
기하 1e-5m/쿼터니언 성분 1e-5, 색 성분 1e-4 이내의 수치 차이만 허용하고 최대 오차를 기록한다.

개발 장면은 seed 8000대/9000대이며 최종 평가와 겹치지 않는다. 중단된 개발 폴더도 보존한다.
기본 평가를 실행하던 중 계산을 여러 프로세스로 분배했으며, 각 장면은 고정된 `run_case`로 독립 실행한다.
스케줄러 해시와 담당 case를 기록한다. 동시 실행한 B 호출 시간을 알고리즘 속도 비교에 사용하지 않는다.

## 재현

Python 3.11 perception 환경에서 다음 명령을 사용한다. 현재 검증한 환경의 실행 파일은 `.venv-perception/Scripts/python.exe`다.
이미 고정된 폴더는 덮어쓰지 않는다. 새 실행에서는 output과 seed 시작점을 바꾼다.

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_planner_search.py freeze --count 80 --start-seed 8100 --output outputs/planner_search/confirmatory
.venv-perception/Scripts/python.exe scripts/benchmark_planner_search.py run --output outputs/planner_search/confirmatory
.venv-perception/Scripts/python.exe scripts/audit_planner_search.py --input outputs/planner_search/confirmatory
.venv-perception/Scripts/python.exe scripts/report_planner_search.py --input outputs/planner_search/confirmatory --output experiments/planner_search/ordinary

.venv-perception/Scripts/python.exe scripts/benchmark_planner_occlusion.py freeze --count 40 --start-seed 9100 --output outputs/planner_search/occlusion
.venv-perception/Scripts/python.exe scripts/benchmark_planner_occlusion.py run --output outputs/planner_search/occlusion
.venv-perception/Scripts/python.exe scripts/audit_planner_search.py --input outputs/planner_search/occlusion
.venv-perception/Scripts/python.exe scripts/report_planner_search.py --input outputs/planner_search/occlusion --output experiments/planner_search/occlusion
```

독립 장면을 분할하려면 `scripts/run_planner_search_shard.py --input ... --shards 3 --shard 0`을 사용한다.
shard 0/1/2를 각각 실행하며, 같은 폴더에 순차 실행기와 shard 실행기를 동시에 돌리지 않는다.

## 기존 증거와의 관계

- [계층 요약의 계산 비용](../hsg_indexed/OVERVIEW.md): 같은 정책·공간 인덱스·부분 갱신 조건에서 HSG 요약의 추가 시간 이점. 계획 선택의 효과와 다르다.
- [동일 ACT 파일럿](../act_conditions/RESULTS.md): 실제 물리 회수에서는 A/B/C 모두 0/4였다. 이 한계는 유지한다.
- 이번 비교의 A/C는 표현과 규칙을 함께 바꾼다. B/C는 숨은 공간 추정 사용의 효과다. 그래프 자료구조만의 본질적 우월성을 뜻하지 않는다.
- 로봇 거리 Greedy 외의 강한 탐색 규칙과도 비교해야 일반적인 planner 우월성을 주장할 수 있다. 현재 결과의 기준선은 사용자가 정의한 Greedy다.

각 결과 폴더에 manifest·프로토콜·환경 버전·정답 감사와 압축된 전체 trial/public input을 보존했다.
카메라 원본은 `outputs/planner_search/`에 있고, 저장소의 압축 입력만으로도 다음 재현 검사를 실행할 수 있다.

```powershell
.venv-perception/Scripts/python.exe scripts/audit_search_replay.py --input experiments/planner_search/ordinary
.venv-perception/Scripts/python.exe scripts/audit_search_replay.py --input experiments/planner_search/occlusion
```

총 360개 평가의 2,031개 public 결정에서 Subgoal과 선택 설명이 모두 재현됐고, public 입력에 시뮬레이터 물체 ID가 없는 것을 확인했다.
관련 테스트 31개도 통과했다.

## 연구 배경 참고

[Mechanical Search (ICRA 2019)](https://ai.stanford.edu/mech-search/multistep/)는 RGB-D 기반 반복 조작으로 가려진 목표를 찾고 회수하는 문제를 다룬다.
해당 논문은 행동 횟수와 회수 성능을 비교한다. 이번 이상적 제거 실험은 그 논문의 프로토콜이나 결과를 재현한 실험이 아니다.

[Hydra (RSS 2022)](https://www.roboticsproceedings.org/rss18/p050.pdf)는 센서로부터 계층적 3D scene graph를 실시간 구성·갱신하는 시스템이다.
이를 근거로 우리 탐색 정책의 우월성이 입증되었다고 주장하지 않는다. 계층 요약의 계산 효과와 작업 정책의 효과를 별도로 평가한다.
