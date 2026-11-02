# 플래너 비교 실험

고정된 52개 장면 × Greedy / Flat SG / HSG의 RGB-D + MuJoCo 실험이다.
모든 방법의 C는 같은 SO101Expert다. 실제 RGB-D A와 ACT를 결합한 4개 장면 × 3방법의 별도 통합 검사도 수행한다.
결과는 [RESULTS.md](RESULTS.md), 수치는 [results.json](results.json), 그래프는 [results.png](results.png)에 둔다.

## 비교 방법

| 방법 | 목표가 식별된 경우 | 목표가 미식별인 경우 |
|---|---|---|
| Greedy | 국소 기하 검사상 위가 빈 목표를 우선 선택. 막혀 있으면 목표 XY에 가까운 위가 빈 선반 물체 선택 | 주어진 선반 힌트 범위에서 팔 앞 중앙에 가까운 위가 빈 선반 물체 선택 |
| Flat SG | 현재 HSG와 같은 적층 의존 관계를 따라 위 물체부터 제거 | 개별 더미의 추정 숨은 물체 수 → 팔 앞 중앙 거리 → ID 순으로 선택 |
| HSG | 기존 `planner.plan_next_subgoal` 그대로 | 선반별 추정 숨은 물체 수 합계 → 개별 더미 숨은 수 → 거리 → ID 순으로 선택 |

세 방법 모두 같은 관측 물체, 위치·크기·색, 알려진 영역 geometry, 목표 속성, 선반 힌트를 받는다.
Greedy는 이를 물체별 영역 라벨과 국소 기하 검사로 사용하며, 전역 방해물 의존 경로를 추적하지 않는다.
PLACE 정책, 버퍼 공간 검사, 재관측 시점, 450 tick/Subgoal, 최대 12 Subgoal은 동일하다.
Flat/HSG의 차이는 선반 단위 점수 하나다. 알려진 선반 하나만 탐색하는 조건에서는 같은 결정을 할 수 있다.
이 비교는 정책의 기여도를 측정하며, 같은 정보를 리스트와 그래프로 저장하는 형식만의 인과 효과를 분리하지 않는다.

## 장면 선정

- 기본 3선반·27개 상자, 원래 크기/질량/마찰/카메라/배치 난수를 유지한다.
- seed 3100부터 차례로 촬영하여 조건별 13개, 총 52개를 선택했다. 후보가 없던 seed도 manifest의 screening에 남긴다.
- 조건은 목표 식별·바로 집기, 목표 식별·물리적 적층 장애, 목표 미식별·선반 힌트 있음, 목표 미식별·선반 힌트 없음이다.
- 초기 RGB-D A와 목표 속성 resolver의 결과로 식별 여부를 나눈다. 식별 조건은 실제 목표 위치 12mm 안의 track인 경우다.
- 미식별 조건은 A에 해당 속성의 track이 없고, GT 접촉상 위에 물체가 있는 목표다. 완전 가림뿐 아니라 색상/geometry 인식 누락도 포함한다.
- 조건별 후보에서 선반 선호를 순환한 뒤 ID 순으로 하나를 정한다. 실행 성공/실패로 장면을 고르거나 제거하지 않는다.
- 도달 가능성·expert 성공 여부로 사전 필터링하지 않았다. 이로 인한 실패도 최종 분모에 남는다.
- GT ID/접촉은 선정 라벨과 실행 후 물리 평가에만 사용한다. A/B/C는 RGB-D track ID, geometry, FK, 두 손가락 촉각만 받는다.
- 목표의 기준 색·크기와 선택 조건의 선반 힌트는 사용자 요청을 구성하는 속성이다. 목표 GT 좌표를 요청에 넣지 않는다.
- 초기 qpos SHA256와 A의 첫 공통 SceneGraph가 manifest와 일치하는지 실행마다 검사한다.
- ACT 통합 검사는 결과를 보기 전에 첫 네 장면 `scene_000`~`scene_003`으로 정했다. 52개 seed 전부 체크포인트 학습/검증 seed와 겹치지 않음을 확인했다.
- 개발용 pilot은 seed 3050~3053으로 별도 저장했고 52개 결과에 합치지 않는다.

## 실행

기존 설치된 Python 3.11 perception 환경에서 수행한 명령:

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_planners.py run --methods greedy flat hsg
training/.venv-rgbd/Scripts/python.exe scripts/benchmark_planners.py run --controller act --checkpoint training/runs/act_rgbd_shelves_absolute/best.pt --device cuda --case-ids scene_000 scene_001 scene_002 scene_003
.venv-perception/Scripts/python.exe scripts/report_planner_benchmark.py --timing-only
.venv-perception/Scripts/python.exe scripts/report_planner_benchmark.py
```

uv로 perception 환경을 선택할 수도 있다:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-perception"
uv run --python 3.11 --extra perception python scripts/benchmark_planners.py run --methods greedy flat hsg
```

기존 결과는 덮어쓰지 않고 재사용한다. 방법/제어기/체크포인트나 고정 manifest가 다른 결과는 거부한다.
정책 코드를 바꿨다면 **반드시 새 `--output` 경로**로 실행한다. 기존 결과와 새 정책 결과를 섞지 않는다.
처음부터 반복하려면 예를 들어 `--output outputs/planner_benchmark/repeat_01`을 모든 run/report 명령에 붙인다.
부동소수점/렌더러 차이로 첫 그래프가 다르면 paired 재현 검사에서 실패한다. 다른 시스템에서는 새 manifest를 만들고 그 안에서 짝을 맞춘다.

추가 평가 장면을 새로 선정할 때는:

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_planners.py prepare --manifest outputs/planner_benchmark/new/manifest.json --output outputs/planner_benchmark/new --count 52 --start-seed 4000
```

`manifest.json`은 결과를 보기 전에 확정한 장면 명세다. 메인 평가는 고정 manifest를 사용한다.
`--methods`로 방법별 프로세스를 실행할 수 있다. 이번 물리 평가는 세 expert 프로세스를 병렬 실행했으며,
전체 wall time은 하드웨어 경합 때문에 속도 우열의 근거로 사용하지 않는다.
별도 B timing은 모든 rollout 종료 뒤 같은 초기 그래프에서 직렬 21회(별도 warm-up 제외), 순서 순환으로 측정한다.

## 산출물과 지표

- `outputs/planner_benchmark/main/<controller>/<method>/<case_id>/result.json`: 모든 계획/완료 이력, 오류, 물리 성공, tick/시간/이동 횟수.
- 같은 폴더의 `plan_inputs.json`: 실제 B 호출에 들어간 공통 그래프·요청·관절 상태.
- `perception/snapshot_*`: 실제 두 카메라 RGB/깊이, 마스크, 추적 상태, scene graph와 변화.
- `results.json`: 실행 원자료 복사, 방법·조건별 요약, paired 비교, Wilson 95% 성공률 구간, McNemar 정확 검정, 버전/소스 해시.
- `timing.json`: 직렬 공통 입력 B 계산 시간과 초기 선택 결과. 실패 반환 시간도 포함한다.

성공은 실제로 요청한 GT 물체가 목적 트레이에 지지되어 있고, 파지 해제·저속·위치 10mm 이내 조건을 만족하는지 평가한다.
관측 기반 완료와 별도로 판정하므로 다른 물체 회수나 잘못된 배치 확인을 성공으로 세지 않는다.
효율 비교는 성공률과 함께 제시하고, 두 방법 모두 성공한 동일 장면의 제거 횟수·simulation time 차이를 별도 계산한다.
제거 횟수는 완료된 비목표 track의 PLACE Subgoal 수다. 충돌로 밀린 물체 수나 모든 물리적 변위를 세는 지표는 아니다.
발견은 B의 계획 경계에서 올바른 물리 목표에 결합된 최초 시점이며, 미발견 실행은 0회로 채우지 않는다.
발견 전 탐색 비용은 해당 시점까지 완료된 비목표 PICK 수로 집계한다. 가림 물체를 집은 직후 PLACE 전에 목표가 드러날 수도 있기 때문이다.
원자료의 `discovery_after_completed_removals`는 당시 완료된 PLACE 수이며, 보고서는 이력과 발견 시각에서 `search_picks_before_discovery`를 별도 계산한다.
성공률이 낮거나 방법이 일찍 종료하면 낮은 전체 평균 시간/이동 횟수만으로 효율이 높다고 주장하지 않는다.
실행 wall time은 환경/모델 생성 뒤 reset부터 종료 판정까지이며 A 파일 저장·읽기도 포함한다. A 시간도 I/O 포함 갱신 시간이다.

52개 장면의 결과로 물체 수 확장성이나 최적 계획을 입증하지 않는다. 실패를 분석한 뒤 바꾼 정책의 검증은 새 seed로 수행해야 한다.
