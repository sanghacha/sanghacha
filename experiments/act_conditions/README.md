# 동일 ACT의 A/B/C 비교

사용자가 확정한 세 조건을 구현했다. 모든 조건에서 같은 ACT 체크포인트와 Subgoal/25D goal encoding을 사용한다.

| 조건 | Planning이 사용하는 정보와 규칙 | 실행 모듈 |
|---|---|---|
| A Baseline | 관측 물체 목록·설정된 영역·국소 기하 후보 검사. robot_base에서 가까운 후보부터 제거 | `planner.robot_greedy` |
| B Ablation | 관측 물체·계층·지지 관계는 유지. 숨은 공간 추정의 계산과 사용은 비활성화 | `planner.hsg_no_hidden` |
| C Ours | B와 같은 planner에 관측 높이 기반 숨은 공간/물체 수 추정과 영역별 순위 사용 | `planner.hsg_with_hidden` |

## 한국어 실험 명세

Baseline 거리는 **고정된 `robot_base` 원점 (0,0,0)과 관측 물체 중심 사이의 3D Euclidean 거리**다.
TCP/그리퍼, 목표 물체, 작업대 중앙과의 거리를 사용하지 않는다. 동률은 track ID로 결정한다.

세 조건의 공통 규칙:

- 현재 관측된 목표의 위가 비어 있으면 목표를 우선 집는다.
- 탐색 후보는 현재 관측된 선반/tabletop 물체 중 국소 적층 geometry 검사상 위가 빈 물체다. tray 안 물체는 탐색에서 제외한다.
- 이 검사는 충돌 없는 실제 접근·IK 또는 ACT 성공을 보장하지 않는다. 세 조건에서 같은 검사만 사용한다.
- 미식별 목표에 대한 선반 힌트는 동일하게 제공한다. 식별 후 목표가 이동한 경우에는 최초 선반 힌트로 현재 목표 접근을 제한하지 않는다.
- 성공적으로 배치 완료한 track은 미식별 목표 탐색에서 다시 선택하지 않는다. 이 execution history는 세 조건에 똑같이 제공한다.
- PLACE는 같은 규칙으로 버퍼 좌우 슬롯 → ID 순 선반의 빈 바닥 칸 → 공통 tabletop 후보를 찾는다. **배치에서 숨은 공간 점수는 A/B/C 모두 사용하지 않는다.**
- B/C는 같은 관측·관계와 planner를 사용한다. B의 `_hidden()`은 추정기를 호출하지 않고 0을 반환한다.
- B/C 모두 숨은 공간 추정치가 0이라고 탐색을 중단하지 않는다. 같은 후보 집합에서 robot-base Greedy로 계속 탐색한다.

## ‘숨은 객체 정보’의 정확한 의미

본 실험은 사용자 답변에 따라 **높이·가림에서 추정한 숨은 공간 단서**를 제거한다.
현재 구현은 주로 관측 물체 밑면 높이·알려진 지지 물체 높이·바닥 높이로 미관측 공간을 추정한다. 별도 학습된 가림 부피/목표 확률 모델은 아니다.
실제로 안 보이는 물체의 정답 ID·좌표를 C에 추가하지 않는다. history-only geometry도 현재 파지 대상으로 넣지 않는다.

B에서도 ACT가 물체에 접근해야 하므로 관측된 위치·크기·자세를 유지한다. 따라서 이는 **숨은 공간 추론 기능의 사용 제거(ablation)**이지,
모든 암묵적인 높이 정보를 센서에서 제거했다는 의미는 아니다. 과거 관측 기억을 제거하는 ablation과도 다르다.

A는 공통 모듈 I/O 때문에 `SceneGraph` 어댑터를 거치지만 의사결정에는 물체 목록과 정적 영역 geometry만 사용하고 관계 간선·숨은 점수를 읽지 않는다.
인식·추적·목표 속성 대응은 공통으로 유지한다. 디버그 inspector에 공통 A의 그래프가 표시되어도 Baseline이 그 관계로 계획한다는 뜻은 아니다.
따라서 A/C는 표현과 규칙을 함께 바꾸는 전체 방식 비교이고, B/C가 추정 단서 사용의 효과를 분리하는 비교다.

## ACT 고정 조건

- `training/runs/act_rgbd_shelves_absolute/best.pt` 하나를 사용한다.
- 체크포인트 SHA256, 소스 SHA256, 장면 manifest, device를 잠근다. 설정이나 소스가 바뀐 기존 출력 경로는 재사용하지 않는다.
- 매 조건/장면마다 새 ACT 인스턴스를 만들고 `model.eval()`로 추론한다. 재학습이나 expert fallback은 하지 않는다.
- 내부 chunk K=16, 기본 실행 cache 4개, temporal ensemble off, public `ActionChunk` 1×6을 유지한다.
- 수치 Subgoal/25D encoding, 카메라 calibration, joint-limit/delta clipping, 제어율·예산도 동일하다.
- 모든 평가 seed가 체크포인트 train/validation seed와 겹치지 않는지 검사한다.
- 같은 모델이어도 planner가 선택한 위치·자세에 따라 ACT 실패율이 달라질 수 있다. 이 결과는 고정 ACT의 시스템 성능이고, 계획 논리 단독의 성능과 구분한다.

## 이번 pilot

새 seed 7200부터 후보 장면을 먼저 촬영했다. 조건별 하나씩 총 4개 장면을 [pilot_manifest.json](pilot_manifest.json)에 실행 전에 고정했다.
목표가 보이는 두 조건은 대조 조건이다. 목표 미식별 두 조건은 제거할 추정 단서가 실제로 존재하는 장면으로 한정한다.
ACT 성공/실패나 어떤 방법이 이기는지는 장면 선정에 사용하지 않았다. 이 pilot은 **단서가 존재하는 조건의 연결 확인**이며 임의 장면 전체의 평균 성능을 대표하지 않는다.

초기 선택은 다음과 같다. 모든 ID는 A의 관측 track ID다.

| 장면 | 조건 | A | B | C |
|---|---|---|---|---|
| seed 7200 | 목표 식별·바로 집기 | i014 | i014 | i014 |
| seed 7201 | 목표 식별·방해물 있음 | i003 | i024 | i024 |
| seed 7202 | 목표 미식별·선반 알려짐 | i014 | i014 | i016 |
| seed 7203 | 목표 미식별·선반 모름 | i004 | i004 | i013 |

이 선택 차이만으로 회수 성공률 향상을 주장하지 않는다. [RESULTS.md](RESULTS.md)와 원자료를 함께 읽는다.
파일럿의 단계별 실패 및 가려진 목표 발견 사례는 [DIAGNOSTICS.md](DIAGNOSTICS.md)에 별도로 정리했다.

## 실험 실행

```powershell
training/.venv-rgbd/Scripts/python.exe scripts/benchmark_act_conditions.py run
.venv-perception/Scripts/python.exe scripts/benchmark_act_conditions.py report
```

규모를 늘릴 때에는 pilot seed를 다시 최종 test로 사용하지 않고 새 manifest/output을 고정한다. 예:

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_act_conditions.py prepare --count 52 --start-seed 7400 --manifest outputs/act_conditions/final/manifest.json --output outputs/act_conditions/final
training/.venv-rgbd/Scripts/python.exe scripts/benchmark_act_conditions.py run --manifest outputs/act_conditions/final/manifest.json --output outputs/act_conditions/final
.venv-perception/Scripts/python.exe scripts/benchmark_act_conditions.py report --manifest outputs/act_conditions/final/manifest.json --output outputs/act_conditions/final --report-dir outputs/act_conditions/final/report
```

출력은 `outputs/act_conditions/pilot/act/<condition>/<scene>/`에 있다.
`condition_audit.json`에는 실제 선택·순위·단서 사용 여부와 ACT 설정/명령 해시가, `result.json`에는 물리적 회수 성공과 실패가 저장된다.
초기 qpos와 첫 RGB-D 그래프를 manifest와 엄격히 비교한다. 정답 객체 정보는 선정 라벨/독립 평가에만 사용한다.

## 기존 GUI로 확인

다음 명령의 planner 모듈만 위 표의 세 모듈로 바꾸면 된다. checkpoint와 다른 인수는 유지한다.

```powershell
training/.venv-rgbd/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --controller act --checkpoint training/runs/act_rgbd_shelves_absolute/best.pt --device cuda --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --seed 101 --planner-module planner.hsg_with_hidden --inspect
```

CLI는 `on_pipeline_event`를 제공하는 planner에 완료 이벤트를 전달한다. GUI 유무와 무관하게 배치 완료 이력이 갱신된다.
기존 planner의 기본 동작은 유지한다. 이전 Greedy/Flat/HSG 결과 및 2×2 계산 실험은 이번 새 A/B/C 결과로 재명명하거나 합치지 않는다.
