# 계층 조회·부분 갱신 비교

이전 [52개 회수 실험](../planner_benchmark/RESULTS.md)은 보존했다. 이번 단계는 **같은 계획을 더 적은 계산으로 유지하는지**를 검증한다.
목표 미식별 시 탐색 점수나 파지·배치 정책은 바꾸지 않았다. 성공률 개선 실험과 구분한다.

최종 결과는 [controlled/RESULTS.md](controlled/RESULTS.md), 그래프는 [controlled/results.png](controlled/results.png),
원자료는 압축 JSON과 [controlled/summary.json](controlled/summary.json)에 있다.

## 실제 구현

`src/planner/indexed.py`의 `IndexedPlanner`가 공통 B 계약을 유지한다.

- 모든 방법: object ID 인덱스, XY 공간 해시, 적층 adjacency, 영역별 membership, 물체별 top/숨은 높이 후보를 사용한다.
- Flat 전체 재계산: 매 입력에서 인덱스와 후보를 다시 구축하고, 탐색할 때 후보 전체에서 영역 합계와 우선순위를 계산한다.
- Flat 부분 갱신: 정확한 변경 감지로 관계와 해당 영역의 물체별 후보를 재사용한다. 조회 시 전체 후보를 집계한다.
- HSG 전체 재계산: 위 자료와 함께 영역별 숨은 수 합계·최선 후보 요약을 매번 구축한다.
- HSG 부분 갱신: 변경된 객체의 이전/새 위치, 연결 관계와 관련 영역의 후보·요약만 갱신한다. 조회 시 영역 요약을 비교하고 선택한 후보를 조회한다.

따라서 HSG는 지역 요약 캐시를 추가로 사용한다. Flat도 같은 캐시를 추가하면 유사한 이점을 얻을 수 있다.
이 비교는 구현한 **계층적 요약·조회 전략의 효과**이며, 그래프 파일 형식 자체가 항상 우월하다는 주장이 아니다.
공간 해시와 adjacency의 공통 최적화를 HSG만의 기여로 계산하지 않는다.

현재 A는 전체 `SceneGraph`를 반환하므로, B의 public snapshot API는 전체 입력을 비교하는 O(N) 비용을 부담한다.
이 비용, 지역 요약 유지 비용, 초기 구축 비용을 숨기지 않고 측정했다. 캐시 때문에 메모리가 늘어날 수 있다.

별도 `DeltaEncoder` / `GraphUpdate` / `plan_update()` API도 제공한다. 정확한 객체 추가·변경·삭제와 metadata, held 변경을 처리한다.
누락/순서 오류 delta는 거부하고, episode 변경·시각 역행은 snapshot API에서 재동기화한다.
pose tolerance는 사용하지 않는다. A가 delta를 제공하려면 해당 생성 비용도 포함해 평가해야 한다.
한 planner 인스턴스에서 snapshot API와 delta API를 섞지 않는다. 이번 성능 결과는 모두 snapshot API다.

## 시뮬레이션에서 사용

기존 명령에 새 B 모듈을 지정한다. 기본 planner는 이전 구현으로 유지했다.

```powershell
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --seed 42 --planner-module planner.hsg_incremental --inspect
```

실제 RGB-D 관측과 함께 사용하는 예:

```powershell
.venv-perception/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --controller expert --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --seed 42 --planner-module planner.hsg_incremental --inspect
```

동일한 인터페이스의 모듈은 `planner.flat_full`, `planner.flat_incremental`, `planner.hsg_full`, `planner.hsg_incremental`이다.
HSG inspector의 이유 설명에 변경 물체·갱신 영역 수를 표시한다. 시각화 비용은 headless 성능 측정에 포함하지 않는다.

## 고정된 계산 실험

- 크기: 관측 객체 27 / 216 / 1,728개, 영역당 9개. 큰 크기는 **합성 그래프**이며 SO-101 가동범위의 물리 장면이 아니다.
- 변화: 한 영역의 물체가 움직이는 국소 변화, 모든 영역이 움직이는 전체 변화. 매 4번째 frame은 변화 없음이다.
- 작업: 목표 미식별 탐색 / 목표가 식별된 장애물 접근. 목표 식별 작업도 음성 대조 조건으로 유지했다.
- 네 방법은 동일 그래프·같은 정책을 사용한다. 호출 순서를 frame별로 순환한다.
- 개발 측정: seed 6100~6102, 36개 stream × 20 frame × 4방법 = 2,880회. `compute.json.gz` / `protocol.json` 보존.
- 알고리즘을 바꾸지 않고 새 seed 6200~6209로 확인 측정: 120개 stream × 20 frame × 4방법 = 9,600회.
- 확인 측정의 조건과 seed는 실행 전 `confirmatory/protocol.json`으로 고정했다. 다른 실험 프로세스와 겹치지 않게 직렬 측정했다.
- v1에서는 frame별 호출 순서와 매 4번째 무변화 frame이 결합되는 문제가 있어, 위 결과를 예비 측정으로 보존했다.
- 최종 v2는 알고리즘을 바꾸지 않고 새 seed 6300~6311(12개), 순서 `(frame + seed) % 4`로 조건별 순서를 균등화했다. 144개 stream × 20 frame × 4방법 = 11,520회다.
- v2는 B 호출 중에만 순환 GC를 끄고 참조 카운트 해제 비용은 유지한다. 다른 방법의 할당이 특정 방법에 GC 비용을 전가하는 영향을 줄이기 위한 microbenchmark 조건이며, 실제 앱의 GC 비용을 포함한 전체 시간으로 해석하지 않는다.
- 첫 frame의 캐시 구축 시간을 따로 기록하고, 이후 시간은 seed별 중앙값으로 요약한다. seed 단위 bootstrap 95% 구간도 제공한다.
- 메모리는 별도 tracemalloc 실행으로 측정한다. 미리 생성한 입력 그래프와 native/GPU 메모리를 제외한 planner의 추가 Python allocation이다.
- 합성 입력 생성 + B 시간도 제공한다. 합성 입력 생성은 실제 RGB-D A 추론 시간이 아니다.

재현:

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_indexed_planners.py compute --report-dir outputs/hsg_indexed/repeat_compute --seeds 6300 6301 6302 6303 6304 6305 6306 6307 6308 6309 6310 6311
.venv-perception/Scripts/python.exe scripts/benchmark_indexed_planners.py report --report-dir outputs/hsg_indexed/repeat_compute
```

기존 계산 결과는 덮어쓰지 않는다. 반복 실험은 새 report 경로를 사용한다.
`compute.json.gz`에는 소스 SHA256, 버전, frame별 계획·작업량·시간·메모리 정보가 있다.

## 새 RGB-D 물리 검증

seed 5100부터 초기 관측 조건으로 선정한 장면 8개(조건별 2개)를 [physical_manifest.json](physical_manifest.json)에 미리 고정했다.
이전 52개와 겹치지 않고, 성공 결과를 보고 장면을 선택하지 않았다. 동일 expert와 실제 2-view RGB-D로 4방법 × 8장면 = 32회 실행했다.
초기 물리 상태와 첫 그래프를 검사하고, expert가 출력한 모든 명령의 SHA256도 비교한다.
이 검증의 목적은 행동이 유지되는지 확인하는 것이며, 8개 장면으로 성공률 일반화를 주장하지 않는다.
ACT의 새 성능 평가는 이번 범위에 포함하지 않는다.

```powershell
.venv-perception/Scripts/python.exe scripts/benchmark_indexed_planners.py physical
.venv-perception/Scripts/python.exe scripts/benchmark_indexed_planners.py report --report-dir experiments/hsg_indexed/controlled
```

물리 원자료는 `outputs/hsg_indexed/physical/expert/<variant>/<scene>/`에 있다.
`indexed_audit.json`은 B 작업량과 실제 명령 해시, `result.json`은 독립 물리 성공 판정과 실행 이력을 담는다.
report는 장면별 네 방법의 명령 해시와 결과가 다르면 중단한다.

추가 회귀 검사로 기존 52개 평가에서 저장한 433개 실제 B 입력에 네 방법을 적용하여 기존 planner와 비교했다.
이는 개발 회귀 검사이며 새 held-out 성능 평가로 세지 않는다. [replay_audit.json](replay_audit.json)에 입력 해시를 보존했다.

```powershell
.venv-perception/Scripts/python.exe scripts/check_indexed_replay.py
.venv-perception/Scripts/python.exe -m unittest discover -s tests -p test_indexed_planner.py -v
```

연구 주장은 결과가 나온 조건으로 한정한다. 전체 변화·이미 식별된 목표에서는 요약 유지 비용이 이득보다 클 수 있다.
성공률 상승, 조작 횟수 감소, 전체 RGB-D→로봇 완료 시간 단축을 B의 계산 실험에서 추론하지 않는다.
