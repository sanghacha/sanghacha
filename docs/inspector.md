# 시뮬레이션과 A/B/C 시각화

기존 파이프라인 명령에 `--inspect`를 붙인다. MuJoCo GUI도 함께 켜진다.
새 패키지는 설치하지 않는다. 기본 GT·전문가 예시:

```powershell
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --seed 42 --inspect
```

RGB-D + 학습된 ACT의 검증한 시연:

```powershell
training/.venv-rgbd/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --controller act --checkpoint training/runs/act_rgbd_shelves_absolute/best.pt --device cuda --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --seed 101 --inspect
```

## 읽는 방법

- 왼쪽 3D 장면은 목표(금색), 현재 실행 대상(청록), 선택한 그래프 노드를 표시한다.
  ID는 `instance_0025 → i025`, `shelf_2_obj_01 → s2.01`로 줄인다. 장식은 관측된 자세만 읽고 물리 상태를 바꾸지 않는다.
- RGB-D 모드의 위쪽 두 작은 영상은 실제 A가 사용한 fixed_top/fixed_oblique의 마지막 RGB-D snapshot이다.
  관측 마스크의 경계를 덧그린다. 물리가 진행되는 동안 매 tick 새로 촬영하는 영상은 아니다.
- **A Scene Graph**는 B로 전달된 공통 객체·영역 그래프의 요약이다. robot/camera 프레임과 stack 그룹은 접어 표시한다.
  `history` 객체는 회색으로 마지막 위치를 보여주며 B의 현재 집기 대상으로 넣지 않는다.
  노드를 클릭하면 전체 ID·상태·영역·좌표를 확인할 수 있다.
- **B**에는 실제 선택한 Subgoal의 이유, 현재 적층의 위→아래 순서, 숨은 공간 후보 점수 또는 배치 슬롯 점유 상태가 나온다.
  GT 그래프의 관계와 RGB-D 위치·크기로 B가 추정한 관계를 구분한다.
  플래너가 아직 반환하지 않은 미래 PICK/PLACE 목록을 만들지 않는다. 실행이 끝나면 다시 관측·계획한다.
- **C**는 현재 PICK/PLACE, Subgoal 시작 물체 자세, 배치 중심, 그리퍼·접촉·파지 상태,
  공통 ActionChunk 크기와 실제 명령, tick 예산을 보여준다.
  여기의 chunk 크기는 공통 인터페이스가 내보낸 실행 명령 수다. ACT 내부에서 예측한 chunk 길이와 다를 수 있다.
- 아래에는 관측 갱신 내역(이동 기준 5mm), 영역 이동/관측 상태 변화와 실제로 완료한 Subgoal 목록이 남는다.

화면의 현재 simulation 시각, A snapshot 시각, B가 계획한 시각을 구분한다.
PICK 재시도나 PLACE 확인 중에 A가 갱신되어도 B의 현재 Subgoal은 완료 전까지 유지된다.
회색 history와 파지 중 FK로 전파한 위치를 현재 카메라 관측과 혼동하지 않는다.

## 조작

| 키/마우스 | 동작 |
|---|---|
| Space | 현재 실행 일시정지/재생. 정지 중에는 물리를 진행하지 않는다. |
| N | 현재 Subgoal을 실행하고, 다음 Subgoal을 선택한 시점에 정지한다. |
| 그래프 노드 클릭 | 해당 ID와 좌표를 검사한다. 제어 목표를 변경하지 않는다. |
| 1~6 | 왼쪽 HUD에서 검사할 관절 선택 |
| 마우스 드래그/휠 | 3D 카메라 회전·이동·확대 |
| Esc | 창 종료. 실행 중이면 파이프라인도 중단한다. |

`--inspect-paused`로 첫 Subgoal 선택 시점부터 멈출 수 있다.
기본적으로 완료/실패 후에도 최종 화면을 유지한다. Esc로 닫는다.
자동 캡처·검사에서는 `--inspect-no-hold --snapshot outputs/inspector/final.png`를 붙여 자동 종료한다.
스크린샷에는 3D와 패널을 모두 저장한다.

기존 `--gui`만 사용한 화면과 수동 `run_simulation.py`의 키보드 조작은 유지한다.
다른 B 구현은 선택적으로 `explain_subgoal(graph, task, subgoal) -> dict`를 제공할 수 있다.
없는 경우 외부 플래너가 이유를 제공하지 않았다고 표시한다. 이유를 추측하지 않는다.

## 검증

실제 RGB-D + ACT seed 101을 inspector와 함께 실행해 6개 Subgoal 완료를 확인했다.
검사 산출물은 `outputs/inspector/review101/`에 있으며 `plan_01.png`~`plan_06.png`, `final.png`와
각 시점의 RGB-D/그래프 기록이 있다. 이는 시각화 동작 확인이며 ACT 성공률의 추가 평가가 아니다.
관측/플래너/완료 이벤트 순서, 숨은 목표 후보 순위, 실제 blocker chain, history 제외,
일시정지·다음 단계 상태와 실패 시 stop을 테스트한다.
