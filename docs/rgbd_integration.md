# RGB-D → A/B/C 통합 실행

`scripts/run_pipeline.py --perception ...`으로 RGB-D 경로를 선택한다. 기본값 `gt`는 기존 동작이다.
현재 색상 상자 실험용 실행 경로이며 실물 로봇에는 연결하지 않는다.

## 실행

[인식 환경 설치](perception.md)의 Python 3.11 `.venv-perception`을 사용한다.
현재 PC에서는 해당 환경을 설치해 검증했다. 저장소 루트에서:

```powershell
.venv-perception/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --case accessible --controller expert --perception rgbd-color --target-spec configs/targets/blue_box.json --seed 300 --gui

.venv-perception/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --controller expert --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --seed 42 --gui
```

`--case blocked --target-spec configs/targets/blue_box.json --seed 42`는 소형 적층 회귀 장면이다.
`--perception-output PATH`로 기록 위치를 지정한다. 생략하면 `outputs/rgbd_pipeline/` 안에 실행별 디렉터리를 만든다.

`rgbd-color`는 **RGB 색상 연결 성분으로 마스크를 만드는 CPU 기준선**이다.
실제 렌더링한 RGB와 미터 depth를 사용하고, 시뮬레이터 segmentation ID·물체 자세·물체 색상 목록은 읽지 않는다.
SAM/DINO 추론 성능을 검증한 결과로 해석하면 안 된다.

## 목표 명세

예시:

```json
{
  "category": "box",
  "reference_rgb": [0.15, 0.35, 0.95],
  "region": "tabletop",
  "size_m": [0.03, 0.03, 0.03]
}
```

RGB는 `[0,1]`이다. 밝기를 정규화한 색도와 영역·선택적 크기로 현재 객체를 매칭한다.
`box`, `cube`, `block`은 이 상자 실험에서 같은 종류로 취급한다.
`region`과 `size_m`은 생략할 수 있다. 같은 조건의 후보가 모호하면 추가 조건이 필요하다는 오류를 낸다.
목표가 없으면 자세를 만들지 않고 B의 숨은 공간 탐색으로 넘긴다.
선반을 지정한 요청은 그 영역을 먼저 탐색한다. 매칭 후에는 추적 ID를 유지하므로 목표를 트레이로 옮겨도
처음 선반 조건 때문에 목표를 잃지 않는다.

공통 `TaskRequest`의 `target_object_id`에는 매칭 전 `unresolved:target`, 이후 `instance_XXXX`가 들어간다.
현재 ID 연결은 색·기하 기반 추적이며, 일반적인 장거리 재식별이나 임의 자연어 이해를 제공하지 않는다.

## 연결과 완료 판정

1. 두 고정 카메라를 동일 물리 시각에 렌더링한다. 촬영·추론 중에는 물리를 진행하지 않는다.
2. 분할 마스크를 기존 점군 재구성·시점 융합·ObjectTracker에 전달한다.
3. `rgbd_hssg`로 원본 그래프를 만들고 `perception.bridge`로 공통 그래프로 변환한다.
   history는 원본에 남지만 현재 집기·배치 장애물로 사용하지 않는다.
4. 공통 오케스트레이터가 목표 매칭 → B의 다음 Subgoal → C 명령을 실행한다.
5. PICK 중 물체가 밀리면 접촉 실패 구간에서 재관측하고 전문가가 접근을 다시 계산한다.
6. PLACE 후 그리퍼 개방·정착을 기다리고 다시 촬영해 배치 위치를 확인한다.
   이동 명령 prior는 관측과 호환되는 유일한 후보의 ID 재연결에만 사용한다.

C와 B는 시뮬레이터 물체 ID를 사용하지 않는다. MuJoCo 관측은 `privileged=False`로 생성해 객체 GT와
GT 접촉 관계를 만들지 않는다. 로봇 관절/FK와 양쪽 jaw의 이진 접촉은 사용할 수 있다.
PICK은 양쪽 접촉과 TCP 12mm 상승, PLACE는 12 tick 정착 후 관측 중심 오차 12mm 미만으로 판정한다.
GT 기반 환경의 실제 지지·속도·10mm 판정과 구분하며, 회귀 테스트는 실행이 끝난 뒤 독립적인 GT 판정도 확인한다.

제어용 자세는 **관측 top surface + 설정된 직립 정사각 상자 크기 후보**를 맞춘 추정이다.
원본 부분 표면 OBB는 그대로 보존하고 `box_model_fit.evidence`에 사용한 prior를 기록한다.
색상 하이라이트의 중복 표면과 바닥·적층 높이 범위를 벗어난 후보를 거른다.
기울어지거나 넘어진 상자, 임의 형상, 관측되지 않은 배치 장애물에는 이 가정이 성립하지 않을 수 있다.

## 기존 GPU 추론 연결

`--perception rgbd-command --perception-command PATH.json`은 외부 프로그램을 인자 배열로 실행한다.
`{inputs}`는 `frame_000/*_rgb.png`와 로컬 calibration/depth가 저장된 촬영 루트,
`{output}`은 예측 루트다. GPU worker가 읽는 입력은 RGB뿐이다.
예를 들어 같은 컴퓨터에 GPU worker 환경이 있으면 JSON은 다음과 같다.

```json
["PATH_TO_GPU_PYTHON", "scripts/infer_multiview_gpu.py", "--inputs", "{inputs}", "--output", "{output}"]
```

프로그램은 기존 worker와 동일한 `frame_000/<camera>/masks.npz`, `detections.json`을 출력해야 한다.
RGB hash 검사를 통과해야 하며, 명령 실패·없는 예측·입력 불일치는 중단한다. 색상 기준선으로 자동 전환하지 않는다.
원격 실행은 RGB만 전송하는 별도 명령 래퍼를 지정할 수 있다. 서버별 전송 설정은 [기존 문서](perception.md)를 따른다.
이번 통합에서 원격 서버나 SAM/DINO 모델 추론을 새로 실행하지는 않았다.

## 기록과 검증 범위

각 snapshot에는 두 카메라 RGB/depth/calibration, 분할 결과, 원본 그래프·delta·tracker state,
`perception_report.json`을 남긴다. 이동 중 FK로 전파한 자세와 재관측한 자세는 구분해야 한다.

```powershell
.venv-perception/Scripts/python.exe -m unittest discover -s tests -p test_rgbd_pipeline.py -v
```

검증은 색상 기준선 + 전문가 C에 대해 accessible seed 300, blocked seed 42, shelves seed 42를 대상으로 한다.
모든 C 호출에서 GT와 GT 접촉 ID가 없는지, 추적 ID가 전달되는지 검사하고 회수 후 독립적인 물리 성공을 확인한다.
숨은 목표 선택·모호한 목표 거부·history 제외·목표 ID 유지·공통 입력 저장/복원도 검사한다.
이 절의 표는 색상 기준선 + 전문가 C의 검증 범위다.
RGB-D 목표를 사용하는 ACT의 수집·학습·실행 결과는 [별도 ACT 검증 기록](../training/RGBD_VALIDATION.md)에 있다.

2026-11-01 로컬 검증 결과:

| 장면 | seed | 완료 Subgoal | 실행 후 독립 GT 물리 검사 |
|---|---:|---:|---|
| accessible | 300 | 2 | 통과 |
| blocked | 42 | 4 | 통과 |
| shelves | 42 | 4 | 통과 |

기존 공통·인식 테스트 76개와 새 통합 테스트 6개, 별도 ACT 학습 테스트 13개가 통과했다.
기존 자동 GUI 테스트 1개는 생략했고, 위 선반 명령은 실제 GUI로 실행해 완료와 최종 화면 저장을 확인했다.
최종 GUI 산출물은 `outputs/rgbd_pipeline/final_gui_seed42/`에 있다.
기존 테스트의 `TemporaryDirectory`는 이 Windows 샌드박스의 권한 제약을 피하도록 테스트 하네스에서만
작업 폴더 아래로 지정했다. 제품 코드의 파일 권한이나 성공 기준을 완화하지 않았다.
