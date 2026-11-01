# SO-101 MuJoCo 공통 개발 환경

카메라 입력부터 인식·객체 추적·SG 생성을 실행하려면 [RGB-D 인식 파이프라인](docs/perception.md)을 참고한다.
이 경로는 별도 Python 3.11 환경과 선택적 GPU worker를 사용하며, 기존 제어/학습 환경과 분리된다.

**MuJoCo 안의 SO-101·테이블·적층 블록·카메라를 공통 환경으로 사용하고,
세 사람이 A/B/C 모듈 하나씩 개발한 뒤 연결하는 프로젝트**이다.
`example.md`의 네 모듈 구조를 유지한다. 실물 로봇과는 통신하지 않는다.

이미 인식·추적된 3D 객체에서 별도로 그래프를 생성하려면
[RGB-D HSSG 패키지](docs/rgbd_hssg.md)를 사용한다. `rgbd_hssg`는 현재의 네 관계와
스냅샷 간 변경 내역을 생성하며 기존 GT 기반 A/B/C 인터페이스는 유지한다.

**RGB-D 인식 결과를 B/C와 연결하는 실행 경로도 제공한다.**
색상 블록용 로컬 분할 또는 외부 SAM/DINO 명령을 선택하며,
설치·목표 지정·실행·검증 범위는 [RGB-D 통합 실행](docs/rgbd_integration.md)을 따른다.
RGB-D와 학습된 ACT를 함께 쓰는 Python 3.11 CUDA 환경 및 평가 결과는
[RGB-D ACT 검증 기록](training/RGBD_VALIDATION.md)에 있다.
`--inspect`를 붙이면 시뮬레이션과 씬그래프 갱신·플래너 선택 이유·현재 Subgoal을 같은 창에서 볼 수 있다.
Space 일시정지, N 다음 단계, 노드 클릭 검사는 [시각화 안내](docs/inspector.md)를 따른다.

## 처음 설치하기

uv와 Python 3.13을 기준으로 저장소 루트에서:

```sh
uv sync --locked
uv run python scripts/setup_assets.py
uv run python scripts/run_simulation.py
```

첫 설치에서 MuJoCo와 공식 SO-101 MJCF/STL 약 18MB를 받는다.
모델은 고정 커밋·checksum으로 검증하며, 원본 라이선스와 출처는 [assets/so101](assets/so101/README.md)에 기록한다.
의존성 버전은 `uv.lock`으로 공유한다. 학습 모델이나 PyTorch는 설치하지 않는다.

## 시뮬레이터에서 확인할 것

- SO-101 5축 + 그리퍼, 테이블, **낮은 선반 3개**.
- 각 선반에 서로 다른 색상/크기의 상자 **9개씩, 총 27개**를 1~3층으로 무작위 배치/적재.
- 적재 선반 앞에 **임시 배치용·목표 회수용 빈 선반 2개**를 둔다. 각각 바닥과 낮은 테두리가 있고 실제 충돌을 처리한다.
- 키보드 **1~6**: 관절 선택, **↑/↓를 누르는 동안 계속 이동**, **키를 놓으면 정지**, **Space**: 초기 목표로 복귀, **R**: 다음 seed로 재배치.
- 화면 왼쪽 위에 선택한 관절·현재 각도·목표 각도·한계를 표시한다. 다른 창으로 포커스를 옮기면 방향키 조작이 중지된다.
- 왼쪽 마우스 드래그: 회전, 오른쪽 드래그: 이동, 휠: 확대/축소.
- 모델의 실제 actuator/접촉 물리로 움직인다. 물체를 손에 강제로 부착하거나 좌표를 순간 이동시키지 않는다.

창을 닫으면 마지막 RGB-D·관절·GT·접촉·graph·subgoal을 `outputs/so101/shelves/seed_42/`에 저장한다.
다른 seed로 생성한 장면은 별도 폴더에 저장한다.

창 없이 실제 RGB-D를 렌더링하고 관절 움직임을 확인하려면:

```sh
uv run python scripts/run_simulation.py --headless --demo motion --seconds 2
uv run python scripts/run_simulation.py --seed 123
uv run python scripts/run_simulation.py --joint-speed 90
```

OpenGL 없이 CPU 물리/GT만 검사하려면 `--headless --physics-only`를 쓴다.
이 경우 `rgb_valid=False`이며 RGB를 인식/정책 입력으로 사용하면 안 된다.
렌더링/GUI는 OpenGL 환경이 필요하다. 현재 검증 플랫폼은 Windows, Python 3.13, MuJoCo 3.17이다.
기본 수동 조작 속도는 팔 60°/초, 그리퍼는 초당 개폐 범위의 80%이다.
`--joint-speed`, `--gripper-speed`로 조절하며 모델의 관절 한계와 step당 명령 제한은 계속 적용된다.
GUI는 눌림/해제를 받는 GLFW 창을 사용한다. [GLFW 입력 안내](https://www.glfw.org/docs/latest/input_guide.html).
RGB-D 렌더러와 GUI가 함께 실행되므로, 조작 창 생성 시 GLFW의 숨김 옵션을 초기화하고 표시를 명시한다.

## 팀원이 수정할 세 함수

| 팀 | 구현 위치 | 공개 함수 |
|---|---|---|
| A | `src/hssg_generator/generator.py` | `generate_scene_graph(observation)` |
| B | `src/planner/planner.py` | `plan_next_subgoal(graph, task, robot_state)` |
| C | `src/action_generator/generator.py` | `predict_action_chunk(observation, subgoal)` |

실제 SO-101 환경의 입력으로 자기 모듈만 검사한다.

```sh
uv run python scripts/check_module.py a
uv run python scripts/check_module.py b
uv run python scripts/check_module.py c
```

다른 Python 모듈을 지정하려면 `--module my_package.my_module`을 붙인다.
현재 A는 MuJoCo GT 접촉 기반 적층 그래프 예시, B는 단순 순서 선택 예시이다.
**C는 체크포인트 없이도 IK·접촉 피드백 전문가로 PICK/PLACE를 수행한다. `HSSG_ACT_CHECKPOINT`를 지정하면 목표 조건부 ACT를 사용한다.**
시연 수집·학습·평가는 루트의 별도 uv 프로젝트 [training/](training/README.md)에서 진행한다.
전문가 제어기는 ACT 학습 결과와 구분한다. 자동 시연 생성도 제공한다.
현재 작업 폴더의 `training/runs/act_grasp/best.pt`는 `accessible`의 상자 B/C에 대해 실제 집기·배치를 검증한 초기 ACT이다.
사용법과 성공/실패 범위는 [모듈 C 검증 기록](training/VALIDATION.md)에 있다.
단독 검사는 입출력 확인이며 물체 회수 성공 평가가 아니다.

## 같은 입력 파일로 독립 개발

시뮬레이터를 매번 실행할 필요 없이 캡처한 한 시점을 그대로 읽는다.

```python
from pipeline.sample_data import load_sample

sample = load_sample("outputs/so101/shelves/seed_42")
# A: sample.observation -> SceneGraph
# B: sample.graph, sample.task, sample.observation.robot_state -> Subgoal
# C: sample.observation, sample.subgoal -> ActionChunk
```

```sh
uv run python scripts/check_module.py b --sample outputs/so101/shelves/seed_42
```

선반 3개의 크기/좌표, 선반당 물체 수, 크기 종류, 적재 층수는 `configs/envs/so101.json`의 `shelf_scene`에서 관리한다.
빈 목표 선반의 크기·바닥·테두리 높이·색상은 같은 파일의 `placement_trays`에서 설정한다.
배치 목표의 z는 빈 선반의 바닥 두께와 물체 높이를 반영한다.
배치는 서로 겹치지 않는 후보 위치와 회전에서 샘플링하고 MuJoCo 물리로 안정화한다.
큰 상자를 아래에 놓고 그보다 작은 상자를 위에 쌓는다. 바닥 목표 `shelf_2_obj_01` 위에는 방해물을 배치한다.
전문가는 현재 상자 장면에서 검증하며, 임의 형상·모든 무작위 배치의 도달 가능성을 보장하지 않는다.
기존 소형 회귀 장면은 `--case blocked`, `--case accessible`, `--case after_move`로 생성한다.
배치는 `--seed`로 재현한다. 저장 폴더에는 `rgb.png`, 숫자 배열 `arrays.npz`,
타입별 데이터·모델 커밋·환경 설정·버전 기록 `sample.json`이 있다. pickle은 사용하지 않는다.

## 각자 개발 후 통합

```python
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.runner import run_pipeline

env = MujocoEnvironment()
try:
    completed = run_pipeline(env, env.task,
                             generate_graph=my_hssg_function,
                             plan_subgoal=my_planner_function,
                             predict_action=my_action_function)
finally:
    env.close()
```

CLI에서도 모듈 경로를 지정할 수 있다.

```sh
uv run python scripts/run_pipeline.py --backend mujoco --action-module my_package.my_policy
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --seed 42 --gui
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --physics-only
```

기본 `auto`는 체크포인트가 있으면 ACT, 없으면 전문가를 선택한다. `--controller expert`는 전문가를 명시한다.
`--controller hold`는 관절 유지 진단용이며 작업을 수행하지 않는다. 실패 시 다른 제어기로 몰래 전환하지 않는다.
전문가는 관절 명령만 출력하고, 실제 접촉으로 집기·들어 올리기·배치·놓기를 수행한다. 물체 위치를 직접 변경하지 않는다.
현재 B의 MuJoCo 예시는 임시 선반의 두 슬롯을 사용해 두 방해물을 분리해서 배치한다.
PICK/PLACE당 450 tick, 최대 Subgoal 12개로 제한하고, 예외/실패/완료 시 stop한다.
연구용 RGB-D 인식·탐색 시스템과 학습된 ACT 성공률은 각각 별도 검증해야 한다.
기존 mock 연결 예제는 `uv run python scripts/run_pipeline.py --backend mock`으로 따로 실행한다.

## 구조와 명세

```text
src/hssg_generator/          A
src/planner/                 B
src/action_generator/        C
src/pipeline/                공용 타입·MuJoCo 환경·실행·입력 저장/로드
configs/envs/so101.json       선반 3개·적재 생성·관절·제어·실행 예산
configs/io/README.md         공통 입출력 명세
assets/so101/                공식 모델 출처와 다운로드 manifest
scripts/                     설치·GUI·모듈 검사·통합 실행
tests/                       CPU 계약·물리·통합 테스트
training/                    별도 uv 프로젝트: 자동 시연·ACT 학습·평가
```

먼저 [입출력 명세](configs/io/README.md)를 맞춘다. 팔 관절은 radian, 그리퍼는 0=open/1=closed이며
실제 모터 원시 단위와 구분한다. 실제 로봇 연결은 포함하지 않는다.
모듈 C의 ACT 학습 환경은 공통 패키지와 분리된 [training/](training/README.md)에 있다.

```sh
uv run python -m unittest discover -s tests -v
```

기본 테스트는 렌더링 없이 CPU 물리로 실행한다. 모델이 없으면 MuJoCo 테스트를 건너뛰므로
전체 검증 전 `setup_assets.py`를 실행한다. 이전 초안은 로컬 `outputs/legacy_v0_1/`에 보존되어 있다.

