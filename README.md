# SO-101 MuJoCo 공통 개발 환경

**MuJoCo 안의 SO-101·테이블·적층 블록·카메라를 공통 환경으로 사용하고,
세 사람이 A/B/C 모듈 하나씩 개발한 뒤 연결하는 프로젝트**이다.
`example.md`의 네 모듈 구조를 유지한다. 실물 로봇과는 통신하지 않는다.

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
**C의 SO-101 시작 코드는 현재 관절을 유지하는 hold baseline이다. ACT/파지 정책은 팀 C의 개발 대상이다.**
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
일반적인 물체의 파지 도달 가능성/경로는 아직 검증하지 않는다.
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
```

기본 C=hold로는 PICK/PLACE를 수행하지 못하며 실행 예산 소진으로 실패한다.
이 결과를 ACT나 목표 추출 성공으로 보고하지 않는다.
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
```

먼저 [입출력 명세](configs/io/README.md)를 맞춘다. 팔 관절은 radian, 그리퍼는 0=open/1=closed이며
실제 모터 원시 단위와 구분한다. 실제 로봇 연결과 ACT 학습은 이 환경 구축에 포함하지 않는다.

```sh
uv run python -m unittest discover -s tests -v
```

기본 테스트는 렌더링 없이 CPU 물리로 실행한다. 모델이 없으면 MuJoCo 테스트를 건너뛰므로
전체 검증 전 `setup_assets.py`를 실행한다. 이전 초안은 로컬 `outputs/legacy_v0_1/`에 보존되어 있다.
