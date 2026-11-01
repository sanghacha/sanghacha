# 입출력 명세 v0.3 — SO-101 공통 개발 환경

공유 타입은 `src/pipeline/types.py` 한 파일에서 관리한다.
세 모듈은 서로의 내부 구현을 import하지 않고 아래 함수와 공통 타입으로 연결한다.

```python
from pipeline.types import Observation, SceneGraph, Subgoal, ActionChunk

# A — hssg_generator
def generate_scene_graph(observation: Observation) -> SceneGraph: ...

# B — planner
def plan_next_subgoal(graph: SceneGraph, task, robot_state) -> Subgoal: ...

# C — action_generator
def predict_action_chunk(observation: Observation, subgoal: Subgoal) -> ActionChunk: ...
```

## 기본 규약

- 3D 좌표는 `robot_base`, 오른손 좌표계, 거리/크기는 m이다.
- `Pose(position, quaternion_xyzw)`의 quaternion은 `[qx,qy,qz,qw]` 단위 quaternion이다.
  MuJoCo wxyz는 `pipeline.types.wxyz_to_xyzw()`로 변환한다.
- object ID는 episode 동안 유지한다. graph/observation의 episode와 timestamp는 일치해야 한다.
- 배열 dtype/shape, NaN/Inf, 잘못된 frame/계층/엣지를 거부한다. 유효하지 않은 입력은 `ValueError`이다.

## 데이터

| 타입 | 필드/의미 |
|---|---|
| `Pose` | `position` xyz, `quaternion_xyzw` |
| `SceneObject` | `object_id`, `pose`, `size` 로컬 xyz extent, `label`, `parent_id` |
| `Observation` | `rgb uint8[H,W,3]`, `robot_state float32[6]`, timestamp/episode/frame/source/robot_id/control_dt, 선택적 depth/GT, 카메라 보정 |
| `Edge` | `source`, `target`, `relation` |
| `SceneGraph` | `nodes`, `edges`, `groups={group_id: parent_id}`, `held_object_id`, timestamp/episode/frame/source |
| `TaskRequest` | `target_object_id`, `destination`, `buffer_pose` — 모두 물체 중심의 목표 Pose |
| `Subgoal` | `subgoal_id`, `primitive`, `object_id`, `object_pose`, `target_ee_pose`, `place_pose`, frame |
| `ActionChunk` | `subgoal_id`, `commands float32[K,6]`, `command_dt`, `robot_id` |

RGB-D 통합에서 사용하는 추가 필드(기존 호출의 기본값은 유지):

- `SceneObject.color_rgb`: 관측 표면의 평균 RGB, 각 채널 `[0,1]`.
- `Observation.perceived_objects`: 현재 추적 객체의 제어용 자세·크기. `ground_truth`와 구분한다.
- `Observation.finger_contacts`: 고정/이동 jaw의 외부 접촉 여부 두 개. 물체의 물리 ID는 포함하지 않는다.
- `SceneGraph.regions`: 영역별 XY 경계, `region_type`, `floor_top_m`. B의 숨은 공간 계산에 사용한다.
- `TaskRequest.target_spec`: 선택적 `TargetSpec(reference_rgb, region, category, size_m, ...)`.
  `TargetResolver`가 관측 객체에 매칭하고 `target_object_id`를 추적 ID로 바꾼다.
  후보가 없으면 미해결 상태로 탐색하고, 모호한 후보는 오류로 중단한다.
  최초 매칭 후에는 ID를 유지하며 초기 영역 조건을 다시 적용하지 않는다.

`RGBDEnvironment`에서는 객체 GT·GT 접촉 ID를 A/B/C에 보내지 않는다.
집은 물체의 이동 중 자세는 최초 관측과 TCP 상대 변환으로 전파한 추정값이다.
PICK은 양쪽 접촉과 TCP 상승, PLACE는 개방 후 정착 시간을 두고 새 RGB-D 관측으로 확인한다.
이 관측 기반 판정과 기존 GT 물리 판정의 차이는 [RGB-D 통합 실행](../../docs/rgbd_integration.md)을 따른다.

`Observation.depth`는 선택적 float32[H,W], m이며 0은 유효하지 않은 깊이를 뜻한다.
`camera_intrinsics`는 float64[3,3], `camera_to_robot_base`는 float64[4,4]이다.
카메라 optical 좌표는 x=오른쪽, y=아래, z=전방으로 정의한다.
`ground_truth`는 선택적 SceneObject tuple이다. 기본 MuJoCo 환경에서는 3개 선반에 있는 27개 물체의 실제 물리 상태를 제공한다.
원본 GT의 `SceneObject.parent_id`는 소속 선반(`shelf_1`~`shelf_3`), 빈 목표 선반(`buffer_tray`/`output_tray`) 또는 `tabletop`이다.
생성된 그래프에서는 `parent_id`를 적층 그룹으로 바꾸고 `groups[적층 그룹]=선반 ID`로 연결한다.
기존 A/B/C 소형 장면의 GT parent가 None인 경우에는 기존 `bin` 그룹을 사용한다.
source는 `mock_gt`, `mujoco_gt`, `rgbd` 중 하나이다. RGB-D 관측은 GT를 노출하지 않는다.
GT가 숨겨진 물체를 아는 실험과 RGB-D에서 관측한 정보만 쓰는 실험을 구분한다.

추가 관측 필드:

- `contacts: tuple[Contact,...]`: entity_a/entity_b, 접촉점 xyz(m), a→b 단위 법선, 법선 힘(N).
  물체는 `shelf_N_obj_XX`(소형 장면은 A/B/C), 선반은 `shelf_N`/`buffer_tray`/`output_tray`, 테이블은 table, 로봇은 해당 geom 이름을 사용한다.
  MuJoCo `data.contact`/`mj_contactForce`에서 읽는다.
- `ee_pose: Pose | None`: 실제 gripperframe site의 자세.
- `rgb_valid: bool`: 실제 카메라 렌더링 여부. `False`는 mock 패턴 또는 physics-only 빈 배열이므로 인식/정책 영상 입력으로 사용하면 안 된다.

`MujocoEnvironment`는 고정된 모델의 base 원점=world 원점으로 구성한다.
SO-101의 관절 순서는 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`이다.
관절은 이름으로 찾아 읽으며, 자유 물체 joint가 포함된 qpos 전체를 로봇 관절이라고 가정하지 않는다.
MuJoCo time은 settle 이후를 episode 0초로 사용한다. 30Hz 제어 한 번에 physics substep 20개를 진행한다.

기본 계층은 workspace → shelf_1/shelf_2/shelf_3 → stack group → object이다.
`on_top_of(A,B)`와 `supports(B,A)`의 방향을 구분한다.
추가 관계는 `occludes`, `inside`이며 가림 관계는 현재 관측 시점 기준이다.
`blocks_grasp`는 지지 관계를 복사하지 않고 Planner에서 선택한 파지 후보/접근 경로에 대해 판정한다.
현재 A는 mock에서는 박스 겹침/높이를 사용하고, MuJoCo에서는 실제 접촉·수직 법선·양의 접촉력·위아래 위치를 함께 검사한다.
일반 물체의 안정성/가림/파지 접근성 전체를 검증하는 알고리즘은 아니다.

## 실행

Planner는 다음 **한 개**의 Subgoal만 반환한다. 이후 단계는 환경 상태를 다시 관측해서 결정한다.
PICK의 `object_pose`는 현재 물체 자세, `target_ee_pose`는 선택한 그리퍼 목표로 서로 다르다.
PLACE만 `place_pose`를 필수로 받는다. 실제 로봇에서는 파지한 물체↔EE 상대 변환을 반영해야 한다.
MuJoCo의 destination/buffer_pose는 빈 선반의 **바닥 위에 놓인 물체 중심**이다.
목표 중심 z는 선반 바닥 두께+목표 물체 높이/2로 계산하고, 다른 높이의 방해물은 Planner에서 그 차이를 보정한다.
MuJoCo 예시 Planner는 11cm 임시 선반에서 x±0.025m 두 슬롯을 후보로 골라 물체 footprint 중복을 검사한다.
다른 크기의 선반/물체를 쓰는 B 구현은 해당 작업 공간에 맞는 후보를 제공해야 한다.
mock은 EE z=물체 중심 z+0.08m를 사용한다.

Action은 absolute joint-position target이다. 첫 5채널은 관절 rad, 마지막은 0=open/1=closed이다.
MuJoCo 환경은 그리퍼 정규화 값을 공식 모델의 hinge angle로 변환한다. 모터 원시 단위는 아니다.
로봇 명령/관측의 `robot_id=so101`과 mock의 `robot_id=mock`은 섞지 않는다.
MuJoCo의 관절 범위는 MJCF actuator ctrlrange에서 읽고, step당 변화량은 `configs/envs/so101.json`에서 설정한다.
범위/변화량 초과는 거부한다. mock의 `xyz=관절값×0.1`은 실제 시뮬레이터에 사용하지 않는다.

파이프라인은 예측한 K개 중 `n_action_steps`만 실행하고 나머지는 버린 뒤 재예측한다.
Subgoal이 완료되면 남은 chunk는 실행하지 않는다. chunk 소진 자체는 작업 성공이 아니다.
모델 내부에 큐/temporal state가 있는 C 구현은 **subgoal_id가 바뀔 때 내부 상태를 초기화**해야 한다.
외부 reset 메서드는 요구하지 않는다. 목표 텐서 인코딩은 C 내부에서 정하며 12D로 고정하지 않는다.

환경의 `subgoal_complete` / `task_complete`가 완료를 판정한다.
MuJoCo PICK은 양쪽 jaw와의 접촉, 기존 지지 해소 및 해당 Subgoal의 초기 높이보다 1cm 이상 lift를 확인한다.
PLACE/회수는 release·장면의 지지 접촉·그리퍼 열림·물체 속도·목표 위치 오차 1cm 이내를 확인한다.
빈 목표 선반 안의 목표에는 해당 선반의 지지 접촉을 요구한다.
현재 기준은 개발용이며 복잡한 물체의 파지 안정성/회전 정밀도 평가는 후속 범위이다.
mock PICK은 held ID, PLACE/최종 회수는 release와 물체 중심 오차≤0.005m를 확인한다.
mock 판정에는 실제 lift 안정성/파지력/회전 오차/충돌 판정이 포함되지 않는다.
Runtime은 실행 예산을 제한하고 성공/실패/예외에서 stop한다. 환경을 생성한 호출자가 close한다.

## 독립 테스트 데이터

`mock_cases.json`은 일반 JSON이다. 커스텀 직렬화 태그나 별도 export 과정이 없다.

```python
from pipeline.mock_data import make_mock_data, make_policy_input

data = make_mock_data("blocked")
# A 입력: data.observation / 기대 결과: data.graph
# B 입력: data.graph, data.task, data.observation.robot_state
# 기대 첫 작업: data.expected_order[0]
observation, subgoal = make_policy_input("PLACE")  # C 입력: 이미 물체를 잡은 상태
```

케이스는 blocked, accessible, after_move, missing_target이다.
expected_order는 논리적 작업 순서이며 시연 trajectory가 아니다.
RGB는 정적인 16×16 합성 패턴이며 인식/정책 학습용 데이터가 아니다.
seed는 mock episode 식별에만 쓰며 배치는 고정되어 있다.

## 실제 SO-101 입력 파일

```sh
uv run python scripts/run_simulation.py --headless --seconds 0.1 --seed 42
```

```python
from pipeline.sample_data import load_sample
sample = load_sample("outputs/so101/shelves/seed_42")
# sample.observation, sample.graph, sample.task, sample.subgoal
```

`arrays.npz`는 RGB/깊이/관절/카메라 배열, `sample.json`은 GT·접촉·EE·graph·목표·설정·버전·모델 커밋이다.
읽을 때 allow_pickle=False를 사용한다. `rgb.png`는 실제 렌더링이 있는 경우에만 저장한다.
코드에 실행 중인 MuJoCo model/data 객체를 넘기지 않으므로 각 모듈이 캡처한 데이터로 독립 개발할 수 있다.
SO-101 C는 `HSSG_ACT_CHECKPOINT`가 없으면 MuJoCo GT 전문가를 사용하고, 지정하면 별도 `training/` 프로젝트의 ACT를 로드한다.
`HSSG_C_BACKEND=expert|act|hold|auto`로 선택할 수 있다. hold는 명시적인 진단용이다.
전문가는 별도 공식 로봇 모델로 IK를 계산하고 실제 관측·접촉으로 단계별 제어한다. 살아 있는 환경의 model/data를 받거나 수정하지 않는다.
계획한 물체의 집기·파지 확인·들어 올리기, 배치 위치 보정·부분 개방·후퇴·완전 개방을 수행한다.
배치가 목표 밖에 정착하면 최대 두 번 재집기를 시도하고, 관절 한계·파지 실패·정체는 오류로 반환한다.
목표 인코딩 `so101-goal-v2`는 primitive one-hot 2, 시작 물체 pose 7, PLACE pose 7, 유효 플래그 1,
PLACE 시작 시 물체↔EE 상대 pose 7, 상대 pose 유효 플래그 1을 합친 25D이다.
상대 pose는 C가 Subgoal 시작 시 물체 자세와 실제 EE 자세로 계산해 고정한다. EE 자세가 없으면 공식 로봇 FK를 사용한다.
ID와 mock EE 목표는 신경망 특징에 포함하지 않는다. 공통 필드와 세 함수 API는 변경하지 않는다.
목표 인코더는 `src/action_generator/goal.py`, 학습/수집/평가 안내는 `training/README.md`이다.
내부 K-step 예측 중 4개를 사용하되, C는 실제 관측 기준으로 명령을 제한하여 한 tick씩 `ActionChunk[1,6]`으로 반환한다.
`training`의 `act-generate`가 실제 성공한 전문가 시연을 자동 저장한다. 학습된 ACT와 전문가 성공률은 구분한다.
기본 실행 예산은 Subgoal당 450 tick이다. 데이터 수집·표시용 `run_pipeline(on_step=...)`은
실행된 각 명령 뒤 `(before_observation, command, after_observation, subgoal)`을 받으며, 생략하면 기존 동작과 같다.

v0.3은 v0.2의 세 함수 API와 기존 필드를 유지하며 실제 환경용 선택적 관측 필드를 추가했다.
v0.1의 `hssg_manipulation.common`/Protocol/codec과는 호환되지 않는다.
필드를 바꾸면 이 문서, `pipeline/types.py`, JSON 예시와 소비자/테스트를 함께 수정한다.
관절 매핑은 SO-101 기준을 사용한다. 위 Goal 인코딩은 모듈 C 내부 v1이며 변경 시 시연·체크포인트 버전도 변경해야 한다.
