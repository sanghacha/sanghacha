# 모듈 C — 목표 조건부 ACT

학습 없이 동작하는 MuJoCo 전문가, 전문가의 자동 시연 생성 → 목표 조건부 ACT 학습 → 체크포인트 추론까지 연결한다.
학습 코드·설정·데이터·체크포인트는 이 루트 `training/` 폴더에 둔다.
A/B의 기본 환경에는 PyTorch를 추가하지 않는 **별도 uv 프로젝트**이다.
아래 명령은 모두 저장소 루트에서 실행한다.

현재는 **시뮬레이터 GT로 물체 자세와 완료 여부를 얻는 기준 구현**이다.
RGB-D 인식·추적·숨은 목표 탐색은 구현하지 않는다. ID를 모델 특징으로 사용하지 않는다.
전문가는 실제 관절 명령과 접촉 물리로 작동한다. 현재 작업 폴더에는 자동 시연으로 학습하고
`accessible` 장면에서 검증한 초기 ACT 체크포인트 `training/runs/act_grasp/best.pt`도 있다.
데이터·가중치는 Git 제외 대상이므로 팀원에게는 별도로 복사하거나 자동 수집·학습 명령으로 생성한다.

## 학습된 ACT 바로 실행

현재 PC의 CUDA 환경을 유지하면서 CPU 추론으로 확인한 실행 예시:

```powershell
uv run --project training --extra cuda python scripts/run_pipeline.py --backend mujoco --case accessible --controller act --checkpoint training/runs/act_grasp/best.pt --device cpu --seed 300 --gui
```

이 명령은 실제 ACT만 사용한다. 실패를 전문가 행동으로 대체하지 않는다.
GPU 추론은 `--device cuda`, GPU 없는 PC의 설치 환경은 `--extra cpu`를 사용한다.
현재 ACT 검증 범위는 고정 카메라의 `accessible` 장면과 3cm 상자 B/C이다.
선반 3개 장면은 아래 전문가로 검증했고, 해당 장면의 ACT는 같은 자동 시연 경로로 별도 학습해야 한다.

## 학습 없이 A/B/C 통합 실행

```powershell
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --seed 42 --gui
```

기본 A/B 예시와 연결해 방해물을 임시 선반에 옮기고 목표를 회수한다. 공통 함수/타입은 유지한다.
`src/action_generator/expert.py`는 Subgoal별 상태를 관리하며, `kinematics.py`의 private 모델로 IK를 계산한다.
현재 상자 장면을 위한 scripted expert이며 학습된 ACT, 일반 형상용 충돌 회피·파지 보장 알고리즘은 아니다.
기본 예산은 Subgoal당 450 tick이고 관절 한계·접촉 실패·정체 시 중단한다.

## 설치와 코드 연결 검사

```powershell
uv sync --project training --extra cpu --locked
uv run --project training --extra cpu act-smoke
```

`act-smoke`는 실제 렌더링·진단 데이터 저장·2 epoch CPU 학습·체크포인트 로드·MuJoCo 명령 실행을 확인한다.
진단 데이터의 `success=False`, `diagnostic=True`이며, 일반 학습/평가에서는 제외·거부한다.
이 결과는 PICK/PLACE 성공이나 정책 성능이 아니다.

NVIDIA GPU로 학습할 때는 `cpu` 대신 `cuda` extra를 사용한다.

```powershell
uv sync --project training --extra cuda --locked
uv run --project training --extra cuda python -c "import torch; print(torch.cuda.is_available())"
```

CPU/CUDA extra는 함께 선택할 수 없다. PyTorch 2.9.0을 공식 CPU/CUDA 12.8 인덱스에서 설치한다.
`uv run`에도 선택한 extra를 붙여야 uv가 다른 PyTorch 빌드로 다시 동기화하지 않는다.
의존성만 설치하며 pretrained 모델은 다운로드하지 않는다.

## 1. 성공 시연 수집

**직접 조작할 필요 없이** 먼저 자동 생성기를 사용할 수 있다.

```powershell
uv run --project training act-generate --case accessible --seeds 0 1 2 3
uv run --project training act-generate --case accessible --seeds 4 5 --target C
uv run --project training act-generate --case blocked --seeds 0 1
```

`--target`으로 같은 장면에서도 조작 대상을 바꿀 수 있다. 기본 저장 위치는 `training/data/demos`이다.
수집은 PyTorch 없이 실행 가능하다. ACT 실행과 달리 전문가를 명시적으로 사용하며 RGB와 실제 실행 명령을 기록한다.
환경의 물리 완료 판정을 통과한 segment만 성공으로 저장한다. 실패한 전체 작업에서도 이미 성공한 segment는 보존한다.
`generation-*.json`에는 전체 작업의 성공/실패·명령 수·저장된 segment를 기록한다. 기본 예산은 Subgoal당 300 tick이다.
`--config`로 바꾸거나 선반 장면으로 확장하면 자동 생성도 실패할 수 있으며, 실패 기록은 기본 학습에서 제외한다.
같은 장면의 모든 Subgoal·다른 목표 시연은 같은 seed 그룹에 속한다. 실행 예산 변경으로 그룹을 우회할 수 없다.

수동 시연을 추가하고 싶을 때는 기존 키보드 수집기를 사용한다.

```powershell
uv run --project training act-collect --case accessible --seed 0
uv run --project training act-collect --case accessible --seed 1
uv run --project training act-collect --case accessible --seed 2 --target C
```

수집 자체에는 PyTorch가 필요 없다. 창은 일시 정지 상태에서 시작한다.

| 조작 | 동작 |
|---|---|
| Enter | 시작/일시 정지. 정지 중에는 시뮬레이션 시간과 기록이 멈춘다 |
| 1~6, ↑/↓ 누르기 | 관절 선택·연속 이동 |
| Space | 현재 Subgoal 시작 관절 목표로 복귀 |
| R | 현재 시연을 실패로 저장하고 다음 seed로 초기화 |
| Esc/창 닫기 | 수집 종료. 미완료 시연은 실패로 저장 |

창 제목과 터미널에서 현재 PICK/PLACE 대상, 위치, 색을 확인한다.
`--target`은 시연할 물체를 바꾸는 환경의 GT ID이며 신경망 입력에는 포함되지 않는다.
양쪽 손가락 접촉·지지 해소·들어 올리기, 또는 배치·놓기·안정화 조건이 3 tick 연속 확인되면 자동 저장한다.
다음 Subgoal은 기존 A/B 예시로 선택한다. 그리퍼를 닫기만 하거나 시간을 소진하면 성공으로 저장하지 않는다.
`--max-steps`는 Subgoal당 상한이며 기본 9000 tick이다. 처음에는 두 단계가 끝나면 종료한다.

여러 seed·목표·초기 배치·가능한 시작 자세에 대한 시연이 필요하다.
특히 같은 장면에서 목표를 바꾼 시연을 모아야 목표 조건을 무시하는 학습을 줄일 수 있다.
기본 `accessible`의 위치 변화는 ±5mm로 작다. 이 데이터만으로 선반 전체나 적층 환경 일반화를 주장할 수 없다.
`--config`로 환경을 바꿀 수 있으나 카메라·제어 설정이 다른 데이터는 별도 데이터셋으로 관리한다.
`blocked`/`shelves`는 기본 PICK/PLACE가 검증된 후 수집 범위를 넓힌다.

## 2. 데이터와 입력 규약

```text
training/data/demos/<segment_uuid>/
  rgb/000000.png ...      # 명령 직전 RGB, 기본 H=120 W=160
  trajectory.npz         # states[T,6], actions[T,6], goal[25], timestamps[T]
  metadata.json          # 최종 저장 완료 표시, 성공·출처·seed·교정·환경·모델 버전
```

한 segment는 한 Subgoal이다. 중간에 종료된 디렉터리는 완료 메타데이터가 없어 로드하지 않는다.
pickle을 사용하지 않는다. 실패 기록도 보존하지만 기본 학습에서는 제외한다.
`actions[t]`는 `states[t]`와 RGB를 관측한 직후 **실제로 실행한 절대 관절 목표**이다.
관절 5개는 rad, gripper는 0=open/1=closed, 제어는 기본 30Hz이다.

`so101-goal-v2`는 모듈 C 내부 규약이며 공통 `Subgoal` 타입은 변경하지 않는다.

| 인덱스 | 내용 |
|---|---|
| 0:2 | PICK/PLACE one-hot |
| 2:5 | Subgoal 시작 시점 물체 중심 xyz, robot_base/m |
| 5:9 | 시작 시점 물체 quaternion xyzw, 동일 회전의 부호를 정규화 |
| 9:12 | PLACE 물체 중심 목적지 xyz |
| 12:16 | PLACE 목적지 quaternion xyzw |
| 16 | PLACE 목적지 유효 여부 |
| 17:20 | PLACE 시작 시 EE 좌표계에서 본 물체 중심 xyz |
| 20:24 | PLACE 시작 시 EE 좌표계에서 본 물체 quaternion xyzw |
| 24 | 위 파지 상대 자세의 유효 여부 |

PICK에서는 9:25가 0이다. 물체 자세와 파지 상대 자세는 Subgoal 수행 중 고정하고 RGB·관절은 매번 갱신한다.
PLACE의 상대 자세는 시작 물체 자세와 실제 EE 자세에서 구한다. 추론 시 EE 자세가 없으면 공식 로봇 FK로 계산한다.
이를 통해 서로 다른 높이/방향으로 잡힌 물체의 배치 조건을 구분한다. 매 시점의 숨은 물체 GT를 모델에 넣지 않는다.
목표 텐서는 `object_id`, `subgoal_id`, mock의 `target_ee_pose`를 사용하지 않는다.
`target_ee_pose`는 기존 파이프라인의 공간 검사 필드로 남는다.
학습·추론 모두 동일한 `action_generator.goal.encode_goal()`을 사용한다.
모델 안에서는 물체/목적지 xyz를 0.3m, 파지 상대 xyz를 0.03m로 나누고,
관절/행동은 **훈련 데이터에서만** 계산한 평균·표준편차로 정규화한다.
이미지는 HWC uint8에서 CHW float32 [0,1]로 변환한다. Depth는 현재 모델 입력에 포함하지 않는다.

## 3. 학습

```powershell
uv run --project training --extra cpu act-train --data training/data/demos --output training/runs/act_v1
# GPU 사용 시
uv run --project training --extra cuda act-train --data training/data/demos --output training/runs/act_gpu_v1 --device cuda
```

설정은 [configs/act.json](configs/act.json)에 있다. `--config`, `--epochs`, `--device`로 변경할 수 있다.
출력 디렉터리가 이미 있으면 덮어쓰지 않는다. 최소 두 개의 다른 scene seed가 필요하다.
같은 초기 장면의 PICK/PLACE·다른 목표 시연은 같은 그룹으로 묶어 학습/검증이 섞이지 않게 한다.
두 seed는 실행 가능한 최소 조건이고, 정책 품질에 충분한 시연 수를 뜻하지 않는다.

모델은 공간 CNN 특징과 관절·목표·잠재변수 토큰을 Transformer에 넣고 미래 16개 명령을 병렬 예측한다.
현재 관절을 action 정규화 단위로 변환한 값에 Transformer의 보정량을 더해 절대 목표를 출력한다.
이 residual head는 매 tick마다 현재 관절 전체를 다시 추정하면서 생기는 위치 오차를 줄이기 위한 설계이다.
학습 시 실제 미래 행동을 CVAE posterior에 넣어 masked L1 + KL loss로 학습한다.
Subgoal 끝을 넘는 미래 행동은 padding 처리하며 loss와 posterior attention에서 제외한다.
검증은 미래 행동을 입력하지 않고 prior z=0으로 수행한다.

```text
training/runs/act_v1/
  config.json, split.json, metrics.csv
  best.pt                 # 검증 prior L1이 가장 작은 모델
  last.pt
```

체크포인트에는 가중치·정규화·목표 버전·관절 제한·카메라 교정·학습 seed 그룹·출처를 함께 저장한다.
현재 모델 체크포인트 버전은 `so101-act-v3`이며, 추론에서 이전 모델과 호환되지 않는다.
`--init-checkpoint`로 동일 구조의 가중치와 정규화를 이어 학습할 수 있다. optimizer는 새로 시작한다.
초기 모델이 새 검증 장면을 학습한 이력이 있으면 거부한다. v2 모델의 17D projection은 25D로 확장해 이어 학습할 수 있다.
v1 데이터는 실제 MuJoCo 모델 커밋이 일치할 때 다음 명령으로 새 폴더에 복사·변환할 수 있다. 원본은 유지한다.

```powershell
uv run --project training python -m so101_act.upgrade_data --source training/data/old --output training/data/upgraded
```

`--grasp-jitter 0.006` 자동 수집은 파지 높이를 달리해 상대 파지 자세가 다른 PLACE 시연을 만든다.
`balance_goals` 설정은 primitive·물체별 frame 수 차이를 보정해 샘플링하고, 검증 L1도 같은 그룹별 평균으로 계산한다.
이 샘플링용 물체 ID는 신경망 입력에 포함하지 않는다.
PICK만 학습한 체크포인트로 PLACE를 요청하면 명시적으로 거부한다.
loss 감소는 파지 성공을 의미하지 않으며, 성공률은 다음 rollout으로 확인한다.

## 4. 평가와 공통 파이프라인 연결

```powershell
uv run --project training --extra cpu act-evaluate --checkpoint training/runs/act_v1/best.pt --case accessible --seeds 100 101 102 --gui

$env:HSSG_ACT_CHECKPOINT = "training/runs/act_v1/best.pt"
$env:HSSG_ACT_DEVICE = "cpu"
uv run --project training --extra cpu python scripts/run_pipeline.py --backend mujoco --case accessible --seed 100
```

GPU 체크포인트도 CPU로 로드할 수 있다. GPU 실행 시 extra와 device를 `cuda`로 바꾼다.
GPU용 예시는 `--config training/configs/act_gpu.json`, 목표별 균형 학습은 `training/configs/act_balanced.json`이다.
batch 64, uint8 이미지 캐시를 사용한다.
이미지 캐시는 PNG 재해독을 줄이는 대신 프레임 수×H×W×3 byte 정도의 RAM을 사용한다.
`HSSG_ACT_CHECKPOINT`가 없으면 SO-101 전문가가 실행된다. mock 경로도 그대로 유지한다.
`HSSG_C_BACKEND=act`는 ACT를 강제하며 체크포인트가 없으면 오류를 낸다. `expert`는 체크포인트와 무관하게 전문가를 사용한다.
`hold`는 진단용이고 `auto`가 기본이다. 명시적인 ACT 실패를 전문가의 행동으로 감추지 않는다.
직접 주입하려면 `ACTPolicy(path).predict_action_chunk`를 `run_pipeline(..., predict_action=...)`에 전달한다.

모델은 K개를 예측하고 처음 4개만 사용한 후 재예측한다.
어댑터는 내부 행동 버퍼에서 **한 tick씩** 꺼내 실제 관절 상태 기준으로 범위·변화량을 제한하여
`ActionChunk[1,6]`을 반환한다. 이 방식으로 각 명령을 실제 관측에 맞춰 검사한다.
episode/Subgoal/목표 조건/시간 연속성이 바뀌면 버퍼를 비운다. temporal aggregation은 구현하지 않는다.
카메라·제어 주기가 훈련 조건과 다르면 거부한다. 완료 판단은 환경에 맡긴다.

평가는 기본적으로 훈련/검증에 쓴 장면 seed를 거부하고, 제한된 rollout의 성공률·tick 수·시간·실패 사유를 JSON으로 저장한다.
현재 A/B도 GT를 사용하는 예시이므로 이 성공률은 전체 RGB-D 탐색 시스템의 성능이 아니다.
CLI의 `--max-steps` 기본값은 Subgoal당 600 tick이다. 공통 파이프라인은 자체 환경 설정 예산을 사용한다.

전문가의 물리 회수 성공도 같은 평가기로 검사할 수 있다. 이 경우 체크포인트/PyTorch는 필요 없다.

```powershell
uv run --project training act-evaluate --controller expert --case shelves --seeds 0 7 42 --physics-only
```

실제로 확인한 전문가/ACT의 결과와 현재 한계는 [검증 기록](VALIDATION.md)에 구분해서 기록한다.

## 검증과 출처

```powershell
uv run --project training --extra cpu python -m unittest discover -s training/tests -v
uv run python -m unittest discover -s tests -v
uv run --project training --extra cpu act-smoke
```

ACT의 action chunking·CVAE 설계를 참고한 독립 구현이다. 작은 CNN을 처음부터 학습하며,
원 논문의 pretrained ResNet 구성·LeRobot 구현·체크포인트와 호환되지 않는다.
이 저장소의 MIT 라이선스를 따른다. 외부 구현을 복사하거나 외부 모델 가중치를 포함하지 않았다.

- Zhao et al., RSS 2023: [논문·프로젝트](https://tonyzhaozh.github.io/aloha/), [공식 구현](https://github.com/tonyzhaozh/act)
- [PyTorch Transformer](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Transformer.html)
- [uv의 CPU/CUDA 의존성 분리](https://docs.astral.sh/uv/guides/integration/pytorch/)
