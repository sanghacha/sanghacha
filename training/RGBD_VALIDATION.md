# RGB-D 조건부 ACT 실행·학습·평가

ACT는 RGB-D에서 생성된 Subgoal과 정책 카메라 RGB, 로봇 관절 상태를 입력받는다.
정책 내부에서 16-step action chunk를 예측하고, 측정 관절 기준 명령 제한을 적용해 실행한다.
전문가로 실패를 대체하는 평가 경로는 없다. 물리적 성공은 실행이 끝난 뒤 별도로 검사한다.

## 통합 환경

기존 Python 3.13 학습 환경은 유지한다. Open3D와 ACT를 함께 사용할 별도 Python 3.11 환경:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'training/.venv-rgbd'
uv sync --project training --python 3.11 --extra cuda --extra perception --locked
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

이후 저장소 루트에서 `training/.venv-rgbd/Scripts/python.exe`를 사용한다.
CPU 환경은 `cuda` 대신 `cpu` extra를 선택한다. CUDA 검증 장치는 RTX 4060, PyTorch 2.9.0+cu128이다.

## RGB-D 시연 수집

```powershell
training/.venv-rgbd/Scripts/python.exe -m so101_act.generate --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --case shelves --seeds 0 1 2 3 --teacher privileged --output training/data/rgbd_shelves --max-steps 450
```

`--teacher observed`는 현재 RGB-D 기하·접촉 기반 전문가다. `--teacher privileged`는 **수집 전용**으로
시뮬레이터의 실제 자세와 접촉을 쓰는 교사다. 후자는 수집 안정성을 위한 정답 행동 생성에만 사용한다.
두 경우 모두 학습 파일의 RGB·관절·25D goal은 RGB-D 관측과 원래 Subgoal에서 얻는다.
교사 물리 ID와 실제 자세는 모델 입력 특징에 포함하지 않는다. 출처는 metadata.provenance.teacher에 기록한다.
실제 물리 완료 검사를 통과한 구간만 학습에 포함하고, 실패한 작업의 성공 구간도 독립적으로 보존한다.
수집 보고서는 매 episode 후 갱신한다. 중단된 `.pending`/미완료 구간은 학습하지 않는다.

`--learner-checkpoint PATH --learner-steps 30`은 수집 중 Subgoal 시작을 ACT로 실행한 뒤 교사로 회복한다.
교사 구간의 실제 실행 명령만 저장한다. 생략한 ACT 명령을 교사 정답으로 기록하지 않는다.
Subgoal 초기 조건과 PLACE의 초기 파지 변환은 prefix 이전 값으로 고정하고 물리 장면별 split도 유지한다.
이 교사 개입은 최종 ACT 평가나 공통 실행에는 사용하지 않는다.

## 학습

```powershell
training/.venv-rgbd/Scripts/python.exe -m so101_act.train --data training/data/rgbd_shelves --output training/runs/my_rgbd_act --config training/configs/act_rgbd.json --init-checkpoint training/runs/act_grasp/best.pt --transfer-camera
```

`--transfer-camera`는 카메라가 달라지는 전이 학습을 명시적으로 허용한다. 로봇·명령 주기·관절 범위·모델 구조 검사는 유지한다.
새 train split에서 정규화 통계를 계산하고 모델의 residual 기준도 함께 갱신한다.
이 옵션을 생략하면 기존 카메라 불일치 거부 동작을 유지한다. 기존 체크포인트가 학습한 장면이 새 validation에 들어가면 거부한다.
`--data ROOT1 ROOT2`로 같은 카메라의 여러 시연 폴더를 합칠 수 있고, 중복 경로·다른 카메라는 거부한다.
같은 물리 장면 seed의 교사 종류·재수집·Subgoal은 같은 split에 속한다.

## ACT만 사용하는 평가

검증한 선반용 실행 예시(학습에 쓰지 않은 개발용 seed 101, ACT 단독):

```powershell
training/.venv-rgbd/Scripts/python.exe scripts/run_pipeline.py --backend mujoco --controller act --checkpoint training/runs/act_rgbd_shelves_absolute/best.pt --device cuda --perception rgbd-color --target-spec configs/targets/shelf_2_yellow_green_box.json --seed 101 --gui
```

seed 42도 완료했으나 학습 장면이므로 시연 확인에 해당한다. seed 101은 두 방해물과 목표를 옮기는 6개 Subgoal을 완료했다.
이것은 임의 배치의 성공 보장이 아니다. 아래의 전체 실패·성공 결과를 함께 본다.

```powershell
training/.venv-rgbd/Scripts/python.exe -m so101_act.evaluate --controller act --checkpoint training/runs/act_grasp/best.pt --perception rgbd-color --target-spec configs/targets/blue_box.json --case accessible --seeds 300 301 302 --device cuda --output training/runs/my_rgbd_evaluation.json
```

평가 파일에는 체크포인트 hash, 장면·seed·인식 방식·관절 명령 수·Subgoal·관측 완료 여부·독립 물리 성공을 기록한다.
실패하면 마지막 Subgoal과 EE/관절/접촉 상태도 기록한다. 기본적으로 체크포인트의 train/validation seed를 거부한다.
`--allow-training-seeds`는 학습 장면 시연 확인용이며, 결과에 이 사실을 명시한다.

`--execute-steps 1`은 매 tick 재예측한다. `--temporal-ensemble`은 매 tick 예측한 겹치는 chunk에서
같은 시각의 명령을 가중 평균한다. 모델 행동만 사용하며 전문가 제어·보정 동작은 추가하지 않는다.
공통 CLI에서는 `--act-execute-steps`, `--act-temporal-ensemble`로 같은 설정을 선택한다.

`ACTConfig.residual_actions` 기본값은 기존 체크포인트의 residual 출력이다.
`act_rgbd_absolute.json`은 정규화된 절대 관절 목표를 직접 예측하는 비교 설정이다.
두 모드는 체크포인트에 기록하며, 출력 방식이 달라지는 전이 학습은 `allow_action_mode_transfer`를 명시해야 한다.

`act_rgbd_velocity.json`은 현재·직전 측정 관절의 차이를 추가 입력으로 쓴다.
차이는 `max_delta` 단위로 정규화하고, 첫 프레임·새 Subgoal·episode·불연속 시각에는 0으로 초기화한다.
학습과 추론 모두 과거 관측만 사용한다. 공통 `robot_state`와 출력 action은 계속 6채널이며
체크포인트의 `include_velocity`가 C 내부의 12차원 조건을 선택한다. 기존 6차원 체크포인트는 그대로 동작한다.

## 성능 해석

현재 인식은 색상 블록용 로컬 RGB 분할과 설정된 직립 상자 크기 prior를 사용한다.
일반 형상 인식, SAM/DINO 서버 추론의 성능, 실제 로봇 성공률로 확대 해석하지 않는다.
낮은 validation L1은 회수 성공을 보장하지 않으므로 반드시 폐루프 물리 실행 결과와 함께 본다.
데이터·체크포인트·대용량 평가 산출물은 Git 제외 대상이다.

기존 `act_grasp/best.pt`의 RGB-D accessible 기준 평가(seed 300/301/302)는 3/3 성공했다.
추가 RGB-D small 학습 모델의 별도 seed 2000/2001/2002 평가는 1/3 성공했으며,
blocked 추가 학습 모델의 초기 seed 42/2000 평가는 0/2였다.
선반 초기 pilot의 seed 42 평가는 chunk 실행과 temporal ensemble 모두 실패했다.
이 실패 결과는 각각 `runs/rgbd_baseline_small.json`, `runs/rgbd_small_finetune_eval.json`,
`runs/rgbd_small_full_blocked_eval.json`, `runs/rgbd_shelves_pilot_eval.json`,
`runs/rgbd_shelves_pilot_ensemble.json`에 보존되어 있다.

공통 CLI의 `--controller act --perception rgbd-color`로 accessible seed 300의 GUI 실행도 성공했다.
최종 화면은 `outputs/rgbd_act/final_accessible300/final.png`에 있다.

## 선반용 추가 학습 결과

원래 선반 시연은 90개 성공 구간, 8,695프레임, 22개 물리 장면 seed 그룹이다.
관측 기반 교사 14개와 privileged 교사 76개 구간이며 모든 모델 입력의 source는 rgbd다.
학습 6,863프레임, 검증 1,832프레임으로 물리 장면별 분리했다.
회복 시연 46개, 3,365프레임을 더한 학습에서는 총 136개 구간·12,060프레임을 사용한다.
회복 시연은 기존 train 그룹 안에 있고 validation은 기존 1,832프레임을 유지한다.

| 체크포인트 | 학습 | seed 42 시연 | 미학습 개발 seed 100/101/102 |
|---|---|---|---|
| act_rgbd_shelves | residual 200 epoch | 첫 방해물 PLACE 실패 | 미실행 |
| act_rgbd_shelves_recovery | 회복 시연 추가, residual 60 epoch | 4 Subgoal 성공 | 0/3 |
| act_rgbd_shelves_absolute | 절대 관절 목표 120 epoch | 4 Subgoal 성공 | 1/3 (101 성공) |

기본 선반 시연 명령은 `act_rgbd_shelves_absolute/best.pt`를 사용한다.
이 파일은 validation 기준으로 선택한 epoch 100이며 SHA256은
`51e247a60ea1d275c86a392a0084c265463b2321e3794562c748693b2225efa3`이다.
회복 모델 best는 epoch 55,
`87485e2984437ab2ec47adf0864f9e2da7372b4c3a57ddd16b525e84388b4819`이다.
각 디렉터리의 integration_manifest.json에 설정·출처·split 정보가 있다.

공통 CLI에서도 absolute 모델 + seed 101의 6 Subgoal, recovery 모델 + seed 42의 4 Subgoal을 GUI로 완료했다.
화면은 `outputs/rgbd_act/final_shelves101/final.png`, `outputs/rgbd_act/final_shelves42/final.png`에 있다.
모델은 개발용 seed로 비교했으므로 별도의 최종 평가 seed 2000/2001/2002 결과는
`runs/rgbd_shelves_absolute_final_test.json`으로 분리한다.
이 최종 평가에서는 0/3 성공했다. 2000은 첫 방해물 배치, 2001은 두 번째 방해물 배치,
2002는 목표 물체 배치에서 예산을 소진했다. 따라서 위 GUI 성공을 임의 배치의 안정적인 성공률로 표현하지 않는다.
현재 한계는 접근 중 오차 누적, 배치 오차, 가림으로 인한 추적 ID 누락이다.
회수 성공은 장면의 다른 물체가 전혀 움직이지 않았다는 의미가 아니며, 최소 이동·최적 계획의 증명도 아니다.

현재 TCP와 목표 사이의 오차를 추가 조건으로 사용하는 `include_goal_error` 비교 모델도 학습했다.
PICK은 관측 목표 중심−TCP, PLACE는 목표 중심−(TCP + 초기 파지 offset의 현재 회전 변환)을 0.1m 단위로 제공한다.
학습은 기록된 관절의 공식 FK로, 추론은 관측 EE/FK로 같은 특징을 계산한다. 이 값으로 IK 명령을 생성하지 않으며,
관절 명령은 ACT가 예측한다. `allow_goal_error_transfer`로 입력 확장을 명시한다.
`act_rgbd_shelves_feedback`은 70 epoch 비교 모델이고, 검사 결과는 `runs/rgbd_shelves_feedback_check.json`에 둔다.
이 비교 모델은 seed 42/101에서 0/2였으므로 성공 시연용 기본 체크포인트로 선택하지 않았다.
이 모델의 후속 검사로 기존 final test 결과를 덮어쓰지 않는다.

소프트웨어 검증은 공통 테스트 84개 통과(GUI 자동 테스트 1개 생략)와 ACT 테스트 21개 통과다.
실제 GUI의 ACT 성공 시연은 별도로 위에 기록했다. 기존 Windows 샌드박스의 임시 폴더 권한 문제는
테스트 하네스에서만 임시 경로를 작업 폴더 아래로 지정해 처리했다.
