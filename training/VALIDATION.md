# 모듈 C 검증 기록 — 2026-11-01

Windows / Python 3.13 / MuJoCo 3.17 / PyTorch 2.9.0+cu128 / RTX 4060에서 확인했다.
범위는 공식 SO-101, 현재 상자/선반 설정, MuJoCo GT 기반 A/B 예시이다.

| 검사 | 결과 |
|---|---|
| 전문가 + A/B, 선반 seed 0~9·42 | 11/11 회수 성공. 각 seed마다 초기화한 실제 물리 rollout |
| 소형 blocked 장면 | PICK(A) → PLACE(A) → PICK(B) → PLACE(B) 성공 |
| GUI 통합 실행 | native 창 `VISIBLE=True`, 자동 PICK/PLACE 및 최종 화면 저장 확인 |
| 자동 시연 생성, accessible seed 0·1 | 2/2 작업 성공, 성공한 PICK/PLACE segment 4개·총 388 frame 저장 |
| 최초 ACT prototype | 100 epoch 학습, accessible 미학습 seed 2에서 300 tick 안에 회수 실패 |
| residual ACT v2 | 100 epoch 학습, 동일 미학습 seed 2에서 300 tick 안에 회수 실패 |
| 최종 ACT v3, CPU | accessible의 B/C 각각 seed 300·301·302: 6/6 회수 성공 |
| 최종 ACT v3, GPU | accessible의 B/C 각각 seed 303: 2/2 회수 성공 |
| 최종 ACT v3, 추가 배치 | accessible B seed 304: 450 tick 시간 초과로 실패 |
| 공통 함수의 ACT GUI 실행 | accessible B seed 300, `predict_action_chunk()`를 통해 집기·배치 완료 |
| C 입력에서 GT 제거 | GT·contacts·held ID를 제거하고 source=rgbd, 물체 ID=track_007로 바꾼 C 입력으로도 회수 성공 |

전문가 성공을 ACT 성공으로 집계하지 않는다. 학습 모델은 코드 연결 검사에 그치지 않고 실제 접촉·해제·지지·목표 오차로 평가했다.
최종 모델의 서로 다른 목표/seed 조합 9개 중 8개가 성공했다. 실패한 seed 304도 포함한 결과이며 GUI 재실행은 중복 집계하지 않는다.
C에 GT를 제거한 검사는 C의 입력 독립성 검사이다. A/B와 완료 판정은 여전히 GT 기반이다.
현재 통합 기본값은 전문가이고, 체크포인트를 명시하면 ACT를 실행한다. ACT 실패를 전문가 행동으로 대체하지 않는다.

최종 학습 데이터는 60회 자동 시연 시도 중 성공한 58회, 116 segment, 10,222 frame이다.
원래 시연과 ±6mm 파지 높이 시연을 함께 사용하고, 20개 장면 seed 그룹을 학습 16개/검증 4개로 분리했다.
PLACE 입력을 25D로 보강해 시작 시 물체↔EE 상대 자세를 넣었다. v2 가중치에서 projection을 확장해 v3를 30 epoch 추가 학습했다.
이전 단계의 가중치와 정규화, 분할 이력은 체크포인트의 `initial_checkpoint`와 각 실행 폴더에 보존한다.
두 episode만 사용한 초기 모델의 실패도 위 표에 남겨 두었다. 단순 loss 감소는 작업 성공이 아니다.
학습·데이터·평가 결과는 Git 제외 디렉터리 `training/data`, `training/runs`에 보존한다.

재현 명령은 저장소 루트 기준이다.

```powershell
uv run python scripts/run_pipeline.py --backend mujoco --controller expert --seed 42 --gui
uv run --project training --extra cuda python scripts/run_pipeline.py --backend mujoco --controller act --checkpoint training/runs/act_grasp/best.pt --device cpu --case accessible --seed 300 --gui
uv run python -m unittest discover -s tests -v
uv run --project training --extra cuda python -m unittest discover -s training/tests -v
uv run --project training act-evaluate --controller expert --case shelves --seeds 0 1 2 3 4 5 6 7 8 9 42 --physics-only
uv run --project training act-generate --case accessible --seeds 0 1
```

현재 작업 폴더에 있는 개별 결과물:

- `outputs/module_c/autonomous_gui.png`: 자율 실행 최종 GUI 이미지
- `outputs/module_c/act_gui.png`: ACT 자율 실행 최종 GUI 이미지
- `outputs/module_c/perception_contract.json`: C 입력의 GT/ID 독립성 검사
- `training/runs/expert_shelves42.json`: seed 42 전문가 평가
- `training/runs/expert_residual/best.pt`: 초기 ACT v2 모델, 회수 성공 미달
- `training/runs/expert_residual/rollout_seed2.json`: ACT v2 실패 기록
- `training/runs/act_grasp/best.pt`: 최종 ACT v3 체크포인트
- `training/runs/act_grasp/final_b_cpu.json`, `final_c_cpu.json`: 최종 CPU 평가
- `training/runs/act_grasp/final_b_cuda.json`, `final_c_cuda.json`: 최종 GPU 평가
- `training/runs/act_grasp/extra_b304_cuda.json`: 추가 배치 실패 기록

최종 체크포인트 SHA-256: `C0004C4F9D21F0F4672D147A77223DE24C73A28490BA80B9A8772C2F0E9CAEB9`.

이 결과는 임의 형상·모든 seed·RGB-D 인식 오차·현실 로봇에서의 성공 보장이 아니다.
관절 한계·파지 실패·배치 정체는 오류와 함께 중단하며 물체 좌표를 직접 변경하지 않는다.
