# RGB-D 인식에서 SG 생성까지 실행하기

현재 main에 있는 `src/perception`에 실행 경로를 연결합니다. `고정 RGB-D 2대 → GPU 마스크/검출 → 로컬 점군·OBB·TSDF → 객체 ID 추적 → rgbd_hssg 생성·변경 로그` 순서입니다. 기존 전문가/ACT 제어기와 training 프로젝트는 유지하며, 이 경로가 기존 A/B/C의 GT 입력을 자동 교체하지는 않습니다.

## 로컬: 별도 Python 3.11 환경

Open3D 0.19.0을 검증한 Python 3.11을 사용합니다. 기본 프로젝트의 `.python-version`은 기존 3.13으로 유지합니다. 저장소 루트의 PowerShell에서:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-perception'
uv sync --locked --extra perception --python 3.11
Remove-Item Env:UV_PROJECT_ENVIRONMENT
.venv-perception\Scripts\python.exe scripts/setup_assets.py
```

이후 명령은 인식 환경의 Python을 명시적으로 사용합니다. CPU 로컬 환경에는 Open3D·SciPy·NetworkX가 설치되며 CUDA PyTorch는 필요하지 않습니다. `perception` extra는 Python 3.13용 설치를 검증하지 않았으므로 해당 extra를 요청할 때는 위처럼 3.11을 지정합니다.

## 1. 촬영

```powershell
.venv-perception\Scripts\python.exe scripts/capture_multiview.py --seed 123 --frames 2 --output outputs/multiview/run_123
```

`fixed_top`과 `fixed_oblique`는 로봇과 독립적인 world 고정 카메라입니다. 같은 snapshot에서 두 카메라 사이에는 physics step이 없습니다. 기본 두 snapshot 사이에는 로봇 관절을 조금 움직이며 물체를 집어 옮기지는 않습니다.

각 frame 디렉터리에 RGB PNG, 미터 depth NPZ, camera K / camera-to-base, 동일 timestamp와 hash를 담은 capture.json을 저장합니다. 객체 GT 위치·이름·접촉은 인식 입력으로 내보내지 않습니다. 로봇 관절 FK, 시뮬레이터 깊이/카메라 자세, 설정된 선반·트레이 경계는 사용합니다.

## 2. GPU 서버 추론

CUDA PyTorch가 설치된 Linux 환경에서 별도 worker를 사용합니다. 이전 검증 환경은 PyTorch 2.8.0+cu128, transformers 4.57.1입니다. CUDA PyTorch 설치는 서버 환경에 맞게 별도로 진행합니다.

```sh
python -m pip install -r requirements-perception-gpu.txt
python scripts/infer_multiview_gpu.py --inputs inputs --output predictions
```

서버 입력은 `inputs/frame_000/fixed_top_rgb.png`, `fixed_oblique_rgb.png` 등 RGB뿐입니다. `scripts/package_rgb.py outputs/multiview/run_123`으로 전송용 rgb_inputs.zip을 만들 수 있습니다. depth, calibration, 로봇·객체 상태는 이 ZIP에 들어가지 않습니다.

worker는 Hugging Face에서 가중치를 내려받아 서버 GPU에서 실행합니다. 외부 추론 API로 이미지를 보내는 방식이 아닙니다. SAM 2.1 Hiera Large의 AMG로 후보를 만들고, Grounding DINO base의 `box. cube. block.` 탐지 결과를 이름 후보로 연결합니다. DINO 박스만 SAM에 넣는 box-prompt 방식은 아닙니다.

출력은 카메라별 masks.npz, detections.json과 입력 RGB hash, 모델 revision·GPU 메모리 기록을 담은 run.json입니다. 이를 로컬 `outputs/multiview/run_123/predictions`로 가져옵니다.

### Windows+WSL 자동 전송

`configs/perception_remote.example.json`을 `configs/perception_remote.json`으로 복사해 실제 호스트·포트·WSL 배포판·로컬 SSH 키 경로를 작성합니다. 이 로컬 설정은 gitignore 대상입니다. 서버에 CUDA PyTorch가 설치된 상태에서 venv를 준비합니다.

```sh
python -m venv --system-site-packages /workspace/hssg-perception/venv
/workspace/hssg-perception/venv/bin/pip install -r requirements-perception-gpu.txt
```

그 다음 로컬에서:

```powershell
.\run_perception.ps1 -Seed 123 -CaptureDirectory outputs/multiview/run_123
```

이 스크립트는 `.venv-perception/Scripts/python.exe`를 기본으로 사용합니다. 다른 Python은 `-PythonExecutable PATH`로 지정합니다. 실행마다 별도 서버 run 디렉터리를 만들고 GPU 추론 후 로컬 재구성까지 수행합니다. 개인키는 서버에 업로드하지 않습니다.

## 3. 로컬 3D 처리와 그래프

```powershell
.venv-perception\Scripts\python.exe scripts/reconstruct_multiview.py --capture outputs/multiview/run_123 --predictions outputs/multiview/run_123/predictions --output outputs/multiview/run_123/reconstruction
```

저장한 예측을 재사용하는 자동 경로는 `run_perception.ps1 -UseSavedPredictions -CaptureDirectory outputs/multiview/run_123`입니다. 기존 촬영만 유지하고 GPU 추론부터 실행하려면 `-UseSavedCapture`를 사용합니다.

출력:

- 객체 후보별 PLY, 부분 표면 OBB, snapshot_tsdf.ply.
- scene_graph.json, graph_delta.json, 카메라별 마스크/ID 이미지.
- report.json, summary.json, tracker_state.json.

그래프는 `perception.reconstruction.export_graph()`가 이미 병합된 `rgbd_hssg.build_scene_graph()`를 호출해 생성합니다. CLI의 변경 로그도 `rgbd_hssg.graph_delta()`를 사용합니다. 관계 종류는 기존 네 가지 그대로입니다. 입력 검증과 stale 의미는 [그래프 패키지 문서](rgbd_hssg.md)를 따릅니다. 기존 `perception.graph_update` 및 로봇 변환 유틸리티는 기존 호출자를 위해 유지합니다.

## 추적·명령 prior·한계

일반 추적은 근거리 위치·색도·기하를 이용합니다. metadata.action_events가 있으면 해당 episode/시각에 한해 예정 목적지 근처의 유일하고 호환되는 관측으로 ID 재연결을 시도합니다. 이벤트에는 `event_id`, `episode_id`, `source_instance_id`, `destination_region`, `expected_position_m`, `started_at`, `completed_at`이 필요합니다. 이 PR은 제어기와의 이벤트 통신을 추가하지 않습니다. 명령 의도만으로 위치를 바꾸지 않으며 관측이 없거나 모호하면 history로 남깁니다.

`--resume-state PATH`로 같은 episode의 추적 상태를 이어갈 수 있습니다. 이 CLI 실행 안의 snapshot들을 비교해 graph_delta를 만듭니다. 이전 그래프를 별도로 로드하지 않으므로 재개 실행의 첫 delta는 이전 실행의 그래프와 비교한 결과가 아닙니다.

OBB는 관측된 부분 표면의 추정치입니다. 누락·중복·병합·ID 변경이 남아 있습니다. TSDF는 snapshot마다 새로 만들며 시간에 걸친 영속 객체 TSDF는 아닙니다. ROI/크기 임계값에는 현재 상자 장면의 가정이 있습니다. supports/contact/occludes/Stack, 일반 장거리 재식별, 학습된 파지 정책을 추가하는 변경은 아닙니다.

## 검증

```powershell
.venv-perception\Scripts\python.exe -m unittest discover -s tests -v
```

인식 extra가 없는 기본 환경에서는 새 인식 전용 테스트를 생략합니다. 캡처 및 실제 모델 추론 검증과 합성 기하 테스트를 구분해 PR 본문에 기록합니다. 모델 가중치, 실제 서버 주소, 대용량 결과는 저장소에 포함하지 않습니다.
