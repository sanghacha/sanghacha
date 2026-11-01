# RGB-D HSSG 생성·갱신 패키지

`rgbd_hssg`는 **인식·추적이 끝난 3D 객체와 로봇/카메라 좌표를 입력받아** NetworkX 그래프와 JSON을 만듭니다. 기존 `hssg_generator`의 GT 기반 A 모듈과 별도 패키지입니다.

입력 객체의 ID와 자세는 호출자가 제공합니다. 이 패키지는 검출·분할, 점군 재구성, 객체 ID 매칭, 이동 명령 기반 재식별, 로봇 제어를 실행하지 않습니다. 물체 GT를 조회하거나 명령 목적지를 객체 위치로 대입하지 않습니다.

## 현재 생성하는 정보

- Workspace, 선반/트레이 영역, 객체 후보, 로봇 베이스/링크/TCP, 고정 카메라 노드.
- `inside`: 영역/로봇→Workspace, 현재 관측한 객체→영역.
- `relative_pose_to`: 객체→로봇 베이스/TCP. `T_reference_object = inverse(T_base_reference) @ T_base_object`이며 translation은 해당 기준 프레임 축에서의 미터 값입니다.
- `kinematic_child_of`: 로봇 링크/TCP→부모 프레임.
- `fixed_in`: 고정 카메라→Workspace.

`inside`는 관측 표면 중심이 설정된 XY 영역 안에 있는지로 계산합니다. 영역이 겹치면 입력 순서상 첫 영역을 선택합니다. 물리적인 접촉·지지 증명이 아닙니다. 그리퍼 거리는 추정 OBB 중심까지의 거리이며 표면 여유 거리나 파지 가능성을 뜻하지 않습니다.

관측되지 않은 `history` 객체는 마지막 기하 정보를 유지하지만 현재 `inside` 관계를 만들지 않습니다. 로봇 상대 자세는 현재 로봇 프레임으로 재계산하며 `object_pose_stale=True`, `object_last_seen`을 함께 기록합니다. 로봇 입력이 없으면 로봇 자세를 만들어 넣지 않습니다. 보정된 카메라는 로봇 상태가 없어도 별도 노드로 보존합니다.

Stack 그룹화, supports/contact/occludes/blocks_access, 기능 관계는 포함하지 않습니다. 객체 OBB의 회전·중심은 부분 표면 추정치일 수 있으므로 `pose_kind`와 관측/추적 근거를 보존합니다.

## 설치와 실행

전체 프로젝트 환경에서는:

```sh
uv sync --locked --extra sg
uv run --extra sg python -m rgbd_hssg examples/rgbd_hssg/before.json --output outputs/rgbd_hssg/before.json
uv run --extra sg python -m rgbd_hssg examples/rgbd_hssg/after.json --previous outputs/rgbd_hssg/before.json --output outputs/rgbd_hssg/after.json --delta-output outputs/rgbd_hssg/delta.json
```

기존 프로젝트의 기본 MuJoCo 의존성은 유지합니다. **패키지 자체의 실행 의존성은 NumPy와 NetworkX뿐**입니다. 시뮬레이터 없이 별도 환경에서 소스만 실행하려면 Python 3.11 이상에서 다음과 같이 사용할 수 있습니다.

```powershell
python -m pip install numpy==2.3.4 "networkx>=3.4,<4"
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m rgbd_hssg examples/rgbd_hssg/before.json --output outputs/rgbd_hssg/before.json
python -m rgbd_hssg examples/rgbd_hssg/after.json --previous outputs/rgbd_hssg/before.json --output outputs/rgbd_hssg/after.json --delta-output outputs/rgbd_hssg/delta.json
```

Linux/macOS에서는 `export PYTHONPATH="$PWD/src"`로 지정합니다.

Python API:

```python
from rgbd_hssg import build_scene_graph, graph_delta, load_snapshot, save_graph

objects, metadata = load_snapshot("examples/rgbd_hssg/before.json")
before = build_scene_graph(objects, metadata)
objects, metadata = load_snapshot("examples/rgbd_hssg/after.json")
after = build_scene_graph(objects, metadata)
changes = graph_delta(before, after)
save_graph(after, "outputs/rgbd_hssg/scene_graph.json")
```

`build_scene_graph()`는 매번 새 그래프를 반환하고 입력을 변경하지 않습니다. `_points`, `_colors`, `_mask` 같은 private 배열은 그래프에서 제외하며 PLY나 이미지를 쓰지 않습니다. 그래프 속성에 필요한 NumPy 값은 JSON 값으로 변환합니다.

## 입력 계약

JSON 최상위는 `objects` 배열과 `metadata` 객체입니다. 실행 가능한 예제 전체는 `examples/rgbd_hssg/before.json`, `after.json`에 있습니다.

각 객체의 필수 필드:

- `instance_id`: 호출자가 유지한 고유 ID.
- `visible_surface_centroid_m`: 영역 소속 판정에 사용하는 관측 표면 중심.
- `obb`: `center_m`, 양수인 `extent_m`, proper 3×3 `rotation`.
- `pose_kind`: 예를 들어 `partial_surface_OBB_estimate_not_registered_6D_pose`.
- `status`: `seen` 또는 `history`.
- `last_seen`: 마지막 관측 시각. `seen`이면 현재 스냅샷 시각과 일치해야 합니다.

`first_seen`, 관측 카메라/마스크 번호, 점 개수, 이름 후보, `tracking_evidence` 등 부가 JSON 속성도 보존합니다. 출력 전용 `id`, `kind`는 입력 객체에 넣지 않습니다.

`metadata` 필드:

- `episode_id`, 비음수 `timestamp`: 같은 episode 안에서 그래프를 비교합니다.
- `frame_id="robot_base"`, `depth_units="m"`: 이미 변환된 베이스 기준 미터 좌표.
- `regions`: 각 영역의 `id`, XY `center`, XY `size`.
- 선택적 `cameras`: 카메라 이름별 `K`, OpenCV 축의 `T_camera_to_base`.
- 선택적 `robot`: 같은 시각의 `T_world_base`, 관절 상태, `links`의 부모와 `T_base_link`, TCP의 부모와 `T_base_tcp`. 예제의 구조를 따릅니다.

좌표 변환, ID 중복, 관절 부모의 누락/순환, 관측 시각, 단위, NaN/Infinity를 검사합니다. 입력의 인식 정확도나 출처 자체를 검증하는 센서는 이 패키지에 없습니다.

## 갱신과 예제의 의미

새 관측을 기존 객체 ID에 대응시키는 일은 상위 인식/추적 모듈의 책임입니다. `graph_delta()`는 전달된 ID를 기준으로 이전/현재 그래프를 비교합니다. episode나 기준 프레임이 다르거나 시각이 역전되면 거부합니다.

출력은 추가/제거 노드, 추가/제거 관계, 속성이 바뀐 관계, 객체 소속/상태/중심 변화입니다. 객체 중심 변화 요약의 기본 임계값은 5 mm이며 조절할 수 있습니다. 작은 변화나 로봇 이동으로 달라진 상대 변환도 `updated_relations` 속성에 기록합니다.

제공 예제는 실제 RGB-D 조작 실험에서 `instance_0020` 하나만 발췌한 것입니다. 상위 추적기는 **이동 명령 prior와 새 RGB-D 관측의 색·크기·위치**로 동일성을 가정했습니다. 그 근거는 after 객체의 `tracking_evidence`에 남아 있습니다. 순수 시각 기반 장거리 재식별 예제가 아닙니다. 실제 전체 장면의 다른 후보에서 발생한 ID 불안정성을 이 발췌 예제가 해결했다는 뜻도 아닙니다. 객체 GT 평가 파일·모델 체크포인트·서버 설정은 예제에 없습니다.

예상 변화는 `inside(instance_0020, shelf_2)` 제거와 `inside(instance_0020, buffer_tray)` 추가, 추정 중심 변위 약 0.19476 m입니다. 패키지는 이 관측 결과를 그래프로 표현하며 이동 명령을 해석하거나 자체적으로 성공을 판정하지 않습니다.

기존 A/B/C의 `pipeline.types.SceneGraph`와 새 패키지의 NetworkX 그래프는 서로 다른 인터페이스입니다. 기존 플래너에 연결하려면 별도 어댑터가 필요하며 이 변경에서 기존 연결을 바꾸지 않습니다.

검증:

```sh
uv run --extra sg python -m unittest discover -s tests -p test_rgbd_hssg.py -v
```
