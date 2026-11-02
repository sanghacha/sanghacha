# System architecture figure

현재 구현된 **RGB-D + indexed HSG planner + ACT** 구성을 그렸다. 기본 CLI 옵션인 GT/expert 구성과 구분한다.

- [고해상도 PNG](system_architecture.png): 4800 × 2820, 400 dpi.
- [편집 가능한 SVG](system_architecture.svg): 텍스트와 도형은 벡터, 관측 예시는 내장된 래스터 이미지.
- [논문 삽입용 PDF](../../output/pdf/system_architecture.pdf): 벡터 도형·임베드된 TrueType 폰트, 1페이지.
- 생성 코드: [render_system_architecture.py](../../scripts/render_system_architecture.py).

## Suggested caption

**Figure X. Closed-loop architecture for hierarchical scene-graph-based manipulation.**
(A) Two fixed RGB-D views are segmented, reconstructed and fused into tracked 3D object hypotheses. A hierarchical scene graph represents the workspace, regions and current object tracks.
(B) Observable target attributes are grounded to an episode-local track by the orchestrator. The indexed HSG planner compares successive graph snapshots, updates affected spatial/support indices and regional summaries, and selects a search, unblocking or placement subgoal.
(C) A goal-conditioned ACT policy encodes the front RGB image, measured joint state and numerical subgoal to predict an action chunk. A short command prefix is cached, and one bounded joint command is issued at each control tick.
The orchestrator verifies subgoal completion using observable feedback and triggers subsequent perception and replanning. The illustrated configuration uses a MuJoCo SO-101 workspace; structured target attributes, configured shelf geometry and an upright-box size vocabulary are available priors.

## 국문 캡션

**그림 X. 계층적 장면 그래프를 활용한 폐루프 물체 조작 시스템 구조.**
모듈 A는 두 고정 카메라의 RGB-D 관측으로 물체의 3D 형상을 추정하고, 다중 시점 융합과 시간적 연관을 통해 추적 ID를 부여한다.
오케스트레이터가 목표 속성을 관측된 물체에 대응시킨 뒤, 모듈 B는 장면 그래프의 변경 부분과 영역 요약을 갱신하여 탐색·방해물 제거·배치 subgoal을 선택한다.
모듈 C의 ACT는 전면 RGB, 측정 관절 상태, 수치화된 subgoal을 조건으로 action chunk를 예측한다.
오케스트레이터는 제어 명령을 순차 실행하고 관측 가능한 피드백으로 완료 여부를 판단하여 재관측과 재계획을 반복한다.

## 그림을 해석할 때의 구현 사항

1. **사용자 입력:** 현재는 JSON `TargetSpec`의 기준 RGB·물체 종류·선택적 영역/크기와 목적지·임시 배치 위치다. 자유 형식 자연어를 해석하는 LLM은 구현되어 있지 않다.
2. **A의 실제 인식:** 그림의 `RGB masks`는 현재 로컬 색상 기반 분할이다. depth back-projection, 두 시점 융합, 알려진 직립 상자 크기 vocabulary를 이용한 fitting을 사용한다. SAM/DINO 서버는 기본 실험 경로가 아니다.
3. **HSSG와 B의 관계:** A는 주로 영역 포함 관계와 추적 상태를 제공한다. B의 `Spatial/support indices`가 관측 위치·크기로 적층 관계를 추정한다. 그림은 A가 GT 접촉이나 완전히 가려진 물체 좌표를 복원한다고 표현하지 않는다.
4. **A → B:** 현재 전달 단위는 전체 공통 `SceneGraph` snapshot이다. 변경 감지는 `IndexedPlanner` 내부 `DeltaEncoder`에서 수행한다. A가 delta만 전송하는 최적화가 이미 연결되어 있다는 뜻이 아니다.
5. **목표 대응:** `TargetResolver`는 실제로 `pipeline.runner`에서 B 호출 전에 수행된다. 그림 B 안의 `Target grounding*`은 개념적인 목표 대응·계획 단계로 묶은 것이며, 별표로 실행 주체를 표시했다.
6. **B → C:** `Subgoal`은 primitive, track ID, 물체 pose, mock EE target, 선택적 배치 pose 등을 포함하는 공통 객체다. ACT는 이 객체를 그대로 neural token으로 받지 않는다. `encode_goal()`이 primitive·초기 물체 pose·배치 pose·파지 상대 pose/validity를 25차원 수치 특징으로 바꾼다. **track ID와 mock EE target은 학습 입력에서 제외**한다. 파지 상대 pose는 PLACE 시작 시 관측 FK/EE pose로 계산한다.
7. **ACT 관측:** 실제 ACT 영상은 고정된 `front` 카메라 RGB다. A의 `fixed_top`/`fixed_oblique` RGB-D와 별도 경로이므로 위쪽 관측 bus를 따로 그렸다. 관절 상태는 5개 팔 관절과 정규화된 그리퍼의 6채널이다.
8. **Chunk 실행:** 현재 기본 absolute checkpoint는 `K=16`의 6채널 관절 목표를 예측한다. 기본으로 첫 4개 예측을 내부 cache에 보관하지만, 매 관측 tick에서 실제 관절 상태를 기준으로 제한한 **공통 ActionChunk 1×6**을 반환한다. Chunk 길이와 실행 prefix는 설정 가능하다. 그림의 진한 4개 열은 기본 실행 prefix를 나타낸다.
9. **피드백:** FK·두 손가락의 binary tactile은 held-object 추정과 완료 판단 등에 사용한다. tactile이 별도 ACT neural input이라는 뜻은 아니다. PICK은 접촉과 lift, PLACE는 release 후 RGB-D 재관측을 활용해 확인한다.
10. **두 시간 척도:** 제어 관측/명령은 30Hz다. A는 주로 계획 경계, release 검증, PICK 재시도에서 새 snapshot을 생성한다. B는 현재 Subgoal이 끝날 때 다음 계획을 선택한다. 매 제어 tick에 전체 HSSG를 새로 생성하는 구조가 아니다.
11. **단순화:** 도식의 그래프는 예시다. 실제 region에는 선반·트레이·tabletop 등이 포함되며 robot/camera frame과 history-only track도 원본 그래프에 존재할 수 있다. history-only geometry는 현재 grasp/collision 대상에 넣지 않고, 파지 중인 물체의 위치는 관측 FK에 따른 추정으로 유지한다. 요약의 `reuse/update`도 동작 예시이며 실제 모든 관측에서 S2만 변한다는 의미는 아니다.
12. **평가 대안:** GT graph, expert controller, Flat/HSG ablation은 교체 가능한 평가 설정이다. 그림은 이 대안들을 주 흐름으로 혼합하지 않고, 구현되어 있는 RGB-D/indexed-HSG/ACT 설정 하나를 표시한다. ACT 학습은 offline이며 이 그림은 온라인 실행 구조다.

## 코드 대응

| 그림 요소 | 실제 코드 |
|---|---|
| RGB-D 관측·관측 기반 완료 판단 | `src/pipeline/rgbd_env.py`, `src/perception/acquisition.py` |
| 색상 분할·3D 복원·융합·추적 | `src/perception/local_masks.py`, `reconstruction.py`, `association.py`, `bridge.py` |
| 원본 HSSG | `src/rgbd_hssg/builder.py` |
| Target grounding | `src/pipeline/target.py`, `runner.py` |
| Snapshot diff·영역 요약·관계 인덱스 | `src/planner/indexed.py` |
| 사용 가능한 새 B 모듈 | `src/planner/hsg_incremental.py` |
| 25D subgoal encoding | `src/action_generator/goal.py` |
| ACT 모델·실행 cache/제한 | `training/so101_act/model.py`, `policy.py` |
| 폐루프 실행 | `src/pipeline/runner.py` |

## 재생성 및 PDF 검수

```powershell
.venv-perception/Scripts/python.exe scripts/render_system_architecture.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/render_pdf_preview.ps1 -Pdf output/pdf/system_architecture.pdf -Output tmp/pdfs/system_architecture_review.png -Width 2400
```

관측 예시는 실제 `outputs/hsg_indexed/gui42/snapshot_000/inputs/frame_000`의 RGB-D이며, 재생성을 위해 `assets/architecture_oblique_*`에 복사했다.
깊이 그림에서 2m 밖의 배경은 표시 범위에서만 마스킹했다. 원본 깊이/실제 A 입력은 변경하지 않았다.
Poppler가 없는 Windows 환경에서 내장 WinRT PDF renderer로 최종 PDF의 1페이지를 렌더링하고 글자·선·배치를 시각 검수했다.
