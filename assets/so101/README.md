# SO-101 모델

원본: Google DeepMind MuJoCo Menagerie의 `robotstudio_so101`.
정확한 커밋·파일 목록·크기·Git blob checksum은 `source.json`에 고정한다.

```sh
uv run python scripts/setup_assets.py
```

다운로드한 MJCF/STL/라이선스는 `upstream/`에 두며 Git에 넣지 않는다.
스크립트는 고정 커밋에서 필요한 파일만 받고 checksum을 확인한다.
로컬 파일이 수정되어 있으면 덮어쓰지 않고 중지한다.

원본 라이선스는 Apache-2.0이며 함께 내려받는 `upstream/LICENSE`를 따른다.
원본 모델은 수정하지 않는다. 작업 장면·물체·카메라·제어 주기는
`pipeline/mujoco_env.py`와 `configs/envs/so101.json`에서 추가/설정한다.

이 모델은 시뮬레이션용이며 실제 하드웨어를 연결하지 않는다.
