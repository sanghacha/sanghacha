"""uv run python scripts/setup_assets.py — 고정된 공식 SO-101 모델만 다운로드한다."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/so101"


def setup_assets() -> None:
    manifest = json.loads((ASSETS / "source.json").read_text(encoding="utf-8"))
    destination = ASSETS / "upstream"
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid asset path")
        path = destination / relative
        if path.exists():
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob == item["git_blob_sha1"]:
                continue
            raise ValueError(f"Modified asset: {path}. Preserve your changes before restoring the upstream file.")
        url = f"{manifest['raw_base']}/{item['path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "hssg-so101-dev"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != item["size"] or blob != item["git_blob_sha1"]:
            raise ValueError(f"Asset checksum mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(relative.as_posix())
    print(f"SO-101 ready: {manifest['commit']}")


if __name__ == "__main__":
    setup_assets()
