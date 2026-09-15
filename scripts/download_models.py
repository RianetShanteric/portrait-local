from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main(root: Path) -> int:
    manifest_path = root / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        destination = root / model.get("install_path", f"models/{model['filename']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256(destination) == model["sha256"]:
            print(f"OK {model['filename']}")
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {model['filename']}...")
        urllib.request.urlretrieve(model["url"], partial)
        actual = sha256(partial)
        if actual != model["sha256"]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Hash mismatch for {model['filename']}: {actual}")
        partial.replace(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
