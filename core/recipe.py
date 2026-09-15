from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def model_identity_from_manifest(root: Path, filename: str) -> dict[str, str]:
    """Return the stable recipe identity for an installed manifest artifact."""
    manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["filename"] == filename:
            return {
                "capability": model["capability"],
                "provider": model["provider"],
                "artifact": model["filename"],
                "version": model["version"],
                "sha256": model["sha256"],
            }
    raise KeyError(f"Model is not listed in manifest: {filename}")


def write_recipe(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
