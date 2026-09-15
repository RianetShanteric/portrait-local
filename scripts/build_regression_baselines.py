from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cli import load_rgb, run_realesrgan, runtime_paths, save_jpeg, save_png
from core.recipe import model_identity_from_manifest, sha256_file, write_recipe
from pipeline.smart import SmartPhotoPipeline


DATASET = ROOT / "tests" / "regression" / "dataset.json"


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8)).save(path, "PNG", optimize=True)


def comparison(before: np.ndarray, after: np.ndarray) -> Image.Image:
    left = Image.fromarray(before, "RGB")
    right = Image.fromarray(after, "RGB")
    limit = 1400
    scale = min(1.0, limit / max(left.width, right.width))
    size = (
        round(max(left.width, right.width) * scale),
        round(max(left.height, right.height) * scale),
    )
    left = left.resize(size, Image.Resampling.LANCZOS)
    right = right.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0] * 2, size[1]), "black")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def inference_summary(
    engine: SmartPhotoPipeline, restoration_used: bool, upscale_used: bool
) -> str:
    parts = [
        engine.segmenter.identity.provider,
        engine.face_detector.identity.provider,
        "OpenCV CPU",
    ]
    if restoration_used:
        parts.append("gfpgan conditional")
    if upscale_used:
        parts.append("real-esrgan conditional")
    return "; ".join(parts)


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    profile = dataset["profile"]
    engine = SmartPhotoPipeline(ROOT, config)
    python, repo, _ = runtime_paths(ROOT)
    temp_parent = ROOT / "runtime" / "temp"
    temp_parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="baseline_", dir=temp_parent) as temp:
        temp_root = Path(temp)
        for index, case in enumerate(dataset["cases"], start=1):
            source = ROOT / "tests" / "regression" / "images" / case["file"]
            if sha256_file(source) != case["stored_sha256"]:
                raise RuntimeError(f"Regression input hash mismatch: {source.name}")

            target = ROOT / "tests" / "regression" / "baseline" / profile / case["id"]
            target.mkdir(parents=True, exist_ok=True)
            rgb = load_rgb(source)
            analysis = engine.analyze(rgb)
            result, decisions = engine.render(rgb, analysis, profile)
            upscale_used = False
            if analysis.metrics["needs_upscale"]:
                smart_path = temp_root / "01_smart" / f"{index:04d}.png"
                save_png(result, smart_path)
                upscaled_path = run_realesrgan(
                    python,
                    repo,
                    smart_path,
                    temp_root / "02_conditional_upscale" / f"{index:04d}",
                    "upscaled",
                    int(config["upscale"]["tile"]),
                    False,
                    outscale=2,
                )
                result = load_rgb(upscaled_path)
                upscale_used = True
            output = target / "result.jpg"
            save_jpeg(result, output, int(config["output"]["jpeg_quality"]))
            save_mask(analysis.person_mask, target / "person-mask.png")
            save_mask(analysis.skin_mask, target / "skin-mask.png")
            save_mask(analysis.eye_mask, target / "eye-mask.png")
            save_mask(analysis.teeth_mask, target / "teeth-mask.png")
            save_mask(analysis.hair_mask, target / "hair-mask.png")
            comparison(rgb, result).save(target / "before-after.jpg", "JPEG", quality=92, subsampling=0, optimize=True)

            restoration_used = max(
                decisions["face_restoration_strengths"], default=0.0
            ) >= 0.015
            restoration_model = (
                engine.face_restorer.identity.__dict__
                if engine.face_restorer is not None and restoration_used
                else None
            )
            evidence = {
                "metrics": analysis.metrics,
                "timings": analysis.timings,
                "decisions": decisions,
                "inference": inference_summary(
                    engine, restoration_used, upscale_used
                ),
            }
            write_recipe(target / "analysis.json", evidence)
            recipe = {
                "schema_version": 1,
                "application_version": config["application_version"],
                "regression_case": case["id"],
                "input": {"name": source.name, "sha256": sha256_file(source)},
                "output": {"name": output.name, "sha256": sha256_file(output)},
                "profile": profile,
                "metrics": analysis.metrics,
                "decisions": decisions,
                "models": {
                    "segmentation": engine.segmenter.identity.__dict__,
                    "face_detection": engine.face_detector.identity.__dict__,
                    "face_restoration": restoration_model,
                    "upscale": (
                        model_identity_from_manifest(ROOT, "RealESRGAN_x4plus.pth")
                        if upscale_used
                        else None
                    ),
                },
            }
            write_recipe(target / "recipe.json", recipe)
            print(f"{case['id']}: people={analysis.metrics['person_count']} faces={analysis.metrics['face_count']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
