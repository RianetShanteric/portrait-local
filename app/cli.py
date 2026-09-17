"""Conservative, all-local portrait batch processor.

The two AI passes are deliberately blended back into the photograph.  This is
not a generative retouching tool: it should clean noise and recover detail
without turning a person into an approximation of themselves.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from core.recipe import model_identity_from_manifest, sha256_file, write_recipe
from pipeline.smart import SmartPhotoPipeline
from app.workflow import (
    CANCELLED_RETURN_CODE,
    CancellationRequested,
    cancellation_requested,
    raise_if_cancelled,
)


class PipelineError(RuntimeError):
    """A problem that the user can fix without inspecting a traceback."""


def load_rgb(path: Path) -> np.ndarray:
    """Read with Pillow so EXIF orientation is honoured and Unicode paths work."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def save_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(path, "PNG", optimize=True)


def save_jpeg(rgb: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(
        path,
        "JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
    )


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    x = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def moderate_auto_tone(rgb: np.ndarray, settings: dict) -> tuple[np.ndarray, float]:
    """Lift a dark JPEG modestly while protecting highlights and colour."""
    if not settings["enabled"]:
        return rgb.copy(), 1.0

    original = rgb.astype(np.float32) / 255.0
    luma = (
        original[:, :, 0] * 0.2126
        + original[:, :, 1] * 0.7152
        + original[:, :, 2] * 0.0722
    )
    median = float(np.quantile(luma, 0.50))
    median = min(max(median, 0.01), 0.99)
    desired_gamma = math.log(settings["target_median_luma"]) / math.log(median)
    gamma = float(np.clip(desired_gamma, settings["gamma_min"], settings["gamma_max"]))

    corrected = np.power(original, gamma)
    highlight_mask = smoothstep(
        settings["highlight_protection_start"], 1.0, luma
    )[:, :, None]
    protection = settings["highlight_protection_strength"] * highlight_mask
    corrected = corrected * (1.0 - protection) + original * protection
    corrected_u8 = np.clip(corrected * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Low-strength CLAHE is applied only to luminance.  It adds a little local
    # separation without the saturated "AI preset" look.
    blend = float(settings["clahe_blend"])
    if blend > 0:
        lab = cv2.cvtColor(corrected_u8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=float(settings["clahe_clip_limit"]), tileGridSize=(16, 16)
        )
        enhanced_l = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = np.clip(
            lab[:, :, 0].astype(np.float32) * (1.0 - blend)
            + enhanced_l.astype(np.float32) * blend,
            0,
            255,
        ).astype(np.uint8)
        corrected_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return corrected_u8, gamma


def run_realesrgan(
    python: Path,
    repo: Path,
    source: Path,
    output_dir: Path,
    suffix: str,
    tile: int,
    with_face_restore: bool,
    outscale: int = 1,
    cancel_file: Path | None = None,
) -> Path:
    raise_if_cancelled(cancel_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(repo / "inference_realesrgan.py"),
        "--input",
        str(source),
        "--output",
        str(output_dir),
        "--model_name",
        "RealESRGAN_x4plus",
        "--model_path",
        str(repo.parents[2] / "models" / "RealESRGAN_x4plus.pth"),
        "--outscale",
        str(outscale),
        "--suffix",
        suffix,
        "--tile",
        str(tile),
        "--ext",
        "png",
    ]
    if with_face_restore:
        command.append("--face_enhance")

    print("  AI: " + ("face restoration" if with_face_restore else "denoise/detail"))
    process = subprocess.Popen(command, cwd=repo)
    while True:
        try:
            return_code = process.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if not cancellation_requested(cancel_file):
                continue
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise CancellationRequested("Обработка остановлена пользователем.")

    if cancellation_requested(cancel_file):
        raise CancellationRequested("Обработка остановлена пользователем.")
    if return_code != 0:
        raise PipelineError("Real-ESRGAN/GFPGAN вернул ошибку.")
    expected = output_dir / f"{source.stem}_{suffix}.png"
    if expected.is_file():
        return expected
    candidates = sorted(output_dir.glob("*.png"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise PipelineError("AI-проход не создал файл результата.")
    return candidates[-1]


def detect_faces(rgb: np.ndarray, model: Path, score_threshold: float) -> list[np.ndarray]:
    """Detect faces only for the conservative post-blend mask."""
    if not model.is_file():
        raise PipelineError(f"Нет модели детекции лиц: {model}")
    height, width = rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (width, height), score_threshold, 0.3, 5000
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    return [row for row in faces if float(row[-1]) >= score_threshold]


def face_mask(shape: tuple[int, int], faces: list[np.ndarray]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    largest_radius = 0
    for face in faces:
        x, y, face_width, face_height = [float(value) for value in face[:4]]
        center = (int(round(x + face_width * 0.50)), int(round(y + face_height * 0.56)))
        axes = (max(1, int(round(face_width * 0.72))), max(1, int(round(face_height * 0.92))))
        largest_radius = max(largest_radius, axes[0], axes[1])
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    if largest_radius:
        blur = max(3, int(largest_radius * 0.38) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def resize_like(rgb: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] == reference.shape[:2]:
        return rgb
    height, width = reference.shape[:2]
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)


def conservative_merge(
    toned: np.ndarray,
    base_ai: np.ndarray,
    face_ai: np.ndarray | None,
    faces: list[np.ndarray],
    settings: dict,
) -> np.ndarray:
    base_ai = resize_like(base_ai, toned)
    toned_f = toned.astype(np.float32)
    base_f = base_ai.astype(np.float32)
    global_amount = float(settings["global_ai_blend"])
    merged = toned_f * (1.0 - global_amount) + base_f * global_amount

    if face_ai is not None and faces:
        face_ai = resize_like(face_ai, toned).astype(np.float32)
        mask = face_mask(toned.shape[:2], faces)[:, :, None]
        # Only the extra change made by GFPGAN reaches the face, and only at a
        # low opacity.  The base image stays the source of identity and skin.
        merged += (face_ai - base_f) * mask * float(settings["face_blend"])
    return np.clip(merged + 0.5, 0, 255).astype(np.uint8)


def runtime_paths(root: Path) -> tuple[Path, Path, Path]:
    python = root / ".venv" / "Scripts" / "python.exe"
    repo = root / "runtime" / "vendor" / "Real-ESRGAN"
    face_model = root / "models" / "face_detection_yunet_2023mar.onnx"
    missing = [str(path) for path in (python, repo / "inference_realesrgan.py", face_model) if not path.exists()]
    if missing:
        raise PipelineError("Набор ещё не установлен. Запусти INSTALL.cmd. Не найдены: " + ", ".join(missing))
    return python, repo, face_model


def unique_run_directory(output_root: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = output_root / stamp
    counter = 2
    while candidate.exists():
        candidate = output_root / f"{stamp}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def unique_output_path(output_dir: Path, source: Path, suffix: str) -> Path:
    """Keep batch outputs distinct when different extensions share one stem."""
    candidate = output_dir / f"{source.stem}{suffix}.jpg"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{source.stem}_{counter}{suffix}.jpg"
        counter += 1
    return candidate


def cancellation_records(
    inputs: list[Path], current_index: int, note: str
) -> list[dict[str, str]]:
    """Describe the current and remaining inputs after cooperative cancellation."""
    records = [
        {
            "input": inputs[current_index].name,
            "output": "",
            "status": "CANCELLED",
            "note": note,
        }
    ]
    records.extend(
        {
            "input": source.name,
            "output": "",
            "status": "CANCELLED",
            "note": "Not started because processing was cancelled.",
        }
        for source in inputs[current_index + 1 :]
    )
    return records


def append_log(log: list[str], message: str) -> None:
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    log.append(f"[{timestamp}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local one-click portrait pipeline")
    parser.add_argument("--root", type=Path, required=True, help="Pipeline directory")
    parser.add_argument("--profile", choices=("natural", "balanced", "strong"), default="balanced")
    parser.add_argument("--debug", action="store_true", help="Keep masks and analysis under work/debug_*")
    parser.add_argument(
        "--cancel-file",
        type=Path,
        help="Stop cooperatively when this local marker file appears",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    cancel_file = args.cancel_file.resolve() if args.cancel_file else None
    config_path = root / "config.json"
    if not config_path.is_file():
        print("Нет config.json.", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = root / "data" / "input"
    output_root = root / "data" / "output"
    work_root = root / "runtime" / "temp"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    extensions = {item.lower() for item in config["input_extensions"]}
    inputs = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name.lower(),
    )
    if not inputs:
        print("В INPUT нет JPG/PNG/WebP. Положи туда фотографии и запусти файл снова.")
        return 2

    try:
        raise_if_cancelled(cancel_file)
        python, repo, face_model = runtime_paths(root)
        engine = SmartPhotoPipeline(root, config)
    except CancellationRequested as error:
        print(str(error), file=sys.stderr)
        return CANCELLED_RETURN_CODE
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Не удалось запустить анализ кадра: {error}", file=sys.stderr)
        return 2

    output_dir = unique_run_directory(output_root)
    run_dir = work_root / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    stage_smart = run_dir / "01_smart"
    stage_upscale = run_dir / "02_conditional_upscale"
    debug_dir = work_root / f"debug_{dt.datetime.now():%Y%m%d_%H%M%S}" if args.debug else None
    run_dir.mkdir(parents=True)
    log: list[str] = []
    records: list[dict[str, str]] = []
    failures = 0
    cancelled = False
    print(f"Найдено фото: {len(inputs)}. Вывод: {output_dir}")
    append_log(log, f"Run started; {len(inputs)} input file(s).")

    for index, source in enumerate(inputs, start=1):
        try:
            raise_if_cancelled(cancel_file)
            print(f"\n[{index}/{len(inputs)}] {source.name}")
            original = load_rgb(source)
            analysis = engine.analyze(original)
            raise_if_cancelled(cancel_file)
            final, decisions = engine.render(original, analysis, args.profile)
            raise_if_cancelled(cancel_file)
            metrics = analysis.metrics
            print(
                f"  scene={metrics['scene']} confidence={metrics['scene_confidence']:.2f} "
                f"people={metrics['person_count']} faces={metrics['face_count']} "
                f"exposure_gap={metrics['exposure_gap']:.3f} noise={metrics['noise']:.4f}"
            )
            if metrics["needs_upscale"]:
                smart_path = stage_smart / f"{index:04d}.png"
                save_png(final, smart_path)
                upscaled_path = run_realesrgan(
                    python,
                    repo,
                    smart_path,
                    stage_upscale / f"{index:04d}",
                    "upscaled",
                    int(config["upscale"]["tile"]),
                    False,
                    outscale=2,
                    cancel_file=cancel_file,
                )
                final = load_rgb(upscaled_path)
                raise_if_cancelled(cancel_file)
            if debug_dir is not None:
                engine.save_debug(
                    debug_dir,
                    f"{index:04d}_{source.name}",
                    original,
                    analysis,
                    decisions,
                )

            raise_if_cancelled(cancel_file)
            destination = unique_output_path(
                output_dir, source, config["output"]["suffix"]
            )
            save_jpeg(final, destination, int(config["output"]["jpeg_quality"]))
            recipe_path = destination.with_suffix(".recipe.json")
            restore_strength = max(
                decisions["face_restoration_strengths"], default=0.0
            )
            restoration_model = (
                engine.face_restorer.identity.__dict__
                if engine.face_restorer is not None and restore_strength >= 0.015
                else None
            )
            upscale_model = (
                model_identity_from_manifest(root, "RealESRGAN_x4plus.pth")
                if metrics["needs_upscale"]
                else None
            )
            recipe = {
                "schema_version": 1,
                "application_version": config["application_version"],
                "input": {"name": source.name, "sha256": sha256_file(source)},
                "output": {"name": destination.name, "sha256": sha256_file(destination)},
                "profile": args.profile,
                "metrics": metrics,
                "decisions": decisions,
                "models": {
                    "segmentation": engine.segmenter.identity.__dict__,
                    "face_detection": engine.face_detector.identity.__dict__,
                    "face_restoration": restoration_model,
                    "upscale": upscale_model,
                },
            }
            write_recipe(recipe_path, recipe)
            note = (
                f"scene={metrics['scene']}; scene_confidence={metrics['scene_confidence']:.2f}; "
                f"people={metrics['person_count']}; faces={metrics['face_count']}; "
                f"exposure_gap={metrics['exposure_gap']:.3f}; subject_lift={decisions.get('subject_lift', 0.0):.3f}; "
                f"noise={metrics['noise']:.4f}; denoise={decisions['denoise_amount']:.3f}; "
                f"sharpen={decisions['sharpen_amount']:.3f}; face_restore_max={restore_strength:.3f}; "
                f"upscale={metrics['needs_upscale']}"
            )
            records.append({"input": source.name, "output": destination.name, "status": "OK", "note": note})
            append_log(log, f"OK {source.name} -> {destination.name}; {note}")
        except CancellationRequested as error:
            cancelled = True
            cancelled_batch = cancellation_records(inputs, index - 1, str(error))
            records.extend(cancelled_batch)
            append_log(log, f"CANCELLED {cancelled_batch[0]['input']}: {error}")
            for record in cancelled_batch[1:]:
                append_log(log, f"CANCELLED {record['input']}: not started")
            break
        except Exception as error:  # Leave the remaining photographs processing.
            failures += 1
            records.append({"input": source.name, "output": "", "status": "ERROR", "note": str(error)})
            append_log(log, f"ERROR {source.name}: {error}")
            print(f"  ERROR: {error}", file=sys.stderr)
            traceback.print_exc()

    keep_work = bool(config["output"]["keep_work_files"]) or failures > 0 or cancelled
    if failures:
        append_log(log, f"Temporary files retained at {run_dir} because at least one image failed.")
    if cancelled:
        append_log(log, f"Run cancelled; temporary files retained at {run_dir}.")
    with (output_dir / "processing_log.txt").open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(log) + "\n")
    with (output_dir / "processing_log.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["input", "output", "status", "note"])
        writer.writeheader()
        writer.writerows(records)

    if not keep_work:
        shutil.rmtree(run_dir)

    if cancelled:
        completed = sum(record["status"] == "OK" for record in records)
        print(f"\nОстановлено: {completed}/{len(inputs)} фото. Временные файлы сохранены в {run_dir}.")
        return CANCELLED_RETURN_CODE
    if failures:
        print(f"\nГотово с ошибками: {len(inputs) - failures}/{len(inputs)}. Временные файлы сохранены в {run_dir}.")
        return 1
    if debug_dir is not None:
        print(f"\nDebug: {debug_dir}")
    print(f"\nГотово: {len(inputs)} фото. Временные файлы удалены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
