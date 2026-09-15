"""Explainable, local scene classification using existing detections and pixels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.contracts import ObjectDetection


ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
DINING = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "dining table"}
CITY = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light", "fire hydrant", "stop sign", "parking meter"}
INDOOR = {"chair", "couch", "bed", "toilet", "tv", "laptop", "keyboard", "mouse", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}


@dataclass(frozen=True)
class SceneResult:
    scene: str
    confidence: float
    tags: list[str]
    scores: dict[str, float]
    reasons: list[str]
    signals: dict[str, float]


def _object_signal(objects: list[ObjectDetection], labels: set[str]) -> float:
    matches = [item for item in objects if item.label in labels]
    return float(min(1.0, sum(item.confidence * min(1.0, item.area_fraction * 8.0 + 0.12) for item in matches)))


def classify_scene(
    rgb: np.ndarray,
    objects: list[ObjectDetection],
    person_count: int,
    face_count: int,
    person_area: float,
) -> SceneResult:
    """Return a conservative primary scene and auditable supporting signals."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    top = slice(0, max(1, round(height * 0.48)))
    sky = (((hue[top] >= 84) & (hue[top] <= 132) & (saturation[top] > 0.12) & (value[top] > 0.28)))
    vegetation = ((hue >= 28) & (hue <= 92) & (saturation > 0.18) & (value > 0.10))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    short = cv2.resize(gray, (min(width, 960), max(1, round(height * min(1.0, 960 / width)))), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(short, 70, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=max(24, min(short.shape) // 10), maxLineGap=10)
    line_count = 0 if lines is None else len(lines)
    line_signal = float(np.clip(line_count / 45.0, 0.0, 1.0))
    median_value = float(np.median(value))
    dark_fraction = float(np.mean(value < 0.18))
    night_signal = float(np.clip((0.34 - median_value) / 0.22, 0.0, 1.0) * 0.72 + np.clip((dark_fraction - 0.25) / 0.55, 0.0, 1.0) * 0.28)
    signals = {
        "sky_fraction_top": float(np.mean(sky)),
        "vegetation_fraction": float(np.mean(vegetation)),
        "line_signal": line_signal,
        "night_signal": night_signal,
        "animal_object_signal": _object_signal(objects, ANIMALS),
        "food_object_signal": _object_signal(objects, FOOD),
        "dining_object_signal": _object_signal(objects, DINING),
        "city_object_signal": _object_signal(objects, CITY),
        "indoor_object_signal": _object_signal(objects, INDOOR),
    }
    landscape = np.clip(signals["sky_fraction_top"] * 1.15 + signals["vegetation_fraction"] * 0.95 - line_signal * 0.22, 0.0, 1.0)
    scores = {
        "portrait": float(np.clip(0.55 + 0.25 * bool(face_count) + min(person_area, 0.35) * 0.45, 0.0, 1.0)) if person_count else 0.0,
        "group": float(np.clip(0.55 + 0.08 * min(person_count, 5), 0.0, 0.96)) if person_count >= 3 else 0.0,
        "landscape": float(landscape),
        "city": float(np.clip(signals["city_object_signal"] * 0.78 + line_signal * 0.30, 0.0, 1.0)),
        "architecture": float(np.clip(line_signal * 0.78 - signals["sky_fraction_top"] * 0.55 - signals["vegetation_fraction"] * 0.18, 0.0, 1.0)),
        "indoor": float(np.clip(signals["indoor_object_signal"] * 0.82 + (1.0 - signals["sky_fraction_top"]) * 0.16, 0.0, 1.0)),
        "night": night_signal,
        "food": float(np.clip(signals["food_object_signal"] * 0.86 + signals["dining_object_signal"] * 0.28, 0.0, 1.0)),
        "animal": signals["animal_object_signal"],
        "object": float(min(0.72, max((item.confidence * min(1.0, item.area_fraction * 4.0 + 0.15) for item in objects if item.label != "person"), default=0.0))),
        "general": 0.22,
    }
    people_are_subject = bool(person_count and (face_count > 0 or person_area >= 0.025))
    if people_are_subject:
        scene = "group" if person_count >= 3 else "portrait"
    else:
        scene = max(scores, key=scores.get)
        if scores[scene] < 0.36:
            scene = "general"
    tags: list[str] = []
    if night_signal >= 0.30:
        tags.append("night")
    if signals["indoor_object_signal"] >= 0.28 and scene != "indoor":
        tags.append("indoor")
    if landscape >= 0.42 and scene not in {"landscape", "city", "architecture"}:
        tags.append("outdoor")
    reasons: list[str] = []
    if person_count:
        reasons.append(f"detected_people={person_count}")
        if face_count:
            reasons.append(f"detected_faces={face_count}")
    for key, label in (
        ("animal_object_signal", "animal_objects"), ("food_object_signal", "food_objects"),
        ("city_object_signal", "city_objects"), ("indoor_object_signal", "indoor_objects"),
        ("sky_fraction_top", "sky"), ("vegetation_fraction", "vegetation"),
        ("line_signal", "structural_lines"), ("night_signal", "low_light"),
    ):
        if signals[key] >= 0.25:
            reasons.append(f"{label}={signals[key]:.2f}")
    if not reasons:
        reasons.append("no_strong_scene_signal")
    confidence = float(np.clip(scores[scene], 0.0, 0.98))
    return SceneResult(scene, confidence, tags, {key: round(float(value), 4) for key, value in scores.items()}, reasons[:5], {key: round(value, 4) for key, value in signals.items()})
