"""Content-aware photographic correction for Portrait Local.

This module deliberately separates analysis from rendering.  Every decision is
derived from the current photograph; profiles only scale the same decisions.
No generative model is used unless a detected face is objectively small or
degraded, and even then only a low-opacity restoration delta is accepted.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.contracts import AnalysisResult as Analysis
from core.contracts import ObjectDetection
from pipeline.scene import classify_scene
from providers import (
    GFPGANFaceRestorer,
    TorchvisionInstancePersonSegmenter,
    TorchvisionPersonSegmenter,
    YuNetFaceDetector,
)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(values, dtype=np.float32)
    x = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _masked_values(values: np.ndarray, mask: np.ndarray | None, threshold: float = 0.5) -> np.ndarray:
    if mask is None:
        return values.reshape(-1)
    selected = values[mask > threshold]
    return selected if selected.size else values.reshape(-1)


def _percentile(values: np.ndarray, mask: np.ndarray | None, q: float) -> float:
    return float(np.quantile(_masked_values(values, mask), q))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[:, :, 0].astype(np.float32) * 0.2126
        + rgb[:, :, 1].astype(np.float32) * 0.7152
        + rgb[:, :, 2].astype(np.float32) * 0.0722
    ) / 255.0


def _sharpness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    values = _masked_values(lap, mask, 0.55)
    if values.size < 64:
        return 0.0
    return float(np.var(values))


def _noise_level(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    source = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(source, (0, 0), 1.0)
    residual = np.abs(source - smooth)
    gradient = np.abs(cv2.Laplacian(source, cv2.CV_32F, ksize=3))
    flat = gradient < float(np.quantile(gradient, 0.60))
    if mask is not None:
        flat &= mask > 0.55
    samples = residual[flat]
    if samples.size < 128:
        samples = residual.reshape(-1)
    return float(np.median(samples) * 1.4826)


def _soft_mask(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    if probability.shape != (height, width):
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
    hard = (probability > 0.42).astype(np.uint8)
    radius = max(3, int(round(math.hypot(width, height) * 0.0025)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
    feather = max(5.0, math.hypot(width, height) * 0.005)
    signed = inside - outside
    soft = np.clip(0.5 + signed / (2.0 * feather), 0.0, 1.0)
    # Preserve model confidence around fine contours while keeping a safe core.
    soft = np.maximum(soft * 0.80, probability * 0.65)
    return cv2.GaussianBlur(np.clip(soft, 0.0, 1.0), (0, 0), max(1.2, feather * 0.16))


def _skin_mask(rgb: np.ndarray, person_mask: np.ndarray, faces: list[dict[str, Any]]) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    hue, sat, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (
        (cr >= 132)
        & (cr <= 181)
        & (cb >= 76)
        & (cb <= 136)
        & (((hue <= 25) | (hue >= 170)))
        & (sat >= 18)
        & (sat <= 190)
        & (value >= 38)
    ).astype(np.float32)
    face_region = np.zeros(skin.shape, dtype=np.float32)
    for face in faces:
        x, y, width, height = face["bbox"]
        center = (int(x + width * 0.5), int(y + height * 0.56))
        axes = (max(1, int(width * 0.63)), max(1, int(height * 0.84)))
        cv2.ellipse(face_region, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    skin *= np.maximum(person_mask, face_region)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(skin, (0, 0), 2.2)


def _landmarks_reliable(face: dict[str, Any]) -> bool:
    points = face.get("landmarks", [])
    if len(points) != 5 or min(face["bbox"][2:]) < 96 or face["confidence"] < 0.82:
        return False
    x, y, width, height = face["bbox"]
    if not all(x - width * 0.08 <= px <= x + width * 1.08 and y - height * 0.08 <= py <= y + height * 1.08 for px, py in points):
        return False
    right_eye, left_eye, nose, right_mouth, left_mouth = points
    eye_distance = math.dist(right_eye, left_eye)
    mouth_distance = math.dist(right_mouth, left_mouth)
    eye_y = (right_eye[1] + left_eye[1]) * 0.5
    mouth_y = (right_mouth[1] + left_mouth[1]) * 0.5
    return bool(
        width * 0.16 <= eye_distance <= width * 0.72
        and width * 0.10 <= mouth_distance <= width * 0.62
        and eye_y < nose[1] < mouth_y
        and mouth_y - eye_y >= height * 0.16
    )


def _portrait_region_masks(
    rgb: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
    faces: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    eyes = np.zeros((height, width), dtype=np.float32)
    mouth_regions = np.zeros((height, width), dtype=np.float32)
    hair = np.zeros((height, width), dtype=np.float32)
    for face in faces:
        if not face["landmarks_reliable"]:
            continue
        x, y, box_width, box_height = face["bbox"]
        right_eye, left_eye, nose, right_mouth, left_mouth = face["landmarks"]
        eye_distance = math.dist(right_eye, left_eye)
        for px, py in (right_eye, left_eye):
            cv2.ellipse(
                eyes,
                (round(px), round(py)),
                (max(2, round(eye_distance * 0.20)), max(2, round(eye_distance * 0.11))),
                0,
                0,
                360,
                1.0,
                -1,
                cv2.LINE_AA,
            )
        mouth_center = (
            round((right_mouth[0] + left_mouth[0]) * 0.5),
            round((right_mouth[1] + left_mouth[1]) * 0.5),
        )
        mouth_width = max(4, round(math.dist(right_mouth, left_mouth) * 0.58))
        cv2.ellipse(mouth_regions, mouth_center, (mouth_width, max(3, round(box_height * 0.085))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        head_center = (round(x + box_width * 0.5), round(y + box_height * 0.20))
        cv2.ellipse(
            hair,
            head_center,
            (max(2, round(box_width * 0.62)), max(2, round(box_height * 0.55))),
            0,
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
        )

    luma = _luma(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    teeth = mouth_regions * ((luma > 0.46) & (saturation < 0.38)).astype(np.float32)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    eyes = cv2.GaussianBlur(eyes, (0, 0), 1.4)
    teeth = cv2.GaussianBlur(teeth, (0, 0), 1.0)
    hair *= person_mask * (1.0 - np.clip(skin_mask * 1.15, 0.0, 1.0))
    hair *= (1.0 - eyes) * (1.0 - mouth_regions)
    hair = cv2.GaussianBlur(np.clip(hair, 0.0, 1.0), (0, 0), 2.0)
    return np.clip(eyes, 0.0, 1.0), np.clip(teeth, 0.0, 1.0), hair


class SmartPhotoPipeline:
    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        self.config = config
        self.model_dir = root / "models"
        analysis_config = config["analysis"]
        self.segmentation_path = self.model_dir / analysis_config["person_segmentation_model"]
        self.face_path = self.model_dir / "face_detection_yunet_2023mar.onnx"
        device = "cuda" if config.get("runtime", {}).get("device", "cuda") == "cuda" else "cpu"
        if analysis_config.get("person_segmentation_provider") == "torchvision-maskrcnn-resnet50-fpn" and self.segmentation_path.is_file():
            self.segmenter = TorchvisionInstancePersonSegmenter(
                self.segmentation_path,
                device=device,
                score_threshold=float(analysis_config.get("person_instance_score_threshold", 0.68)),
            )
        else:
            fallback_path = self.model_dir / analysis_config.get(
                "person_segmentation_fallback_model",
                "deeplabv3_resnet50_coco-cd0a2569.pth",
            )
            self.segmenter = TorchvisionPersonSegmenter(fallback_path, device=device)
        self.face_detector = YuNetFaceDetector(self.face_path, float(config["analysis"]["face_detection_threshold"]))
        self.face_restorer = None

    def _segment_people(self, rgb: np.ndarray) -> tuple[np.ndarray, list[float], list[ObjectDetection]]:
        height, width = rgb.shape[:2]
        result = self.segmenter.segment_people(rgb)
        threshold = float(self.config["analysis"]["person_confidence"])
        if not result.confidences or max(result.confidences) < threshold:
            return np.zeros((height, width), dtype=np.float32), [], result.objects
        return _soft_mask(result.mask, width, height), result.confidences, result.objects

    def _detect_faces(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        height, width = rgb.shape[:2]
        detected = self.face_detector.detect(rgb)
        faces: list[dict[str, Any]] = []
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        luma = _luma(rgb)
        for detection in detected:
            x, y, box_width, box_height = detection.bbox
            crop_gray = gray[y : y + box_height, x : x + box_width]
            crop_luma = luma[y : y + box_height, x : x + box_width]
            sharpness = _sharpness(crop_gray)
            noise = _noise_level(crop_gray)
            min_side = min(box_width, box_height)
            size_quality = float(np.clip((min_side - 64.0) / 190.0, 0.0, 1.0))
            sharp_quality = float(np.clip((sharpness - 35.0) / 170.0, 0.0, 1.0))
            noise_quality = float(np.clip(1.0 - noise / 0.035, 0.0, 1.0))
            quality = 0.40 * size_quality + 0.40 * sharp_quality + 0.20 * noise_quality
            faces.append(
                {
                    "bbox": [x, y, box_width, box_height],
                    "confidence": detection.confidence,
                    "brightness": float(np.median(crop_luma)),
                    "contrast": float(np.quantile(crop_luma, 0.90) - np.quantile(crop_luma, 0.10)),
                    "sharpness": sharpness,
                    "noise": noise,
                    "quality": float(np.clip(quality, 0.0, 1.0)),
                    "landmarks": [[float(px), float(py)] for px, py in detection.landmarks],
                }
            )
            faces[-1]["landmarks_reliable"] = _landmarks_reliable(faces[-1])
        return faces

    def analyze(self, rgb: np.ndarray) -> Analysis:
        started = time.perf_counter()
        height, width = rgb.shape[:2]
        person_mask, person_confidence, objects = self._segment_people(rgb)
        segmented_at = time.perf_counter()
        faces = self._detect_faces(rgb)
        faces_at = time.perf_counter()
        background_mask = 1.0 - person_mask
        skin_mask = _skin_mask(rgb, person_mask, faces)
        eye_mask, teeth_mask, hair_mask = _portrait_region_masks(rgb, person_mask, skin_mask, faces)
        luma = _luma(rgb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        has_people = float(np.mean(person_mask > 0.50)) > 0.005
        person_for_stats = person_mask if has_people else None
        background_for_stats = background_mask if has_people else None

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        neutral = (saturation < 0.20) & (luma > 0.15) & (luma < 0.85)
        if np.count_nonzero(neutral) > 512:
            cast_a = float(np.median(lab[:, :, 1][neutral]) - 128.0)
            cast_b = float(np.median(lab[:, :, 2][neutral]) - 128.0)
        else:
            cast_a = cast_b = 0.0

        metrics: dict[str, Any] = {
            "width": width,
            "height": height,
            "has_people": has_people,
            "person_count": max(len(person_confidence), len(faces)),
            "person_confidences": person_confidence,
            "person_area": float(np.mean(person_mask)),
            "face_count": len(faces),
            "landmark_face_count": sum(bool(face["landmarks_reliable"]) for face in faces),
            "eye_area": float(np.mean(eye_mask)),
            "teeth_area": float(np.mean(teeth_mask)),
            "hair_area": float(np.mean(hair_mask)),
            "global_median": _percentile(luma, None, 0.50),
            "person_median": _percentile(luma, person_for_stats, 0.50),
            "background_median": _percentile(luma, background_for_stats, 0.50),
            "highlight_fraction": float(np.mean(luma > 0.975)),
            "shadow_fraction": float(np.mean(luma < 0.055)),
            "dynamic_range": _percentile(luma, None, 0.98) - _percentile(luma, None, 0.02),
            "global_saturation": float(np.median(saturation)),
            "person_saturation": _percentile(saturation, person_for_stats, 0.50),
            "background_saturation": _percentile(saturation, background_for_stats, 0.50),
            "background_saturation_p85": _percentile(saturation, background_for_stats, 0.85),
            "noise": _noise_level(gray),
            "sharpness": _sharpness(gray),
            "neutral_cast_a": cast_a,
            "neutral_cast_b": cast_b,
        }
        metrics["exposure_gap"] = metrics["background_median"] - metrics["person_median"] if has_people else 0.0
        metrics["needs_upscale"] = bool(
            min(width, height) < int(self.config["analysis"]["conditional_upscale_min_side"])
            and (metrics["sharpness"] < 75.0 or min(width, height) < 520)
        )
        scene = classify_scene(
            rgb,
            objects,
            int(metrics["person_count"]),
            int(metrics["face_count"]),
            float(metrics["person_area"]),
        )
        metrics.update(
            {
                "scene": scene.scene,
                "scene_confidence": scene.confidence,
                "scene_tags": scene.tags,
                "scene_reasons": scene.reasons,
                "scene_scores": scene.scores,
                "scene_signals": scene.signals,
                # Bounding boxes are needed transiently for inference only. The
                # recipe retains the minimum useful, non-biometric evidence.
                "detected_objects": [
                    {
                        "label": item.label,
                        "confidence": round(item.confidence, 4),
                        "area_fraction": round(item.area_fraction, 5),
                    }
                    for item in objects
                ],
            }
        )
        # Raw facial landmarks are ephemeral processing data. Recipes retain
        # the gate result and quality evidence, but not landmark coordinates.
        metrics["faces"] = [
            {key: value for key, value in face.items() if key != "landmarks"}
            for face in faces
        ]
        ended = time.perf_counter()
        return Analysis(
            person_mask=person_mask,
            background_mask=background_mask,
            skin_mask=skin_mask,
            eye_mask=eye_mask,
            teeth_mask=teeth_mask,
            hair_mask=hair_mask,
            faces=faces,
            metrics=metrics,
            timings={
                "person_segmentation_seconds": segmented_at - started,
                "face_analysis_seconds": faces_at - segmented_at,
                "metrics_seconds": ended - faces_at,
                "analysis_total_seconds": ended - started,
            },
        )

    @staticmethod
    def _apply_subject_tone(rgb: np.ndarray, analysis: Analysis, profile: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        metrics = analysis.metrics
        if not metrics["has_people"]:
            return rgb.copy(), {"subject_lift": 0.0, "subject_gamma": 1.0}
        gap = float(metrics["exposure_gap"])
        lift = float(
            np.clip(
                (gap - 0.025) * 0.72 * profile["intensity"],
                0.0,
                profile["subject_lift_cap"],
            )
        )
        person_median = float(np.clip(metrics["person_median"], 0.04, 0.94))
        target = float(np.clip(person_median + lift, 0.05, 0.86))
        gamma = math.log(target) / math.log(person_median) if lift > 0.002 else 1.0
        gamma = float(np.clip(gamma, 0.72, 1.0))

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lum = lab[:, :, 0] / 255.0
        corrected = np.power(np.clip(lum, 0.0, 1.0), gamma)
        highlight_guard = 1.0 - _smoothstep(0.70, 0.94, lum)
        mask = analysis.person_mask * highlight_guard
        lab[:, :, 0] = np.clip((lum * (1.0 - mask) + corrected * mask) * 255.0, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB), {"subject_lift": lift, "subject_gamma": gamma}

    @staticmethod
    def _control_background(rgb: np.ndarray, analysis: Analysis, profile: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        metrics = analysis.metrics
        subject_scene = metrics.get("scene") in {"portrait", "group"}
        if not subject_scene:
            mask = np.ones(analysis.background_mask.shape, dtype=np.float32)
        else:
            mask = analysis.background_mask
        strength = float(profile["background_control"])
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lum = lab[:, :, 0] / 255.0
        highlight_need = float(np.clip((metrics["highlight_fraction"] - 0.012) * 1.20, 0.0, 0.055))
        if metrics["background_median"] > 0.54:
            highlight_need = max(highlight_need, min(0.055, (metrics["background_median"] - 0.54) * 0.20))
        highlight_need *= strength
        shoulder = _smoothstep(0.68, 0.92, lum)
        clipped_guard = 1.0 - 0.86 * _smoothstep(0.94, 1.0, lum)
        rolloff = shoulder * clipped_guard * highlight_need
        lab[:, :, 0] = np.clip((lum - rolloff * mask) * 255.0, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation_gap = metrics["background_saturation"] - metrics["person_saturation"] if subject_scene else 0.0
        relative_reduction = float(np.clip((saturation_gap - 0.025) * 0.30, 0.0, 0.060))
        vivid_reduction = float(np.clip((metrics["background_saturation_p85"] - 0.48) * 0.40, 0.0, 0.080))
        saturation_reduction = max(relative_reduction, vivid_reduction) * strength
        normalized_saturation = hsv[:, :, 1] / 255.0
        vivid_mask = _smoothstep(0.38, 0.78, normalized_saturation)
        hsv[:, :, 1] *= 1.0 - analysis.background_mask * vivid_mask * saturation_reduction
        # Luminance changes can make HSV saturation rise even when S itself was
        # reduced.  Enforce a final content-derived ceiling: vivid background
        # must not end up more saturated than it was in the source.
        post_saturation = hsv[:, :, 1] / 255.0
        post_p85 = _percentile(post_saturation, analysis.background_mask, 0.85)
        target_background_p85 = max(0.0, metrics["background_saturation_p85"] - 0.015)
        saturation_guard = float(
            np.clip(
                (post_p85 - target_background_p85) / max(post_p85, 0.01) + 0.004,
                0.0,
                0.050,
            )
        )
        if saturation_guard > 0:
            hsv[:, :, 1] *= 1.0 - analysis.background_mask * vivid_mask * saturation_guard
            saturation_reduction += saturation_guard
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result, {"background_highlight_rolloff": highlight_need, "background_saturation_reduction": saturation_reduction}

    @staticmethod
    def _safe_white_balance(rgb: np.ndarray, analysis: Analysis) -> tuple[np.ndarray, dict[str, float]]:
        cast_a = float(analysis.metrics["neutral_cast_a"])
        cast_b = float(analysis.metrics["neutral_cast_b"])
        magnitude = math.hypot(cast_a, cast_b)
        if magnitude < 4.0:
            return rgb.copy(), {"white_balance_a": 0.0, "white_balance_b": 0.0}
        correction_a = float(np.clip(-cast_a * 0.18, -3.0, 3.0))
        correction_b = float(np.clip(-cast_b * 0.18, -3.0, 3.0))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        skin_guard = 1.0 - analysis.skin_mask * 0.78
        lab[:, :, 1] = np.clip(lab[:, :, 1] + correction_a * skin_guard, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + correction_b * skin_guard, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB), {"white_balance_a": correction_a, "white_balance_b": correction_b}

    @staticmethod
    def _adaptive_detail(rgb: np.ndarray, analysis: Analysis, profile: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        noise = float(analysis.metrics["noise"])
        sharpness = float(analysis.metrics["sharpness"])
        result = rgb.copy()
        denoise_amount = float(np.clip((noise - 0.010) / 0.030, 0.0, 0.38)) * profile["denoise_strength"]
        if denoise_amount > 0.025:
            filtered = cv2.bilateralFilter(result, 5, 7.0 + noise * 180.0, 3.5)
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gradient = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            detail_guard = 1.0 - _smoothstep(0.025, 0.16, gradient)
            protected = np.maximum(analysis.skin_mask, analysis.teeth_mask)
            skin_guard = 1.0 - protected * 0.58
            mix = (denoise_amount * detail_guard * skin_guard)[:, :, None]
            result = np.clip(result.astype(np.float32) * (1.0 - mix) + filtered.astype(np.float32) * mix, 0, 255).astype(np.uint8)

        blur_need = float(np.clip((105.0 - sharpness) / 105.0, 0.0, 1.0))
        sharpen_amount = 0.26 * blur_need * profile["sharpen_strength"]
        if sharpen_amount > 0.02:
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
            lum = lab[:, :, 0]
            blurred = cv2.GaussianBlur(lum, (0, 0), 1.15)
            detail = np.clip(lum - blurred, -10.0, 10.0)
            protected = np.maximum(analysis.skin_mask, analysis.teeth_mask)
            skin_guard = 1.0 - protected * 0.72
            if analysis.metrics.get("scene") in {"portrait", "group"}:
                region = 0.25 + analysis.person_mask * 0.75
            else:
                region = np.ones_like(lum)
            lab[:, :, 0] = np.clip(lum + detail * sharpen_amount * skin_guard * region, 0, 255)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result, {"denoise_amount": denoise_amount, "sharpen_amount": sharpen_amount}

    @staticmethod
    def _portrait_region_detail(
        rgb: np.ndarray,
        analysis: Analysis,
        profile: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, float]]:
        if not np.any(analysis.eye_mask > 0.05) and not np.any(analysis.hair_mask > 0.05):
            return rgb, {
                "eye_detail_amount": 0.0,
                "hair_detail_amount": 0.0,
                "teeth_protected": bool(np.any(analysis.teeth_mask > 0.05)),
            }
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        luminance = lab[:, :, 0]
        blurred = cv2.GaussianBlur(luminance, (0, 0), 0.85)
        detail = np.clip(luminance - blurred, -7.0, 7.0)
        intensity = float(profile["intensity"])
        face_sharpness = [float(face["sharpness"]) for face in analysis.faces if face["landmarks_reliable"]]
        median_face_sharpness = float(np.median(face_sharpness)) if face_sharpness else 0.0
        blur_need = float(np.clip((150.0 - median_face_sharpness) / 150.0, 0.0, 1.0))
        eye_amount = (0.035 + 0.065 * blur_need) * intensity
        hair_amount = (0.030 + 0.050 * blur_need) * intensity
        region = analysis.eye_mask * eye_amount + analysis.hair_mask * hair_amount
        region *= 1.0 - analysis.teeth_mask
        lab[:, :, 0] = np.clip(luminance + detail * region, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        # LAB round-tripping can quantize otherwise untouched RGB pixels. Keep
        # the operation strictly local, including byte-for-byte outside masks.
        result[region <= 1e-5] = rgb[region <= 1e-5]
        return result, {
            "eye_detail_amount": eye_amount,
            "hair_detail_amount": hair_amount,
            "teeth_protected": bool(np.any(analysis.teeth_mask > 0.05)),
        }

    @staticmethod
    def _general_photo_tone(rgb: np.ndarray, analysis: Analysis, profile: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        median = float(np.clip(analysis.metrics["global_median"], 0.03, 0.97))
        target = median
        if median < 0.27 and analysis.metrics["highlight_fraction"] < 0.025:
            target = min(0.34, median + profile["subject_lift_cap"] * 0.65)
        elif median > 0.72:
            target = max(0.66, median - 0.04 * profile["background_control"])
        gamma = math.log(target) / math.log(median) if abs(target - median) > 0.005 else 1.0
        gamma = float(np.clip(gamma, 0.82, 1.12))
        source = rgb.astype(np.float32) / 255.0
        corrected = np.power(source, gamma)
        guard = 1.0 - _smoothstep(0.76, 0.98, _luma(rgb))[:, :, None]
        result = source * (1.0 - guard) + corrected * guard
        return np.clip(result * 255.0, 0, 255).astype(np.uint8), {"general_gamma": gamma}

    def _restoration_strengths(self, analysis: Analysis, cap: float) -> list[float]:
        strengths: list[float] = []
        for face in analysis.faces:
            quality = float(face["quality"])
            size = min(face["bbox"][2], face["bbox"][3])
            if quality >= 0.66 and size >= 150:
                strengths.append(0.0)
                continue
            need = float(np.clip((0.67 - quality) / 0.67, 0.0, 1.0))
            if size >= 180 and face["sharpness"] >= 95 and face["noise"] < 0.025:
                need *= 0.10
            strengths.append(min(cap, cap * need))
        return strengths

    def _apply_conditional_face_restoration(
        self, rgb: np.ndarray, analysis: Analysis, cap: float
    ) -> tuple[np.ndarray, list[float]]:
        strengths = self._restoration_strengths(analysis, cap)
        if not strengths or max(strengths) < 0.015:
            return rgb, strengths
        if self.face_restorer is None:
            self.face_restorer = GFPGANFaceRestorer(self.model_dir / "GFPGANv1.3.pth")
        restored = self.face_restorer.restore(rgb).astype(np.float32)
        base = rgb.astype(np.float32)
        result = base.copy()
        height, width = rgb.shape[:2]
        for face, strength in zip(analysis.faces, strengths):
            if strength < 0.015:
                continue
            x, y, box_width, box_height = face["bbox"]
            mask = np.zeros((height, width), dtype=np.float32)
            center = (int(x + box_width * 0.5), int(y + box_height * 0.56))
            axes = (max(1, int(box_width * 0.68)), max(1, int(box_height * 0.88)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, max(axes) * 0.16))[:, :, None]
            result += (restored - base) * mask * strength
        return np.clip(result, 0, 255).astype(np.uint8), strengths

    def render(self, rgb: np.ndarray, analysis: Analysis, profile_name: str) -> tuple[np.ndarray, dict[str, Any]]:
        profile = dict(self.config["profiles"][profile_name])
        scene = str(analysis.metrics.get("scene", "general"))
        tags = set(analysis.metrics.get("scene_tags", []))
        policy = {"tone_scale": 1.0, "denoise_scale": 1.0, "sharpen_scale": 1.0, "background_scale": 1.0}
        if "night" in tags or scene == "night":
            policy.update({"tone_scale": 0.58, "denoise_scale": 1.12, "sharpen_scale": 0.68, "background_scale": 0.62})
        elif scene in {"city", "architecture"}:
            policy.update({"denoise_scale": 0.82, "sharpen_scale": 1.10, "background_scale": 0.90})
        elif scene in {"food", "landscape"}:
            policy.update({"sharpen_scale": 0.90, "background_scale": 0.62})
        profile["subject_lift_cap"] *= policy["tone_scale"]
        profile["denoise_strength"] *= policy["denoise_scale"]
        profile["sharpen_strength"] *= policy["sharpen_scale"]
        profile["background_control"] *= policy["background_scale"]
        started = time.perf_counter()
        decisions: dict[str, Any] = {
            "profile": profile_name,
            "scene": scene,
            "scene_confidence": analysis.metrics.get("scene_confidence", 0.0),
            "scene_policy": policy,
        }
        if scene in {"portrait", "group"}:
            result, values = self._apply_subject_tone(rgb, analysis, profile)
            decisions.update(values)
        else:
            result, values = self._general_photo_tone(rgb, analysis, profile)
            decisions.update(values)
        result, values = self._safe_white_balance(result, analysis)
        decisions.update(values)
        result, values = self._control_background(result, analysis, profile)
        decisions.update(values)
        result, values = self._adaptive_detail(result, analysis, profile)
        decisions.update(values)
        result, values = self._portrait_region_detail(result, analysis, profile)
        decisions.update(values)
        result, strengths = self._apply_conditional_face_restoration(
            result, analysis, float(profile["face_restore_cap"])
        )
        decisions["face_restoration_strengths"] = strengths
        decisions["upscale_requested"] = bool(analysis.metrics["needs_upscale"])
        decisions["render_seconds"] = time.perf_counter() - started
        return result, decisions

    @staticmethod
    def save_debug(
        debug_dir: Path,
        source_name: str,
        rgb: np.ndarray,
        analysis: Analysis,
        decisions: dict[str, Any],
    ) -> None:
        target = debug_dir / Path(source_name).stem
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(analysis.person_mask * 255, 0, 255).astype(np.uint8)).save(target / "person-mask.png")
        Image.fromarray(np.clip(analysis.background_mask * 255, 0, 255).astype(np.uint8)).save(target / "background-mask.png")
        Image.fromarray(np.clip(analysis.skin_mask * 255, 0, 255).astype(np.uint8)).save(target / "skin-mask.png")
        Image.fromarray(np.clip(analysis.eye_mask * 255, 0, 255).astype(np.uint8)).save(target / "eye-mask.png")
        Image.fromarray(np.clip(analysis.teeth_mask * 255, 0, 255).astype(np.uint8)).save(target / "teeth-mask.png")
        Image.fromarray(np.clip(analysis.hair_mask * 255, 0, 255).astype(np.uint8)).save(target / "hair-mask.png")
        overlay = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(overlay)
        for index, face in enumerate(analysis.faces, 1):
            x, y, width, height = face["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=(103, 232, 194), width=3)
            draw.text((x + 4, y + 4), f"F{index} q={face['quality']:.2f}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            if face["landmarks_reliable"]:
                for point_index, (px, py) in enumerate(face["landmarks"]):
                    colour = (80, 180, 255) if point_index < 2 else (255, 190, 70)
                    radius = max(2, round(min(width, height) * 0.012))
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colour)
        overlay.save(target / "faces-overlay.jpg", quality=94)
        payload = {"metrics": analysis.metrics, "timings": analysis.timings, "decisions": decisions, "inference": "Torchvision DeepLabV3 CUDA; YuNet CPU; OpenCV CPU; GFPGAN CUDA conditional"}
        (target / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
