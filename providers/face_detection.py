from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.contracts import FaceDetection, ModelIdentity


class YuNetFaceDetector:
    identity = ModelIdentity("face-detection", "opencv-yunet", "face_detection_yunet_2023mar.onnx", "2023mar", "8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4")

    def __init__(self, model_path: Path, threshold: float) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing face detector: {model_path}")
        self.model_path = model_path
        self.threshold = threshold

    def detect(self, rgb: np.ndarray) -> list[FaceDetection]:
        height, width = rgb.shape[:2]
        detector = cv2.FaceDetectorYN.create(str(self.model_path), "", (width, height), self.threshold, 0.3, 5000)
        _, rows = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if rows is None:
            return []
        found: list[FaceDetection] = []
        for row in rows:
            x, y, box_width, box_height = [int(round(value)) for value in row[:4]]
            x, y = max(0, x), max(0, y)
            box_width, box_height = min(box_width, width - x), min(box_height, height - y)
            if box_width >= 8 and box_height >= 8:
                landmarks = tuple(
                    (float(row[index]), float(row[index + 1]))
                    for index in range(4, 14, 2)
                )
                found.append(FaceDetection((x, y, box_width, box_height), float(row[-1]), landmarks))
        return found
