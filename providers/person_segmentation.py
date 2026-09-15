from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet50

from core.contracts import ModelIdentity, ObjectDetection, PersonSegmentation


class TorchvisionPersonSegmenter:
    """Permissively licensed semantic person segmentation, kept behind a provider."""

    identity = ModelIdentity(
        capability="person-segmentation",
        provider="torchvision-deeplabv3-resnet50",
        artifact="deeplabv3_resnet50_coco-cd0a2569.pth",
        version="torchvision-0.22.0",
        sha256="CD0A25694C4A0F7106B38F4938BF90A874F2F241CC410B8F63C7024399538F06",
    )

    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing segmentation model: {model_path}")
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def segment_people(self, rgb: np.ndarray) -> PersonSegmentation:
        height, width = rgb.shape[:2]
        scale = min(1.0, 768.0 / max(height, width))
        resized = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)["out"]
            probabilities = torch.softmax(logits, dim=1)[0, 15].float().cpu().numpy()
        probability = cv2.resize(probabilities, (width, height), interpolation=cv2.INTER_LINEAR)
        confidence = float(np.quantile(probability[probability > 0.35], 0.75)) if np.any(probability > 0.35) else 0.0
        return PersonSegmentation(mask=np.clip(probability, 0.0, 1.0), confidences=[confidence] if confidence else [])


class TorchvisionInstancePersonSegmenter:
    """Mask R-CNN person instances with a merged photographic subject mask."""

    identity = ModelIdentity(
        capability="person-instance-segmentation",
        provider="torchvision-maskrcnn-resnet50-fpn",
        artifact="maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        version="torchvision-0.22.0-coco-v1",
        sha256="BF2D0C1EFBC936EEEE2BC95A48E80EBC86B891F61B0106485937FC29F9315FC0",
    )

    def __init__(self, model_path: Path, device: str = "cuda", score_threshold: float = 0.68) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing instance segmentation model: {model_path}")
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.score_threshold = float(score_threshold)
        self.model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=91,
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def segment_people(self, rgb: np.ndarray) -> PersonSegmentation:
        height, width = rgb.shape[:2]
        scale = min(1.0, 1280.0 / max(height, width))
        resized = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div_(255.0).to(self.device)
        with torch.inference_mode():
            prediction = self.model([tensor])[0]

        labels = prediction["labels"].detach().cpu().numpy()
        scores = prediction["scores"].detach().float().cpu().numpy()
        boxes = prediction["boxes"].detach().float().cpu().numpy()
        masks = prediction["masks"].detach().float().cpu().numpy()[:, 0]
        accepted: list[np.ndarray] = []
        confidences: list[float] = []
        objects: list[ObjectDetection] = []
        categories = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
        minimum_area = resized.shape[0] * resized.shape[1] * 0.00035
        object_threshold = max(0.60, self.score_threshold - 0.08)
        resized_height, resized_width = resized.shape[:2]
        for label, score, box, mask in zip(labels, scores, boxes, masks):
            label_id = int(label)
            confidence = float(score)
            if confidence >= object_threshold and 0 <= label_id < len(categories):
                x1, y1, x2, y2 = (float(value) for value in box)
                area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, resized_width * resized_height)
                if area_fraction >= 0.00035:
                    objects.append(
                        ObjectDetection(
                            label=str(categories[label_id]),
                            confidence=confidence,
                            bbox=(
                                round(x1 / scale),
                                round(y1 / scale),
                                round((x2 - x1) / scale),
                                round((y2 - y1) / scale),
                            ),
                            area_fraction=float(area_fraction),
                        )
                    )
            if label_id != 1 or confidence < self.score_threshold:
                continue
            if float(np.count_nonzero(mask >= 0.50)) < minimum_area:
                continue
            accepted.append(mask)
            confidences.append(confidence)

        if not accepted:
            return PersonSegmentation(mask=np.zeros((height, width), dtype=np.float32), confidences=[], objects=objects)
        merged = np.max(np.stack(accepted, axis=0), axis=0)
        merged = cv2.resize(merged, (width, height), interpolation=cv2.INTER_LINEAR)
        return PersonSegmentation(mask=np.clip(merged, 0.0, 1.0), confidences=confidences, objects=objects)
