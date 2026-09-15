from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ModelIdentity:
    capability: str
    provider: str
    artifact: str
    version: str
    sha256: str


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_fraction: float


@dataclass(frozen=True)
class PersonSegmentation:
    mask: np.ndarray
    confidences: list[float]
    objects: list[ObjectDetection] = field(default_factory=list)


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass
class AnalysisResult:
    person_mask: np.ndarray
    background_mask: np.ndarray
    skin_mask: np.ndarray
    eye_mask: np.ndarray
    teeth_mask: np.ndarray
    hair_mask: np.ndarray
    faces: list[dict[str, Any]]
    metrics: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


class SegmentationProvider(Protocol):
    identity: ModelIdentity

    def segment_people(self, rgb: np.ndarray) -> PersonSegmentation: ...


class FaceDetectionProvider(Protocol):
    identity: ModelIdentity

    def detect(self, rgb: np.ndarray) -> list[FaceDetection]: ...


class FaceRestorationProvider(Protocol):
    identity: ModelIdentity

    def restore(self, rgb: np.ndarray) -> np.ndarray: ...
