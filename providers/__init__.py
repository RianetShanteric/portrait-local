"""Replaceable local model providers."""

from .face_detection import YuNetFaceDetector
from .person_segmentation import TorchvisionInstancePersonSegmenter, TorchvisionPersonSegmenter
from .restoration import GFPGANFaceRestorer

__all__ = [
    "YuNetFaceDetector",
    "TorchvisionInstancePersonSegmenter",
    "TorchvisionPersonSegmenter",
    "GFPGANFaceRestorer",
]
