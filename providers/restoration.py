from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from core.contracts import ModelIdentity


class GFPGANFaceRestorer:
    identity = ModelIdentity("face-restoration", "gfpgan", "GFPGANv1.3.pth", "1.3", "C953A88F2727C85C3D9AE72E2BD4846BBAF59FE6972AD94130E23E7017524A70")

    def __init__(self, model_path: Path) -> None:
        from gfpgan import GFPGANer

        project_root = model_path.parent.parent
        auxiliary = project_root / "gfpgan" / "weights"
        required = [
            model_path,
            auxiliary / "detection_Resnet50_Final.pth",
            auxiliary / "parsing_parsenet.pth",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise RuntimeError(f"Face restoration models are not installed: {names}. Run INSTALL.cmd.")

        # GFPGAN 1.3 resolves its two facexlib weights from a hard-coded path
        # relative to the process working directory. Initialise from the project
        # root only after proving that every file is already local; this prevents
        # an inference run from silently reaching the network.
        previous_cwd = Path.cwd()
        try:
            os.chdir(project_root)
            self.engine = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        finally:
            os.chdir(previous_cwd)

    def restore(self, rgb: np.ndarray) -> np.ndarray:
        _, _, restored = self.engine.enhance(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
