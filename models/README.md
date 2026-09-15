# Models

Model files are not committed to source control.

`manifest.json` is the canonical inventory. The installer downloads every
artifact to its declared local path and accepts it only when SHA-256 matches.

The 0.4.0 manifest currently contains only:

- instance and semantic person segmentation;
- YuNet face detection;
- GFPGAN face restoration and its facexlib auxiliaries;
- conditional Real-ESRGAN upscale.

Adaptive denoise and sharpening in 0.4.0 are non-generative image operations,
not a separately downloaded denoise model. Diffusion, identity, depth,
relighting and other model families are not part of this release.
