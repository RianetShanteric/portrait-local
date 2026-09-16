# Changelog

This changelog records shipped project behavior. Planned work is listed in
[`ROADMAP.md`](<ROADMAP.md>) and is not presented as part of the current
release.

## [0.4.0] - Public Portfolio Snapshot

This is a pre-1.0 public portfolio snapshot of the bounded local-first MVP,
not a production or hosted-service release.

### Added

- local-first Windows desktop UI and batch CLI;
- JPG/JPEG/PNG/WebP input with new JPEG output;
- Natural, Balanced and Strong processing profiles;
- instance-aware person segmentation with an offline DeepLabV3 fallback;
- YuNet face analysis and geometry-gated detail protection;
- bounded, explainable scene classification;
- conservative masked enhancement with adaptive denoise and sharpening;
- conditional GFPGAN face restoration and Real-ESRGAN 2x upscale;
- processing recipes, isolated output folders and optional debug masks;
- attributed public regression assets and machine-readable baseline evidence;
- CPU-safe GitHub Actions checks.

### Validation

- CPU-safe contract and scene-analysis tests run in GitHub Actions;
- CUDA processing, conditional restoration/upscale and clean installation were
  validated locally on an NVIDIA RTX 5070 with the documented environment.

### Boundaries

RAW development, photographic styles, generative or edit modes, dedicated AI
denoise/deblur models, queue cancellation/retry, series consistency and
additional model families are outside the `0.4.0` scope.

Model weights are downloaded during setup and are not redistributed in this
repository. Their provenance and integrity information are documented in
`models/manifest.json` and `THIRD_PARTY_NOTICES.md`.
