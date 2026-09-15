# Third-Party Notices

Portrait Local is licensed under Apache-2.0. This does not replace or weaken
the licenses of third-party components and model artifacts.

The entries below describe the shipped working prototype. Exact package
versions, upstream URLs, artifact hashes and model-license notes are recorded
in `models/manifest.json`.

Model weights are deliberately not committed to this repository. The installer
downloads them from their upstream URLs and verifies SHA-256 before use. This
manifest is provenance and integrity information, not a grant to redistribute
the weights; review each upstream artifact's terms before bundling or sharing
it.

Python packages are installed from their upstream indexes rather than vendored
into this repository. Their own licenses remain applicable; dependency names
and versions are declared in `pyproject.toml` and captured in
`requirements.lock`.

## Torchvision DeepLabV3

- Purpose: person segmentation.
- Project: https://github.com/pytorch/vision
- Model: DeepLabV3 ResNet-50.
- Upstream code license: BSD 3-Clause.
- Distribution gate: keep the exact model hash and pretrained-data provenance
  in `models/manifest.json`.

## Torchvision Mask R-CNN

- Purpose: primary instance-aware person segmentation and counting.
- Project: https://github.com/pytorch/vision
- Model: Mask R-CNN ResNet-50 FPN, COCO V1.
- Upstream code license: BSD 3-Clause.
- Pretrained data: COCO; its dataset terms and provenance remain separate from
  the Torchvision code license.
- Exact artifact URL, size and SHA-256 are recorded in `models/manifest.json`.

## GFPGAN

- Purpose: conditional face restoration.
- Project: https://github.com/TencentARC/GFPGAN
- Upstream license: Apache License 2.0, with additional third-party components
  listed in the upstream license file.
- Distribution gate: retain upstream notices and review the exact weight and
  transitive component licenses.

## facexlib

- Purpose: RetinaFace detection and ParseNet compositing inside GFPGAN.
- Project: https://github.com/xinntao/facexlib
- Upstream license: MIT.
- Runtime policy: both auxiliary weights are hash-pinned and installed in
  advance; photo processing must never let GFPGAN download them implicitly.

## Real-ESRGAN

- Purpose: conditional super-resolution.
- Project: https://github.com/xinntao/Real-ESRGAN
- Upstream license: BSD 3-Clause.
- Distribution gate: retain copyright, license and disclaimer notices.

## OpenCV Zoo / YuNet

- Purpose: face detection.
- Project: https://github.com/opencv/opencv_zoo
- Repository license: Apache License 2.0. The upstream project requires checking
  the license of each individual model as well.
- Distribution gate: record the exact YuNet artifact, source, hash and model
  license before bundling it.

## Regression Photographs

- `couple_outdoors_public_domain.jpg`: National Cancer Institute; public domain
  as a work of the U.S. federal government.
- `indoor_moderator_public_domain.jpg`: Bill Branson / National Cancer
  Institute; public domain as a work of the U.S. federal government.
- `longfellow_group_public_domain.jpg`: National Cancer Institute; public
  domain as a work of the U.S. federal government.
- `back_view_seated_cc0.jpg`: OutdoorShooter00; CC0 1.0 Universal.
- `glacier_landscape_public_domain.jpg`: National Park Service / NPS Natural
  Resources; public domain as a U.S. federal government work.
- `city_night_public_domain.jpg`: Chris Spielmann / National Cancer Institute;
  public domain as a U.S. federal government work.

Exact source pages, download and stored hashes, transformations and coverage
details are recorded in `tests/regression/dataset.json` and
`tests/regression/README.md`. Repository derivatives contain no EXIF metadata.

This notice is an engineering inventory, not legal advice. It must be updated
whenever a dependency, model or redistributable asset is added or replaced.
