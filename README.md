# PortraitLocal

[![CI](https://github.com/RianetShanteric/portrait-local/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RianetShanteric/portrait-local/actions/workflows/ci.yml)

PortraitLocal is a local-first Windows photo enhancement application built
around a conservative AI/CV pipeline. It analyzes each image, selects bounded
scene-aware processing stages, and produces a new enhanced image without
uploading user photos to a cloud service.

## Features

- JPG/JPEG/PNG/WebP input with new JPEG output;
- Natural, Balanced and Strong processing profiles;
- automatic people, face and scene analysis;
- explainable policies for portrait, group, landscape, city, architecture,
  indoor, night, food, animal and object scenes;
- conservative identity-preserving enhancement with masked corrections;
- conditional GFPGAN face restoration and conditional Real-ESRGAN 2x upscale;
- before/after preview and export in the desktop UI;
- batch processing through the CLI;
- processing recipes, logs and optional debug masks for inspection.

## How It Works

```text
input
  -> people / face / image analysis
  -> explainable scene policy
  -> conservative enhancement
  -> optional face restoration
  -> optional 2x upscale
  -> preview and export
```

Heavy AI stages are conditional. Face restoration is used only when the face
quality and geometry gates justify it, and Real-ESRGAN is used only for small
or degraded inputs. Each run records the selected stages and model identities
in a processing recipe.

Scene classification combines Mask R-CNN object detections with measurable
image signals such as structure, sky, vegetation and low-light distribution.
It is intentionally explainable and bounded; it is not a dense semantic
segmentation system or a promise of perfect scene recognition.

## Tech Stack

- Python 3.12;
- PyTorch and CUDA;
- `torchvision` Mask R-CNN and DeepLabV3;
- OpenCV YuNet and Pillow;
- CustomTkinter desktop UI;
- GFPGAN and facexlib for conditional face restoration;
- Real-ESRGAN for conditional super-resolution.

Model implementations are accessed through provider interfaces so the
processing policy does not depend directly on a concrete provider.

## Local-first and Privacy

User photographs are read and processed locally. After initial installation,
the enhancement path does not require cloud photo processing or network calls.
Model weights are stored locally and deliberately excluded from this
repository. The repository contains no user photographs or runtime data.

## Installation and Running

The validated platform is Windows with Python 3.12 x64, Git, a compatible
NVIDIA driver and a CUDA-capable GPU. Internet access is required during the
initial package, vendor-code and model setup. The installer uses pinned CUDA
Torch packages and verifies every model artifact against the SHA-256 manifest.

From a checkout:

```powershell
.\INSTALL.cmd
.\PORTRAIT_UI.cmd
```

The UI creates the local virtual environment when needed. Put photographs in
`data/input`; each run creates a separate folder under `data/output` and keeps
the input files unchanged.

## CLI

After installation, place input images in `data/input` and run:

```powershell
.\.venv\Scripts\python.exe -m app.cli --root (Get-Location).Path --profile balanced
```

Use `natural`, `balanced` or `strong` for the profile. `RUN_PORTRAITS.cmd`
provides the same Balanced-profile batch entry point.

## Validation and Regression Evidence

The repository includes 19 automated tests and six attributed regression
scenes covering portrait, group, back-facing person, indoor, landscape and
night-city cases. The CUDA path, clean installation path and conditional
restoration/upscale stages were validated on an NVIDIA RTX 5070. The
regression inventory and reproducible visual/machine-readable evidence are in
[`tests/regression`](<tests/regression/README.md>); source attribution and
model provenance are recorded in
[`THIRD_PARTY_NOTICES.md`](<THIRD_PARTY_NOTICES.md>).

### Public demonstration

The following documented regression comparison is included for a visual
example. It uses a public-domain source image, and the complete baseline folder
contains the corresponding input/output hashes, recipe and attribution:

![Couple outdoors documented before-and-after regression comparison](<tests/regression/baseline/balanced/couple_outdoors/before-after.jpg>)

Run the development checks after installation:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

GitHub Actions runs the CPU-safe contract and scene-analysis tests without
model weights or CUDA. The full test suite additionally contains pipeline
checks that require local model weights and is intended for the validated local
NVIDIA environment.

## Limitations and Scope

- Windows and NVIDIA/CUDA are the validated configuration; support for other
  operating systems, GPUs or CPU-only end-user installation is not claimed;
- model weights are downloaded at setup time and are not redistributed here;
- scene policies are conservative heuristics and can fall back to general
  processing when evidence is weak;
- RAW workflows, photographic styles, generative/edit modes, dedicated AI
  denoise/deblur models, queue cancellation/retry and series consistency are
  outside the current release scope;
- the current workflow accepts common raster formats and exports a new JPEG.

## Project Documents

- [`PRODUCT.md`](<PRODUCT.md>) describes the current product scope;
- [`ROADMAP.md`](<ROADMAP.md>) separates future candidates from implemented
  behavior;
- [`CHANGELOG.md`](<CHANGELOG.md>) records shipped release content;
- [`SECURITY.md`](<SECURITY.md>) describes safe vulnerability reporting and
  handling of local photo data.

## License

Original PortraitLocal source is licensed under Apache-2.0. Third-party
software, model artifacts and regression images retain their own terms; see
[`LICENSE`](<LICENSE>) and
[`THIRD_PARTY_NOTICES.md`](<THIRD_PARTY_NOTICES.md>) before redistributing
anything beyond the source repository.
