# Roadmap

PortraitLocal is currently a bounded, local-first photo enhancement MVP. This
document describes product direction and planned work; it is not evidence that
an unimplemented capability already exists.

The released baseline is `v0.4.0` at commit `c215e64`. Future milestones are
sequenced by product value and acceptance risk, not by a promise of delivery
dates. A milestone should be released only when its acceptance criteria are
met on the supported workflow.

## Product principles

- Keep photo processing local after the initial installation.
- Preserve original files and make every destructive-looking action explicit.
- Prefer one-click workflows and conservative automatic decisions.
- Skip an operation when the analysis cannot justify it.
- Keep ordinary enhancement non-generative.
- Never let generative processing silently replace ordinary enhancement.
- Keep model implementations behind capability-oriented provider contracts.
- Prefer measured quality, identity preservation and reliability over model
  count.

## Released / current: v0.4.0

`v0.4.0` is a complete, standalone MVP for conservative automatic
enhancement of common raster photographs.

The released workflow is:

```text
Import -> Analyze -> Enhance -> Preview -> Export
```

The current product includes:

- Windows desktop UI and batch CLI;
- JPG, JPEG, PNG and WebP input with new JPEG output;
- Natural, Balanced and Strong profiles;
- Mask R-CNN instance-aware person segmentation with an offline DeepLabV3
  fallback;
- YuNet face analysis and geometry-gated eye, hair and teeth protection;
- explainable scene classification using existing detections and measurable
  image signals;
- conservative policies for portrait, group, landscape, city, architecture,
  indoor, night, food, animal and object scenes;
- adaptive non-generative denoise and sharpening;
- conditional GFPGAN face restoration and conditional Real-ESRGAN 2x upscale;
- isolated output folders, processing logs, recipes and optional debug masks;
- attributed public regression assets and machine-readable evidence.

The scene layer is intentionally bounded and explainable. It is not dense
semantic segmentation and it does not promise perfect scene recognition.

The current release does not include RAW development, photographic styles,
generative or interactive editing, dedicated AI denoise/deblur model families,
queue cancellation/retry, series consistency, automatic photo selection,
composition analysis or a model manager.

## Planned milestones

### v0.5.0 — Workflow Reliability

#### Goal

Make the existing multi-file workflow observable and dependable without
changing the enhancement scope.

#### User value

Users should be able to process a batch, understand what is happening, stop a
run safely and recover from an individual failure without guessing which files
were completed.

#### Major scope

- per-file and per-stage progress reporting;
- explicit queue state and completion/error status;
- cooperative cancellation between processing stages and subprocess work;
- retry of failed work in a fresh, isolated run;
- clear handling of partial success and preserved outputs;
- stable CLI status and exit behavior for automation.

#### Architectural implications

Add a small workflow/job-control layer around the existing CLI and pipeline.
Keep `SmartPhotoPipeline` as the enhancement engine. UI state should observe
structured workflow events rather than duplicate processing logic.

#### Dependencies

- the existing CLI, UI and isolated output-run structure;
- no new model family or mandatory runtime dependency;
- Windows subprocess behavior must be understood and tested.

#### Risks

- cancellation can leave a subprocess or temporary file in an ambiguous state;
- stale UI state can report success for an older run;
- retry can accidentally overwrite or mix outputs if run identity is weak.

#### Acceptance criteria

- a real supported batch of at least six images completes through the UI and
  CLI;
- progress identifies the current file and reaches a terminal state;
- cancellation leaves input files unchanged and does not present partial output
  as a completed result;
- retry creates a clean run and records the final status of each item;
- one failed item does not corrupt successful items;
- existing regression, packaging and local GPU checks remain valid.

#### Explicitly excluded

Persistent job databases, resume-after-restart, parallel inference, photoshoot
consistency, RAW, styles, generative editing and new model providers.

### v0.6.0 — Quality Rescue

#### Goal

Extend the existing adaptive quality handling with a measured, conditional
quality-rescue capability for genuinely noisy, blurred or compressed inputs.

#### User value

Weak source images should receive targeted help while already-clean images and
faces remain protected from unnecessary processing.

#### Major scope

- separate luminance and chroma noise assessment;
- conditional denoise for cases that justify it;
- conservative deblur or detail recovery where evidence supports it;
- bounded JPEG/compression artifact handling;
- before/after quality gates with a safe fallback to the current result;
- model and operation decisions recorded in the recipe.

#### Architectural implications

Introduce a capability-oriented quality assessment and rescue provider. Keep
the current CPU operations as a valid fallback; the new stage must be
selectable, measurable and skippable rather than unconditionally inserted into
the pipeline.

#### Dependencies

- separately licensed model artifacts, if a model is required;
- manifest entries, hash verification and installer support;
- labelled public or synthetic cases for clean, noisy, blurred and compressed
  images;
- local VRAM and performance measurements.

#### Risks

Over-smoothing skin or hair, hallucinated detail, model-license ambiguity,
VRAM pressure and quality metrics that reward a visually worse result.

#### Acceptance criteria

- clean inputs demonstrably skip unnecessary rescue stages;
- noisy, blurred and compressed cases show an accepted, bounded improvement;
- face geometry, skin texture and hair edges do not regress on the protected
  cases;
- failed or low-confidence rescue falls back safely;
- input hashes remain unchanged and recipes record stage decisions and model
  identities;
- the local GPU smoke and regression set pass without network access during
  processing.

#### Explicitly excluded

RAW, photographic styles, generative editing, background changes, series
consistency and personalization.

### v0.7.0 — Photo Set Consistency

#### Goal

Turn sequential batch processing into a reliable workflow for related images
from the same photo set.

#### User value

A set of photographs should look like one coherent session instead of a group
of independently corrected frames.

#### Major scope

- session or series grouping;
- shared exposure and white-balance references;
- consistent skin tone and conservative color treatment;
- individual analysis and outlier-safe fallback for every frame;
- session-aware recipes and output reporting.

#### Architectural implications

Add session-level analysis above the existing per-image pipeline. Per-image
decisions remain explicit; shared values must be recorded separately from
frame-specific measurements. Parallel inference should be considered only
after memory and quality behavior are measured.

#### Dependencies

- reliable queue behavior from `v0.5.0`;
- a public or synthetic labelled set of related photographs;
- stable output and recipe identity across a session.

#### Risks

Propagating a bad reference across the set, flattening intentional lighting
differences, and increasing VRAM or processing-time requirements.

#### Acceptance criteria

- a documented set of at least 20 related images is processed;
- exposure, white balance and skin-tone consistency improve without erasing
  intentional scene differences;
- outlier frames fall back to safe per-image behavior;
- cancellation, retry, input preservation and output isolation remain valid;
- session and frame decisions are auditable in the recipes.

#### Explicitly excluded

Automatic best-shot selection, composition correction, RAW development,
photographic styles and generative editing.

### v0.8.0 — Smart Selection

#### Goal

Provide conservative recommendations for the strongest photographs in a set
without deleting or hiding originals.

#### User value

Users should be able to review a shortlist based on visible technical reasons
instead of inspecting every near-duplicate manually.

#### Major scope

- blur and focus assessment;
- closed-eye and face-visibility signals where applicable;
- exposure and technical-quality scoring;
- duplicate and near-duplicate grouping;
- Best, Good and Review recommendations with explanations.

#### Architectural implications

Implement selection as a read-only analysis layer that consumes existing image
features and session context. Keep it separate from enhancement and export;
selection must never silently delete or overwrite an original.

#### Dependencies

- session grouping from `v0.7.0`;
- a labelled evaluation set with known acceptable and unacceptable frames;
- UI review states and clear user confirmation boundaries.

#### Risks

False rejection of meaningful images, bias toward technical sharpness, and
confusing a recommendation with an automatic destructive action.

#### Acceptance criteria

- every recommendation has an auditable reason;
- measured false-reject and false-accept rates are documented on the labelled
  set;
- no original is deleted, moved or overwritten automatically;
- low-confidence cases remain in Review;
- selection works independently of the enhancement choice.

#### Explicitly excluded

Generative editing, automatic deletion, identity embeddings, composition
correction and RAW development.

### v0.9.0 — Composition Assist

#### Goal

Offer conservative, reviewable composition suggestions for high-confidence
cases.

#### User value

Users can correct an obvious horizon, perspective or crop issue without turning
the product into a manual editor.

#### Major scope

- horizon detection and optional correction;
- perspective suggestions;
- face-aware and subject-aware crop suggestions;
- before/after preview and explicit confirmation;
- safe fallback to the unmodified composition.

#### Architectural implications

Keep composition analysis and suggested transforms separate from enhancement.
Every candidate transform should carry confidence, preview state and recipe
metadata. Automatic application is allowed only above a documented threshold.

#### Dependencies

- stable UI preview and workflow control;
- labelled composition cases;
- reliable preservation of the original image and pre-transform result.

#### Risks

Cropping important context, cutting people or changing facial geometry, and
turning a suggestion into an irreversible default.

#### Acceptance criteria

- low-confidence cases remain unchanged;
- high-confidence suggestions are visible before confirmation;
- originals remain byte-for-byte unchanged;
- accepted transforms do not distort faces or subjects;
- the CLI and UI report whether a composition suggestion was applied.

#### Explicitly excluded

Full manual editing, generative scene extension, background replacement,
photographic styles and RAW development.

### v1.0.0 — One-Click Local Editor

#### Goal

Consolidate the accepted capabilities into a stable, documented, raster-first
local product with a dependable one-click workflow.

#### User value

The user can import a photo or photo set, choose a bounded mode, process it
locally, review the result and export without understanding model internals.

#### Major scope

- the validated `v0.4.0` enhancement pipeline;
- reliable batch workflow and cancellation/retry;
- quality-rescue stages accepted in `v0.6.0`;
- photo-set consistency accepted in `v0.7.0`;
- optional smart selection and composition assistance when confidence is high;
- stable recipes, installation guidance, privacy boundaries and release checks.

#### Architectural implications

Stabilize capability contracts, workflow state, recipe schema, model
provenance, quality gates and supported-platform documentation. Avoid a broad
rewrite; introduce new seams only where a released capability needs them.

#### Dependencies

All preceding milestones must have independent acceptance evidence. The final
release also requires a clean supported installation, local GPU acceptance for
GPU-dependent paths, regression evidence, UI/CLI smoke checks and a reviewed
public repository.

#### Risks

The release can become an artificial bundle of unrelated features. Integration
must not hide weak individual acceptance behind a single end-to-end demo.

#### Acceptance criteria

- clean installation and launch on the documented Windows/Python/CUDA setup;
- complete supported raster workflow through both UI and CLI;
- regression and photo-set evidence pass with input preservation;
- no network photo processing after installation;
- quality and identity safeguards are visible in recipes and release notes;
- CPU CI, local GPU checks and manual UI evidence have clearly separated
  reports;
- public repository contains no weights, runtime data, credentials or private
  photographs.

#### Explicitly excluded

RAW development, photographic styles, generative or interactive editing,
background generation, clothing editing, full identity-engine research,
personalization, variants and a model manager.

## Post-v1.0 direction

The order and exact version numbers below are intentionally not frozen.

### RAW development

RAW is a separate input and color-management domain, not a small extension of
the raster importer. A future milestone should cover demosaic, camera white
balance, exposure, highlight recovery, lens correction, chromatic aberration,
color management, metadata handling and a clean hand-off into the existing
enhancement pipeline.

### Photographic looks and style packs

Start with a small set of non-generative photographic looks applied after
intelligent enhancement. Reusable style packs should follow only after the
base style contract, preview behavior, import/export format and attribution
rules are stable.

### Identity safety and creative operations

Before any operation can reconstruct or generate pixels near a person, require
an identity-safety gate with before/after validation, a documented threshold,
automatic rejection or rollback and recipe evidence. Only then consider
relighting, cleanup, background operations, generative fill or other creative
editing.

### Restoration and super-resolution

Future restoration may expand from the current conditional face restoration to
damage repair, colorization and broader recovery. Super-resolution may add
target resolution and additional scale/model choices. These remain separate
quality-risk tracks, not automatic additions to ordinary enhancement.

### Personalization and larger photo workflows

Local preferences, variants, Smart Photoshoot and a capability-oriented Model
Manager may become valuable after the core workflow is stable and user demand
justifies their complexity. They should build on the selection, session and
recipe contracts rather than introduce parallel product flows.

## Long-term product vision

PortraitLocal may eventually become a fully local photography workspace with a
simple automatic path and explicit creative modes. The broad vision includes
RAW development, conservative enhancement, photographic looks, series
processing, selection, restoration, super-resolution, relighting, cleanup,
background operations, generative editing and local personalization.

The vision does not change the core constraints: originals remain safe, local
processing is the default, creative changes are explicit, model weights retain
their own terms, and a feature is not considered shipped until its actual
implementation and acceptance evidence exist.

## Roadmap reconciliation

The previous numbered roadmap is historical context, not a binding version map.

- The old `0.1.x`, `0.2.0`, `0.2.1`, `0.3.0` and `0.4.0` work is represented by
  the released product and its regression evidence.
- The old `0.5.0 AI Denoise & Detail` is split: workflow reliability is now
  `v0.5.0`, while dedicated quality rescue is `v0.6.0`.
- The old `0.7.0 Batch & Photoshoot` is split between workflow reliability and
  photo-set consistency because basic batch processing already exists.
- The old `1.0.0` bundled RAW, batch, photosets and composition into one large
  promise. The new `v1.0.0` is deliberately raster-first and integrates only
  capabilities that have passed independent acceptance.
- Styles, RAW, generative editing, identity infrastructure, personalization
  and model management remain future tracks rather than hidden requirements
  for the current release.
