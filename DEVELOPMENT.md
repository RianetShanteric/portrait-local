# Development and Release Workflow

This repository is developed as a small, reviewable open-source project. The
current public baseline is PortraitLocal `v0.4.0` at commit `c215e64`. Product
versioning, roadmap status and development status must remain separate: a
documentation or process change does not create a new application release.

## Working agreement

- Do not make real development changes directly on `main`.
- Start each task from an up-to-date, clean baseline.
- Keep a task within its approved product scope.
- Preserve original photographs and existing privacy boundaries.
- Prefer a small, understandable change over an architectural rewrite.
- All project documentation, commit messages and release notes are written in
  English.

## Branch naming

Use a conventional prefix followed by a concise task description:

- `feat/` for a user-visible capability;
- `fix/` for a defect within the current scope;
- `docs/` for documentation and process work;
- `refactor/` for a behavior-preserving structural change;
- `test/` for regression or validation coverage;
- `chore/` for maintenance that does not change product behavior.

Examples:

```text
docs/roadmap-development-workflow
feat/workflow-reliability
fix/output-name-collision
test/quality-rescue-regressions
```

## Task lifecycle

The normal lifecycle is:

```text
task
  -> branch
  -> implementation
  -> meaningful commits
  -> relevant local checks
  -> diff review
  -> push
  -> pull request
  -> required CI
  -> merge
  -> release when appropriate
```

The task description is the scope boundary. If a required change would alter
the product scope, dependencies or model configuration beyond that task, stop
and request approval before implementing it.

## Commit hygiene

- Commits should represent coherent logical changes.
- Required CI and branch protection must not be bypassed.

## Validation tiers

### GitHub Actions CI

GitHub-hosted CI runs on an ordinary Ubuntu runner and is intentionally limited
to checks that do not need NVIDIA hardware or downloaded model weights. The
current workflow covers:

- CPU-safe contract tests;
- scene-analysis tests;
- dependency consistency with `pip check`;
- Python compilation;
- wheel build.

CI does not download model weights, run heavy inference or prove CUDA, GPU
performance, model quality or Windows installer behavior.

### Local release acceptance

GPU-dependent capabilities require local acceptance on the documented supported
environment. Depending on the release, this can include:

- a clean installation with the supported Python and CUDA dependencies;
- model artifacts verified against the expected manifest hashes;
- real CLI and UI processing;
- real regression images and release-specific cases;
- input-file preservation and output metadata checks;
- conditional-stage evidence when restoration or upscale is in scope;
- installer and launcher smoke checks;
- a reviewed diff and publication-hygiene scan.

Local GPU acceptance must be reported separately from the GitHub CI result. A
green CPU check must never be described as GPU validation.

## Release expectations

Before merging a release-bearing change:

1. confirm the implementation matches the approved scope;
2. add or update regression evidence for changed behavior;
3. run the relevant CPU-safe checks and local hardware checks;
4. review the complete diff, including staged new files;
5. run whitespace and repository-hygiene checks;
6. update English documentation and the changelog without overstating support;
7. obtain review and required CI before merging.

Create a tag or GitHub Release only when the corresponding application version
is actually implemented, accepted and documented. Documentation/process work
alone remains under `Unreleased` and does not bump the application version.

## Repository hygiene

Never commit:

- model weights or downloaded vendor/runtime artifacts;
- virtual environments, caches, temporary files or generated logs;
- user photographs or private input/output data;
- private regression inputs without documented permission;
- credentials, tokens, private keys or `.env` files;
- machine-specific paths or local state;
- build artifacts, installers or oversized binaries unless a release decision
  explicitly requires them.

Keep `.gitignore`, `SECURITY.md`, the model manifest and third-party notices in
sync with the actual repository contents. Model manifests record provenance and
integrity information; they do not grant permission to redistribute upstream
weights.

## Scope and safety

The ordinary enhancement path is local and non-generative. New creative or
generative capabilities must be explicit, separately accepted and must not
silently replace ordinary enhancement. Operations that can change a person's
identity or reconstruct pixels near a person require an appropriate safety
gate before they can become user-facing functionality.

When evidence is incomplete, report the capability as `NOT VERIFIED` instead of
converting a static check, mock or CPU run into a stronger claim.
