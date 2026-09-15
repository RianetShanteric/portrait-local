# Scripts

Scripted installation, model download, integrity verification and local
validation entry points belong here. Scripts must be safe to run repeatedly.

`INSTALL.cmd` is the supported application installer. `requirements.lock` is a
post-install environment snapshot; CUDA Torch wheels and model artifacts are
resolved separately by the installer rather than by a generic `pip install
-r` path.

- `build_regression_baselines.py` verifies the licensed regression inputs and
  rebuilds the auditable Balanced-profile evidence folders locally.
