# Data

Runtime input and output directories are created under this location when the
application runs. Their contents are intentionally excluded from Git.
Any per-user recipes or styles kept under `data/recipes` or `data/styles` are
runtime data and are excluded as well.

Original files are read-only inputs and must never be overwritten or deleted.
