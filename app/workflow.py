"""Shared cooperative workflow-control primitives for the local app."""

from __future__ import annotations

import re
import uuid
from pathlib import Path


CANCELLED_RETURN_CODE = 3


class CancellationRequested(RuntimeError):
    """The user asked the current local run to stop."""


def new_cancellation_marker(root: Path) -> Path:
    """Return a fresh per-run marker path under the local runtime directory."""
    return root / "runtime" / "temp" / f"ui_cancel_{uuid.uuid4().hex}.flag"


def cancellation_requested(cancel_file: Path | None) -> bool:
    """Return whether a cooperating caller has requested cancellation."""
    return cancel_file is not None and cancel_file.is_file()


def raise_if_cancelled(cancel_file: Path | None) -> None:
    if cancellation_requested(cancel_file):
        raise CancellationRequested("Обработка остановлена пользователем.")


def parse_progress_line(line: str) -> tuple[int, int, str] | None:
    """Parse the CLI's stable per-input progress marker for the UI."""
    match = re.match(r"^\[(\d+)/(\d+)\]\s*(.*)$", line.strip())
    if not match:
        return None
    current, total = int(match.group(1)), int(match.group(2))
    if current < 1 or total < 1 or current > total:
        return None
    return current, total, match.group(3).strip()
