import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import app.cli as cli
from app.workflow import (
    CANCELLED_RETURN_CODE,
    CancellationRequested,
    cancellation_requested,
    new_cancellation_marker,
    parse_progress_line,
    raise_if_cancelled,
)


def test_cancel_marker_is_cooperative_and_local(tmp_path: Path) -> None:
    marker = tmp_path / "cancel.flag"

    assert not cancellation_requested(marker)
    raise_if_cancelled(marker)

    marker.touch()

    assert cancellation_requested(marker)
    with pytest.raises(CancellationRequested):
        raise_if_cancelled(marker)


def test_cancellation_marker_paths_are_fresh_and_isolated(tmp_path: Path) -> None:
    first = new_cancellation_marker(tmp_path)
    second = new_cancellation_marker(tmp_path)

    assert first != second
    assert first.parent == tmp_path / "runtime" / "temp"
    first.parent.mkdir(parents=True)
    first.touch()
    assert cancellation_requested(first)
    assert not cancellation_requested(second)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[2/6] portrait.jpg", (2, 6, "portrait.jpg")),
        ("  [1/1]  ", (1, 1, "")),
        ("pipeline output", None),
        ("[0/3] not-started.jpg", None),
        ("[4/3] invalid.jpg", None),
        ("[-1/3] invalid.jpg", None),
        ("[1/0] invalid.jpg", None),
        ("[x/3] invalid.jpg", None),
    ],
)
def test_progress_parser_accepts_only_stable_cli_markers(
    line: str, expected: tuple[int, int, str] | None
) -> None:
    assert parse_progress_line(line) == expected


def test_cancellation_return_code_is_distinct_from_processing_failure() -> None:
    assert CANCELLED_RETURN_CODE == 3
    assert CANCELLED_RETURN_CODE not in (0, 1, 2)


def test_cancellation_before_work_returns_without_initializing_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "config.json").write_text(
        json.dumps({"input_extensions": [".jpg"]}), encoding="utf-8"
    )
    input_dir = root / "data" / "input"
    input_dir.mkdir(parents=True)
    source = input_dir / "portrait.jpg"
    source.write_bytes(b"source")
    marker = root / "cancel.flag"
    marker.touch()

    def unexpected_runtime_lookup(_root: Path) -> tuple[Path, Path, Path]:
        raise AssertionError("cancelled startup must not initialize runtime paths")

    monkeypatch.setattr(cli, "runtime_paths", unexpected_runtime_lookup)
    monkeypatch.setattr(
        sys,
        "argv",
        ["portrait-local", "--root", str(root), "--cancel-file", str(marker)],
    )

    assert cli.main() == CANCELLED_RETURN_CODE
    assert source.read_bytes() == b"source"
    assert not list((root / "data" / "output").iterdir())


def test_realesrgan_process_is_terminated_when_marker_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cancel.flag"

    class FakeProcess:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.terminated = False

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                marker.touch()
                raise subprocess.TimeoutExpired("realesrgan", timeout)
            return -15

        def terminate(self) -> None:
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(CancellationRequested):
        cli.run_realesrgan(
            tmp_path / "python.exe",
            tmp_path / "Real-ESRGAN",
            tmp_path / "source.png",
            tmp_path / "output",
            "upscaled",
            0,
            False,
            cancel_file=marker,
        )

    assert process.terminated is True
    assert process.wait_calls == 2


def test_realesrgan_kills_process_when_terminate_does_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cancel.flag"

    class StubbornProcess:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                marker.touch()
            if self.wait_calls <= 2:
                raise subprocess.TimeoutExpired("realesrgan", timeout)
            return -9

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = StubbornProcess()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(CancellationRequested):
        cli.run_realesrgan(
            tmp_path / "python.exe",
            tmp_path / "Real-ESRGAN",
            tmp_path / "source.png",
            tmp_path / "output",
            "upscaled",
            0,
            False,
            cancel_file=marker,
        )

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 3


def test_realesrgan_startup_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_to_start(*args: object, **kwargs: object) -> None:
        raise OSError("cannot start Real-ESRGAN")

    monkeypatch.setattr(cli.subprocess, "Popen", fail_to_start)

    with pytest.raises(OSError, match="cannot start Real-ESRGAN"):
        cli.run_realesrgan(
            tmp_path / "python.exe",
            tmp_path / "Real-ESRGAN",
            tmp_path / "source.png",
            tmp_path / "output",
            "upscaled",
            0,
            False,
        )


def test_realesrgan_nonzero_exit_remains_a_processing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedProcess:
        def wait(self, timeout: float | None = None) -> int:
            return 1

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())

    with pytest.raises(cli.PipelineError, match="Real-ESRGAN/GFPGAN"):
        cli.run_realesrgan(
            tmp_path / "python.exe",
            tmp_path / "Real-ESRGAN",
            tmp_path / "source.png",
            tmp_path / "output",
            "upscaled",
            0,
            False,
        )


def test_realesrgan_cancellation_wins_over_a_simultaneous_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cancel.flag"

    class FailedProcess:
        def wait(self, timeout: float | None = None) -> int:
            marker.touch()
            return 1

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())

    with pytest.raises(CancellationRequested):
        cli.run_realesrgan(
            tmp_path / "python.exe",
            tmp_path / "Real-ESRGAN",
            tmp_path / "source.png",
            tmp_path / "output",
            "upscaled",
            0,
            False,
            cancel_file=marker,
        )


def test_cancellation_records_mark_current_and_remaining_inputs() -> None:
    inputs = [Path("first.jpg"), Path("second.jpg"), Path("third.jpg")]

    records = cli.cancellation_records(inputs, 1, "user requested stop")

    assert records == [
        {
            "input": "second.jpg",
            "output": "",
            "status": "CANCELLED",
            "note": "user requested stop",
        },
        {
            "input": "third.jpg",
            "output": "",
            "status": "CANCELLED",
            "note": "Not started because processing was cancelled.",
        },
    ]


def test_cancellation_writes_truthful_status_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "config.json").write_text(
        json.dumps(
            {
                "input_extensions": [".jpg"],
                "output": {"keep_work_files": False},
            }
        ),
        encoding="utf-8",
    )
    input_dir = root / "data" / "input"
    input_dir.mkdir(parents=True)
    first = input_dir / "first.jpg"
    second = input_dir / "second.jpg"
    first.write_bytes(b"first input")
    second.write_bytes(b"second input")
    marker = root / "cancel.flag"

    class FakeEngine:
        def analyze(self, _rgb: np.ndarray) -> None:
            marker.touch()
            return None

    monkeypatch.setattr(
        cli,
        "runtime_paths",
        lambda _root: (Path("python.exe"), Path("Real-ESRGAN"), Path("face.onnx")),
    )
    monkeypatch.setattr(cli, "SmartPhotoPipeline", lambda _root, _config: FakeEngine())
    monkeypatch.setattr(cli, "load_rgb", lambda _path: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(
        sys,
        "argv",
        ["portrait-local", "--root", str(root), "--cancel-file", str(marker)],
    )

    assert cli.main() == CANCELLED_RETURN_CODE
    assert first.read_bytes() == b"first input"
    assert second.read_bytes() == b"second input"

    output_runs = sorted((root / "data" / "output").iterdir())
    assert len(output_runs) == 1
    with (output_runs[0] / "processing_log.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["status"] for row in rows] == ["CANCELLED", "CANCELLED"]
    assert rows[0]["input"] == "first.jpg"
    assert rows[1]["note"] == "Not started because processing was cancelled."
    assert not list(output_runs[0].glob("*.jpg"))
    assert list((root / "runtime" / "temp").glob("run_*"))


def test_retry_primitives_create_fresh_isolated_run_directories(tmp_path: Path) -> None:
    first = cli.unique_run_directory(tmp_path)
    second = cli.unique_run_directory(tmp_path)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.parent == second.parent == tmp_path
