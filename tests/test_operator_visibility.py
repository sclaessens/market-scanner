from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import run_full_pipeline, run_scan


def test_run_full_pipeline_step_prints_neutral_success(monkeypatch, capsys):
    calls = []

    def fake_run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_full_pipeline.subprocess, "run", fake_run)

    run_full_pipeline.run_step("Example step", ["python", "example.py"])

    output = capsys.readouterr().out
    assert calls == [["python", "example.py"]]
    assert "Pipeline step started: Example step" in output
    assert "Command: python example.py" in output
    assert "Pipeline step completed: Example step" in output


def test_run_full_pipeline_step_prints_failure_context(monkeypatch, capsys):
    def fake_run(command):
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_full_pipeline.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_full_pipeline.run_step("Example step", ["python", "example.py"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 7
    assert "Pipeline step failed: Example step" in output
    assert "Return code: 7" in output


def test_scan_artifact_message_includes_optional_row_count():
    assert (
        run_scan.format_artifact_message(run_scan.SCANNER_RANKED_FILE, row_count=12)
        == f"Artifact written: {run_scan.SCANNER_RANKED_FILE} rows=12"
    )
    assert (
        run_scan.format_artifact_message(run_scan.TELEGRAM_MESSAGE_FILE)
        == f"Artifact written: {run_scan.TELEGRAM_MESSAGE_FILE}"
    )


def test_scan_progress_uses_neutral_operational_language(capsys):
    run_scan.print_scan_progress(
        processed_count=25,
        total_count=100,
        setup_count=4,
        failed_count=1,
    )

    output = capsys.readouterr().out
    assert "Scanner progress: processed=25/100" in output
    assert "setup_rows_collected=4" in output
    assert "failed_rows=1" in output
    assert "tradeable" not in output.lower()
    assert "conviction" not in output.lower()
    assert "urgency" not in output.lower()
