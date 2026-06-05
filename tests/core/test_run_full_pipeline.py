from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WRAPPER = REPO_ROOT / "scripts" / "run_full_pipeline.py"
CANONICAL_DRY_RUN_POINTER = "canonical app dry-run boundary"


def test_legacy_full_pipeline_wrapper_fails_closed():
    completed = subprocess.run(
        [sys.executable, str(LEGACY_WRAPPER)],
        check=False,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert completed.returncode == 2
    assert "Legacy full pipeline execution is disabled." in completed.stdout
    assert CANONICAL_DRY_RUN_POINTER in completed.stdout
    assert completed.stderr == ""


def test_legacy_full_pipeline_wrapper_accepts_old_args_but_still_fails_closed(
    tmp_path: Path,
):
    completed = subprocess.run(
        [
            sys.executable,
            str(LEGACY_WRAPPER),
            "--fundamentals-history-path",
            str(tmp_path / "fundamentals_history.csv"),
            "--fundamental-metrics-output-path",
            str(tmp_path / "fundamental_metrics.csv"),
            "--fundamental-analysis-output-path",
            str(tmp_path / "fundamental_analysis.csv"),
        ],
        check=False,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert completed.returncode == 2
    assert "Legacy full pipeline execution is disabled." in completed.stdout
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_legacy_full_pipeline_wrapper_no_longer_invokes_legacy_scan_script():
    source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "scripts/run_scan.py" not in source
    assert "_build_run_scan_command" not in source


def test_legacy_full_pipeline_wrapper_does_not_expose_production_behavior():
    source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    blocked_terms = {
        "send_daily_summary",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "requests",
        "write_reporting_outputs",
        "build_final_decisions",
        "build_reporting_layer",
    }

    for term in blocked_terms:
        assert term not in source
