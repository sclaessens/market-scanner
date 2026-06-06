from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUNNER_DIR = REPO_ROOT / "scripts"
ARCHIVED_RUNNER_DIR = REPO_ROOT / "archive" / "legacy_runtime" / "scripts"
LEGACY_WRAPPER = ARCHIVED_RUNNER_DIR / "run_full_pipeline.py"
LEGACY_SCAN_SCRIPT = ARCHIVED_RUNNER_DIR / "run_scan.py"
CANONICAL_DRY_RUN_POINTER = "canonical app dry-run boundary"


def test_archived_full_pipeline_wrapper_is_static_fail_closed_reference():
    source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    assert "Legacy full pipeline execution is disabled." in source
    assert CANONICAL_DRY_RUN_POINTER in source
    assert "return 2" in source
    assert "if __name__ == \"__main__\":" in source
    assert "raise SystemExit(main(sys.argv[1:]))" in source


def test_archived_full_pipeline_wrapper_preserves_legacy_arg_names_statically():
    source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    assert "--fundamentals-history-path" in source
    assert "--fundamental-metrics-output-path" in source
    assert "--fundamental-analysis-output-path" in source
    assert "parse_args(" in source


def test_legacy_full_pipeline_wrapper_no_longer_invokes_legacy_scan_script():
    source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    assert "_build_run_scan_command" not in source
    assert "run_scan.py" not in source


def test_legacy_runtime_scripts_are_archived_not_active():
    assert not (ACTIVE_RUNNER_DIR / "run_scan.py").exists()
    assert not (ACTIVE_RUNNER_DIR / "run_full_pipeline.py").exists()
    assert LEGACY_SCAN_SCRIPT.exists()
    assert LEGACY_WRAPPER.exists()


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
