from __future__ import annotations

from pathlib import Path

from market_scanner.app import CANONICAL_ENTRYPOINT, main


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "daily-market-scan.yml"
ACTIVE_RUNNER_DIR = REPO_ROOT / "scripts"
ARCHIVED_RUNNER_DIR = REPO_ROOT / "archive" / "legacy_runtime" / "scripts"
LEGACY_RUNNER = ARCHIVED_RUNNER_DIR / "run_scan.py"
LEGACY_WRAPPER = ARCHIVED_RUNNER_DIR / "run_full_pipeline.py"

FORBIDDEN_OPERATOR_TERMS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "urgency",
    "scoring",
    "target-price",
    "tradeability",
    "recommendation",
}


def test_canonical_dry_run_operator_output_is_neutral(capsys):
    exit_code = main(["--dry-run"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Canonical app dry-run completed." in output.out
    assert f"entrypoint={CANONICAL_ENTRYPOINT}" in output.out
    assert "legacy_runners_invoked=False" in output.out
    assert "provider_calls_made=False" in output.out
    assert "production_data_writes=False" in output.out
    assert "reports_generated=False" in output.out
    assert "telegram_artifacts_created=False" in output.out
    assert output.err == ""

    for term in FORBIDDEN_OPERATOR_TERMS:
        assert term not in output.out


def test_canonical_execute_operator_output_fails_closed(capsys):
    exit_code = main(["--execute"])

    output = capsys.readouterr()
    assert exit_code == 2
    assert output.out == ""
    assert "Only dry-run canonical app planning is approved." in output.err


def test_daily_workflow_operator_path_uses_canonical_dry_run_only():
    workflow = WORKFLOW_FILE.read_text(encoding="utf-8")

    assert "PYTHONPATH=src python -m market_scanner.app --dry-run" in workflow
    assert "run_scan.py" not in workflow
    assert "run_full_pipeline.py" not in workflow
    assert "TELEGRAM_BOT_TOKEN" not in workflow
    assert "TELEGRAM_CHAT_ID" not in workflow
    assert "reports/daily/telegram_message.txt" not in workflow
    assert "git add data/" not in workflow


def test_legacy_runtime_scripts_remain_non_canonical_static_targets():
    legacy_runner_source = LEGACY_RUNNER.read_text(encoding="utf-8")
    legacy_wrapper_source = LEGACY_WRAPPER.read_text(encoding="utf-8")

    assert not (ACTIVE_RUNNER_DIR / "run_scan.py").exists()
    assert not (ACTIVE_RUNNER_DIR / "run_full_pipeline.py").exists()
    assert LEGACY_RUNNER.exists()
    assert LEGACY_WRAPPER.exists()
    assert "Pipeline run started: market scan" in legacy_runner_source
    assert "Legacy full pipeline execution is disabled." in legacy_wrapper_source
    assert "canonical app dry-run boundary" in legacy_wrapper_source
    assert "market_scanner.app" not in legacy_wrapper_source
