from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from market_engine.run_reports import (
    MARKET_ENGINE_INTERPRETATION_REPORT_FORMAT_VERSION,
    build_market_engine_interpretation_report,
)
from market_engine.run_reports.interpretation_report import run_command


def test_happy_path_generates_markdown_and_summary_for_multiple_tickers(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "MSFT")
    _write_ticker_artifact(input_root, "AMD")

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="run22-report",
        generated_at="2026-06-23T12:30:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8")
    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))

    assert result.report_format_version == MARKET_ENGINE_INTERPRETATION_REPORT_FORMAT_VERSION
    assert result.included_tickers == ("AMD", "MSFT")
    assert result.skipped_tickers == ()
    assert "# Market Engine Interpretation Report" in markdown
    assert "## Scope And Safety" in markdown
    assert "## Per-Ticker Sections" in markdown
    assert "### AMD" in markdown
    assert "### MSFT" in markdown
    assert summary["included_tickers"] == ["AMD", "MSFT"]
    assert summary["parsed_ticker_count"] == 2
    assert summary["report_format_version"] == MARKET_ENGINE_INTERPRETATION_REPORT_FORMAT_VERSION


def test_missing_manifest_is_reported_as_skipped(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD", include_manifest=False)

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="missing-manifest",
        generated_at="2026-06-23T12:30:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers == ({"ticker": "AMD", "reason": "manifest.json is missing"},)


def test_missing_dry_run_is_reported_as_skipped(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD", include_dry_run=False)

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="missing-dry-run",
        generated_at="2026-06-23T12:30:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers == ({"ticker": "AMD", "reason": "dry_run.json is missing"},)


def test_malformed_json_is_reported_as_skipped(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    ticker_dir = input_root / "AMD"
    ticker_dir.mkdir(parents=True)
    (ticker_dir / "dry_run.json").write_text("{not json", encoding="utf-8")
    (ticker_dir / "manifest.json").write_text(json.dumps(_manifest_payload("AMD")), encoding="utf-8")

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="malformed-json",
        generated_at="2026-06-23T12:30:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers[0]["ticker"] == "AMD"
    assert "invalid JSON" in result.skipped_tickers[0]["reason"]


def test_ticker_ordering_is_deterministic(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "NVDA")
    _write_ticker_artifact(input_root, "AMD")
    _write_ticker_artifact(input_root, "MSFT")

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="ordered",
        generated_at="2026-06-23T12:30:00Z",
    )

    assert result.included_tickers == ("AMD", "MSFT", "NVDA")


def test_markdown_avoids_advisory_language_while_summary_keeps_guardrail_metadata(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD")

    result = build_market_engine_interpretation_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="guardrails",
        generated_at="2026-06-23T12:30:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8").lower()
    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))

    for forbidden in ("buy", "sell", "hold", "target price", "stop-loss", "take-profit"):
        assert forbidden not in markdown
    assert "advisory_language_guardrail" in summary
    assert "buy" in summary["advisory_language_guardrail"]


def test_cli_writes_report_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD")
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_command(
        [
            "--input-artifact-root",
            str(input_root),
            "--output-root",
            str(tmp_path / "reports"),
            "--report-run-id",
            "cli-report",
            "--generated-at",
            "2026-06-23T12:30:00Z",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "markdown_report_path=" in stdout.getvalue()
    assert "summary_json_path=" in stdout.getvalue()


def _write_ticker_artifact(
    root: Path,
    ticker: str,
    *,
    include_dry_run: bool = True,
    include_manifest: bool = True,
) -> None:
    ticker_dir = root / ticker
    ticker_dir.mkdir(parents=True)
    if include_dry_run:
        (ticker_dir / "dry_run.json").write_text(
            json.dumps(_dry_run_artifact(ticker), indent=2),
            encoding="utf-8",
        )
    if include_manifest:
        (ticker_dir / "manifest.json").write_text(
            json.dumps(_manifest_payload(ticker), indent=2),
            encoding="utf-8",
        )


def _dry_run_artifact(ticker: str) -> dict[str, object]:
    return {
        "artifact_format_version": "market-engine-local-dry-run-artifact-v1",
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": "2026-06-23T12:30:00Z",
        "non_production_artifact": True,
        "source_dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
        "source_dry_run_id": f"fixture-{ticker.lower()}",
        "source_input_mode": "cached_source_snapshot",
        "source_run_state": "dry_run_completed",
        "payload": {
            "dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
            "dry_run_id": f"fixture-{ticker.lower()}",
            "input_mode": "cached_source_snapshot",
            "ticker": ticker,
            "run_state": "dry_run_completed",
            "blocked_stage": None,
            "missing_data_summary": [],
            "stale_data_summary": [],
            "stage_results": [
                {"stage_name": "source_context", "status": "completed"},
                {"stage_name": "fundamental_observations", "status": "completed"},
                {"stage_name": "delivery_reporting", "status": "completed"},
            ],
            "delivery_report_reference": {
                "cached_source_reference": {
                    "source_snapshot_reference": f"sec_companyfacts/run/raw/{ticker}_companyfacts.json",
                    "source_snapshot_path": f"/tmp/{ticker}_companyfacts.json",
                    "source_snapshot_root": "/tmp",
                }
            },
        },
    }


def _manifest_payload(ticker: str) -> dict[str, object]:
    return {
        "manifest_format_version": "market-engine-local-dry-run-artifact-manifest-v1",
        "artifact_count": 1,
        "artifact_created_at": "2026-06-23T12:30:00Z",
        "non_production_artifact": True,
        "source_dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
        "source_dry_run_id": f"fixture-{ticker.lower()}",
        "source_input_mode": "cached_source_snapshot",
        "source_run_state": "dry_run_completed",
        "artifacts": [{"artifact_relative_path": "dry_run.json"}],
    }
