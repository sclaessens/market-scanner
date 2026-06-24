from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from market_engine.output_reports import (
    MARKET_ENGINE_READABLE_OPERATOR_REPORT_FORMAT_VERSION,
    ReadableOperatorReportError,
    build_readable_operator_report,
)
from market_engine.output_reports.readable_operator_report import run_command


REQUIRED_MARKDOWN_SECTIONS = (
    "## 1. Report Metadata",
    "## 2. Source Artifact Boundary",
    "## 3. Non-Actionable Boundary",
    "## 4. Universe Coverage",
    "## 5. Artifact Integrity Summary",
    "## 6. Stage Completion Summary",
    "## 7. Per-Ticker Operator Summaries",
    "## 8. Missing-Data And Stale-Data Notes",
    "## 9. Blocked And Skipped Ticker Notes",
    "## 10. Provenance Summary",
    "## 11. Human-Review Checklist",
    "## 12. Safe Next-Step Candidate",
    "## 13. Appendix: Machine-Readable Summary Reference",
)


def test_happy_path_generates_operator_markdown_and_summary_for_multiple_tickers(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "MSFT")
    _write_ticker_artifact(input_root, "AMD")

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="operator-report",
        generated_at="2026-06-24T12:00:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8")
    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))

    assert result.report_format_version == MARKET_ENGINE_READABLE_OPERATOR_REPORT_FORMAT_VERSION
    assert result.included_tickers == ("AMD", "MSFT")
    assert result.completed_tickers == ("AMD", "MSFT")
    assert result.skipped_tickers == ()
    assert summary["report_format_version"] == MARKET_ENGINE_READABLE_OPERATOR_REPORT_FORMAT_VERSION
    assert summary["included_tickers"] == ["AMD", "MSFT"]
    assert summary["completed_tickers"] == ["AMD", "MSFT"]
    assert summary["non_actionable_boundary"] is True
    assert summary["advisory_language_guardrail"] == {
        "forbidden_action_terms_checked": True,
        "operator_report_contains_trading_instruction": False,
    }
    for section in REQUIRED_MARKDOWN_SECTIONS:
        assert section in markdown


def test_ticker_ordering_is_deterministic(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "NVDA")
    _write_ticker_artifact(input_root, "AMD")
    _write_ticker_artifact(input_root, "MSFT")

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="ordered",
        generated_at="2026-06-24T12:00:00Z",
    )

    assert result.included_tickers == ("AMD", "MSFT", "NVDA")


def test_missing_manifest_produces_explicit_skipped_ticker_reason(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD", include_manifest=False)

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="missing-manifest",
        generated_at="2026-06-24T12:00:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers == ({"ticker": "AMD", "reason": "manifest.json is missing"},)
    assert result.artifact_missing_tickers == (
        {"ticker": "AMD", "reason": "manifest.json is missing"},
    )


def test_missing_dry_run_produces_explicit_skipped_ticker_reason(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD", include_dry_run=False)

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="missing-dry-run",
        generated_at="2026-06-24T12:00:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers == ({"ticker": "AMD", "reason": "dry_run.json is missing"},)


def test_malformed_json_produces_explicit_skipped_ticker_reason(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    ticker_dir = input_root / "AMD"
    ticker_dir.mkdir(parents=True)
    (ticker_dir / "dry_run.json").write_text("{not json", encoding="utf-8")
    (ticker_dir / "manifest.json").write_text(json.dumps(_manifest_payload("AMD")), encoding="utf-8")

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="malformed-json",
        generated_at="2026-06-24T12:00:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers[0]["ticker"] == "AMD"
    assert "invalid JSON" in result.skipped_tickers[0]["reason"]
    assert result.malformed_artifact_tickers[0]["ticker"] == "AMD"


def test_unsupported_format_version_skips_ticker_explicitly(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD", artifact_format_version="unsupported")

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="unsupported",
        generated_at="2026-06-24T12:00:00Z",
    )

    assert result.included_tickers == ()
    assert result.skipped_tickers == (
        {"ticker": "AMD", "reason": "unsupported dry_run artifact format version"},
    )


def test_missing_input_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReadableOperatorReportError, match="does not exist"):
        build_readable_operator_report(
            input_artifact_root=tmp_path / "missing",
            output_root=tmp_path / "reports",
            report_run_id="missing-root",
            generated_at="2026-06-24T12:00:00Z",
        )


def test_unsafe_report_run_id_fails_closed(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD")

    with pytest.raises(ReadableOperatorReportError, match="safe path segment"):
        build_readable_operator_report(
            input_artifact_root=input_root,
            output_root=tmp_path / "reports",
            report_run_id="../bad",
            generated_at="2026-06-24T12:00:00Z",
        )


def test_output_overwrite_is_refused(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD")
    output_dir = tmp_path / "reports" / "existing-report"
    output_dir.mkdir(parents=True)

    with pytest.raises(ReadableOperatorReportError, match="already exists"):
        build_readable_operator_report(
            input_artifact_root=input_root,
            output_root=tmp_path / "reports",
            report_run_id="existing-report",
            generated_at="2026-06-24T12:00:00Z",
        )


def test_missing_stale_blocked_provenance_and_zero_markers_are_preserved(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(
        input_root,
        "AMD",
        run_state="dry_run_blocked",
        blocked_stage="source_context",
        blocked_reasons=["source_context_missing"],
        missing_data_summary=["revenue"],
        stale_data_summary=["source_snapshot_stale"],
        include_numeric_zero=True,
    )

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="markers",
        generated_at="2026-06-24T12:00:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8")
    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))

    assert result.blocked_tickers[0]["ticker"] == "AMD"
    assert result.missing_data_notes_present is True
    assert result.stale_data_notes_present is True
    assert result.provenance_references_present is True
    assert "source_context_missing" in markdown
    assert "source_snapshot_stale" in markdown
    assert "Numeric zero values preserved: `yes`" in markdown
    assert summary["blocked_notes_present"] is True


def test_normal_report_text_does_not_create_forbidden_advisory_language(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _write_ticker_artifact(input_root, "AMD")

    result = build_readable_operator_report(
        input_artifact_root=input_root,
        output_root=tmp_path / "reports",
        report_run_id="guardrail",
        generated_at="2026-06-24T12:00:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8").lower()
    for forbidden in (
        "buy",
        "sell",
        "hold",
        "entry price",
        "exit price",
        "target price",
        "target weight",
        "position size",
        "allocation",
        "conviction",
        "urgency",
        "tradeability",
        "ranking",
        "score",
        "broker-ready",
        "execution instruction",
        "delivery instruction",
    ):
        assert forbidden not in markdown


def test_cli_report_generation_works(tmp_path: Path) -> None:
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
            "2026-06-24T12:00:00Z",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "operator_report_path=" in stdout.getvalue()
    assert "summary_json_path=" in stdout.getvalue()


def _write_ticker_artifact(
    root: Path,
    ticker: str,
    *,
    include_dry_run: bool = True,
    include_manifest: bool = True,
    artifact_format_version: str = "market-engine-local-dry-run-artifact-v1",
    run_state: str = "dry_run_completed",
    blocked_stage: str | None = None,
    blocked_reasons: list[str] | None = None,
    missing_data_summary: list[str] | None = None,
    stale_data_summary: list[str] | None = None,
    include_numeric_zero: bool = False,
) -> None:
    ticker_dir = root / ticker
    ticker_dir.mkdir(parents=True)
    if include_dry_run:
        (ticker_dir / "dry_run.json").write_text(
            json.dumps(
                _dry_run_artifact(
                    ticker,
                    artifact_format_version=artifact_format_version,
                    run_state=run_state,
                    blocked_stage=blocked_stage,
                    blocked_reasons=blocked_reasons or [],
                    missing_data_summary=missing_data_summary or [],
                    stale_data_summary=stale_data_summary or [],
                    include_numeric_zero=include_numeric_zero,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if include_manifest:
        (ticker_dir / "manifest.json").write_text(
            json.dumps(_manifest_payload(ticker, run_state=run_state), indent=2),
            encoding="utf-8",
        )


def _dry_run_artifact(
    ticker: str,
    *,
    artifact_format_version: str,
    run_state: str,
    blocked_stage: str | None,
    blocked_reasons: list[str],
    missing_data_summary: list[str],
    stale_data_summary: list[str],
    include_numeric_zero: bool,
) -> dict[str, object]:
    return {
        "artifact_format_version": artifact_format_version,
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": "2026-06-24T12:00:00Z",
        "non_production_artifact": True,
        "source_dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
        "source_dry_run_id": f"fixture-{ticker.lower()}",
        "source_input_mode": "cached_source_snapshot",
        "source_run_state": run_state,
        "payload": {
            "dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
            "dry_run_id": f"fixture-{ticker.lower()}",
            "generated_at": "2026-06-24T12:00:00Z",
            "input_mode": "cached_source_snapshot",
            "ticker": ticker,
            "run_state": run_state,
            "blocked_stage": blocked_stage,
            "blocked_reasons": blocked_reasons,
            "missing_data_summary": missing_data_summary,
            "stale_data_summary": stale_data_summary,
            "stage_results": [
                {"stage_name": "source_context", "status": "completed"},
                {"stage_name": "fundamental_observations", "status": "completed"},
                {
                    "stage_name": "portfolio_review",
                    "status": "blocked" if blocked_stage else "completed",
                },
            ],
            "portfolio_context_reference": {
                "cash": 0 if include_numeric_zero else 100,
            },
            "delivery_report_reference": {
                "cached_source_reference": {
                    "source_snapshot_reference": f"sec_companyfacts/run/raw/{ticker}_companyfacts.json",
                    "source_snapshot_path": f"/tmp/{ticker}_companyfacts.json",
                    "source_snapshot_root": "/tmp",
                }
            },
        },
    }


def _manifest_payload(ticker: str, *, run_state: str = "dry_run_completed") -> dict[str, object]:
    return {
        "manifest_format_version": "market-engine-local-dry-run-artifact-manifest-v1",
        "artifact_count": 1,
        "artifact_created_at": "2026-06-24T12:00:00Z",
        "non_production_artifact": True,
        "source_dry_run_format_version": "market-engine-end-to-end-dry-run-v1",
        "source_dry_run_id": f"fixture-{ticker.lower()}",
        "source_input_mode": "cached_source_snapshot",
        "source_run_state": run_state,
        "artifacts": [{"artifact_relative_path": "dry_run.json"}],
    }
