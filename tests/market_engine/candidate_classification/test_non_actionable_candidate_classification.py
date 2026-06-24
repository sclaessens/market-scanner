from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from market_engine.candidate_classification import (
    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION,
    CandidateClassificationError,
    CandidateClassificationInput,
    build_candidate_classification_report,
    classify_non_actionable_candidate_from_readable_output,
)
from market_engine.candidate_classification.non_actionable_candidate_classification import (
    run_command,
)


REQUIRED_OUTPUT_FAMILIES = [
    "setup_detection",
    "analysis_review",
    "recommendation_review",
    "portfolio_review",
    "decision_engine_handoff",
    "delivery_reporting",
]


def test_complete_readable_operator_context_produces_review_bucket(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(operator_root, included_tickers=["AMD", "MSFT"])

    result = build_candidate_classification_report(
        input_operator_report_root=operator_root,
        output_root=tmp_path / "reports",
        candidate_classification_run_id="candidate-run",
        generated_at="2026-06-24T13:00:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8")
    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))

    assert result.candidate_classification_format_version == MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
    assert result.classified_tickers == ("AMD", "MSFT")
    assert result.bucket_counts["ready_for_manual_candidate_review"] == 2
    assert summary["candidate_classification_format_version"] == MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
    assert summary["per_ticker_classifications"][0]["candidate_bucket"] == "ready_for_manual_candidate_review"
    assert "Non-actionable candidate classification: `ready_for_manual_candidate_review`" in markdown
    assert "human triage only" in markdown


def test_missing_readable_output_is_insufficient_evidence() -> None:
    classification = classify_non_actionable_candidate_from_readable_output(
        CandidateClassificationInput(
            ticker="AMD",
            operator_report_format_version="market-engine-readable-operator-report-v1",
            run_state=None,
            output_families_present=(),
            missing_data_notes_present=False,
            stale_data_notes_present=False,
            blocked_notes_present=False,
            provenance_references_present=False,
            numeric_zero_evidence_present=False,
            skipped_reason="missing_readable_output",
        )
    )

    assert classification.candidate_bucket == "unclassified_due_to_insufficient_evidence"
    assert "missing_readable_output" in classification.blocking_reasons


def test_incomplete_dry_run_requires_blocked_state_review() -> None:
    classification = classify_non_actionable_candidate_from_readable_output(
        _candidate_input(run_state="dry_run_blocked", blocked_notes_present=True)
    )

    assert classification.candidate_bucket == "requires_blocked_state_review"
    assert "incomplete_dry_run" in classification.blocking_reasons
    assert classification.safety_flags.blocked_state_detected is True


def test_stale_data_notes_require_stale_data_review() -> None:
    classification = classify_non_actionable_candidate_from_readable_output(
        _candidate_input(stale_data_notes_present=True)
    )

    assert classification.candidate_bucket == "requires_stale_data_review"
    assert "stale_data_notes_present" in classification.blocking_reasons
    assert classification.safety_flags.stale_data_detected is True


def test_actionable_language_is_detected_and_unclassified() -> None:
    classification = classify_non_actionable_candidate_from_readable_output(
        _candidate_input(source_text="Operator text says buy now.")
    )

    assert classification.candidate_bucket == "unclassified_due_to_unsupported_input"
    assert "actionable_language_detected" in classification.blocking_reasons
    assert classification.safety_flags.actionable_language_detected is True


def test_format_version_and_evidence_references_are_machine_readable(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(operator_root, included_tickers=["AMD"])

    result = build_candidate_classification_report(
        input_operator_report_root=operator_root,
        output_root=tmp_path / "reports",
        candidate_classification_run_id="evidence",
        generated_at="2026-06-24T13:00:00Z",
    )

    summary = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))
    ticker_payload = summary["per_ticker_classifications"][0]

    assert summary["candidate_classification_format_version"] == MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
    assert ticker_payload["evidence_references"]
    assert ticker_payload["safety_flags"]["actionable_language_detected"] is False


def test_markdown_output_does_not_create_forbidden_action_language(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(operator_root, included_tickers=["AMD"])

    result = build_candidate_classification_report(
        input_operator_report_root=operator_root,
        output_root=tmp_path / "reports",
        candidate_classification_run_id="safe-text",
        generated_at="2026-06-24T13:00:00Z",
    )

    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8").lower()
    for forbidden in (
        "buy",
        "sell",
        "hold",
        "target price",
        "entry price",
        "stop-loss",
        "take-profit",
        "allocation",
        "position size",
        "conviction",
        "urgency",
        "ranking",
        "broker-ready",
        "execution instruction",
    ):
        assert forbidden not in markdown


def test_unsupported_operator_summary_version_is_unclassified(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(
        operator_root,
        included_tickers=["AMD"],
        report_format_version="unsupported",
    )

    result = build_candidate_classification_report(
        input_operator_report_root=operator_root,
        output_root=tmp_path / "reports",
        candidate_classification_run_id="unsupported",
        generated_at="2026-06-24T13:00:00Z",
    )

    assert result.per_ticker_classifications[0].candidate_bucket == "unclassified_due_to_unsupported_input"
    assert result.unclassified_tickers[0]["ticker"] == "UNKNOWN"


def test_missing_operator_summary_fails_closed(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    operator_root.mkdir()

    with pytest.raises(CandidateClassificationError, match="summary is missing"):
        build_candidate_classification_report(
            input_operator_report_root=operator_root,
            output_root=tmp_path / "reports",
            candidate_classification_run_id="missing-summary",
            generated_at="2026-06-24T13:00:00Z",
        )


def test_unsafe_run_id_and_overwrite_fail_closed(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(operator_root, included_tickers=["AMD"])

    with pytest.raises(CandidateClassificationError, match="safe path segment"):
        build_candidate_classification_report(
            input_operator_report_root=operator_root,
            output_root=tmp_path / "reports",
            candidate_classification_run_id="../bad",
            generated_at="2026-06-24T13:00:00Z",
        )

    existing = tmp_path / "reports" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(CandidateClassificationError, match="already exists"):
        build_candidate_classification_report(
            input_operator_report_root=operator_root,
            output_root=tmp_path / "reports",
            candidate_classification_run_id="existing",
            generated_at="2026-06-24T13:00:00Z",
        )


def test_cli_generates_candidate_classification_report(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator"
    _write_operator_report(operator_root, included_tickers=["AMD"])
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_command(
        [
            "--input-operator-report-root",
            str(operator_root),
            "--output-root",
            str(tmp_path / "reports"),
            "--candidate-classification-run-id",
            "cli-candidate",
            "--generated-at",
            "2026-06-24T13:00:00Z",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "candidate_classification_report_path=" in stdout.getvalue()
    assert "summary_json_path=" in stdout.getvalue()


def _candidate_input(
    *,
    run_state: str = "dry_run_completed",
    missing_data_notes_present: bool = False,
    stale_data_notes_present: bool = False,
    blocked_notes_present: bool = False,
    source_text: str = "",
) -> CandidateClassificationInput:
    return CandidateClassificationInput(
        ticker="AMD",
        operator_report_format_version="market-engine-readable-operator-report-v1",
        run_state=run_state,
        output_families_present=tuple(REQUIRED_OUTPUT_FAMILIES),
        missing_data_notes_present=missing_data_notes_present,
        stale_data_notes_present=stale_data_notes_present,
        blocked_notes_present=blocked_notes_present,
        provenance_references_present=True,
        numeric_zero_evidence_present=True,
        source_text=source_text,
    )


def _write_operator_report(
    root: Path,
    *,
    included_tickers: list[str],
    report_format_version: str = "market-engine-readable-operator-report-v1",
) -> None:
    root.mkdir(parents=True)
    (root / "operator_report.md").write_text(
        "# Readable Operator Report\n\nLocal artifact summary for human triage.\n",
        encoding="utf-8",
    )
    payload = {
        "report_format_version": report_format_version,
        "report_run_id": "operator-fixture",
        "generated_at": "2026-06-24T12:00:00Z",
        "input_artifact_root": "/tmp/artifacts",
        "interpretation_report_root": None,
        "included_tickers": included_tickers,
        "skipped_tickers": [],
        "blocked_tickers": [],
        "completed_tickers": included_tickers,
        "output_families_present": REQUIRED_OUTPUT_FAMILIES,
        "missing_data_notes_present": False,
        "stale_data_notes_present": False,
        "blocked_notes_present": False,
        "provenance_references_present": True,
        "numeric_zero_evidence_present": True,
        "non_actionable_boundary": True,
        "advisory_language_guardrail": {
            "forbidden_action_terms_checked": True,
            "operator_report_contains_trading_instruction": False,
        },
    }
    (root / "operator_report_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
