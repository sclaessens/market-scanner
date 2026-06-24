from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION = (
    "market-engine-candidate-classification-v1"
)
EXPECTED_OPERATOR_REPORT_FORMAT_VERSION = (
    "market-engine-readable-operator-report-v1"
)
DEFAULT_CANDIDATE_CLASSIFICATION_OUTPUT_ROOT = Path("artifacts/market_engine")
CANDIDATE_CLASSIFICATION_MARKDOWN_FILENAME = "candidate_classification_report.md"
CANDIDATE_CLASSIFICATION_SUMMARY_FILENAME = "candidate_classification_summary.json"
NON_ACTIONABLE_BOUNDARY = (
    "Candidate classification is for human review triage only and does not "
    "authorize market participation, capital guidance, external system "
    "actions, portfolio mutation, or watchlist mutation."
)

ALLOWED_CANDIDATE_BUCKETS = (
    "ready_for_manual_candidate_review",
    "requires_missing_data_review",
    "requires_stale_data_review",
    "requires_blocked_state_review",
    "requires_source_coverage_review",
    "requires_portfolio_context_review",
    "requires_human_interpretation_review",
    "unclassified_due_to_malformed_artifact",
    "unclassified_due_to_unsupported_input",
    "unclassified_due_to_insufficient_evidence",
)

FORBIDDEN_ACTION_TERMS = (
    "buy",
    "sell",
    "hold",
    "koop",
    "verkoop",
    "houden",
    "target price",
    "entry price",
    "exit price",
    "stop-loss",
    "take-profit",
    "allocation",
    "position size",
    "conviction score",
    "urgency",
    "ranking",
    "order advice",
    "order generation",
    "broker-ready order",
    "koopwaardig",
    "nu kopen",
    "breakout kopen",
    "instappen op",
)

REQUIRED_OUTPUT_FAMILIES = (
    "setup_detection",
    "analysis_review",
    "recommendation_review",
    "portfolio_review",
    "decision_engine_handoff",
    "delivery_reporting",
)


class CandidateClassificationError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateEvidenceReference:
    reference_type: str
    reference: str


@dataclass(frozen=True)
class CandidateSafetyFlags:
    actionable_language_detected: bool
    unsupported_input_detected: bool
    malformed_input_detected: bool
    stale_data_detected: bool
    blocked_state_detected: bool


@dataclass(frozen=True)
class CandidateClassificationInput:
    ticker: str
    operator_report_format_version: str | None
    run_state: str | None
    output_families_present: tuple[str, ...]
    missing_data_notes_present: bool
    stale_data_notes_present: bool
    blocked_notes_present: bool
    provenance_references_present: bool
    numeric_zero_evidence_present: bool
    skipped_reason: str | None = None
    malformed_artifact: bool = False
    unsupported_input: bool = False
    source_text: str = ""


@dataclass(frozen=True)
class CandidateTickerClassification:
    ticker: str
    candidate_bucket: str
    candidate_rationale: str
    evidence_references: tuple[CandidateEvidenceReference, ...]
    blocking_reasons: tuple[str, ...]
    safety_flags: CandidateSafetyFlags


@dataclass(frozen=True)
class CandidateClassificationReportResult:
    candidate_classification_format_version: str
    candidate_classification_run_id: str
    generated_at: str | None
    input_operator_report_root: str | None
    input_artifact_root: str | None
    input_interpretation_report_root: str | None
    output_directory: str
    markdown_report_path: str
    summary_json_path: str
    included_tickers: tuple[str, ...]
    classified_tickers: tuple[str, ...]
    unclassified_tickers: tuple[dict[str, str], ...]
    bucket_counts: dict[str, int]
    per_ticker_classifications: tuple[CandidateTickerClassification, ...]
    missing_data_notes_present: bool
    stale_data_notes_present: bool
    blocked_notes_present: bool
    provenance_references_present: bool
    numeric_zero_evidence_present: bool
    non_actionable_boundary: bool
    advisory_language_guardrail: dict[str, bool]

    def to_summary_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["per_ticker_classifications"] = [
            _classification_payload(item)
            for item in self.per_ticker_classifications
        ]
        return payload


def classify_non_actionable_candidate_from_readable_output(
    candidate_input: CandidateClassificationInput,
) -> CandidateTickerClassification:
    blocking_reasons: list[str] = []
    evidence_references = _evidence_references(candidate_input)
    actionable_language_detected = _contains_forbidden_action_language(
        candidate_input.source_text,
    )

    if actionable_language_detected:
        blocking_reasons.append("actionable_language_detected")
    if candidate_input.malformed_artifact:
        blocking_reasons.append("malformed_artifact")
    if candidate_input.unsupported_input:
        blocking_reasons.append("contract_version_mismatch")
    if candidate_input.skipped_reason:
        blocking_reasons.append(candidate_input.skipped_reason)
    if candidate_input.operator_report_format_version not in {
        EXPECTED_OPERATOR_REPORT_FORMAT_VERSION,
        None,
    }:
        blocking_reasons.append("operator_report_contract_version_mismatch")
    if candidate_input.run_state and candidate_input.run_state not in {
        "dry_run_completed",
        "completed",
    }:
        blocking_reasons.append("incomplete_dry_run")
    if candidate_input.missing_data_notes_present:
        blocking_reasons.append("missing_data_notes_present")
    if candidate_input.stale_data_notes_present:
        blocking_reasons.append("stale_data_notes_present")
    if candidate_input.blocked_notes_present:
        blocking_reasons.append("blocked_notes_present")

    missing_families = tuple(
        family
        for family in REQUIRED_OUTPUT_FAMILIES
        if family not in set(candidate_input.output_families_present)
    )
    if missing_families:
        blocking_reasons.extend(f"missing_{family}" for family in missing_families)

    bucket = _bucket_for_reasons(blocking_reasons, candidate_input)
    return CandidateTickerClassification(
        ticker=candidate_input.ticker,
        candidate_bucket=bucket,
        candidate_rationale=_rationale_for_bucket(bucket),
        evidence_references=evidence_references,
        blocking_reasons=tuple(sorted(set(blocking_reasons))),
        safety_flags=CandidateSafetyFlags(
            actionable_language_detected=actionable_language_detected,
            unsupported_input_detected=(
                candidate_input.unsupported_input
                or "contract_version_mismatch" in blocking_reasons
                or "operator_report_contract_version_mismatch" in blocking_reasons
            ),
            malformed_input_detected=candidate_input.malformed_artifact,
            stale_data_detected=candidate_input.stale_data_notes_present,
            blocked_state_detected=(
                candidate_input.blocked_notes_present
                or "incomplete_dry_run" in blocking_reasons
            ),
        ),
    )


def build_candidate_classification_report(
    *,
    input_operator_report_root: str | Path,
    output_root: str | Path = DEFAULT_CANDIDATE_CLASSIFICATION_OUTPUT_ROOT,
    candidate_classification_run_id: str,
    generated_at: str | None = None,
    input_artifact_root: str | Path | None = None,
    input_interpretation_report_root: str | Path | None = None,
) -> CandidateClassificationReportResult:
    operator_root = _validated_input_root(Path(input_operator_report_root))
    artifact_root = _validated_optional_root(input_artifact_root)
    interpretation_root = _validated_optional_root(input_interpretation_report_root)
    safe_run_id = _safe_path_segment(
        candidate_classification_run_id,
        field_name="candidate_classification_run_id",
    )
    output_directory = _prepare_output_directory(Path(output_root), safe_run_id)
    operator_summary = _read_operator_summary(operator_root)
    operator_markdown = _read_optional_text(operator_root / "operator_report.md")
    classifications = tuple(
        classify_non_actionable_candidate_from_readable_output(item)
        for item in _inputs_from_operator_summary(
            operator_summary,
            operator_markdown=operator_markdown,
        )
    )
    bucket_counts = {
        bucket: sum(1 for item in classifications if item.candidate_bucket == bucket)
        for bucket in ALLOWED_CANDIDATE_BUCKETS
    }
    unclassified = tuple(
        {
            "ticker": item.ticker,
            "candidate_bucket": item.candidate_bucket,
            "blocking_reasons": ",".join(item.blocking_reasons),
        }
        for item in classifications
        if item.candidate_bucket.startswith("unclassified")
    )
    markdown_path = output_directory / CANDIDATE_CLASSIFICATION_MARKDOWN_FILENAME
    summary_path = output_directory / CANDIDATE_CLASSIFICATION_SUMMARY_FILENAME
    result = CandidateClassificationReportResult(
        candidate_classification_format_version=(
            MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
        ),
        candidate_classification_run_id=safe_run_id,
        generated_at=generated_at,
        input_operator_report_root=operator_root.as_posix(),
        input_artifact_root=artifact_root.as_posix() if artifact_root else None,
        input_interpretation_report_root=(
            interpretation_root.as_posix() if interpretation_root else None
        ),
        output_directory=output_directory.as_posix(),
        markdown_report_path=markdown_path.as_posix(),
        summary_json_path=summary_path.as_posix(),
        included_tickers=tuple(item.ticker for item in classifications),
        classified_tickers=tuple(
            item.ticker
            for item in classifications
            if not item.candidate_bucket.startswith("unclassified")
        ),
        unclassified_tickers=unclassified,
        bucket_counts=bucket_counts,
        per_ticker_classifications=classifications,
        missing_data_notes_present=any(
            "missing_data_notes_present" in item.blocking_reasons
            for item in classifications
        ),
        stale_data_notes_present=any(
            item.safety_flags.stale_data_detected for item in classifications
        ),
        blocked_notes_present=any(
            item.safety_flags.blocked_state_detected for item in classifications
        ),
        provenance_references_present=any(
            item.evidence_references for item in classifications
        ),
        numeric_zero_evidence_present=bool(
            operator_summary.get("numeric_zero_evidence_present")
        ),
        non_actionable_boundary=True,
        advisory_language_guardrail={
            "forbidden_action_terms_checked": True,
            "candidate_classification_contains_trading_instruction": False,
        },
    )
    _write_text(markdown_path, _render_markdown_report(result))
    _write_json(summary_path, result.to_summary_payload())
    return result


def _inputs_from_operator_summary(
    summary: Mapping[str, Any],
    *,
    operator_markdown: str,
) -> tuple[CandidateClassificationInput, ...]:
    format_version = _optional_text(summary.get("report_format_version"))
    if format_version != EXPECTED_OPERATOR_REPORT_FORMAT_VERSION:
        return (
            CandidateClassificationInput(
                ticker="UNKNOWN",
                operator_report_format_version=format_version,
                run_state=None,
                output_families_present=(),
                missing_data_notes_present=False,
                stale_data_notes_present=False,
                blocked_notes_present=False,
                provenance_references_present=False,
                numeric_zero_evidence_present=False,
                unsupported_input=True,
                source_text=operator_markdown + json.dumps(summary, sort_keys=True),
            ),
        )

    included = tuple(str(ticker) for ticker in summary.get("included_tickers") or ())
    skipped = summary.get("skipped_tickers") or ()
    blocked = summary.get("blocked_tickers") or ()
    completed = set(str(ticker) for ticker in summary.get("completed_tickers") or ())
    output_families = tuple(str(item) for item in summary.get("output_families_present") or ())
    inputs: list[CandidateClassificationInput] = []
    for ticker in sorted(included):
        ticker_blocked = _ticker_in_records(ticker, blocked)
        inputs.append(
            CandidateClassificationInput(
                ticker=ticker,
                operator_report_format_version=format_version,
                run_state="dry_run_completed" if ticker in completed else "not_completed",
                output_families_present=output_families,
                missing_data_notes_present=bool(summary.get("missing_data_notes_present")),
                stale_data_notes_present=bool(summary.get("stale_data_notes_present")),
                blocked_notes_present=bool(summary.get("blocked_notes_present")) or ticker_blocked,
                provenance_references_present=bool(
                    summary.get("provenance_references_present")
                ),
                numeric_zero_evidence_present=bool(
                    summary.get("numeric_zero_evidence_present")
                ),
                source_text=operator_markdown,
            )
        )
    for item in sorted(
        (record for record in skipped if isinstance(record, Mapping)),
        key=lambda record: str(record.get("ticker", "")),
    ):
        inputs.append(
            CandidateClassificationInput(
                ticker=str(item.get("ticker") or "UNKNOWN"),
                operator_report_format_version=format_version,
                run_state=None,
                output_families_present=output_families,
                missing_data_notes_present=bool(summary.get("missing_data_notes_present")),
                stale_data_notes_present=bool(summary.get("stale_data_notes_present")),
                blocked_notes_present=True,
                provenance_references_present=bool(
                    summary.get("provenance_references_present")
                ),
                numeric_zero_evidence_present=bool(
                    summary.get("numeric_zero_evidence_present")
                ),
                skipped_reason=str(item.get("reason") or "skipped"),
                malformed_artifact="malformed" in str(item.get("reason") or ""),
                source_text=operator_markdown,
            )
        )
    return tuple(inputs)


def _bucket_for_reasons(
    blocking_reasons: Sequence[str],
    candidate_input: CandidateClassificationInput,
) -> str:
    reasons = set(blocking_reasons)
    if "actionable_language_detected" in reasons:
        return "unclassified_due_to_unsupported_input"
    if candidate_input.malformed_artifact or "malformed_artifact" in reasons:
        return "unclassified_due_to_malformed_artifact"
    if (
        candidate_input.unsupported_input
        or "contract_version_mismatch" in reasons
        or "operator_report_contract_version_mismatch" in reasons
    ):
        return "unclassified_due_to_unsupported_input"
    if "stale_data_notes_present" in reasons:
        return "requires_stale_data_review"
    if "missing_data_notes_present" in reasons:
        return "requires_missing_data_review"
    if "blocked_notes_present" in reasons or "incomplete_dry_run" in reasons:
        return "requires_blocked_state_review"
    if "missing_readable_output" in reasons:
        return "unclassified_due_to_insufficient_evidence"
    if any(reason.startswith("missing_source") for reason in reasons):
        return "requires_source_coverage_review"
    if any(reason.startswith("missing_portfolio") for reason in reasons):
        return "requires_portfolio_context_review"
    if any(reason.startswith("missing_") for reason in reasons):
        return "unclassified_due_to_insufficient_evidence"
    if not candidate_input.provenance_references_present:
        return "requires_human_interpretation_review"
    return "ready_for_manual_candidate_review"


def _rationale_for_bucket(bucket: str) -> str:
    return {
        "ready_for_manual_candidate_review": (
            "Readable output contains complete local artifact context for human follow-up."
        ),
        "requires_missing_data_review": (
            "Classification requires review because upstream missing-data markers are present."
        ),
        "requires_stale_data_review": (
            "Classification requires review because upstream stale-data markers are present."
        ),
        "requires_blocked_state_review": (
            "Classification requires review because an upstream stage did not complete."
        ),
        "requires_source_coverage_review": (
            "Classification requires review because source coverage evidence is incomplete."
        ),
        "requires_portfolio_context_review": (
            "Classification requires review because portfolio-context evidence is incomplete."
        ),
        "requires_human_interpretation_review": (
            "Readable output exists but needs human interpretation before narrower triage."
        ),
        "unclassified_due_to_malformed_artifact": (
            "Classification is unavailable because an artifact could not be read safely."
        ),
        "unclassified_due_to_unsupported_input": (
            "Classification is unavailable because the input is unsupported by contract."
        ),
        "unclassified_due_to_insufficient_evidence": (
            "Classification is unavailable because required readable evidence is incomplete."
        ),
    }[bucket]


def _evidence_references(
    candidate_input: CandidateClassificationInput,
) -> tuple[CandidateEvidenceReference, ...]:
    references = [
        CandidateEvidenceReference(
            reference_type="ticker",
            reference=candidate_input.ticker,
        )
    ]
    references.extend(
        CandidateEvidenceReference(reference_type="output_family", reference=family)
        for family in sorted(candidate_input.output_families_present)
    )
    if candidate_input.provenance_references_present:
        references.append(
            CandidateEvidenceReference(
                reference_type="provenance",
                reference="operator_report_provenance_present",
            )
        )
    return tuple(references)


def _render_markdown_report(result: CandidateClassificationReportResult) -> str:
    lines = [
        "# Candidate Classification Report",
        "",
        "## 1. Classification Metadata",
        "",
        f"* Candidate classification format version: `{result.candidate_classification_format_version}`",
        f"* Candidate classification run id: `{result.candidate_classification_run_id}`",
        f"* Generated at: `{result.generated_at or 'not supplied'}`",
        f"* Included ticker count: `{len(result.included_tickers)}`",
        f"* Classified ticker count: `{len(result.classified_tickers)}`",
        f"* Unclassified ticker count: `{len(result.unclassified_tickers)}`",
        "* Local-only marker: `true`",
        "",
        "## 2. Source Artifact Boundary",
        "",
        f"* Input operator report root: `{result.input_operator_report_root or 'not supplied'}`",
        f"* Input artifact root: `{result.input_artifact_root or 'not supplied'}`",
        f"* Input interpretation report root: `{result.input_interpretation_report_root or 'not supplied'}`",
        "* Approved source: existing local readable output and dry-run summaries.",
        "",
        "## 3. Non-Actionable Boundary",
        "",
        NON_ACTIONABLE_BOUNDARY,
        "",
        "## 4. Input Coverage Summary",
        "",
        f"* Included tickers: `{', '.join(result.included_tickers) or 'none'}`",
        f"* Classified tickers: `{', '.join(result.classified_tickers) or 'none'}`",
        f"* Unclassified tickers: `{_unclassified_text(result.unclassified_tickers)}`",
        "",
        "## 5. Classification Method",
        "",
        "The classifier applies fixed review-only buckets to existing readable output markers: missing-data, stale-data, blocked-state, malformed-artifact, unsupported-input, stage-family completeness, provenance, and advisory-language safety.",
        "",
        "## 6. Candidate Bucket Summary",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for bucket in ALLOWED_CANDIDATE_BUCKETS:
        lines.append(f"| {bucket} | {result.bucket_counts.get(bucket, 0)} |")
    lines.extend(["", "## 7. Per-Ticker Candidate Classifications", ""])
    for item in result.per_ticker_classifications:
        lines.extend(_classification_markdown(item))
    lines.extend(
        [
            "## 8. Unclassified, Blocked, Skipped, Stale, And Malformed Ticker Notes",
            "",
            f"* Unclassified tickers: `{_unclassified_text(result.unclassified_tickers)}`",
            f"* Missing-data notes present: `{_yes_no(result.missing_data_notes_present)}`",
            f"* Stale-data notes present: `{_yes_no(result.stale_data_notes_present)}`",
            f"* Blocked notes present: `{_yes_no(result.blocked_notes_present)}`",
            "",
            "## 9. Missing-Data And Stale-Data Notes",
            "",
            "Missing and stale markers are preserved from the readable operator summary. They are never repaired, inferred, or converted into zero values.",
            "",
            "## 10. Provenance Summary",
            "",
            f"* Provenance references present: `{_yes_no(result.provenance_references_present)}`",
            f"* Numeric-zero evidence present: `{_yes_no(result.numeric_zero_evidence_present)}`",
            "",
            "## 11. Human-Review Checklist",
            "",
            "* Input operator report was checked.",
            "* Local summary metadata was checked.",
            "* Candidate buckets were reviewed as human-review buckets only.",
            "* Missing-data markers were preserved.",
            "* Stale-data markers were preserved.",
            "* Blocked-state markers were preserved.",
            "* Provenance markers were preserved where available.",
            "* Non-actionable boundary is visible.",
            "",
            "## 12. Safe Next-Step Candidate",
            "",
            "Human review of candidate-classification outputs before any later contract sprint.",
            "",
            "## 13. Appendix: Machine-Readable Summary Reference",
            "",
            f"* Summary JSON: `{result.summary_json_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def _classification_markdown(item: CandidateTickerClassification) -> list[str]:
    return [
        f"### {item.ticker}",
        "",
        f"* Non-actionable candidate classification: `{item.candidate_bucket}`",
        f"* Rationale: {item.candidate_rationale}",
        f"* Evidence references: `{', '.join(reference.reference for reference in item.evidence_references) or 'none'}`",
        f"* Blocking reasons: `{', '.join(item.blocking_reasons) or 'none'}`",
        f"* Safety flag - actionable language detected: `{_yes_no(item.safety_flags.actionable_language_detected)}`",
        "",
        "This classification is intended for human triage only and does not authorize market participation, capital guidance, external system actions, portfolio mutation, or watchlist mutation.",
        "",
    ]


def _classification_payload(item: CandidateTickerClassification) -> dict[str, Any]:
    payload = asdict(item)
    payload["evidence_references"] = [
        asdict(reference) for reference in item.evidence_references
    ]
    payload["safety_flags"] = asdict(item.safety_flags)
    return payload


def _contains_forbidden_action_language(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in FORBIDDEN_ACTION_TERMS)


def _ticker_in_records(ticker: str, records: Any) -> bool:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return False
    return any(
        isinstance(record, Mapping) and str(record.get("ticker")) == ticker
        for record in records
    )


def _validated_input_root(path: Path) -> Path:
    if any(part == ".." for part in path.parts):
        raise CandidateClassificationError(
            "Input root may not contain parent traversal."
        )
    resolved = path.resolve()
    if not resolved.exists():
        raise CandidateClassificationError(f"Input root does not exist: {path}")
    if not resolved.is_dir():
        raise CandidateClassificationError(f"Input root is not a directory: {path}")
    return resolved


def _validated_optional_root(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return _validated_input_root(Path(path))


def _prepare_output_directory(output_root: Path, run_id: str) -> Path:
    if any(part == ".." for part in output_root.parts):
        raise CandidateClassificationError(
            "Output root may not contain parent traversal."
        )
    root = output_root.resolve()
    output_directory = (root / run_id).resolve()
    try:
        output_directory.relative_to(root)
    except ValueError as exc:
        raise CandidateClassificationError("Output path escaped the output root.") from exc
    if output_directory.exists():
        raise CandidateClassificationError(
            f"Candidate classification output directory already exists: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    return output_directory


def _read_operator_summary(operator_root: Path) -> dict[str, Any]:
    summary_path = operator_root / "operator_report_summary.json"
    if not summary_path.exists():
        raise CandidateClassificationError(
            f"Operator report summary is missing: {summary_path}"
        )
    return _read_json_object(summary_path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateClassificationError(f"{path.name} contains invalid JSON") from exc
    except OSError as exc:
        raise CandidateClassificationError(f"{path.name} cannot be read") from exc
    if not isinstance(payload, dict):
        raise CandidateClassificationError(f"{path.name} must contain a JSON object")
    return payload


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _safe_path_segment(value: str, *, field_name: str) -> str:
    safe = value.strip()
    if not safe or safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise CandidateClassificationError(f"{field_name} is not a safe path segment.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in safe):
        raise CandidateClassificationError(f"{field_name} contains unsafe characters.")
    return safe


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise CandidateClassificationError(
            f"Candidate classification report file already exists: {path}"
        )
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CandidateClassificationError(
            f"Candidate classification summary file already exists: {path}"
        )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _unclassified_text(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "none"
    return "; ".join(
        f"{item.get('ticker', 'UNKNOWN')}:{item.get('candidate_bucket', 'unknown')}"
        for item in items
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        result = build_candidate_classification_report(
            input_operator_report_root=args.input_operator_report_root,
            output_root=args.output_root,
            candidate_classification_run_id=args.candidate_classification_run_id,
            generated_at=args.generated_at,
            input_artifact_root=args.input_artifact_root,
            input_interpretation_report_root=args.input_interpretation_report_root,
        )
    except CandidateClassificationError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(f"candidate_classification_report_path={result.markdown_report_path}", file=stdout)
    print(f"summary_json_path={result.summary_json_path}", file=stdout)
    print(f"classified_tickers={','.join(result.classified_tickers)}", file=stdout)
    print(f"unclassified_ticker_count={len(result.unclassified_tickers)}", file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate non-actionable candidate classification from local readable operator output."
    )
    parser.add_argument("--input-operator-report-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_CANDIDATE_CLASSIFICATION_OUTPUT_ROOT))
    parser.add_argument("--candidate-classification-run-id", required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--input-artifact-root", default=None)
    parser.add_argument("--input-interpretation-report-root", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
