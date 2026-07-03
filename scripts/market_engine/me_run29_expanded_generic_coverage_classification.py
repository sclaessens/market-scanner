from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION,
)
from market_engine.source_support.cached_source_coverage import (
    CACHED_SOURCE_COVERAGE_CONTRACT_VERSION,
    CachedSourceCoverageClassification,
    CachedSourceCoverageInput,
    ReadinessStatus,
    SourceFamily,
    TargetCapability,
    classify_cached_source_coverage_batch,
)
from market_engine.source_support.staging_validation_coverage_adapter import (
    adapt_staging_validation_to_cached_source_coverage_input,
)


SUMMARY_FORMAT = "market-engine-me-run29-expanded-generic-coverage-summary-v1"
DEFAULT_TARGET_CAPABILITY = TargetCapability.RECOMMENDATION_REVIEW
NEXT_SPRINT = "ME-GV01 - Define The Governor investment evaluation contract"


class MeRun29Error(ValueError):
    """Raised when ME-RUN29 evidence cannot be classified safely."""


def run_expanded_generic_coverage_classification(
    *,
    input_evidence_path: str | Path,
    run_id: str,
    classification_timestamp: str,
    artifact_root: str | Path,
    target_capability: TargetCapability = DEFAULT_TARGET_CAPABILITY,
) -> dict[str, Any]:
    evidence_path = Path(input_evidence_path)
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-RUN29 artifact root already exists: {root}")

    evidence = _load_evidence(evidence_path)
    entries = _ordered_entries(evidence)
    coverage_inputs = tuple(
        adapt_staging_validation_to_cached_source_coverage_input(
            entry,
            universe_supported=True,
            target_capability=target_capability,
            source_family_hint=_source_family_hint(entry),
        )
        for entry in entries
    )
    batch = classify_cached_source_coverage_batch(coverage_inputs)
    results = tuple(
        _result_payload(entry=entry, coverage_input=coverage_input, classification=item)
        for entry, coverage_input, item in zip(
            entries,
            coverage_inputs,
            batch.classifications,
            strict=True,
        )
    )
    blocker_counts = Counter(
        blocker.code.value
        for classification in batch.classifications
        for blocker in classification.blockers
    )
    reserved_state_counts = _reserved_state_counts(batch.classifications)
    if any(reserved_state_counts.values()):
        raise MeRun29Error("reserved authority state became reachable")

    summary = {
        "summary_format": SUMMARY_FORMAT,
        "run_id": _required_text(run_id, "run_id"),
        "contract_version": CACHED_SOURCE_COVERAGE_CONTRACT_VERSION,
        "input_evidence_source": _display_path(evidence_path),
        "input_evidence_kind": evidence.get(
            "evidence_kind",
            "cached_source_staging_validation",
        ),
        "input_evidence_notice": evidence.get("fixture_notice"),
        "classification_timestamp": _required_text(
            classification_timestamp,
            "classification_timestamp",
        ),
        "target_capability": target_capability.value,
        "tickers_total": len({item.ticker for item in coverage_inputs}),
        "source_families_total": len(
            {
                evidence_item.source_family.value
                for coverage_input in coverage_inputs
                for evidence_item in coverage_input.source_evidence
            }
        ),
        "staging_entries_total": len(entries),
        "coverage_counts": {
            item.status: item.count for item in batch.coverage_counts
        },
        "readiness_counts": {
            item.status: item.count for item in batch.readiness_counts
        },
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "dominant_blockers": tuple(
            {
                "reason": reason,
                "count": count,
            }
            for reason, count in sorted(
                blocker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "recommendation_eligible_count": sum(
            item.recommendation_review_allowed
            for item in batch.classifications
        ),
        "reserved_state_counts": reserved_state_counts,
        "actionable_count": batch.actionable_count,
        "decision_ready_count": sum(
            item.decision_engine_handoff_allowed
            for item in batch.classifications
        ),
        "de_ready_count": batch.de_ready_count,
        "results": results,
        "forbidden_side_effects_confirmed": {
            "provider_calls_performed": False,
            "network_used": False,
            "source_acquisition_performed": False,
            "snapshot_import_performed": False,
            "production_write_performed": False,
            "telegram_or_email_sent": False,
            "portfolio_written": False,
            "watchlist_written": False,
            "broker_action_performed": False,
            "governor_invoked": False,
            "dispatch_station_invoked": False,
            "decision_engine_invoked": False,
        },
        "classification_boundary": (
            "Refinery/RUN evidence only. No recommendation, scoring, allocation, "
            "delivery, execution, Governor, Dispatch Station, or Decision Engine "
            "authority is invoked."
        ),
        "next_sprint": NEXT_SPRINT,
    }
    summary = json.loads(json.dumps(summary, sort_keys=True))
    root.mkdir(parents=True)
    _write_json(root / "coverage_classification_summary.json", summary)
    (root / "coverage_classification_report.md").write_text(
        _markdown_report(summary),
        encoding="utf-8",
    )
    return summary


def _load_evidence(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MeRun29Error(f"input evidence is unreadable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MeRun29Error("input evidence is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise MeRun29Error("input evidence must be a JSON object")
    if (
        payload.get("report_format_version")
        != CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION
    ):
        raise MeRun29Error("input evidence report format is unsupported")
    return payload


def _ordered_entries(evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_entries = evidence.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise MeRun29Error("input evidence entries must be a non-empty list")
    if not all(isinstance(entry, Mapping) for entry in raw_entries):
        raise MeRun29Error("input evidence entries must be JSON objects")
    return tuple(
        sorted(
            raw_entries,
            key=lambda entry: (
                str(entry.get("ticker") or ""),
                str(entry.get("market") or ""),
                str(entry.get("source_family") or ""),
                str(entry.get("snapshot_id") or ""),
            ),
        )
    )


def _source_family_hint(entry: Mapping[str, Any]) -> SourceFamily | None:
    value = entry.get("generic_source_family_hint")
    if value is None:
        return None
    if not isinstance(value, str):
        raise MeRun29Error("generic source-family hint must be text or null")
    try:
        return SourceFamily(value)
    except ValueError as exc:
        raise MeRun29Error(
            f"generic source-family hint is unsupported: {value}"
        ) from exc


def _result_payload(
    *,
    entry: Mapping[str, Any],
    coverage_input: CachedSourceCoverageInput,
    classification: CachedSourceCoverageClassification,
) -> dict[str, Any]:
    evidence = coverage_input.source_evidence
    generic_family = evidence[0].source_family if evidence else None
    family_result = next(
        (
            item
            for item in classification.source_family_results
            if item.source_family is generic_family
        ),
        None,
    )
    return {
        "ticker": classification.ticker,
        "market": classification.market,
        "staging_source_family": entry.get("source_family"),
        "generic_source_family": (
            generic_family.value if generic_family is not None else None
        ),
        "staging_validation_status": entry["staging_validation_status"],
        "staging_issues": tuple(entry.get("issues", ())),
        "coverage_status": classification.coverage_status.value,
        "readiness_status": classification.readiness_status.value,
        "source_family_coverage_status": (
            family_result.coverage_status.value
            if family_result is not None
            else None
        ),
        "source_family_requirement_satisfied": (
            family_result.requirement_satisfied
            if family_result is not None
            else False
        ),
        "blockers": tuple(
            {
                "reason": blocker.code.value,
                "source_family": (
                    blocker.source_family.value
                    if blocker.source_family is not None
                    else None
                ),
                "pipeline_stage": blocker.pipeline_stage.value,
            }
            for blocker in classification.blockers
        ),
        "recommendation_review_allowed": (
            classification.recommendation_review_allowed
        ),
        "actionable": classification.actionable,
        "decision_engine_handoff_allowed": (
            classification.decision_engine_handoff_allowed
        ),
        "de_ready": classification.de_ready,
        "adapted_coverage_input": asdict(coverage_input),
        "classification": asdict(classification),
    }


def _reserved_state_counts(
    classifications: Sequence[CachedSourceCoverageClassification],
) -> dict[str, int]:
    return {
        "actionable": sum(item.actionable for item in classifications),
        "actionable_review": sum(
            item.readiness_status is ReadinessStatus.ACTIONABLE
            for item in classifications
        ),
        "decision_ready": sum(
            item.decision_engine_handoff_allowed
            for item in classifications
        ),
        "de_ready": sum(item.de_ready for item in classifications),
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ME-RUN29 Expanded Generic Coverage Classification",
        "",
        "## Purpose",
        "",
        (
            "Classify deterministic cached-source staging-validation evidence "
            "through the ME-SA14 adapter and ME-SA13 generic classifier."
        ),
        "",
        "## Input Evidence",
        "",
        f"- Source: `{summary['input_evidence_source']}`",
        f"- Evidence kind: `{summary['input_evidence_kind']}`",
        f"- Notice: {summary['input_evidence_notice'] or 'Not supplied.'}",
        f"- Classification timestamp: `{summary['classification_timestamp']}`",
        f"- Target capability: `{summary['target_capability']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Tickers | {summary['tickers_total']} |",
        f"| Generic source families | {summary['source_families_total']} |",
        f"| Staging entries | {summary['staging_entries_total']} |",
        f"| Recommendation-eligible | {summary['recommendation_eligible_count']} |",
        f"| Actionable | {summary['actionable_count']} |",
        f"| Decision-ready | {summary['decision_ready_count']} |",
        f"| DE-ready | {summary['de_ready_count']} |",
        "",
        "## Per Ticker and Source Family",
        "",
        (
            "| Ticker | Market | Staging family | Generic family | Validation | "
            "Family coverage | Aggregate coverage | Readiness | Actionable |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            f"| {result['ticker']} | {result['market'] or '-'} | "
            f"{result['staging_source_family'] or '-'} | "
            f"{result['generic_source_family'] or '-'} | "
            f"{result['staging_validation_status']} | "
            f"{result['source_family_coverage_status'] or '-'} | "
            f"{result['coverage_status']} | {result['readiness_status']} | "
            f"{str(result['actionable']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Dominant Blockers",
            "",
            "| Blocker | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{item['reason']}` | {item['count']} |"
        for item in summary["dominant_blockers"]
    )
    lines.extend(
        [
            "",
            "## Reserved-State Confirmation",
            "",
            "All reserved-state counts are zero:",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in summary["reserved_state_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Governance and Non-Goals",
            "",
            summary["classification_boundary"],
            "",
            "The run performs no acquisition, provider or network access, "
            "snapshot import, production write, delivery, portfolio/watchlist "
            "mutation, broker action, scoring, ranking, or recommendation-state "
            "upgrade.",
            "",
            "## Next Sprint",
            "",
            f"`{summary['next_sprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _required_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise MeRun29Error(f"{field_name} must be non-empty text without padding")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic ME-RUN29 generic coverage classification from "
            "staging-validation evidence."
        )
    )
    parser.add_argument("--input-evidence", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--classification-timestamp", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    summary = run_expanded_generic_coverage_classification(
        input_evidence_path=args.input_evidence,
        run_id=args.run_id,
        classification_timestamp=args.classification_timestamp,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
