from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from market_engine.batch_status.artifact_discovery import (
    CANONICAL_SELECTION_RULE,
    ArtifactCandidate,
    DiscoveryResult,
    select_canonical_artifacts,
)


STATUS_INDEX_SCHEMA_VERSION = "market-engine-ticker-status-index-v1"
STATUS_INDEX_ARTIFACT_TYPE = "market-engine-ticker-status-index"
MANIFEST_SCHEMA_VERSION = "market-engine-batch-status-run-manifest-v1"
FAILURES_SCHEMA_VERSION = "market-engine-batch-status-failures-v1"


def build_ticker_status_index(
    discovery: DiscoveryResult,
    *,
    run_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    emitted_at = generated_at or _generated_at_utc()
    canonical_by_ticker = select_canonical_artifacts(discovery.candidates)
    tickers = [
        _ticker_status_row(
            ticker,
            canonical,
            candidate_artifact_count=sum(
                1 for candidate in discovery.candidates if candidate.ticker == ticker
            ),
            invalid_candidate_count=sum(
                1
                for candidate in discovery.candidates
                if candidate.ticker == ticker and not candidate.valid
            ),
        )
        for ticker, canonical in sorted(canonical_by_ticker.items())
    ]
    summary = _summary(tickers, discovery)
    return {
        "schema_version": STATUS_INDEX_SCHEMA_VERSION,
        "artifact_type": STATUS_INDEX_ARTIFACT_TYPE,
        "run_id": run_id,
        "generated_at": emitted_at,
        "artifact_root": discovery.artifact_root,
        "summary": summary,
        "tickers": tickers,
    }


def write_batch_status_outputs(
    index: Mapping[str, Any],
    discovery: DiscoveryResult,
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_summary = discovery.summary_dict()
    failures = {
        "schema_version": FAILURES_SCHEMA_VERSION,
        "failures": list(discovery.failures),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-batch-status-run-manifest",
        "run_id": run_id,
        "generated_at": index.get("generated_at"),
        "artifact_root": index.get("artifact_root"),
        "outputs": {
            "ticker_status_index_json": "ticker_status_index.json",
            "ticker_status_index_md": "ticker_status_index.md",
            "discovery_summary": "discovery_summary.json",
            "failures": "failures.json",
        },
        "baseline_guardrail": {
            "openai_api_required": False,
            "provider_invocation_allowed": False,
            "source_acquisition_performed": False,
            "ranking_performed": False,
            "recommendation_semantics_performed": False,
        },
    }

    _write_json(output_dir / "ticker_status_index.json", index)
    (output_dir / "ticker_status_index.md").write_text(
        render_ticker_status_markdown(index),
        encoding="utf-8",
    )
    _write_json(output_dir / "discovery_summary.json", discovery_summary)
    _write_json(output_dir / "failures.json", failures)
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def render_ticker_status_markdown(index: Mapping[str, Any]) -> str:
    summary = index.get("summary") or {}
    status_counts = summary.get("status_counts") or {}
    rows = [
        "# Market Engine Ticker Status Index",
        "",
        f"Run ID: `{index.get('run_id')}`",
        f"Generated at: `{index.get('generated_at')}`",
        f"Artifact root: `{index.get('artifact_root')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Tickers total | {summary.get('tickers_total', 0)} |",
        f"| Valid artifacts | {summary.get('artifacts_valid', 0)} |",
        f"| Invalid artifacts | {summary.get('artifacts_invalid', 0)} |",
        f"| Blocked | {status_counts.get('blocked', 0)} |",
        f"| Stale | {status_counts.get('stale', 0)} |",
        f"| Review ready | {status_counts.get('review_ready', 0)} |",
        f"| Descriptive only | {status_counts.get('descriptive_only', 0)} |",
        f"| Invalid artifact | {status_counts.get('invalid_artifact', 0)} |",
        "",
        "## Ticker Statuses",
        "",
        "| Ticker | Status | Readiness | Stale | Actionable | DE Ready | Blocked stage | Missing data | Artifact |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in index.get("tickers") or ():
        rows.append(
            "| "
            + " | ".join(
                (
                    _md(row.get("ticker")),
                    _md(row.get("status")),
                    _md(row.get("readiness_level")),
                    _yes_no(row.get("context_stale")),
                    _yes_no(row.get("actionable_review_allowed")),
                    _yes_no(row.get("decision_engine_ready")),
                    _md(row.get("blocked_stage")),
                    _md(", ".join(row.get("missing_data_summary") or ())),
                    _md(row.get("artifact_path")),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _ticker_status_row(
    ticker: str,
    candidate: ArtifactCandidate,
    *,
    candidate_artifact_count: int,
    invalid_candidate_count: int,
) -> dict[str, Any]:
    payload = candidate.payload
    readiness = _mapping(payload.get("analysis_context_readiness"))
    blocked_reasons = _strings(payload.get("blocked_reasons"))
    missing_data_summary = _strings(payload.get("missing_data_summary"))
    readiness_blocked_reasons = _strings(readiness.get("blocked_reasons"))
    evidence_families_missing = _strings(readiness.get("evidence_families_missing"))
    status = _status(candidate, payload, readiness, blocked_reasons)
    return {
        "ticker": ticker,
        "status": status,
        "readiness_level": _string(readiness.get("readiness_level")),
        "actionable_review_allowed": bool(readiness.get("actionable_review_allowed")),
        "decision_engine_ready": bool(readiness.get("decision_engine_ready")),
        "context_stale": bool(readiness.get("context_stale")),
        "blocked_stage": _string(payload.get("blocked_stage")),
        "blocked_reasons": blocked_reasons,
        "readiness_blocked_reasons": readiness_blocked_reasons,
        "missing_data_summary": missing_data_summary,
        "evidence_families_missing": evidence_families_missing,
        "input_mode": candidate.input_mode,
        "dry_run_id": candidate.dry_run_id,
        "artifact_created_at": candidate.artifact_created_at,
        "artifact_path": candidate.artifact_path,
        "artifact_sha256": candidate.sha256,
        "candidate_artifact_count": candidate_artifact_count,
        "invalid_candidate_count": invalid_candidate_count,
        "invalid_reasons": list(candidate.invalid_reasons),
        "provenance": _provenance(payload),
    }


def _status(
    candidate: ArtifactCandidate,
    payload: Mapping[str, Any],
    readiness: Mapping[str, Any],
    blocked_reasons: Iterable[str],
) -> str:
    if not candidate.valid:
        return "invalid_artifact"
    if _string(payload.get("blocked_stage")) or tuple(blocked_reasons):
        return "blocked"
    if bool(readiness.get("context_stale")):
        return "stale"
    if bool(readiness.get("actionable_review_allowed")) or bool(
        readiness.get("decision_engine_ready")
    ):
        return "review_ready"
    return "descriptive_only"


def _summary(
    tickers: list[Mapping[str, Any]],
    discovery: DiscoveryResult,
) -> dict[str, Any]:
    status_counts = _counts(row.get("status") for row in tickers)
    readiness_counts = _counts(row.get("readiness_level") for row in tickers)
    tickers_with_valid_artifact = sum(1 for row in tickers if not row.get("invalid_reasons"))
    return {
        "tickers_total": len(tickers),
        "tickers_with_valid_artifact": tickers_with_valid_artifact,
        "tickers_without_valid_artifact": len(tickers) - tickers_with_valid_artifact,
        "artifacts_discovered": len(discovery.candidates),
        "artifacts_valid": len(discovery.valid_candidates),
        "artifacts_invalid": len(discovery.invalid_candidates),
        "status_counts": status_counts,
        "readiness_counts": readiness_counts,
        "stale_count": sum(1 for row in tickers if row.get("context_stale")),
        "actionable_review_allowed_count": sum(
            1 for row in tickers if row.get("actionable_review_allowed")
        ),
        "decision_engine_ready_count": sum(
            1 for row in tickers if row.get("decision_engine_ready")
        ),
    }


def _provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _mapping(payload.get("provenance_summary"))
    recommendation = _mapping(provenance.get("recommendation_review"))
    source_context = _mapping(provenance.get("source_context"))
    return {
        "source_refresh_snapshot_id": _string(source_context.get("source_refresh_snapshot_id")),
        "recommendation_review_state": _string(recommendation.get("state")),
        "recommendation_review_category": _string(recommendation.get("category")),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generated_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _md(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value).replace("|", "\\|")


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
