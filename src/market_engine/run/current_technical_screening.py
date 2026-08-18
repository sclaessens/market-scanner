from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import pandas as pd

from market_engine.run.full_canonical_universe_analysis import (
    MISSING_FULL_ADVICE_EVIDENCE,
    _candidate_score,
    _detect_technical_setup,
    _rank_candidates,
    _screening_from_setup_context,
)
from market_engine.source_refresh.advisory_ohlc_history import (
    DEFAULT_POLICY_PATH,
    DEFAULT_UNIVERSE_SNAPSHOT,
    AdvisoryHistoryIssue,
    _canonical_json,
    _clock_now,
    _effective_analytic_authority_status,
    _sha256,
    _sha256_file,
    _utc_text,
    load_advisory_ohlc_history,
)
from market_engine.source_refresh.advisory_price_evidence import (
    DEFAULT_POLICY_PATH as DEFAULT_PRICE_POLICY_PATH,
    load_advisory_price_artifact,
)
from market_engine.portfolio_review.manual_transaction_ledger import load_ledger, rebuild_positions
from market_engine.data.data11_execution import (
    ValidatedDownstreamAuthorityState,
    ValidatedExecutionProof,
    load_downstream_after_authority,
    validated_after_payload,
)
from market_engine.data.data11_governance import load_downstream_prestate, validate_approval_decision


SCREENING_VERSION = "market-engine-current-technical-screening-v1"
SCREENING_MANIFEST_VERSION = "market-engine-current-technical-screening-manifest-v1"
RANKING_VERSION = "market-engine-current-technical-candidate-ranking-v1"
RECONCILIATION_VERSION = "market-engine-technical-price-reconciliation-v1"
HANDOFF_VERSION = "market-engine-run33-grounded-candidate-input-v1"
HANDOFF_MANIFEST_VERSION = "market-engine-run33-grounded-candidate-input-manifest-v1"
DEFAULT_SCREENING_ROOT = Path("artifacts/market_engine/current_technical_screening_runs")
DEFAULT_HANDOFF_ROOT = Path("artifacts/market_engine/run33_handoff_runs")
DEFAULT_RUN30_RANKING = Path("artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z/candidate_ranking.json")
DEFAULT_PRESTATE_AUTHORITY = Path("config/market_engine/data11_downstream_prestate_authority.json")
DEFAULT_READINESS_REPORT = Path("artifacts/market_engine/run_evidence/me-run31-after-me-data06-review-fix-20260718T113254Z/advice_readiness_report.json")
DEFAULT_FUNDAMENTAL_STATUS = Path("artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z/per_ticker_fundamental_status.json")
DEFAULT_DATA11_ROOT = Path("artifacts/market_engine/run_evidence/me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z")
SCREENING_POLICY_VERSION = "market-engine-current-technical-screening-policy-v1"
DEFAULT_SCREENING_POLICY = Path("config/market_engine/current_technical_screening_policy.json")


class CurrentScreeningIssue(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


_RUN33_TOKEN = object()


@dataclass(frozen=True)
class _ValidatedRun33HandoffContext:
    _token: object
    manifest: Mapping[str, Any]
    candidate_input: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    downstream_authority: ValidatedDownstreamAuthorityState | None


DEFAULT_APPROVAL_DECISIONS = tuple(
    DEFAULT_DATA11_ROOT / "approval_candidates" / ticker / "approval_candidate.json"
    for ticker in ("ASH", "BIO", "CI")
)


def run_current_technical_screening(
    *, run_id: str, history_artifact_root: str | Path,
    output_root: str | Path = DEFAULT_SCREENING_ROOT,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    history_policy_path: str | Path = DEFAULT_POLICY_PATH,
    screening_policy_path: str | Path = DEFAULT_SCREENING_POLICY,
    _clock: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    destination = _output_destination(output_root, DEFAULT_SCREENING_ROOT, run_id)
    history = load_advisory_ohlc_history(
        history_artifact_root, universe_path=universe_path, policy_path=history_policy_path, _clock=_clock
    )
    policy_source = Path(screening_policy_path)
    policy = _load_screening_policy(policy_source)
    index_rows, ranked = _derive_screening(history, top_limit=policy["top_candidate_limit"])
    ranking_records = ranked[: policy["top_candidate_limit"]]
    old = _load_old_ranking(DEFAULT_RUN30_RANKING)
    old_symbols = [str(row.get("symbol")) for row in old[: policy["top_candidate_limit"]]]
    new_symbols = [str(row["symbol"]) for row in ranking_records]
    cutoff_sessions = list(history.manifest["expected_last_completed_sessions"])
    history_binding = {"run_id": history.manifest["run_id"], "artifact_sha256": history.manifest["artifact_sha256"], "observations_sha256": history.manifest["observations_sha256"], "manifest_file_sha256": _sha256_file(history.root / "manifest.json")}
    policy_binding = {"schema_version": policy["schema_version"], "policy_id": policy["policy_id"], "path": policy_source.as_posix(), "sha256": _sha256_file(policy_source)}
    universe_index = {"schema_version": "market-engine-current-technical-universe-index-v1", "run_id": run_id, "records": index_rows}
    ranking = {"schema_version": RANKING_VERSION, "run_id": run_id, "ranking_scope": "technical_setup_screening", "cutoff_sessions": cutoff_sessions, "history_binding": history_binding, "screening_policy_binding": policy_binding, "records": ranking_records, "eligible_total": len(ranked), "requested_top_limit": policy["top_candidate_limit"], "ranking_gap": max(0, policy["top_candidate_limit"] - len(ranking_records))}
    payloads: dict[str, Any] = {
        "universe_analysis_index.json": universe_index,
        "setup_detection_summary.json": {"schema_version": "market-engine-current-technical-setup-summary-v1", "counts": dict(sorted(Counter(row.get("setup_detection", {}).get("setup_state", "blocked") for row in index_rows).items()))},
        "analysis_outcome_distribution.json": {"schema_version": "market-engine-current-technical-outcome-distribution-v1", "counts": dict(sorted(Counter(row["output_label"] for row in index_rows).items()))},
        "blocker_report.json": {"schema_version": "market-engine-current-technical-blocker-report-v1", "counts": dict(sorted(Counter(code for row in index_rows for code in row["blockers"]).items())), "records": [{"instrument_id": row["instrument_id"], "reason_codes": row["blockers"]} for row in index_rows if row["blockers"]]},
        "candidate_ranking.json": ranking,
        "old_vs_new_screening_drift.json": {"schema_version": "market-engine-current-technical-run30-drift-v1", "old_run30_is_audit_only": True, "old_top_symbols": old_symbols, "new_top_symbols": new_symbols, "retained": sorted(set(old_symbols) & set(new_symbols)), "added": sorted(set(new_symbols) - set(old_symbols)), "removed": sorted(set(old_symbols) - set(new_symbols))},
    }
    authority_usable = _effective_analytic_authority_status(history) == "usable"
    manifest_base = {"schema_version": SCREENING_MANIFEST_VERSION, "artifact_version": SCREENING_VERSION, "run_id": run_id, "generated_at": _utc_text(_clock_now(_clock)), "history_binding": history_binding, "universe_sha256": history.manifest["universe_sha256"], "history_policy_sha256": history.manifest["history_policy_sha256"], "screening_policy_binding": policy_binding, "cutoff_sessions": cutoff_sessions, "instrument_count": len(index_rows), "screened_count": sum(row["screening_status"] == "completed" for row in index_rows), "ranking_count": len(ranking_records), "candidate_ranking_sha256": _sha256(_canonical_json(ranking) + b"\n"), "universe_index_sha256": _sha256(_canonical_json(universe_index) + b"\n"), "run_status": "completed_with_blockers" if authority_usable and any(row["screening_status"] == "blocked" for row in index_rows) else ("completed" if authority_usable else "blocked_history_authority"), "analytic_authority_status": "usable" if authority_usable else "unusable", "authority_boundary": "technical_classification_only"}
    payloads["manifest.json"] = {**manifest_base, "artifact_sha256": _sha256(_canonical_json(manifest_base))}
    payloads["candidate_ranking.md"] = _ranking_markdown(ranking_records)
    payloads["top_candidates.md"] = _ranking_markdown(ranking_records)
    _write_artifact(destination, payloads)
    return payloads["manifest.json"], destination


def _derive_screening(history: Any, *, top_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authority_usable = _effective_analytic_authority_status(history) == "usable"
    index_rows: list[dict[str, Any]] = []
    for instrument in sorted(history.universe["instruments"], key=lambda row: str(row["instrument_id"])):
        instrument_id = str(instrument["instrument_id"])
        status = history.effective_status[instrument_id]
        blockers: list[str] = []
        entry: dict[str, Any] = {
            "instrument_id": instrument_id, "symbol": instrument["symbol"], "source_symbol": instrument["source_symbol"],
            "history_status": status, "screening_status": "blocked", "ranking_eligible": False,
            "candidate_score": 0, "output_label": "unable_to_analyse", "confidence": "low",
            "blockers": blockers, "missing_evidence": list(MISSING_FULL_ADVICE_EVIDENCE), "full_advice_ready": False,
            "ranking_scope": "technical_setup_screening", "setup_detection": {}, "setup_price_market_context": {},
            "score_components": {}, "positive_components": {}, "penalties": {}, "raw_score": 0,
            "traceability": {}, "exclusion_reasons": ["current_history_not_eligible"],
        }
        if not authority_usable:
            blockers.append("history_analytic_authority_unusable")
        elif status != "fresh":
            blockers.append(f"history_{status}")
        else:
            try:
                series = history.series[instrument_id]
                frame = _frame(series["bars"])
                setup = _detect_technical_setup(frame)
                context = {key: setup[key] for key in ("trend_state", "setup_state", "price_position", "risk_state")}
                screening = _screening_from_setup_context(context)
                inspection = {"artifactpath": f"{history.root.as_posix()}/{next(row['series_file'] for row in history.index if row['instrument_id'] == instrument_id)}", "start_date": frame.iloc[0]["Date"], "end_date": frame.iloc[-1]["Date"], "row_count": len(frame)}
                score = _candidate_score(setup, context, inspection, screening)
                entry.update({"screening_status": "completed", "ranking_eligible": score["eligible"], "candidate_score": score["score"], "output_label": screening["label"], "confidence": screening["confidence"], "blockers": screening["blockers"], "missing_evidence": screening["missing_evidence"], "setup_detection": setup, "setup_price_market_context": context, "score_components": score["score_components"], "positive_components": score["positive_components"], "penalties": score["penalties"], "raw_score": score["raw_score"], "traceability": score["traceability"], "exclusion_reasons": score["exclusion_reasons"]})
            except Exception as exc:
                entry["blockers"] = ["technical_calculation_failed"]
                entry["error_type"] = type(exc).__name__
        index_rows.append(entry)
    return index_rows, _rank_candidates(index_rows)


def build_run33_grounded_handoff(
    *, run_id: str, screening_root: str | Path, history_root: str | Path, price_root: str | Path,
    output_root: str | Path = DEFAULT_HANDOFF_ROOT,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    history_policy_path: str | Path = DEFAULT_POLICY_PATH,
    price_policy_path: str | Path = DEFAULT_PRICE_POLICY_PATH,
    screening_policy_path: str | Path = DEFAULT_SCREENING_POLICY,
    portfolio_ledger_path: str | Path | None = None,
    approval_decision_paths: Sequence[str | Path] = DEFAULT_APPROVAL_DECISIONS,
    downstream_authority: ValidatedDownstreamAuthorityState | None = None,
    _clock: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    destination = _output_destination(output_root, DEFAULT_HANDOFF_ROOT, run_id)
    now = _clock_now(_clock)
    for candidate in (Path(screening_root), Path(history_root), Path(price_root)):
        if candidate.is_symlink() or any(path.is_symlink() for path in candidate.rglob("*")):
            raise CurrentScreeningIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in authority artifacts")
    history = load_advisory_ohlc_history(history_root, universe_path=universe_path, policy_path=history_policy_path, _clock=lambda: now)
    screening_manifest, ranking, universe_index = _load_screening(screening_root, history, screening_policy_path=screening_policy_path)
    price = load_advisory_price_artifact(price_root, universe_path=universe_path, policy_path=price_policy_path, trusted_now=_utc_text(now))
    after_payload = validated_after_payload(downstream_authority)
    if downstream_authority is not None and after_payload is None:
        raise CurrentScreeningIssue("DOWNSTREAM_AUTHORITY_INVALID", "downstream authority must be a private validated state")
    prestate = load_downstream_prestate()
    if prestate.get("measurement_status") != "measured":
        raise CurrentScreeningIssue("DOWNSTREAM_PRESTATE_INVALID", "canonical downstream prestate is invalid")
    derived = _derive_handoff(
        run_id=run_id,
        history=history,
        screening_manifest=screening_manifest,
        ranking=ranking,
        universe_index=universe_index,
        price=price,
        screening_root=Path(screening_root), history_root=Path(history_root), price_root=Path(price_root),
        approval_decision_paths=approval_decision_paths,
        downstream_payload=after_payload,
        prestate=prestate,
        portfolio_binding=_portfolio_binding(portfolio_ledger_path),
    )
    manifest_base = {
        "schema_version": HANDOFF_MANIFEST_VERSION,
        "run_id": run_id,
        "generated_at": _utc_text(now),
        **derived["manifest_semantics"],
    }
    manifest = {**manifest_base, "artifact_sha256": _sha256(_canonical_json(manifest_base))}
    candidate_input = {"schema_version": HANDOFF_VERSION, "run_id": run_id, "records": derived["records"]}
    payloads = {"manifest.json": manifest, "technical_price_reconciliation.json": derived["reconciliation"], "run33_candidate_input.json": candidate_input}
    _write_artifact(destination, payloads)
    return manifest, destination


def load_validated_run33_handoff(
    handoff_root: str | Path,
    *,
    screening_root: str | Path,
    history_root: str | Path,
    price_root: str | Path,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    history_policy_path: str | Path = DEFAULT_POLICY_PATH,
    price_policy_path: str | Path = DEFAULT_PRICE_POLICY_PATH,
    screening_policy_path: str | Path = DEFAULT_SCREENING_POLICY,
    portfolio_ledger_path: str | Path | None = None,
    approval_decision_paths: Sequence[str | Path] = DEFAULT_APPROVAL_DECISIONS,
    downstream_after_authority_path: str | Path | None = None,
    execution_proof: ValidatedExecutionProof | None = None,
    downstream_repository_root: str | Path = ".",
    _clock: Callable[[], datetime] | None = None,
) -> _ValidatedRun33HandoffContext:
    if isinstance(handoff_root, Mapping):
        raise CurrentScreeningIssue("CALLER_CONTENT_FORBIDDEN", "RUN33 handoff authority requires an artifact path")
    root = Path(handoff_root)
    _validate_handoff_files(root)
    manifest = _json(root / "manifest.json")
    candidate_input = _json(root / "run33_candidate_input.json")
    reconciliation = _json(root / "technical_price_reconciliation.json")
    integrity = dict(manifest); artifact_sha = integrity.pop("artifact_sha256", None)
    if manifest.get("schema_version") != HANDOFF_MANIFEST_VERSION or artifact_sha != _sha256(_canonical_json(integrity)):
        raise CurrentScreeningIssue("HANDOFF_INTEGRITY_INVALID", "handoff manifest integrity is invalid")
    now = _clock_now(_clock)
    try:
        generated_at = datetime.fromisoformat(str(manifest.get("generated_at", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CurrentScreeningIssue("HANDOFF_TIME_INVALID", "handoff generation time is invalid") from exc
    if generated_at.tzinfo is None or generated_at.astimezone(UTC) > now or _utc_text(generated_at) != manifest.get("generated_at"):
        raise CurrentScreeningIssue("HANDOFF_TIME_INVALID", "handoff generation time is invalid")
    history = load_advisory_ohlc_history(history_root, universe_path=universe_path, policy_path=history_policy_path, _clock=lambda: now)
    screening_manifest, ranking, universe_index = _load_screening(screening_root, history, screening_policy_path=screening_policy_path)
    price = load_advisory_price_artifact(price_root, universe_path=universe_path, policy_path=price_policy_path, trusted_now=_utc_text(now))
    downstream = None
    if downstream_after_authority_path is not None or execution_proof is not None:
        downstream = load_downstream_after_authority(
            downstream_after_authority_path or "",
            execution_proof=execution_proof,
            repository_root=downstream_repository_root,
        )
        if downstream is None:
            raise CurrentScreeningIssue("DOWNSTREAM_AUTHORITY_INVALID", "execution proof and downstream after-authority are invalid")
    prestate = load_downstream_prestate()
    if prestate.get("measurement_status") != "measured":
        raise CurrentScreeningIssue("DOWNSTREAM_PRESTATE_INVALID", "canonical downstream prestate is invalid")
    derived = _derive_handoff(
        run_id=str(manifest["run_id"]),
        history=history,
        screening_manifest=screening_manifest,
        ranking=ranking,
        universe_index=universe_index,
        price=price,
        screening_root=Path(screening_root), history_root=Path(history_root), price_root=Path(price_root),
        approval_decision_paths=approval_decision_paths,
        downstream_payload=validated_after_payload(downstream),
        prestate=prestate,
        portfolio_binding=_portfolio_binding(portfolio_ledger_path),
    )
    if candidate_input != {"schema_version": HANDOFF_VERSION, "run_id": manifest["run_id"], "records": derived["records"]}:
        raise CurrentScreeningIssue("HANDOFF_SEMANTIC_REPLAY_INVALID", "candidate input differs from validated evidence")
    if reconciliation != derived["reconciliation"]:
        raise CurrentScreeningIssue("HANDOFF_SEMANTIC_REPLAY_INVALID", "price reconciliation differs from validated evidence")
    for key, value in derived["manifest_semantics"].items():
        if manifest.get(key) != value:
            raise CurrentScreeningIssue("HANDOFF_SEMANTIC_REPLAY_INVALID", f"handoff manifest {key} differs from validated evidence")
    return _ValidatedRun33HandoffContext(_RUN33_TOKEN, manifest, candidate_input, reconciliation, downstream)


def validated_run33_handoff_payload(value: Any) -> Mapping[str, Any] | None:
    if (
        isinstance(value, _ValidatedRun33HandoffContext)
        and value._token is _RUN33_TOKEN
        and value.manifest.get("status") == "ready_for_run33"
    ):
        return value.candidate_input
    return None


def _derive_handoff(
    *, run_id: str, history: Any, screening_manifest: Mapping[str, Any], ranking: Mapping[str, Any],
    universe_index: Mapping[str, Any], price: Mapping[str, Any], screening_root: Path,
    history_root: Path, price_root: Path, approval_decision_paths: Sequence[str | Path],
    downstream_payload: Mapping[str, Any] | None, prestate: Mapping[str, Any],
    portfolio_binding: Mapping[str, Any],
) -> dict[str, Any]:
    price_records = {row["instrument_id"]: row for row in price["observations"]["records"]}
    technical_rows = {row["instrument_id"]: row for row in universe_index["records"]}
    checkpoint = _data11_checkpoint(approval_decision_paths, downstream_payload)
    authoritative_state = downstream_payload if checkpoint["status"] == "ready_for_run33" else prestate
    by_ticker = authoritative_state.get("by_ticker") or {}
    expected_identity = {(str(row["symbol"]), str(row["instrument_id"])) for row in history.universe["instruments"]}
    actual_identity = {(str(ticker), str(row.get("instrument_id"))) for ticker, row in by_ticker.items()}
    if actual_identity != expected_identity:
        raise CurrentScreeningIssue("DOWNSTREAM_UNIVERSE_INVALID", "downstream state does not reconcile all canonical identities")
    reconciliation_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    for instrument in sorted(history.universe["instruments"], key=lambda row: str(row["instrument_id"])):
        instrument_id, ticker = str(instrument["instrument_id"]), str(instrument["symbol"])
        technical = technical_rows[instrument_id]
        fundamental = by_ticker[ticker]
        price_row = price_records.get(instrument_id)
        effective_price = price["effective_freshness"].get(instrument_id, {"status": "missing"})
        reasons: list[str] = []
        reconciliation_status = "blocked"
        latest_close = history.series.get(instrument_id, {}).get("bars", [{}])[-1].get("close") if instrument_id in history.series else None
        if price_row is None:
            reasons.append("ADVISORY_PRICE_MISSING")
        else:
            error_map = {"INSTRUMENT_IDENTITY_MISMATCH": "PRICE_IDENTITY_MISMATCH", "CURRENCY_MISMATCH": "PRICE_CURRENCY_MISMATCH", "CURRENCY_INVALID": "PRICE_CURRENCY_MISMATCH"}
            if price_row.get("error_code") in error_map: reasons.append(error_map[price_row["error_code"]])
            if price_row.get("canonical_ticker") != ticker: reasons.append("PRICE_IDENTITY_MISMATCH")
            if price_row.get("currency") != instrument["currency"]: reasons.append("PRICE_CURRENCY_MISMATCH")
            if effective_price["status"] != "fresh": reasons.append("ADVISORY_PRICE_NOT_FRESH")
            last_session = next((row["last_session"] for row in history.index if row["instrument_id"] == instrument_id), None)
            if price_row.get("observation_timestamp") and price_row["observation_timestamp"][:10] != last_session: reasons.append("PRICE_SESSION_MISMATCH")
            if not reasons and (latest_close is None or Decimal(price_row["price"]) != Decimal(latest_close)): reasons.append("PRICE_CLOSE_MISMATCH")
            if not reasons: reconciliation_status = "passed"
        reconciliation_rows.append({"instrument_id": instrument_id, "canonical_ticker": ticker, "status": reconciliation_status, "history_close": latest_close, "advisory_price": price_row.get("price") if price_row and effective_price["status"] == "fresh" else None, "reason_codes": reasons})
        conditions = {
            "current_technical_history": history.effective_status[instrument_id] == "fresh" and history.manifest["analytic_authority_status"] == "usable",
            "current_technical_screening": technical["screening_status"] == "completed",
            "fresh_advisory_price": effective_price["status"] == "fresh",
            "technical_price_reconciliation": reconciliation_status == "passed",
            "authoritative_fundamental_context": fundamental.get("overall_fundamental_status") == "complete" and fundamental.get("canonical_advice_input_ready") is True,
            "identity_reconciliation": price_row is not None and price_row.get("canonical_ticker") == ticker,
            "data11_checkpoint_approved": checkpoint["status"] == "ready_for_run33",
        }
        failed = [key.upper() for key, passed in conditions.items() if not passed]
        handoff_rows.append({"instrument_id": instrument_id, "canonical_ticker": ticker, "eligible_for_run33": all(conditions.values()), "reason_codes": failed, "technical_evidence_status": technical["screening_status"], "price_evidence_status": effective_price["status"], "fundamental_evidence_status": str(fundamental.get("overall_fundamental_status") or "missing"), "portfolio_context_status": portfolio_binding["status"], "conditions": conditions})
    reconciliation = {"schema_version": RECONCILIATION_VERSION, "records": reconciliation_rows, "counts": dict(sorted(Counter(row["status"] for row in reconciliation_rows).items()))}
    downstream_binding = None if downstream_payload is None else {
        "authority_path": downstream_payload["authority_path"],
        "authority_sha256": downstream_payload["authority_sha256"],
        "data06_run_id": downstream_payload["data06_run_id"],
        "run31_run_id": downstream_payload["run31_run_id"],
        "payload_sha256": _sha256(_canonical_json(downstream_payload)),
    }
    candidate_input = {"schema_version": HANDOFF_VERSION, "run_id": run_id, "records": handoff_rows}
    semantics = {"status": checkpoint["status"], "eligible_count": sum(row["eligible_for_run33"] for row in handoff_rows), "candidate_input_sha256": _sha256(_canonical_json(candidate_input) + b"\n"), "screening_manifest_sha256": _sha256_file(screening_root / "manifest.json"), "candidate_ranking_sha256": _sha256_file(screening_root / "candidate_ranking.json"), "universe_index_sha256": _sha256_file(screening_root / "universe_analysis_index.json"), "history_manifest_sha256": _sha256_file(history_root / "manifest.json"), "price_manifest_sha256": _sha256_file(price_root / "advisory_price_manifest.json"), "price_observations_sha256": _sha256_file(price_root / "advisory_price_observations.json"), "reconciliation_sha256": _sha256(_canonical_json(reconciliation) + b"\n"), "fundamental_prestate_authority": {"path": prestate["authority_path"], "sha256": prestate["authority_sha256"]}, "validated_downstream_authority": downstream_binding, "data11_checkpoint": checkpoint, "canonical_universe_sha256": history.manifest["universe_sha256"], "history_policy_sha256": history.manifest["history_policy_sha256"], "screening_policy_binding": screening_manifest["screening_policy_binding"], "price_policy_sha256": price["manifest"]["freshness_policy_sha256"], "portfolio_context_binding": portfolio_binding, "downstream_execution": {"data07_calls": 0, "data06_calls": 0, "run31_calls": 0, "run33_calls": 0}, "authority_boundary": "input_handoff_only_no_decision_authority"}
    return {"manifest_semantics": semantics, "reconciliation": reconciliation, "records": handoff_rows}


def _load_screening(
    root_value: str | Path,
    history: Any,
    *,
    screening_policy_path: str | Path = DEFAULT_SCREENING_POLICY,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(root_value, Mapping): raise CurrentScreeningIssue("CALLER_CONTENT_FORBIDDEN", "screening authority requires an artifact path")
    root = Path(root_value); checksums = _json(root / "checksum_index.json")
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise CurrentScreeningIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in screening artifacts")
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "checksum_index.json")
    expected_files = sorted({
        "manifest.json", "universe_analysis_index.json", "setup_detection_summary.json",
        "analysis_outcome_distribution.json", "blocker_report.json", "candidate_ranking.json",
        "old_vs_new_screening_drift.json", "candidate_ranking.md", "top_candidates.md",
    })
    if checksums.get("schema_version") != "market-engine-checksum-index-v1" or actual != expected_files or actual != sorted(checksums.get("files", {})):
        raise CurrentScreeningIssue("SCREENING_INTEGRITY_INVALID", "checksum index does not enumerate screening files exactly")
    for name, digest in checksums.get("files", {}).items():
        if _sha256_file(root / name) != digest: raise CurrentScreeningIssue("SCREENING_INTEGRITY_INVALID", f"checksum mismatch for {name}")
    manifest, ranking, index = _json(root / "manifest.json"), _json(root / "candidate_ranking.json"), _json(root / "universe_analysis_index.json")
    if manifest.get("schema_version") != SCREENING_MANIFEST_VERSION or ranking.get("schema_version") != RANKING_VERSION:
        raise CurrentScreeningIssue("SCREENING_CONTRACT_INVALID", "screening contract is invalid")
    integrity = dict(manifest); artifact_sha = integrity.pop("artifact_sha256", None)
    if artifact_sha != _sha256(_canonical_json(integrity)):
        raise CurrentScreeningIssue("SCREENING_INTEGRITY_INVALID", "screening manifest integrity is invalid")
    binding = manifest.get("history_binding", {})
    if binding.get("artifact_sha256") != history.manifest["artifact_sha256"] or ranking.get("history_binding") != binding:
        raise CurrentScreeningIssue("HISTORY_BINDING_INVALID", "screening does not bind the loaded history")
    if manifest.get("candidate_ranking_sha256") != _sha256_file(root / "candidate_ranking.json") or manifest.get("universe_index_sha256") != _sha256_file(root / "universe_analysis_index.json"):
        raise CurrentScreeningIssue("SCREENING_BINDING_INVALID", "manifest does not bind ranking and universe index")
    policy_source = Path(screening_policy_path)
    policy = _load_screening_policy(policy_source)
    expected_policy_binding = {"schema_version": policy["schema_version"], "policy_id": policy["policy_id"], "path": policy_source.as_posix(), "sha256": _sha256_file(policy_source)}
    if manifest.get("screening_policy_binding") != expected_policy_binding or ranking.get("screening_policy_binding") != expected_policy_binding:
        raise CurrentScreeningIssue("SCREENING_POLICY_BINDING_INVALID", "screening policy binding changed")
    expected = {str(row["instrument_id"]) for row in history.universe["instruments"]}
    actual_ids = [str(row.get("instrument_id")) for row in index.get("records", [])]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected: raise CurrentScreeningIssue("SCREENING_UNIVERSE_INVALID", "screening universe is incomplete")
    replayed_rows, replayed_ranked = _derive_screening(history, top_limit=policy["top_candidate_limit"])
    replayed_ranking = replayed_ranked[: policy["top_candidate_limit"]]
    if index.get("records") != replayed_rows or ranking.get("records") != replayed_ranking:
        raise CurrentScreeningIssue("SCREENING_SEMANTIC_REPLAY_INVALID", "screening differs from history-derived calculations")
    if ranking.get("eligible_total") != len(replayed_ranked) or ranking.get("requested_top_limit") != policy["top_candidate_limit"] or ranking.get("ranking_gap") != max(0, policy["top_candidate_limit"] - len(replayed_ranking)):
        raise CurrentScreeningIssue("SCREENING_SEMANTIC_REPLAY_INVALID", "ranking totals differ from replayed policy")
    return manifest, ranking, index


def _frame(bars: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame([{"Date": row["session"], "Open": Decimal(row["open"]), "High": Decimal(row["high"]), "Low": Decimal(row["low"]), "Close": Decimal(row["close"]), "Volume": int(row["volume"]) if row["volume"] is not None else pd.NA} for row in bars])
    frame["Volume"] = pd.array(frame["Volume"], dtype="Int64")
    return frame


def _write_artifact(destination: Path, payloads: Mapping[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, payload in payloads.items():
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str): data = payload.encode()
        else: data = _canonical_json(payload) + b"\n"
        path.write_bytes(data); checksums[name] = _sha256(data)
    (destination / "checksum_index.json").write_bytes(_canonical_json({"schema_version": "market-engine-checksum-index-v1", "files": checksums}) + b"\n")


def _validate_handoff_files(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise CurrentScreeningIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in RUN33 handoff artifacts")
    checksums = _json(root / "checksum_index.json")
    expected = {"manifest.json", "technical_price_reconciliation.json", "run33_candidate_input.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "checksum_index.json"}
    if checksums.get("schema_version") != "market-engine-checksum-index-v1" or set(checksums.get("files", {})) != expected or actual != expected:
        raise CurrentScreeningIssue("HANDOFF_INTEGRITY_INVALID", "handoff files are not exactly enumerated")
    for name, digest in checksums["files"].items():
        if _sha256_file(root / name) != digest:
            raise CurrentScreeningIssue("HANDOFF_INTEGRITY_INVALID", f"checksum mismatch for {name}")


def _output_destination(output_root: str | Path, approved: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}", run_id): raise CurrentScreeningIssue("RUN_ID_INVALID", "run_id is invalid")
    root = Path(output_root)
    if root.is_absolute() or root != approved or ".." in root.parts: raise CurrentScreeningIssue("OUTPUT_PATH_INVALID", "output root is not approved")
    repository = _repository_root(); destination = (repository / root / run_id).resolve(); allowed = (repository / approved).resolve()
    if allowed not in destination.parents or destination.exists(): raise CurrentScreeningIssue("OUTPUT_PATH_INVALID", "output destination is unsafe or exists")
    cursor = repository
    for part in root.parts:
        cursor /= part
        if cursor.is_symlink(): raise CurrentScreeningIssue("OUTPUT_PATH_INVALID", "symlink output paths are forbidden")
    return destination


def _repository_root() -> Path:
    return Path.cwd().resolve()


def _ranking_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Current technical candidates", "", "Technical classification only; this is not recommendation or execution authority.", "", "| Rank | Instrument | Ticker | Score |", "|---:|---|---|---:|"]
    lines.extend(f"| {row['rank']} | {row['instrument_id']} | {row['symbol']} | {row['candidate_score']} |" for row in rows)
    return "\n".join(lines) + "\n"


def _load_old_ranking(path: Path) -> list[dict[str, Any]]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return []
    return list(value.get("candidates") or value.get("records") or []) if isinstance(value, dict) else []


def _load_screening_policy(path: Path) -> dict[str, Any]:
    policy = _json(path)
    required = {
        "schema_version", "policy_id", "indicator_contract", "setup_contract", "scoring_contract",
        "ranking_contract", "top_candidate_limit", "threshold_authority",
    }
    if set(policy) != required or policy.get("schema_version") != SCREENING_POLICY_VERSION:
        raise CurrentScreeningIssue("SCREENING_POLICY_INVALID", "screening policy contract is invalid")
    if (
        policy.get("top_candidate_limit") != 25
        or policy.get("ranking_contract") != "score-descending-then-symbol-then-instrument-id-v1"
        or policy.get("setup_contract") != "run30-detect-technical-setup-v1"
        or policy.get("scoring_contract") != "run30-candidate-score-v1"
        or policy.get("threshold_authority") != "src/market_engine/run/full_canonical_universe_analysis.py"
    ):
        raise CurrentScreeningIssue("SCREENING_POLICY_INVALID", "screening ranking semantics changed")
    indicators = policy.get("indicator_contract")
    if indicators != {
        "moving_average_sessions": [20, 50, 200],
        "average_true_range_proxy_sessions": 20,
        "high_low_window_sessions": 20,
        "support_window_excludes_latest_bar": True,
    }:
        raise CurrentScreeningIssue("SCREENING_POLICY_INVALID", "screening indicator semantics changed")
    return policy


def _json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc: raise CurrentScreeningIssue("ARTIFACT_READ_INVALID", f"cannot read {path.name}") from exc
    if not isinstance(value, dict): raise CurrentScreeningIssue("ARTIFACT_READ_INVALID", f"{path.name} must contain an object")
    return value


def _load_bound_json(path: Path) -> dict[str, Any]: return {"path": path.as_posix(), "sha256": _sha256_file(path), "payload_sha256": _sha256(_canonical_json(_json(path)))}


def _fundamental_statuses(path: Path) -> dict[str, str]:
    report = _json(path); records = report.get("tickers") or []
    result = {}
    for row in records:
        if isinstance(row, Mapping) and row.get("instrument_id"): result[str(row["instrument_id"])] = str(row.get("overall_fundamental_status") or "missing")
    return result


def _data11_checkpoint(
    decision_paths: Sequence[str | Path],
    downstream_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not decision_paths:
        raise CurrentScreeningIssue("DATA11_APPROVAL_INVALID", "at least one approval candidate is required")
    validations = []
    display = {}
    for raw_path in decision_paths:
        path = Path(raw_path)
        validation = validate_approval_decision(path)
        validations.append({
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "validation_status": validation.get("validation_status"),
            "reason_codes": list(validation.get("reason_codes") or []),
            "decision_id": validation.get("decision_id"),
            "ticker": validation.get("ticker"),
        })
        display[path.parent.name] = _json(path).get("decision")
    approved = [row for row in validations if row["validation_status"] == "approved"]
    pending = all("APPROVAL_PENDING" in row["reason_codes"] for row in validations)
    approval_binding = downstream_payload.get("approval_binding") if downstream_payload is not None else None
    downstream_matches_approval = isinstance(approval_binding, Mapping) and any(
        row["sha256"] == approval_binding.get("decision_sha256")
        and row["decision_id"] == approval_binding.get("decision_id")
        and row["ticker"] == approval_binding.get("ticker")
        for row in approved
    )
    if downstream_payload is not None and downstream_matches_approval:
        status = "ready_for_run33"
    elif downstream_payload is not None:
        status = "conditional_blocked_invalid_downstream_authority"
    elif approved:
        status = "conditional_blocked_pending_downstream_refresh"
    elif pending:
        status = "conditional_blocked_pending_data11_approval"
    else:
        status = "conditional_blocked_invalid_downstream_authority"
    return {
        "status": status,
        "display_decisions": display,
        "approval_validations": validations,
        "validated_downstream_authority_present": downstream_payload is not None,
    }


def _portfolio_binding(path: str | Path | None) -> dict[str, Any]:
    if path is None: return {"status": "not_supplied", "ledger_sha256": None}
    if isinstance(path, Mapping): raise CurrentScreeningIssue("PORTFOLIO_CONTEXT_INVALID", "caller projections are forbidden")
    loaded = load_ledger(path); projection = rebuild_positions(path)
    return {"status": "authoritative_ledger_bound", "ledger_sha256": _sha256_file(Path(path)), "ledger_digest": projection["ledger_digest"], "event_count": len(loaded["events"])}


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description="Build current technical screening from validated advisory history")
    parser.add_argument("--run-id", required=True); parser.add_argument("--history-artifact-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_SCREENING_ROOT.as_posix())
    args = parser.parse_args(argv)
    try: manifest, path = run_current_technical_screening(run_id=args.run_id, history_artifact_root=args.history_artifact_root, output_root=args.output_root)
    except (CurrentScreeningIssue, AdvisoryHistoryIssue) as exc:
        print(json.dumps({"status": "blocked", "code": getattr(exc, "code", type(exc).__name__)}), file=stderr); return 2
    print(json.dumps({"status": manifest["run_status"], "artifact_path": path.as_posix()}, sort_keys=True), file=stdout); return 0


if __name__ == "__main__": raise SystemExit(run_command())
