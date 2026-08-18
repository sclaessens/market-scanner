from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

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
    _sha256,
    _sha256_file,
    load_advisory_ohlc_history,
)
from market_engine.source_refresh.advisory_price_evidence import (
    DEFAULT_POLICY_PATH as DEFAULT_PRICE_POLICY_PATH,
    load_advisory_price_artifact,
)
from market_engine.portfolio_review.manual_transaction_ledger import load_ledger, rebuild_positions


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


class CurrentScreeningIssue(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def run_current_technical_screening(
    *, run_id: str, history_artifact_root: str | Path,
    output_root: str | Path = DEFAULT_SCREENING_ROOT,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    history_policy_path: str | Path = DEFAULT_POLICY_PATH,
    trusted_now: str | None = None, top_limit: int = 25,
) -> tuple[dict[str, Any], Path]:
    destination = _output_destination(output_root, DEFAULT_SCREENING_ROOT, run_id)
    history = load_advisory_ohlc_history(history_artifact_root, universe_path=universe_path, policy_path=history_policy_path, trusted_now=trusted_now)
    lag_blocked = bool(history.manifest["provider_session_lag"]["detected"])
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
        if lag_blocked:
            blockers.append("widespread_provider_session_lag")
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
    ranked = _rank_candidates(index_rows)
    ranking_records = ranked[:top_limit]
    old = _load_old_ranking(DEFAULT_RUN30_RANKING)
    old_symbols = [str(row.get("symbol")) for row in old[:top_limit]]
    new_symbols = [str(row["symbol"]) for row in ranking_records]
    cutoff_sessions = list(history.manifest["expected_last_completed_sessions"])
    history_binding = {"run_id": history.manifest["run_id"], "artifact_sha256": history.manifest["artifact_sha256"], "observations_sha256": history.manifest["observations_sha256"], "manifest_file_sha256": _sha256_file(history.root / "manifest.json")}
    universe_index = {"schema_version": "market-engine-current-technical-universe-index-v1", "run_id": run_id, "records": index_rows}
    ranking = {"schema_version": RANKING_VERSION, "run_id": run_id, "ranking_scope": "technical_setup_screening", "cutoff_sessions": cutoff_sessions, "history_binding": history_binding, "records": ranking_records, "eligible_total": len(ranked), "requested_top_limit": top_limit, "ranking_gap": max(0, top_limit - len(ranking_records))}
    payloads: dict[str, Any] = {
        "universe_analysis_index.json": universe_index,
        "setup_detection_summary.json": {"schema_version": "market-engine-current-technical-setup-summary-v1", "counts": dict(sorted(Counter(row.get("setup_detection", {}).get("setup_state", "blocked") for row in index_rows).items()))},
        "analysis_outcome_distribution.json": {"schema_version": "market-engine-current-technical-outcome-distribution-v1", "counts": dict(sorted(Counter(row["output_label"] for row in index_rows).items()))},
        "blocker_report.json": {"schema_version": "market-engine-current-technical-blocker-report-v1", "counts": dict(sorted(Counter(code for row in index_rows for code in row["blockers"]).items())), "records": [{"instrument_id": row["instrument_id"], "reason_codes": row["blockers"]} for row in index_rows if row["blockers"]]},
        "candidate_ranking.json": ranking,
        "old_vs_new_screening_drift.json": {"schema_version": "market-engine-current-technical-run30-drift-v1", "old_run30_is_audit_only": True, "old_top_symbols": old_symbols, "new_top_symbols": new_symbols, "retained": sorted(set(old_symbols) & set(new_symbols)), "added": sorted(set(new_symbols) - set(old_symbols)), "removed": sorted(set(old_symbols) - set(new_symbols))},
    }
    manifest_base = {"schema_version": SCREENING_MANIFEST_VERSION, "artifact_version": SCREENING_VERSION, "run_id": run_id, "generated_at": trusted_now or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"), "history_binding": history_binding, "universe_sha256": history.manifest["universe_sha256"], "history_policy_sha256": history.manifest["history_policy_sha256"], "screening_policy": "unchanged_run30_technical_setup_and_ranking_semantics", "cutoff_sessions": cutoff_sessions, "instrument_count": len(index_rows), "screened_count": sum(row["screening_status"] == "completed" for row in index_rows), "ranking_count": len(ranking_records), "candidate_ranking_sha256": _sha256(_canonical_json(ranking) + b"\n"), "universe_index_sha256": _sha256(_canonical_json(universe_index) + b"\n"), "run_status": "blocked_history_freshness" if lag_blocked else ("completed_with_blockers" if any(row["screening_status"] == "blocked" for row in index_rows) else "completed"), "authority_boundary": "technical_classification_only"}
    payloads["manifest.json"] = {**manifest_base, "artifact_sha256": _sha256(_canonical_json(manifest_base))}
    payloads["candidate_ranking.md"] = _ranking_markdown(ranking_records)
    payloads["top_candidates.md"] = _ranking_markdown(ranking_records)
    _write_artifact(destination, payloads)
    return payloads["manifest.json"], destination


def build_run33_grounded_handoff(
    *, run_id: str, screening_root: str | Path, history_root: str | Path, price_root: str | Path,
    output_root: str | Path = DEFAULT_HANDOFF_ROOT,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    history_policy_path: str | Path = DEFAULT_POLICY_PATH,
    price_policy_path: str | Path = DEFAULT_PRICE_POLICY_PATH,
    trusted_now: str | None = None, portfolio_ledger_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    destination = _output_destination(output_root, DEFAULT_HANDOFF_ROOT, run_id)
    for candidate in (Path(screening_root), Path(history_root), Path(price_root)):
        if candidate.is_symlink() or any(path.is_symlink() for path in candidate.rglob("*")):
            raise CurrentScreeningIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in authority artifacts")
    history = load_advisory_ohlc_history(history_root, universe_path=universe_path, policy_path=history_policy_path, trusted_now=trusted_now)
    screening_manifest, ranking, universe_index = _load_screening(screening_root, history)
    price = load_advisory_price_artifact(price_root, universe_path=universe_path, policy_path=price_policy_path, trusted_now=trusted_now)
    price_records = {row["instrument_id"]: row for row in price["observations"]["records"]}
    technical_rows = {row["instrument_id"]: row for row in universe_index["records"]}
    fundamental = _fundamental_statuses(DEFAULT_FUNDAMENTAL_STATUS)
    checkpoint = _data11_checkpoint(DEFAULT_DATA11_ROOT)
    portfolio_binding = _portfolio_binding(portfolio_ledger_path)
    reconciliation_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    for instrument in sorted(history.universe["instruments"], key=lambda row: str(row["instrument_id"])):
        instrument_id, ticker = str(instrument["instrument_id"]), str(instrument["symbol"])
        technical = technical_rows[instrument_id]
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
            "current_technical_history": history.effective_status[instrument_id] == "fresh" and not history.manifest["provider_session_lag"]["detected"],
            "current_technical_screening": technical["screening_status"] == "completed",
            "fresh_advisory_price": effective_price["status"] == "fresh",
            "technical_price_reconciliation": reconciliation_status == "passed",
            "authoritative_fundamental_context": fundamental.get(instrument_id) == "available",
            "identity_reconciliation": price_row is not None and price_row.get("canonical_ticker") == ticker,
            "data11_checkpoint_approved": checkpoint["status"] == "approved_with_refreshed_downstream_authority",
        }
        failed = [key.upper() for key, passed in conditions.items() if not passed]
        handoff_rows.append({"instrument_id": instrument_id, "canonical_ticker": ticker, "eligible_for_run33": all(conditions.values()), "reason_codes": failed, "technical_evidence_status": technical["screening_status"], "price_evidence_status": effective_price["status"], "fundamental_evidence_status": fundamental.get(instrument_id, "missing"), "portfolio_context_status": portfolio_binding["status"], "conditions": conditions})
    reconciliation = {"schema_version": RECONCILIATION_VERSION, "records": reconciliation_rows, "counts": dict(sorted(Counter(row["status"] for row in reconciliation_rows).items()))}
    authority = _load_bound_json(DEFAULT_PRESTATE_AUTHORITY)
    manifest_base = {"schema_version": HANDOFF_MANIFEST_VERSION, "run_id": run_id, "status": "conditional_blocked_pending_data11_approval" if checkpoint["status"] == "pending" else "conditional_blocked_pending_refreshed_downstream_authority", "eligible_count": sum(row["eligible_for_run33"] for row in handoff_rows), "screening_manifest_sha256": _sha256_file(Path(screening_root) / "manifest.json"), "candidate_ranking_sha256": _sha256_file(Path(screening_root) / "candidate_ranking.json"), "universe_index_sha256": _sha256_file(Path(screening_root) / "universe_analysis_index.json"), "history_manifest_sha256": _sha256_file(Path(history_root) / "manifest.json"), "price_manifest_sha256": _sha256_file(Path(price_root) / "advisory_price_manifest.json"), "price_observations_sha256": _sha256_file(Path(price_root) / "advisory_price_observations.json"), "reconciliation_sha256": _sha256(_canonical_json(reconciliation)), "fundamental_prestate_authority": authority, "fundamental_status_sha256": _sha256_file(DEFAULT_FUNDAMENTAL_STATUS), "data11_checkpoint": checkpoint, "canonical_universe_sha256": history.manifest["universe_sha256"], "history_policy_sha256": history.manifest["history_policy_sha256"], "price_policy_sha256": price["manifest"]["freshness_policy_sha256"], "portfolio_context_binding": portfolio_binding, "downstream_execution": {"data07_calls": 0, "data06_calls": 0, "run31_calls": 0, "run33_calls": 0}, "authority_boundary": "input_handoff_only_no_decision_authority"}
    manifest = {**manifest_base, "artifact_sha256": _sha256(_canonical_json(manifest_base))}
    payloads = {"manifest.json": manifest, "technical_price_reconciliation.json": reconciliation, "run33_candidate_input.json": {"schema_version": HANDOFF_VERSION, "run_id": run_id, "records": handoff_rows}}
    _write_artifact(destination, payloads)
    return manifest, destination


def _load_screening(root_value: str | Path, history: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(root_value, Mapping): raise CurrentScreeningIssue("CALLER_CONTENT_FORBIDDEN", "screening authority requires an artifact path")
    root = Path(root_value); checksums = _json(root / "checksum_index.json")
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise CurrentScreeningIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in screening artifacts")
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "checksum_index.json")
    if checksums.get("schema_version") != "market-engine-checksum-index-v1" or actual != sorted(checksums.get("files", {})):
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
    expected = {str(row["instrument_id"]) for row in history.universe["instruments"]}
    actual_ids = [str(row.get("instrument_id")) for row in index.get("records", [])]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected: raise CurrentScreeningIssue("SCREENING_UNIVERSE_INVALID", "screening universe is incomplete")
    return manifest, ranking, index


def _frame(bars: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"Date": row["session"], "Open": Decimal(row["open"]), "High": Decimal(row["high"]), "Low": Decimal(row["low"]), "Close": Decimal(row["close"]), "Volume": int(row["volume"]) if row["volume"] is not None else 0} for row in bars])


def _write_artifact(destination: Path, payloads: Mapping[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, payload in payloads.items():
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str): data = payload.encode()
        else: data = _canonical_json(payload) + b"\n"
        path.write_bytes(data); checksums[name] = _sha256(data)
    (destination / "checksum_index.json").write_bytes(_canonical_json({"schema_version": "market-engine-checksum-index-v1", "files": checksums}) + b"\n")


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


def _data11_checkpoint(root: Path) -> dict[str, Any]:
    files = [root / "approval_candidates" / ticker / "approval_candidate.json" for ticker in ("ASH", "BIO", "CI")]
    decisions = {path.parent.name: _json(path).get("decision") for path in files}
    status = "approval_complete_downstream_refresh_required" if decisions and set(decisions.values()) == {"approved"} else "pending"
    return {"status": status, "decisions": decisions, "bindings": {path.parent.name: _sha256_file(path) for path in files}, "refreshed_downstream_authority_present": False}


def _portfolio_binding(path: str | Path | None) -> dict[str, Any]:
    if path is None: return {"status": "not_supplied", "ledger_sha256": None}
    if isinstance(path, Mapping): raise CurrentScreeningIssue("PORTFOLIO_CONTEXT_INVALID", "caller projections are forbidden")
    loaded = load_ledger(path); projection = rebuild_positions(path)
    return {"status": "authoritative_ledger_bound", "ledger_sha256": _sha256_file(Path(path)), "ledger_digest": projection["ledger_digest"], "event_count": len(loaded["events"])}


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description="Build current technical screening from validated advisory history")
    parser.add_argument("--run-id", required=True); parser.add_argument("--history-artifact-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_SCREENING_ROOT.as_posix()); parser.add_argument("--trusted-now")
    args = parser.parse_args(argv)
    try: manifest, path = run_current_technical_screening(run_id=args.run_id, history_artifact_root=args.history_artifact_root, output_root=args.output_root, trusted_now=args.trusted_now)
    except (CurrentScreeningIssue, AdvisoryHistoryIssue) as exc:
        print(json.dumps({"status": "blocked", "code": getattr(exc, "code", type(exc).__name__)}), file=stderr); return 2
    print(json.dumps({"status": manifest["run_status"], "artifact_path": path.as_posix()}, sort_keys=True), file=stdout); return 0


if __name__ == "__main__": raise SystemExit(run_command())
