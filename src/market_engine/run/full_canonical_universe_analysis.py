from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import pandas as pd

from market_engine.data.complete_local_market_dataset import DEFAULT_CANONICAL_CONFIG
from market_engine.data.incremental_market_data_refresh import determine_safe_cutoff_date
from market_engine.data.local_market_data_universe import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MIN_HISTORY_ROWS,
    DEFAULT_PRICE_HISTORY_ROOT,
    build_universe_snapshot,
    inspect_price_history,
)


SCHEMA_VERSION = "market-engine-run30-full-canonical-universe-analysis-v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/universe_analysis_runs")
DEFAULT_TOP_CANDIDATE_LIMIT = 25

SETUP_STATES = (
    "breakout_candidate",
    "pullback_watch",
    "extended_wait",
    "weak_setup",
    "no_clear_setup",
)
ADVICE_LABELS = (
    "buy_candidate",
    "wait_for_price",
    "watchlist",
    "avoid_for_now",
    "unable_to_analyse",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")


def run_full_canonical_universe_analysis(
    *,
    run_id: str,
    universe_path: str | Path = DEFAULT_CANONICAL_CONFIG,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    cutoff_date: str | None = None,
    top_candidate_limit: int = DEFAULT_TOP_CANDIDATE_LIMIT,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    allow_overwrite: bool = True,
) -> tuple[dict[str, Any], Path]:
    effective_cutoff = cutoff_date or determine_safe_cutoff_date()
    started = time.perf_counter()
    universe = build_universe_snapshot(universe_path, price_history_root=price_history_root)
    entries = [
        _analyse_instrument(
            instrument,
            price_history_root=price_history_root,
            cutoff_date=effective_cutoff,
            min_history_rows=min_history_rows,
        )
        for instrument in sorted(universe["instruments"], key=lambda row: str(row["instrument_id"]))
    ]
    runtime_seconds = time.perf_counter() - started
    ranked = _rank_candidates(entries)
    artifacts = _artifacts(
        run_id=run_id,
        universe=universe,
        universe_path=universe_path,
        price_history_root=price_history_root,
        cutoff_date=effective_cutoff,
        entries=entries,
        ranked=ranked,
        runtime_seconds=runtime_seconds,
        top_candidate_limit=top_candidate_limit,
    )
    output_dir = _write_artifacts(
        artifacts,
        output_root=output_root,
        run_id=run_id,
        allow_overwrite=allow_overwrite,
    )
    return artifacts, output_dir


def _analyse_instrument(
    instrument: Mapping[str, Any],
    *,
    price_history_root: str | Path,
    cutoff_date: str,
    min_history_rows: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "instrument_id": instrument["instrument_id"],
        "symbol": instrument["symbol"],
        "source_symbol": instrument["source_symbol"],
        "asset_type": instrument["asset_type"],
        "universe_memberships": instrument["universe_memberships"],
    }
    try:
        inspection = inspect_price_history(
            instrument,
            price_history_root=price_history_root,
            required_forward_date=cutoff_date,
            min_history_rows=min_history_rows,
        )
        status = inspection["snapshotstatus"]
        if status != "valid_current_snapshot":
            return {
                **base,
                "final_processing_status": _processing_status(status),
                "analysis_status": "not_analysed",
                "advice_label": "unable_to_analyse",
                "confidence": "low",
                "candidate_score": None,
                "ranking_eligible": False,
                "blockers": _blockers_from_inspection(inspection),
                "missing_evidence": _missing_evidence_from_inspection(inspection),
                "price_history": _price_history_reference(inspection),
                "setup_detection": _blocked_setup(status),
                "analysis_outcome": "blocked",
                "runtime_seconds": _elapsed(started),
            }

        frame = _read_price_frame(Path(price_history_root) / f"{instrument['source_symbol']}.csv")
        setup = _detect_setup(frame)
        advice = _advice_from_setup(setup)
        ranking = _candidate_score(setup, inspection, advice)
        return {
            **base,
            "final_processing_status": "eligible_analyzed",
            "analysis_status": "analysed",
            "advice_label": advice["label"],
            "confidence": advice["confidence"],
            "candidate_score": ranking["score"],
            "ranking_eligible": ranking["eligible"],
            "blockers": advice["blockers"],
            "missing_evidence": advice["missing_evidence"],
            "price_history": _price_history_reference(inspection),
            "setup_detection": setup,
            "analysis_outcome": advice["analysis_outcome"],
            "score_components": ranking["components"],
            "traceability": ranking["traceability"],
            "runtime_seconds": _elapsed(started),
        }
    except Exception as exc:
        return {
            **base,
            "final_processing_status": "failed",
            "analysis_status": "failed",
            "advice_label": "unable_to_analyse",
            "confidence": "low",
            "candidate_score": None,
            "ranking_eligible": False,
            "blockers": [f"{type(exc).__name__}: {exc}"],
            "missing_evidence": ["valid_analysis_input"],
            "price_history": None,
            "setup_detection": _blocked_setup("technical_failure"),
            "analysis_outcome": "failed",
            "runtime_seconds": _elapsed(started),
        }


def _read_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", "Close", "High", "Low", "Open", "Volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing price columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    for column in ("Close", "High", "Low", "Open", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("Date").reset_index(drop=True)


def _detect_setup(frame: pd.DataFrame) -> dict[str, Any]:
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    latest_close = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean())
    high20 = float(high.tail(20).max())
    low20 = float(low.tail(20).min())
    atr20 = float((high.tail(20) - low.tail(20)).mean())
    if atr20 <= 0 or math.isnan(atr20):
        atr20 = max(latest_close * 0.01, 0.01)

    if latest_close > ma20 and latest_close > ma50 and (ma50 > ma200 or latest_close > ma200):
        trend_state = "uptrend"
    elif latest_close < ma20 and latest_close < ma50 and (ma50 < ma200 or latest_close < ma200):
        trend_state = "downtrend"
    else:
        trend_state = "sideways"

    distance_from_ma20_atr = (latest_close - ma20) / atr20
    distance_from_high_pct = (high20 - latest_close) / high20 if high20 else 0.0
    support_break_pct = (latest_close - low20) / low20 if low20 else 0.0
    if trend_state == "downtrend":
        setup_state = "weak_setup"
    elif trend_state == "uptrend" and distance_from_high_pct <= 0.03:
        setup_state = "breakout_candidate"
    elif trend_state == "uptrend" and -0.75 <= distance_from_ma20_atr <= 1.25:
        setup_state = "pullback_watch"
    elif trend_state == "uptrend" and distance_from_ma20_atr > 1.25:
        setup_state = "extended_wait"
    else:
        setup_state = "no_clear_setup"

    if support_break_pct < -0.02:
        price_position = "below_support_or_breakdown"
    elif distance_from_ma20_atr > 1.75:
        price_position = "above_preferred_entry"
    elif -0.75 <= distance_from_ma20_atr <= 1.25:
        price_position = "near_entry_zone"
    else:
        price_position = "fair_zone"

    volatility_pct = atr20 / latest_close if latest_close else 1.0
    if volatility_pct >= 0.06:
        risk_state = "high"
    elif volatility_pct >= 0.035:
        risk_state = "elevated"
    else:
        risk_state = "normal"

    return {
        "status": "completed",
        "trend_state": trend_state,
        "setup_state": setup_state,
        "price_position": price_position,
        "risk_state": risk_state,
        "latest_close": round(latest_close, 6),
        "ma20": round(ma20, 6),
        "ma50": round(ma50, 6),
        "ma200": round(ma200, 6),
        "twenty_day_high": round(high20, 6),
        "twenty_day_low": round(low20, 6),
        "atr20": round(atr20, 6),
        "distance_from_ma20_atr": round(distance_from_ma20_atr, 6),
        "volatility_pct": round(volatility_pct, 6),
    }


def _advice_from_setup(setup: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    missing: list[str] = ["fundamental_context", "portfolio_context", "market_context"]
    if setup["trend_state"] == "downtrend" or setup["risk_state"] == "high" or setup["price_position"] == "below_support_or_breakdown":
        return {
            "label": "avoid_for_now",
            "confidence": "medium",
            "analysis_outcome": "risk_or_trend_blocked",
            "blockers": ["weak_or_high_risk_setup"],
            "missing_evidence": missing,
        }
    if setup["setup_state"] == "breakout_candidate" and setup["price_position"] in {"near_entry_zone", "fair_zone"} and setup["risk_state"] == "normal":
        return {
            "label": "buy_candidate",
            "confidence": "medium",
            "analysis_outcome": "constructive_price_setup",
            "blockers": blockers,
            "missing_evidence": missing,
        }
    if setup["setup_state"] in {"breakout_candidate", "pullback_watch", "extended_wait"}:
        return {
            "label": "wait_for_price",
            "confidence": "medium" if setup["trend_state"] == "uptrend" else "low",
            "analysis_outcome": "constructive_but_not_entry_ready",
            "blockers": ["price_or_risk_not_preferred"],
            "missing_evidence": missing,
        }
    return {
        "label": "watchlist",
        "confidence": "low",
        "analysis_outcome": "inconclusive_setup",
        "blockers": ["no_clear_setup"],
        "missing_evidence": missing,
    }


def _candidate_score(
    setup: Mapping[str, Any],
    inspection: Mapping[str, Any],
    advice: Mapping[str, Any],
) -> dict[str, Any]:
    missing_evidence = list(advice.get("missing_evidence") or [])
    components = {
        "trend": {"uptrend": 35, "sideways": 10, "downtrend": -30}[str(setup["trend_state"])],
        "setup": {
            "breakout_candidate": 25,
            "pullback_watch": 22,
            "extended_wait": 8,
            "no_clear_setup": 0,
            "weak_setup": -30,
        }[str(setup["setup_state"])],
        "price_position": {
            "near_entry_zone": 20,
            "fair_zone": 12,
            "above_preferred_entry": -8,
            "below_support_or_breakdown": -35,
        }[str(setup["price_position"])],
        "risk": {"normal": 12, "elevated": -12, "high": -40}[str(setup["risk_state"])],
        "history_depth": min(10, int((int(inspection.get("row_count") or 0) / DEFAULT_MIN_HISTORY_ROWS) * 10)),
        "missing_evidence_penalty": -5 * len(missing_evidence),
    }
    raw_score = sum(components.values())
    score = max(0, min(100, raw_score))
    eligible = advice["label"] in {"buy_candidate", "wait_for_price", "watchlist"} and score >= 35
    return {
        "score": score,
        "eligible": eligible,
        "components": components,
        "traceability": {
            "price_history_path": inspection.get("artifactpath"),
            "start_date": inspection.get("start_date"),
            "end_date": inspection.get("end_date"),
            "row_count": inspection.get("row_count"),
            "setup_fields": {
                "trend_state": setup["trend_state"],
                "setup_state": setup["setup_state"],
                "price_position": setup["price_position"],
                "risk_state": setup["risk_state"],
            },
        },
    }


def _rank_candidates(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        {
            "rank": 0,
            "instrument_id": row["instrument_id"],
            "symbol": row["symbol"],
            "source_symbol": row["source_symbol"],
            "candidate_score": row["candidate_score"],
            "advice_label": row["advice_label"],
            "confidence": row["confidence"],
            "setup_detection": row["setup_detection"],
            "score_components": row.get("score_components") or {},
            "traceability": row.get("traceability") or {},
            "blockers": row.get("blockers") or [],
            "missing_evidence": row.get("missing_evidence") or [],
        }
        for row in entries
        if row.get("ranking_eligible")
    ]
    candidates.sort(key=lambda row: (-int(row["candidate_score"]), str(row["symbol"]), str(row["instrument_id"])))
    for index, row in enumerate(candidates, start=1):
        row["rank"] = index
    return candidates


def _artifacts(
    *,
    run_id: str,
    universe: Mapping[str, Any],
    universe_path: str | Path,
    price_history_root: str | Path,
    cutoff_date: str,
    entries: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    top_candidate_limit: int,
) -> dict[str, Any]:
    summary = _summary(universe, entries, ranked)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "market-engine-run30-full-canonical-universe-analysis",
        "run_id": run_id,
        "generated_at": _generated_at_from_run_id(run_id),
        "input": {
            "canonical_universe_path": Path(universe_path).as_posix(),
            "universe_version": universe["universe_version"],
            "price_history_root": Path(price_history_root).as_posix(),
            "cutoff_date": cutoff_date,
        },
        "outputs": {
            "universe_analysis_index": "universe_analysis_index.json",
            "candidate_ranking": "candidate_ranking.json",
            "top_candidates": "top_candidates.md",
        },
        "guardrails": {
            "provider_invocation_performed": False,
            "deterministic_advice_labels_produced": True,
            "provider_or_model_advice_generation_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "telegram_delivery_performed": False,
            "synthetic_forward_data_used": False,
        },
        "status": "completed_with_blockers" if summary["blocker_counts"] else "completed_successfully",
    }
    throughput = _throughput_report(entries, runtime_seconds)
    ranking = {
        "schema_version": "market-engine-run30-candidate-ranking-v1",
        "run_id": run_id,
        "ranking_policy": {
            "missing_data_positive_evidence": False,
            "blocked_or_incomplete_evidence_excluded": True,
            "tie_breakers": ["candidate_score desc", "symbol asc", "instrument_id asc"],
        },
        "candidate_count": len(ranked),
        "candidates": list(ranked),
    }
    top = list(ranked[:top_candidate_limit])
    return {
        "manifest": manifest,
        "universe_analysis_index": {
            "schema_version": "market-engine-run30-universe-analysis-index-v1",
            "run_id": run_id,
            "summary": summary,
            "instruments": list(entries),
        },
        "universe_analysis_summary": _render_universe_summary(run_id, summary, throughput),
        "throughput_report": throughput,
        "setup_detection_summary": _setup_summary(entries),
        "analysis_outcome_distribution": _analysis_distribution(entries),
        "blocker_report": _blocker_report(entries),
        "candidate_ranking": ranking,
        "candidate_ranking_markdown": _render_candidate_ranking(ranking),
        "top_candidates": _render_top_candidates(top, summary),
        "unable_to_analyse": _render_unable_to_analyse(entries),
    }


def _summary(
    universe: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(row["final_processing_status"]) for row in entries)
    advice = Counter(str(row["advice_label"]) for row in entries)
    confidence = Counter(str(row["confidence"]) for row in entries)
    blockers = Counter(
        blocker
        for row in entries
        for blocker in row.get("blockers") or ()
    )
    attempted = len(entries)
    return {
        "total_canonical_instruments": universe["summary"]["total_instruments"],
        "attempted_instruments": attempted,
        "eligible_analyzed": statuses.get("eligible_analyzed", 0),
        "insufficient_history": statuses.get("insufficient_history", 0),
        "blocked": statuses.get("blocked", 0),
        "failed": statuses.get("failed", 0),
        "unsupported_mapping": statuses.get("unsupported_mapping", 0),
        "final_processing_status_counts": dict(sorted(statuses.items())),
        "advice_label_counts": {label: advice.get(label, 0) for label in ADVICE_LABELS},
        "confidence_counts": {level: confidence.get(level, 0) for level in CONFIDENCE_LEVELS},
        "ranked_candidates": len(ranked),
        "blocker_counts": dict(sorted(blockers.items())),
    }


def _throughput_report(entries: Sequence[Mapping[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    attempted = len(entries)
    per_ticker = runtime_seconds / attempted if attempted else 0.0
    return {
        "schema_version": "market-engine-run30-throughput-report-v1",
        "total_runtime_seconds": round(runtime_seconds, 6),
        "attempted_instruments": attempted,
        "average_runtime_seconds_per_ticker": round(per_ticker, 8),
        "tickers_per_second": round(attempted / runtime_seconds, 6) if runtime_seconds else 0.0,
        "per_ticker_runtime": [
            {
                "instrument_id": row["instrument_id"],
                "symbol": row["symbol"],
                "runtime_seconds": row["runtime_seconds"],
            }
            for row in entries
        ],
    }


def _setup_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    setup = Counter(str((row.get("setup_detection") or {}).get("setup_state")) for row in entries)
    trend = Counter(str((row.get("setup_detection") or {}).get("trend_state")) for row in entries)
    risk = Counter(str((row.get("setup_detection") or {}).get("risk_state")) for row in entries)
    return {
        "schema_version": "market-engine-run30-setup-detection-summary-v1",
        "setup_state_counts": {state: setup.get(state, 0) for state in SETUP_STATES},
        "trend_state_counts": dict(sorted(trend.items())),
        "risk_state_counts": dict(sorted(risk.items())),
    }


def _analysis_distribution(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-run30-analysis-outcome-distribution-v1",
        "analysis_outcome_counts": dict(sorted(Counter(str(row["analysis_outcome"]) for row in entries).items())),
        "advice_label_counts": {label: Counter(str(row["advice_label"]) for row in entries).get(label, 0) for label in ADVICE_LABELS},
        "confidence_counts": {level: Counter(str(row["confidence"]) for row in entries).get(level, 0) for level in CONFIDENCE_LEVELS},
    }


def _blocker_report(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocker_counts = Counter(blocker for row in entries for blocker in row.get("blockers") or ())
    missing_counts = Counter(item for row in entries for item in row.get("missing_evidence") or ())
    return {
        "schema_version": "market-engine-run30-blocker-report-v1",
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "missing_evidence_counts": dict(sorted(missing_counts.items())),
        "blocked_instruments": [
            {
                "instrument_id": row["instrument_id"],
                "symbol": row["symbol"],
                "final_processing_status": row["final_processing_status"],
                "blockers": row.get("blockers") or [],
                "missing_evidence": row.get("missing_evidence") or [],
            }
            for row in entries
            if row["final_processing_status"] != "eligible_analyzed" or row.get("blockers")
        ],
    }


def _render_universe_summary(run_id: str, summary: Mapping[str, Any], throughput: Mapping[str, Any]) -> str:
    rows = [
        "# ME-RUN30 Universe Analysis Summary",
        "",
        f"Run ID: {run_id}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total canonical instruments | {summary['total_canonical_instruments']} |",
        f"| Attempted instruments | {summary['attempted_instruments']} |",
        f"| Eligible analysed | {summary['eligible_analyzed']} |",
        f"| Insufficient history | {summary['insufficient_history']} |",
        f"| Blocked | {summary['blocked']} |",
        f"| Failed | {summary['failed']} |",
        f"| Ranked candidates | {summary['ranked_candidates']} |",
        f"| Runtime seconds | {throughput['total_runtime_seconds']} |",
        f"| Tickers per second | {throughput['tickers_per_second']} |",
        "",
        "## Advice Distribution",
        "",
        "| Advice | Count |",
        "|---|---:|",
    ]
    for label, count in summary["advice_label_counts"].items():
        rows.append(f"| {label} | {count} |")
    rows.append("")
    return "\n".join(rows)


def _render_candidate_ranking(ranking: Mapping[str, Any]) -> str:
    rows = [
        "# ME-RUN30 Candidate Ranking",
        "",
        "| Rank | Symbol | Score | Advice | Confidence | Setup | Trend | Price position | Risk |",
        "|---:|---|---:|---|---|---|---|---|---|",
    ]
    for row in ranking["candidates"]:
        setup = row["setup_detection"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(row["rank"]),
                    _md(row["symbol"]),
                    str(row["candidate_score"]),
                    _md(row["advice_label"]),
                    _md(row["confidence"]),
                    _md(setup["setup_state"]),
                    _md(setup["trend_state"]),
                    _md(setup["price_position"]),
                    _md(setup["risk_state"]),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _render_top_candidates(candidates: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    rows = [
        "# ME-RUN30 Top Candidate Review Package",
        "",
        f"Ranked candidates: {summary['ranked_candidates']}",
        "",
    ]
    if not candidates:
        return "\n".join(rows + ["No review candidates passed the deterministic ranking gate.", ""])
    for row in candidates:
        setup = row["setup_detection"]
        trace = row["traceability"]
        rows.extend(
            [
                f"## {row['rank']}. {row['symbol']}",
                "",
                f"- Score: {row['candidate_score']}",
                f"- Advice label: {row['advice_label']}",
                f"- Confidence: {row['confidence']}",
                f"- Setup: {setup['setup_state']} / {setup['trend_state']} / {setup['price_position']} / {setup['risk_state']}",
                f"- Evidence: {trace.get('price_history_path')} rows={trace.get('row_count')} through {trace.get('end_date')}",
                f"- Missing evidence: {', '.join(row.get('missing_evidence') or []) or 'none'}",
                f"- Blockers: {', '.join(row.get('blockers') or []) or 'none'}",
                "",
            ]
        )
    return "\n".join(rows)


def _render_unable_to_analyse(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        "# ME-RUN30 Unable To Analyse",
        "",
        "| Symbol | Status | Blockers | Missing evidence |",
        "|---|---|---|---|",
    ]
    for row in entries:
        if row["final_processing_status"] == "eligible_analyzed":
            continue
        rows.append(
            f"| {_md(row['symbol'])} | {_md(row['final_processing_status'])} | {_md(', '.join(row.get('blockers') or []))} | {_md(', '.join(row.get('missing_evidence') or []))} |"
        )
    rows.append("")
    return "\n".join(rows)


def _write_artifacts(
    artifacts: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", artifacts["manifest"])
    _write_json(output_dir / "universe_analysis_index.json", artifacts["universe_analysis_index"])
    (output_dir / "universe_analysis_summary.md").write_text(artifacts["universe_analysis_summary"], encoding="utf-8")
    _write_json(output_dir / "throughput_report.json", artifacts["throughput_report"])
    _write_json(output_dir / "setup_detection_summary.json", artifacts["setup_detection_summary"])
    _write_json(output_dir / "analysis_outcome_distribution.json", artifacts["analysis_outcome_distribution"])
    _write_json(output_dir / "blocker_report.json", artifacts["blocker_report"])
    _write_json(output_dir / "candidate_ranking.json", artifacts["candidate_ranking"])
    (output_dir / "candidate_ranking.md").write_text(artifacts["candidate_ranking_markdown"], encoding="utf-8")
    (output_dir / "top_candidates.md").write_text(artifacts["top_candidates"], encoding="utf-8")
    (output_dir / "unable_to_analyse.md").write_text(artifacts["unable_to_analyse"], encoding="utf-8")
    return output_dir


def _processing_status(snapshotstatus: str) -> str:
    if snapshotstatus in {"insufficient_history", "insufficient_forward_data", "stale_snapshot"}:
        return "insufficient_history"
    if snapshotstatus == "unsupported_symbol_mapping":
        return "unsupported_mapping"
    if snapshotstatus in {"missing_price_history", "validation_failed"}:
        return "blocked"
    return "blocked"


def _blockers_from_inspection(inspection: Mapping[str, Any]) -> list[str]:
    return [str(inspection.get("blocker") or inspection.get("snapshotstatus") or "unknown_blocker")]


def _missing_evidence_from_inspection(inspection: Mapping[str, Any]) -> list[str]:
    status = str(inspection.get("snapshotstatus") or "")
    if status == "missing_price_history":
        return ["local_price_history"]
    if status in {"insufficient_history", "insufficient_forward_data", "stale_snapshot"}:
        return ["sufficient_local_price_history"]
    if status == "validation_failed":
        return ["valid_local_price_history"]
    if status == "unsupported_symbol_mapping":
        return ["supported_source_mapping"]
    return ["complete_analysis_evidence"]


def _price_history_reference(inspection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifactpath": inspection.get("artifactpath"),
        "snapshotstatus": inspection.get("snapshotstatus"),
        "start_date": inspection.get("start_date"),
        "end_date": inspection.get("end_date"),
        "row_count": inspection.get("row_count"),
        "checksum": inspection.get("checksum"),
    }


def _blocked_setup(reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "trend_state": "unknown",
        "setup_state": "no_clear_setup",
        "price_position": "unknown",
        "risk_state": "unknown",
    }


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 8)


def _generated_at_from_run_id(run_id: str) -> str:
    marker = run_id.rsplit("-", 1)[-1]
    try:
        return datetime.strptime(marker, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        artifacts, output_dir = run_full_canonical_universe_analysis(
            run_id=args.run_id,
            universe_path=args.universe,
            price_history_root=args.price_history_root,
            output_root=args.output_root,
            cutoff_date=args.cutoff_date,
            top_candidate_limit=args.top_candidate_limit,
            allow_overwrite=args.allow_overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=stderr)
        return 2
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output_dir.as_posix(),
                "summary": artifacts["universe_analysis_index"]["summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        file=stdout,
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ME-RUN30 full canonical-universe analysis and candidate ranking.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--universe", default=DEFAULT_CANONICAL_CONFIG.as_posix())
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--cutoff-date", default=None)
    parser.add_argument("--top-candidate-limit", type=int, default=DEFAULT_TOP_CANDIDATE_LIMIT)
    parser.add_argument("--allow-overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
