from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.evaluation.advice_outcomes import (
    build_advice_outcome_evaluation,
    price_history_snapshot_summary,
)


ADVICE_OUTCOME_REFRESH_SCHEMA_VERSION = "market-engine-advice-outcome-refresh-run-v1"
ADVICE_OUTCOME_REFRESH_MANIFEST_SCHEMA_VERSION = (
    "market-engine-advice-outcome-refresh-manifest-v1"
)


def build_advice_outcome_refresh(
    evaluation_artifact: str | Path,
    *,
    price_history_root: str | Path,
    run_id: str,
    ticker_filter: Sequence[str] | None = None,
) -> dict[str, Any]:
    evaluation_path = Path(evaluation_artifact)
    price_root = Path(price_history_root)
    previous = _load_previous_outcome_index(evaluation_path)
    selected_previous = _select_unresolved(previous, ticker_filter=ticker_filter)
    advice_index_path = _advice_index_path(previous)
    advice_exists = advice_index_path is not None and advice_index_path.exists()
    horizons = previous.get("horizons") or []
    refreshed_by_ticker: dict[str, Mapping[str, Any]] = {}
    if advice_exists:
        refreshed = build_advice_outcome_evaluation(
            advice_index_path,
            price_data_root=price_root,
            run_id=run_id,
            horizons=horizons,
            ticker_filter=[row["ticker"] for row in selected_previous],
        )
        refreshed_by_ticker = {
            row["ticker"]: row for row in refreshed["advice_outcome_index"]["tickers"]
        }

    outcomes = []
    for previous_row in selected_previous:
        ticker = previous_row["ticker"]
        snapshot = price_history_snapshot_summary(price_root, ticker)
        if not advice_exists:
            new_row = None
            new_status = "blocked"
            new_blocker = "missing_evaluation_context"
            explanation = "Refresh could not load the original advice index from the previous evaluation context."
        else:
            new_row = refreshed_by_ticker.get(ticker)
            if new_row is None:
                new_status = "blocked"
                new_blocker = "unknown_ticker"
                explanation = "Ticker was selected from unresolved outcomes but was not found in the original advice index."
            else:
                new_status, new_blocker = _row_status(new_row)
                explanation = _explanation(new_status, new_blocker, snapshot)
        outcomes.append(
            {
                "ticker": ticker,
                "advice": previous_row.get("advice"),
                "advice_evaluation_identifier": {
                    "previous_evaluation_run_id": previous.get("run_id"),
                    "previous_evaluation_artifact": evaluation_path.as_posix(),
                    "advice_index_path": advice_index_path.as_posix()
                    if advice_index_path is not None
                    else None,
                },
                "previous_status": _previous_status(previous_row),
                "previous_blocker": _previous_blocker(previous_row),
                "used_snapshot": snapshot["path"],
                "snapshot_status": snapshot["status"],
                "snapshot_last_available_date": snapshot["last_available_date"],
                "new_status": new_status,
                "new_blocker": new_blocker,
                "resolved": new_status == "resolved",
                "outcome_metrics": (new_row or {}).get("outcomes", {}),
                "explanation": explanation,
            }
        )

    refresh_index = {
        "schema_version": ADVICE_OUTCOME_REFRESH_SCHEMA_VERSION,
        "artifact_type": "market-engine-advice-outcome-refresh-run",
        "run_id": run_id,
        "generated_at": _generated_at_from_run_id_or_previous(run_id, previous),
        "run_type": "scheduled_future_outcome_refresh_local_snapshots",
        "input_mode": "existing_evaluation_artifact",
        "input": {
            "evaluation_artifact_path": evaluation_path.as_posix(),
            "previous_evaluation_run_id": previous.get("run_id"),
            "advice_index_path": advice_index_path.as_posix()
            if advice_index_path is not None
            else None,
            "price_history_root": price_root.as_posix(),
            "ticker_filter": [ticker.upper() for ticker in ticker_filter or ()],
        },
        "horizons": horizons,
        "summary": _summary(outcomes),
        "outcomes": outcomes,
    }
    return {
        "manifest": _manifest(refresh_index),
        "refresh_index": refresh_index,
        "refresh_report": render_refresh_report(refresh_index),
        "missing_price_history": _missing_price_history(refresh_index),
    }


def write_advice_outcome_refresh(
    refresh: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", refresh["manifest"])
    _write_json(output_dir / "refresh_outcome_index.json", refresh["refresh_index"])
    _write_json(output_dir / "missing_price_history.json", refresh["missing_price_history"])
    (output_dir / "refresh_report.md").write_text(
        refresh["refresh_report"],
        encoding="utf-8",
    )
    return output_dir


def run_advice_outcome_refresh(
    evaluation_artifact: str | Path,
    *,
    price_history_root: str | Path,
    output_root: str | Path,
    run_id: str,
    ticker_filter: Sequence[str] | None = None,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    refresh = build_advice_outcome_refresh(
        evaluation_artifact,
        price_history_root=price_history_root,
        run_id=run_id,
        ticker_filter=ticker_filter,
    )
    output_dir = write_advice_outcome_refresh(
        refresh,
        output_root=output_root,
        run_id=run_id,
        allow_overwrite=allow_overwrite,
    )
    return refresh, output_dir


def render_refresh_report(refresh_index: Mapping[str, Any]) -> str:
    summary = refresh_index["summary"]
    rows = [
        "# Advice Outcome Refresh Report",
        "",
        f"Run ID: {refresh_index['run_id']}",
        f"Previous evaluation: {refresh_index['input']['evaluation_artifact_path']}",
        f"Price history root: {refresh_index['input']['price_history_root']}",
        f"Generated at: {refresh_index['generated_at']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Selected outcomes | {summary['selected_outcomes']} |",
        f"| Resolved | {summary['resolved']} |",
        f"| Still unresolved | {summary['still_unresolved']} |",
        f"| Insufficient forward data | {summary['insufficient_forward_data']} |",
        f"| Missing price history | {summary['missing_price_history']} |",
        f"| Other blockers | {summary['other_blockers']} |",
        "",
        "## Missing Price History",
        "",
    ]
    missing = summary["missing_price_history_tickers"]
    rows.append(", ".join(missing) if missing else "None")
    rows.extend(
        [
            "",
            "## Refresh Outcomes",
            "",
            "| Ticker | Advice | Previous status | Previous blocker | Snapshot | Last price date | New status | New blocker | Explanation |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in refresh_index["outcomes"]:
        rows.append(
            "| "
            + " | ".join(
                (
                    _md(row["ticker"]),
                    _md(row.get("advice")),
                    _md(row["previous_status"]),
                    _md(row.get("previous_blocker")),
                    _md(row.get("used_snapshot") or "none"),
                    _md(row.get("snapshot_last_available_date") or ""),
                    _md(row["new_status"]),
                    _md(row.get("new_blocker") or ""),
                    _md(row["explanation"]),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _load_previous_outcome_index(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "advice_outcome_index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_type") != "market-engine-advice-outcome-index":
        raise ValueError(f"unsupported evaluation artifact: {path}")
    return data


def _select_unresolved(
    previous: Mapping[str, Any],
    *,
    ticker_filter: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    allowed = {ticker.upper() for ticker in ticker_filter or ()}
    selected = []
    for row in previous.get("tickers") or ():
        ticker = str(row.get("ticker") or "").upper()
        if allowed and ticker not in allowed:
            continue
        if any(outcome.get("status") == "unresolved" for outcome in row.get("outcomes", {}).values()):
            selected.append(row)
    return sorted(selected, key=lambda row: row["ticker"])


def _advice_index_path(previous: Mapping[str, Any]) -> Path | None:
    path = (previous.get("input") or {}).get("advice_index_path")
    return Path(path) if path else None


def _previous_status(row: Mapping[str, Any]) -> str:
    return "resolved" if _row_has_resolved_horizon(row) else "unresolved"


def _previous_blocker(row: Mapping[str, Any]) -> str | None:
    reasons = sorted(
        {
            str(outcome.get("reason"))
            for outcome in (row.get("outcomes") or {}).values()
            if outcome.get("status") == "unresolved" and outcome.get("reason")
        }
    )
    return ",".join(reasons) if reasons else None


def _row_status(row: Mapping[str, Any]) -> tuple[str, str | None]:
    if _row_has_resolved_horizon(row):
        return "resolved", None
    reasons = sorted(
        {
            str(outcome.get("reason"))
            for outcome in (row.get("outcomes") or {}).values()
            if outcome.get("status") == "unresolved" and outcome.get("reason")
        }
    )
    return "unresolved", ",".join(reasons) if reasons else "unknown"


def _row_has_resolved_horizon(row: Mapping[str, Any]) -> bool:
    return any(outcome.get("status") == "resolved" for outcome in (row.get("outcomes") or {}).values())


def _explanation(
    new_status: str,
    new_blocker: str | None,
    snapshot: Mapping[str, Any],
) -> str:
    if new_status == "resolved":
        return "Outcome resolved using the selected local price-history snapshot."
    if new_blocker == "missing_price_history":
        return "No suitable local price-history CSV was found for this ticker."
    if new_blocker == "insufficient_forward_data":
        return "Local price history exists but still does not extend far enough beyond the advice date."
    if snapshot["status"] != "available":
        return "Local price-history snapshot is unavailable or invalid."
    return "Outcome remains unresolved after local refresh."


def _summary(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blockers = Counter(
        outcome.get("new_blocker")
        for outcome in outcomes
        if outcome.get("new_status") != "resolved" and outcome.get("new_blocker")
    )
    missing = sorted(
        outcome["ticker"]
        for outcome in outcomes
        if outcome.get("new_blocker") == "missing_price_history"
    )
    resolved = sum(1 for outcome in outcomes if outcome["resolved"])
    still_unresolved = len(outcomes) - resolved
    return {
        "selected_outcomes": len(outcomes),
        "resolved": resolved,
        "still_unresolved": still_unresolved,
        "insufficient_forward_data": blockers.get("insufficient_forward_data", 0),
        "missing_price_history": blockers.get("missing_price_history", 0),
        "other_blockers": sum(
            count
            for blocker, count in blockers.items()
            if blocker not in {"insufficient_forward_data", "missing_price_history"}
        ),
        "missing_price_history_tickers": missing,
        "blocker_counts": dict(sorted(blockers.items())),
        "snapshot_references": sorted(
            {
                outcome["used_snapshot"]
                for outcome in outcomes
                if outcome.get("used_snapshot")
            }
        ),
    }


def _missing_price_history(refresh_index: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        {
            "ticker": row["ticker"],
            "advice": row.get("advice"),
            "previous_blocker": row.get("previous_blocker"),
            "new_blocker": row.get("new_blocker"),
        }
        for row in refresh_index["outcomes"]
        if row.get("new_blocker") == "missing_price_history"
    ]
    return {
        "schema_version": "market-engine-advice-outcome-refresh-missing-price-history-v1",
        "run_id": refresh_index["run_id"],
        "missing": missing,
        "tickers": [row["ticker"] for row in missing],
    }


def _manifest(refresh_index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ADVICE_OUTCOME_REFRESH_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-advice-outcome-refresh-manifest",
        "run_id": refresh_index["run_id"],
        "generated_at": refresh_index["generated_at"],
        "input": refresh_index["input"],
        "outputs": {
            "refresh_outcome_index_json": "refresh_outcome_index.json",
            "refresh_report_md": "refresh_report.md",
            "missing_price_history_json": "missing_price_history.json",
        },
        "baseline_guardrail": {
            "openai_api_required": False,
            "provider_invocation_allowed": False,
            "live_source_acquisition_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "scheduler_implemented": False,
        },
    }


def _generated_at_from_run_id_or_previous(run_id: str, previous: Mapping[str, Any]) -> str:
    marker = run_id.rsplit("-", 1)[-1]
    if len(marker) == 16 and marker.endswith("Z") and "T" in marker:
        return (
            f"{marker[0:4]}-{marker[4:6]}-{marker[6:8]}T"
            f"{marker[9:11]}:{marker[11:13]}:{marker[13:15]}Z"
        )
    return str(previous.get("generated_at") or "")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")
