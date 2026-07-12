from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ADVICE_OUTCOME_INDEX_SCHEMA_VERSION = "market-engine-advice-outcome-index-v1"
LABEL_PERFORMANCE_SUMMARY_SCHEMA_VERSION = "market-engine-label-performance-summary-v1"
UNRESOLVED_OUTCOMES_SCHEMA_VERSION = "market-engine-unresolved-advice-outcomes-v1"
EVALUATION_RUN_MANIFEST_SCHEMA_VERSION = "market-engine-evaluation-run-manifest-v1"

DEFAULT_HORIZONS = (
    {"name": "1w", "trading_days": 5},
    {"name": "1m", "trading_days": 21},
    {"name": "3m", "trading_days": 63},
)

EXPECTED_DIRECTIONS = {
    "buy_candidate": "up_or_constructive",
    "wait_for_price": "avoid_overpaying",
    "watchlist": "neutral_observe",
    "avoid_for_now": "weak_or_down",
    "hold_existing": "stable_or_up",
    "take_loss_review": "weak_or_down",
    "unable_to_advise": "no_expectation",
}


def build_advice_outcome_evaluation(
    advice_index_path: str | Path,
    *,
    price_data_root: str | Path,
    run_id: str,
    horizons: Sequence[Mapping[str, int | str]] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    advice_path = Path(advice_index_path)
    price_root = Path(price_data_root)
    advice_index = json.loads(advice_path.read_text(encoding="utf-8"))
    generated_at = _generated_at_from_run_id(run_id)
    anchor = _advice_anchor(advice_index)
    normalized_horizons = _normalize_horizons(horizons)

    ticker_rows = []
    for row in _advice_rows(advice_index):
        ticker_rows.append(
            _evaluate_ticker(
                row,
                advice_date=anchor["advice_date"],
                anchor_source=anchor["anchor_source"],
                price_data_root=price_root,
                horizons=normalized_horizons,
            )
        )

    outcome_index = {
        "schema_version": ADVICE_OUTCOME_INDEX_SCHEMA_VERSION,
        "artifact_type": "market-engine-advice-outcome-index",
        "run_id": run_id,
        "generated_at": generated_at,
        "input": {
            "advice_index_path": advice_path.as_posix(),
            "advice_index_run_id": advice_index.get("run_id"),
            "advice_index_generated_at": advice_index.get("generated_at"),
            "price_data_root": price_root.as_posix(),
        },
        "horizons": normalized_horizons,
        "summary": _summary(ticker_rows, normalized_horizons),
        "tickers": ticker_rows,
    }
    label_summary = build_label_performance_summary(outcome_index)
    unresolved = build_unresolved_outcomes(outcome_index)
    return {
        "manifest": _manifest(outcome_index),
        "advice_outcome_index": outcome_index,
        "label_performance_summary": label_summary,
        "rule_feedback_report": render_rule_feedback_report(outcome_index, label_summary),
        "advice_outcome_report": render_advice_outcome_report(outcome_index),
        "unresolved_outcomes": unresolved,
    }


def write_advice_outcome_evaluation(
    evaluation: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "manifest.json", evaluation["manifest"])
    _write_json(output_dir / "advice_outcome_index.json", evaluation["advice_outcome_index"])
    (output_dir / "advice_outcome_report.md").write_text(
        evaluation["advice_outcome_report"],
        encoding="utf-8",
    )
    _write_json(
        output_dir / "label_performance_summary.json",
        evaluation["label_performance_summary"],
    )
    (output_dir / "rule_feedback_report.md").write_text(
        evaluation["rule_feedback_report"],
        encoding="utf-8",
    )
    _write_json(output_dir / "unresolved_outcomes.json", evaluation["unresolved_outcomes"])
    return output_dir


def run_advice_outcome_evaluation(
    advice_index_path: str | Path,
    *,
    price_data_root: str | Path,
    output_root: str | Path,
    run_id: str,
    horizons: Sequence[Mapping[str, int | str]] = DEFAULT_HORIZONS,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    evaluation = build_advice_outcome_evaluation(
        advice_index_path,
        price_data_root=price_data_root,
        run_id=run_id,
        horizons=horizons,
    )
    output_dir = write_advice_outcome_evaluation(
        evaluation,
        output_root=output_root,
        run_id=run_id,
        allow_overwrite=allow_overwrite,
    )
    return evaluation, output_dir


def build_label_performance_summary(outcome_index: Mapping[str, Any]) -> dict[str, Any]:
    horizons = [horizon["name"] for horizon in outcome_index["horizons"]]
    labels: dict[str, Any] = {}
    for row in outcome_index["tickers"]:
        label = row["advice"]
        labels.setdefault(
            label,
            {
                "count": 0,
                "resolved_count": 0,
                "unresolved_count": 0,
                "average_return_by_horizon": {name: None for name in horizons},
                "preliminary_outcome_counts": {},
            },
        )
        labels[label]["count"] += 1

    returns_by_label_horizon: dict[tuple[str, str], list[float]] = defaultdict(list)
    preliminary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in outcome_index["tickers"]:
        label = row["advice"]
        if _has_resolved_horizon(row):
            labels[label]["resolved_count"] += 1
        else:
            labels[label]["unresolved_count"] += 1
        preliminary = row["label_interpretation"]["preliminary_outcome"]
        preliminary_counts[label][preliminary] += 1
        for horizon_name, outcome in row["outcomes"].items():
            if outcome["status"] == "resolved":
                returns_by_label_horizon[(label, horizon_name)].append(outcome["return_pct"])

    for label, data in labels.items():
        for horizon_name in horizons:
            returns = returns_by_label_horizon.get((label, horizon_name), [])
            data["average_return_by_horizon"][horizon_name] = (
                round(sum(returns) / len(returns), 2) if returns else None
            )
        data["preliminary_outcome_counts"] = dict(sorted(preliminary_counts[label].items()))

    return {
        "schema_version": LABEL_PERFORMANCE_SUMMARY_SCHEMA_VERSION,
        "run_id": outcome_index["run_id"],
        "labels": dict(sorted(labels.items())),
        "limitations": [
            "Outcome data may be unresolved when local price history does not extend beyond the advice date.",
            "This is preliminary outcome tracking, not a full backtest.",
        ],
    }


def build_unresolved_outcomes(outcome_index: Mapping[str, Any]) -> dict[str, Any]:
    unresolved_rows = []
    reason_counts: Counter[str] = Counter()
    for row in outcome_index["tickers"]:
        reasons = sorted(
            {
                _reason_value(outcome)
                for outcome in row["outcomes"].values()
                if outcome["status"] == "unresolved"
            }
        )
        if not reasons:
            continue
        for reason in reasons:
            reason_counts[reason] += 1
        unresolved_rows.append(
            {
                "ticker": row["ticker"],
                "advice": row["advice"],
                "reasons": reasons,
                "price_source_path": row.get("price_source_path"),
            }
        )

    return {
        "schema_version": UNRESOLVED_OUTCOMES_SCHEMA_VERSION,
        "run_id": outcome_index["run_id"],
        "unresolved": unresolved_rows,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def render_advice_outcome_report(outcome_index: Mapping[str, Any]) -> str:
    summary = outcome_index["summary"]
    input_data = outcome_index["input"]
    rows = [
        "# Advice Outcome Report",
        "",
        f"Run ID: {outcome_index['run_id']}",
        f"Input advice index: {input_data['advice_index_path']}",
        f"Price data root: {input_data['price_data_root']}",
        f"Generated at: {outcome_index['generated_at']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Tickers total | {summary['tickers_total']} |",
        f"| Resolved outcomes | {summary['resolved_outcomes']} |",
        f"| Unresolved outcomes | {summary['unresolved_outcomes']} |",
        f"| 1w resolved | {summary['resolved_by_horizon'].get('1w', 0)} |",
        f"| 1m resolved | {summary['resolved_by_horizon'].get('1m', 0)} |",
        f"| 3m resolved | {summary['resolved_by_horizon'].get('3m', 0)} |",
        "",
        "## Outcome Table",
        "",
        "| Ticker | Advice | Confidence | Advice date | Entry price | 1w | 1m | 3m | Preliminary outcome |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in outcome_index["tickers"]:
        rows.append(
            "| "
            + " | ".join(
                (
                    _md(row["ticker"]),
                    _md(row["advice"]),
                    _md(row.get("confidence")),
                    _md(row["advice_date"]),
                    _md(_format_price(row.get("entry_price"))),
                    _md(_format_horizon(row["outcomes"].get("1w"))),
                    _md(_format_horizon(row["outcomes"].get("1m"))),
                    _md(_format_horizon(row["outcomes"].get("3m"))),
                    _md(row["label_interpretation"]["preliminary_outcome"]),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def render_rule_feedback_report(
    outcome_index: Mapping[str, Any],
    label_summary: Mapping[str, Any],
) -> str:
    summary = outcome_index["summary"]
    rows = [
        "# Advice Rule Feedback Report",
        "",
        f"Run ID: {outcome_index['run_id']}",
        f"Input advice batch: {outcome_index['input']['advice_index_path']}",
        f"Generated at: {outcome_index['generated_at']}",
        "",
        "## Evaluation readiness",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Tickers evaluated | {summary['tickers_total']} |",
        f"| Resolved 1w outcomes | {summary['resolved_by_horizon'].get('1w', 0)} |",
        f"| Resolved 1m outcomes | {summary['resolved_by_horizon'].get('1m', 0)} |",
        f"| Resolved 3m outcomes | {summary['resolved_by_horizon'].get('3m', 0)} |",
        f"| Unresolved outcomes | {summary['unresolved_outcomes']} |",
        "",
        "## Label Feedback",
        "",
    ]
    labels = label_summary["labels"]
    for label in sorted(labels):
        data = labels[label]
        counts = data["preliminary_outcome_counts"]
        rows.extend(
            [
                f"### {label}",
                "",
                f"- Count: {data['count']}",
                f"- Resolved: {data['resolved_count']}",
                f"- Preliminary supportive: {counts.get('supportive', 0)}",
                f"- Preliminary adverse: {counts.get('adverse', 0)}",
                f"- Interpretation: {EXPECTED_DIRECTIONS.get(label, 'no_expectation')}",
                f"- Suggested rule feedback: {_feedback_for_label(label, data, summary)}",
                "",
            ]
        )
    rows.extend(
        [
            "## Limitations",
            "",
            "- No live data was acquired.",
            "- Outcomes are unresolved where local price history does not extend beyond advice date.",
            "- This is not a full backtest.",
            "",
        ]
    )
    return "\n".join(rows)


def parse_horizons(raw: str | None) -> tuple[dict[str, int], ...]:
    if not raw:
        return tuple(dict(horizon) for horizon in DEFAULT_HORIZONS)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    names = ("1w", "1m", "3m")
    return tuple(
        {"name": names[index] if index < len(names) else f"{days}d", "trading_days": days}
        for index, days in enumerate(values)
    )


def _evaluate_ticker(
    row: Mapping[str, Any],
    *,
    advice_date: date | None,
    anchor_source: str,
    price_data_root: Path,
    horizons: Sequence[Mapping[str, int | str]],
) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    base = {
        "ticker": ticker,
        "advice": row.get("advice") or "unable_to_advise",
        "confidence": row.get("confidence"),
        "advice_date": advice_date.isoformat() if advice_date else None,
        "anchor_source": anchor_source,
        "entry_price": None,
        "price_source_path": None,
        "outcomes": {},
        "label_interpretation": {
            "expected_direction": EXPECTED_DIRECTIONS.get(row.get("advice"), "no_expectation"),
            "preliminary_outcome": "unresolved",
        },
    }
    if advice_date is None:
        base["outcomes"] = _unresolved_horizons(horizons, "missing_advice_date")
        return base

    price_path = resolve_price_history_path(price_data_root, ticker)
    if price_path is None:
        base["outcomes"] = _unresolved_horizons(horizons, "missing_price_history")
        return base
    base["price_source_path"] = price_path.as_posix()

    history = _read_price_history(price_path)
    if history["status"] != "ok":
        base["outcomes"] = _unresolved_horizons(horizons, history["reason"])
        return base

    rows = history["rows"]
    anchor_index = _anchor_index(rows, advice_date)
    if anchor_index is None:
        base["outcomes"] = _unresolved_horizons(horizons, "insufficient_forward_data")
        return base

    entry = rows[anchor_index]
    base["entry_price"] = round(entry["price"], 4)
    outcomes = {}
    for horizon in horizons:
        horizon_name = str(horizon["name"])
        trading_days = int(horizon["trading_days"])
        end_index = anchor_index + trading_days
        if end_index >= len(rows):
            outcomes[horizon_name] = {
                "status": "unresolved",
                "reason": "insufficient_forward_data",
            }
            continue
        end = rows[end_index]
        return_pct = ((end["price"] - entry["price"]) / entry["price"]) * 100
        outcomes[horizon_name] = {
            "status": "resolved",
            "trading_days": trading_days,
            "end_date": end["date"].isoformat(),
            "end_price": round(end["price"], 4),
            "return_pct": round(return_pct, 2),
            "direction": _direction(return_pct),
        }
    base["outcomes"] = outcomes
    base["label_interpretation"]["preliminary_outcome"] = _preliminary_outcome(
        str(base["advice"]),
        outcomes,
        horizons,
    )
    return base


def resolve_price_history_path(price_data_root: str | Path, ticker: str) -> Path | None:
    root = Path(price_data_root)
    candidates = [
        root / f"{ticker}.csv",
        root / f"{ticker.upper()}.csv",
        root / f"{ticker.lower()}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for directory in (root / ticker, root / ticker.upper(), root / ticker.lower()):
        if directory.is_dir():
            matches = sorted(directory.glob("*.csv"))
            if matches:
                return matches[0]
    matches = sorted(path for path in root.glob("**/*.csv") if path.stem.upper() == ticker.upper())
    return matches[0] if matches else None


def _read_price_history(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_headers = reader.fieldnames or []
        columns = {_normalize_column(header): header for header in raw_headers}
        if "date" not in columns:
            return {"status": "error", "reason": "invalid_price_history"}
        price_key = None
        for candidate in ("adj_close", "adjusted_close", "close"):
            if candidate in columns:
                price_key = columns[candidate]
                break
        if price_key is None:
            return {"status": "error", "reason": "missing_close_price"}
        date_key = columns["date"]
        rows = []
        for raw in reader:
            try:
                value = raw.get(price_key)
                raw_date = raw.get(date_key)
                if not value or not raw_date:
                    continue
                rows.append(
                    {
                        "date": date.fromisoformat(raw_date[:10]),
                        "price": float(value),
                    }
                )
            except (TypeError, ValueError):
                continue
    rows = sorted({(item["date"], item["price"]) for item in rows})
    parsed_rows = [{"date": item_date, "price": price} for item_date, price in rows]
    if not parsed_rows:
        return {"status": "error", "reason": "invalid_price_history"}
    return {"status": "ok", "rows": parsed_rows}


def _summary(rows: Sequence[Mapping[str, Any]], horizons: Sequence[Mapping[str, int | str]]) -> dict[str, Any]:
    resolved_by_horizon = {
        str(horizon["name"]): sum(
            1
            for row in rows
            if row["outcomes"].get(str(horizon["name"]), {}).get("status") == "resolved"
        )
        for horizon in horizons
    }
    unresolved_reasons: Counter[str] = Counter()
    for row in rows:
        for reason in {
            _reason_value(outcome)
            for outcome in row["outcomes"].values()
            if outcome["status"] == "unresolved"
        }:
            unresolved_reasons[reason] += 1
    label_counts = Counter(row["advice"] for row in rows)
    preliminary_counts = Counter(
        row["label_interpretation"]["preliminary_outcome"] for row in rows
    )
    return {
        "tickers_total": len(rows),
        "resolved_outcomes": sum(1 for row in rows if _has_resolved_horizon(row)),
        "unresolved_outcomes": sum(1 for row in rows if not _has_resolved_horizon(row)),
        "resolved_by_horizon": resolved_by_horizon,
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_performance": dict(sorted(preliminary_counts.items())),
    }


def _manifest(outcome_index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-advice-outcome-evaluation-run-manifest",
        "run_id": outcome_index["run_id"],
        "generated_at": outcome_index["generated_at"],
        "input": {
            "advice_index_path": outcome_index["input"]["advice_index_path"],
            "price_data_root": outcome_index["input"]["price_data_root"],
        },
        "outputs": {
            "advice_outcome_index_json": "advice_outcome_index.json",
            "advice_outcome_report_md": "advice_outcome_report.md",
            "label_performance_summary_json": "label_performance_summary.json",
            "rule_feedback_report_md": "rule_feedback_report.md",
            "unresolved_outcomes_json": "unresolved_outcomes.json",
        },
        "baseline_guardrail": {
            "openai_api_required": False,
            "provider_invocation_allowed": False,
            "live_source_acquisition_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
        },
        "next_baseline_sprint": "ME-EVAL02 - Extend outcome tracking with scheduled/future refresh or local snapshot import",
    }


def _advice_anchor(advice_index: Mapping[str, Any]) -> dict[str, Any]:
    generated_at = advice_index.get("generated_at")
    parsed = _parse_datetime(generated_at)
    if parsed:
        return {"advice_date": parsed.date(), "anchor_source": "advice_index.generated_at"}
    run_id_date = _parse_run_id_date(str(advice_index.get("run_id") or ""))
    if run_id_date:
        return {"advice_date": run_id_date, "anchor_source": "advice_index.run_id"}
    return {"advice_date": None, "anchor_source": "unresolved_missing_advice_date"}


def _generated_at_from_run_id(run_id: str) -> str:
    parsed = _parse_run_id_datetime(run_id)
    if parsed:
        return parsed.isoformat().replace("+00:00", "Z")
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_run_id_date(run_id: str) -> date | None:
    parsed = _parse_run_id_datetime(run_id)
    return parsed.date() if parsed else None


def _parse_run_id_datetime(run_id: str) -> datetime | None:
    match = re.search(r"(\d{8}T\d{6}Z)", run_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _normalize_horizons(horizons: Sequence[Mapping[str, int | str]]) -> tuple[dict[str, int], ...]:
    return tuple(
        {"name": str(horizon["name"]), "trading_days": int(horizon["trading_days"])}
        for horizon in horizons
    )


def _advice_rows(advice_index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(advice_index.get("tickers") or (), key=lambda row: str(row.get("ticker") or ""))


def _anchor_index(rows: Sequence[Mapping[str, Any]], advice_date: date) -> int | None:
    eligible = [index for index, row in enumerate(rows) if row["date"] <= advice_date]
    if eligible:
        return eligible[-1]
    return 0 if rows else None


def _preliminary_outcome(
    advice: str,
    outcomes: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[Mapping[str, int | str]],
) -> str:
    resolved = [
        outcomes[str(horizon["name"])]
        for horizon in horizons
        if outcomes.get(str(horizon["name"]), {}).get("status") == "resolved"
    ]
    if not resolved:
        return "unresolved"
    return_pct = resolved[-1]["return_pct"]
    if advice == "buy_candidate":
        if return_pct > 2:
            return "supportive"
        if return_pct < -2:
            return "adverse"
        return "neutral"
    if advice == "wait_for_price":
        if return_pct > 5:
            return "possibly_too_conservative"
        if return_pct < -5:
            return "supportive_wait"
        return "reasonable_wait"
    if advice == "watchlist":
        return "observed_no_judgment"
    if advice == "avoid_for_now":
        if return_pct < -2:
            return "supportive"
        if return_pct > 2:
            return "adverse"
        return "neutral"
    if advice == "hold_existing":
        return "supportive" if return_pct >= 0 else "adverse"
    if advice == "take_loss_review":
        return "supportive" if return_pct < -2 else "neutral_or_adverse"
    if advice == "unable_to_advise":
        return "no_outcome_judgment"
    return "unresolved"


def _feedback_for_label(label: str, data: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    if summary["unresolved_outcomes"] > summary["resolved_outcomes"]:
        return "Need future/local price history before rule quality can be judged."
    counts = data["preliminary_outcome_counts"]
    resolved = data["resolved_count"]
    if label == "buy_candidate" and counts.get("adverse", 0) > resolved / 2:
        return "Review buy_candidate setup/price thresholds."
    if label == "wait_for_price" and counts.get("possibly_too_conservative", 0) > resolved / 2:
        return "Wait threshold may be too conservative."
    if label == "avoid_for_now" and counts.get("adverse", 0) > resolved / 2:
        return "Avoid rule may be too strict."
    if label == "avoid_for_now" and counts.get("supportive", 0):
        return "Avoid rule was directionally supportive."
    return "No rule change indicated by the available preliminary outcomes."


def _unresolved_horizons(
    horizons: Sequence[Mapping[str, int | str]],
    reason: str,
) -> dict[str, dict[str, str]]:
    return {
        str(horizon["name"]): {"status": "unresolved", "reason": reason}
        for horizon in horizons
    }


def _has_resolved_horizon(row: Mapping[str, Any]) -> bool:
    return any(outcome["status"] == "resolved" for outcome in row["outcomes"].values())


def _reason_value(outcome: Mapping[str, Any]) -> str:
    reason = str(outcome.get("reason") or "unknown")
    return reason.removeprefix("unresolved_")


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _direction(return_pct: float) -> str:
    if return_pct > 0:
        return "up"
    if return_pct < 0:
        return "down"
    return "flat"


def _format_horizon(outcome: Mapping[str, Any] | None) -> str:
    if not outcome:
        return "unresolved"
    if outcome["status"] == "resolved":
        return f"{outcome['return_pct']:.2f}%"
    return f"unresolved:{_reason_value(outcome)}"


def _format_price(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
