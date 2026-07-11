from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.advice.deterministic_advice import (
    ADVICE_LABELS,
    build_advice_index,
    render_advice_markdown,
)


ADVICE_BATCH_MANIFEST_SCHEMA_VERSION = "market-engine-advice-batch-manifest-v1"
ADVICE_BATCH_SUMMARY_SCHEMA_VERSION = "market-engine-advice-batch-summary-v1"
ADVICE_BATCH_INDEX_SCHEMA_VERSION = "market-engine-advice-batch-index-v1"

LABEL_REPORTS = {
    "buy_candidate": "buy_candidates.md",
    "wait_for_price": "wait_for_price.md",
    "watchlist": "watchlist.md",
    "avoid_for_now": "avoid_for_now.md",
    "unable_to_advise": "unable_to_advise.md",
}

OUTCOME_TRACKING_LABELS = {
    "buy_candidate",
    "wait_for_price",
    "avoid_for_now",
    "hold_existing",
    "take_loss_review",
}


def build_advice_batch(
    ticker_status_index_path: str | Path,
    *,
    run_id: str,
    generated_at: str | None = None,
    target_universe_path: str | Path | None = None,
    target_size: int | None = None,
    max_tickers: int | None = None,
) -> dict[str, Any]:
    input_path = Path(ticker_status_index_path)
    status_index = json.loads(input_path.read_text(encoding="utf-8"))
    effective_generated_at = generated_at or _generated_at_utc()
    status_tickers = _status_tickers(status_index)
    selected_status_tickers = status_tickers[:max_tickers] if max_tickers else status_tickers
    selected_status_index = {
        **status_index,
        "tickers": selected_status_tickers,
        "summary": {
            **(status_index.get("summary") or {}),
            "tickers_total": len(selected_status_tickers),
        },
    }
    selected_index_path = _write_selected_status_index_if_needed(
        input_path,
        selected_status_index,
        max_tickers=max_tickers,
        run_id=run_id,
    )
    advice_index = build_advice_index(
        selected_index_path,
        run_id=run_id,
        generated_at=effective_generated_at,
    )
    advice_index["schema_version"] = ADVICE_BATCH_INDEX_SCHEMA_VERSION
    advice_index["artifact_type"] = "market-engine-deterministic-advice-batch-index"
    advice_index["input"] = {
        **(advice_index.get("input") or {}),
        "source_ticker_status_index_path": input_path.as_posix(),
        "target_universe_path": Path(target_universe_path).as_posix()
        if target_universe_path
        else None,
        "target_size": _target_size(
            target_size=target_size,
            target_universe_path=target_universe_path,
            status_ticker_count=len(selected_status_tickers),
        ),
    }
    target_tickers = _target_tickers(target_universe_path)
    coverage = _coverage(
        target_size=advice_index["input"]["target_size"],
        status_ticker_count=len(selected_status_tickers),
        advice_ticker_count=len(advice_index.get("tickers") or ()),
        target_tickers=target_tickers,
        status_tickers=[row["ticker"] for row in selected_status_tickers if row.get("ticker")],
    )
    summary = _batch_summary(
        run_id=run_id,
        target_size=coverage["target_tickers"],
        tickers_in_status_index=coverage["tickers_in_status_index"],
        tickers_with_advice=coverage["tickers_with_advice"],
        coverage_percentage=coverage["coverage_percentage"],
        advice_index=advice_index,
    )
    return {
        "run_id": run_id,
        "generated_at": effective_generated_at,
        "input": {
            "ticker_status_index_path": input_path.as_posix(),
            "ticker_status_index_run_id": status_index.get("run_id"),
            "target_universe_path": Path(target_universe_path).as_posix()
            if target_universe_path
            else None,
            "target_size": coverage["target_tickers"],
            "max_tickers": max_tickers,
        },
        "advice_index": advice_index,
        "summary": summary,
        "coverage": coverage,
    }


def write_advice_batch_outputs(
    batch: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    advice_index = batch["advice_index"]
    summary = batch["summary"]
    manifest = _manifest(batch)

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "advice_index.json", advice_index)
    (output_dir / "advice_index.md").write_text(
        render_advice_markdown(advice_index),
        encoding="utf-8",
    )
    _write_json(output_dir / "advice_summary.json", summary)
    for label, filename in LABEL_REPORTS.items():
        (output_dir / filename).write_text(
            render_label_report(advice_index, label),
            encoding="utf-8",
        )
    (output_dir / "missing_data_report.md").write_text(
        render_missing_data_report(advice_index),
        encoding="utf-8",
    )
    (output_dir / "coverage_report.md").write_text(
        render_coverage_report(batch),
        encoding="utf-8",
    )
    return output_dir


def run_advice_batch(
    ticker_status_index_path: str | Path,
    *,
    output_root: str | Path,
    run_id: str,
    generated_at: str | None = None,
    target_universe_path: str | Path | None = None,
    target_size: int | None = None,
    max_tickers: int | None = None,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    batch = build_advice_batch(
        ticker_status_index_path,
        run_id=run_id,
        generated_at=generated_at,
        target_universe_path=target_universe_path,
        target_size=target_size,
        max_tickers=max_tickers,
    )
    output_dir = write_advice_batch_outputs(
        batch,
        output_root=output_root,
        run_id=run_id,
        allow_overwrite=allow_overwrite,
    )
    return batch, output_dir


def render_label_report(advice_index: Mapping[str, Any], label: str) -> str:
    if label not in LABEL_REPORTS:
        raise ValueError(f"unsupported label report: {label}")
    title = {
        "buy_candidate": "Buy Candidates",
        "wait_for_price": "Wait For Price",
        "watchlist": "Watchlist",
        "avoid_for_now": "Avoid For Now",
        "unable_to_advise": "Unable To Advise",
    }[label]
    rows = [row for row in _advice_rows(advice_index) if row.get("advice") == label]
    output = [f"# {title}", ""]
    if not rows:
        empty_line = {
            "buy_candidate": "No buy candidates in this run.",
            "wait_for_price": "No wait-for-price tickers in this run.",
            "watchlist": "No watchlist tickers in this run.",
            "avoid_for_now": "No avoid-for-now tickers in this run.",
            "unable_to_advise": "No unable-to-advise tickers in this run.",
        }[label]
        return "\n".join(output + [empty_line, ""])

    if label == "unable_to_advise":
        output.extend(
            [
                "| Ticker | Reason | Missing for advice | Next action |",
                "|---|---|---|---|",
            ]
        )
        for row in rows:
            output.append(
                "| "
                + " | ".join(
                    (
                        _md(row.get("ticker")),
                        _md(row.get("primary_reason")),
                        _md(", ".join(row.get("missing_for_buy_candidate") or ())),
                        _md(row.get("next_action")),
                    )
                )
                + " |"
            )
        output.append("")
        return "\n".join(output)

    if label == "avoid_for_now":
        output.extend(
            [
                "| Ticker | Confidence | Reason | Blockers | Next action |",
                "|---|---|---|---|---|",
            ]
        )
        for row in rows:
            output.append(
                "| "
                + " | ".join(
                    (
                        _md(row.get("ticker")),
                        _md(row.get("confidence")),
                        _md(row.get("primary_reason")),
                        _md(", ".join(row.get("blockers") or ())),
                        _md(row.get("next_action")),
                    )
                )
                + " |"
            )
        output.append("")
        return "\n".join(output)

    missing_header = (
        "Missing for buy candidate"
        if label != "buy_candidate"
        else "Missing for better advice"
    )
    output.extend(
        [
            f"| Ticker | Confidence | Reason | {missing_header} | Next action |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        output.append(
            "| "
            + " | ".join(
                (
                    _md(row.get("ticker")),
                    _md(row.get("confidence")),
                    _md(row.get("primary_reason")),
                    _md(", ".join(row.get("missing_for_buy_candidate") or ())),
                    _md(row.get("next_action")),
                )
            )
            + " |"
        )
    output.append("")
    return "\n".join(output)


def render_missing_data_report(advice_index: Mapping[str, Any]) -> str:
    summary = advice_index.get("summary") or {}
    top_missing = summary.get("top_missing_for_buy_candidate") or {}
    rows = [
        "# Missing Data Report",
        "",
        "## Top Missing Inputs For Buy Candidate",
        "",
        "| Missing input | Count |",
        "|---|---:|",
    ]
    if top_missing:
        for missing, count in sorted(top_missing.items()):
            rows.append(f"| {_md(missing)} | {count} |")
    else:
        rows.append("| None | 0 |")
    rows.extend(
        [
            "",
            "## Missing Inputs By Ticker",
            "",
            "| Ticker | Advice | Missing for buy candidate | Blockers |",
            "|---|---|---|---|",
        ]
    )
    for row in _advice_rows(advice_index):
        rows.append(
            "| "
            + " | ".join(
                (
                    _md(row.get("ticker")),
                    _md(row.get("advice")),
                    _md(", ".join(row.get("missing_for_buy_candidate") or ())),
                    _md(", ".join(row.get("blockers") or ())),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def render_coverage_report(batch: Mapping[str, Any]) -> str:
    coverage = batch["coverage"]
    summary = batch["summary"]
    input_info = batch["input"]
    readiness = summary["evaluation_readiness"]
    rows = [
        "# Market Engine Advice Batch Coverage Report",
        "",
        f"Run ID: `{batch.get('run_id')}`",
        f"Generated at: `{batch.get('generated_at')}`",
        f"Input ticker status index: `{input_info.get('ticker_status_index_path')}`",
            f"Target universe: `{input_info.get('target_universe_path') or 'not provided'}`",
        f"Target size: `{coverage.get('target_tickers')}`",
        "",
        "## Coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Target tickers | {coverage.get('target_tickers')} |",
        f"| Tickers in status index | {coverage.get('tickers_in_status_index')} |",
        f"| Tickers with advice labels | {coverage.get('tickers_with_advice')} |",
        f"| Tickers missing artifact/status | {coverage.get('tickers_missing_artifact_or_status')} |",
        f"| Coverage percentage | {coverage.get('coverage_percentage'):.2f}% |",
        "",
        "## Advice Distribution",
        "",
        "| Advice | Count |",
        "|---|---:|",
    ]
    for label in ADVICE_LABELS:
        rows.append(f"| {label} | {summary['advice_counts'].get(label, 0)} |")
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- What worked: {coverage.get('tickers_with_advice')} ticker(s) received deterministic advice labels from the available status index.",
            f"- What is still missing: {coverage.get('tickers_missing_artifact_or_status')} target ticker(s) lack status/advice coverage, and top missing inputs are { _missing_phrase(summary.get('top_missing_for_buy_candidate') or {}) }.",
            f"- Whether this is ready for evaluation: {readiness.get('reason')}",
            f"- Recommended next sprint: {summary.get('recommended_next_sprint')}",
            "",
        ]
    )
    return "\n".join(rows)


def _batch_summary(
    *,
    run_id: str,
    target_size: int,
    tickers_in_status_index: int,
    tickers_with_advice: int,
    coverage_percentage: float,
    advice_index: Mapping[str, Any],
) -> dict[str, Any]:
    advice_summary = advice_index.get("summary") or {}
    advice_counts = {
        label: (advice_summary.get("advice_counts") or {}).get(label, 0)
        for label in ADVICE_LABELS
    }
    readiness = _evaluation_readiness(advice_counts)
    return {
        "schema_version": ADVICE_BATCH_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "target_size": target_size,
        "tickers_in_status_index": tickers_in_status_index,
        "tickers_with_advice": tickers_with_advice,
        "coverage_percentage": coverage_percentage,
        "advice_counts": advice_counts,
        "confidence_counts": advice_summary.get("confidence_counts") or {},
        "watchlist_or_better_count": advice_summary.get("watchlist_or_better_count", 0),
        "buy_candidate_count": advice_counts.get("buy_candidate", 0),
        "unable_to_advise_count": advice_counts.get("unable_to_advise", 0),
        "top_missing_for_buy_candidate": advice_summary.get("top_missing_for_buy_candidate")
        or {},
        "evaluation_readiness": readiness,
        "recommended_next_sprint": readiness["recommended_next_sprint"],
    }


def _evaluation_readiness(advice_counts: Mapping[str, int]) -> dict[str, Any]:
    ready = any(advice_counts.get(label, 0) > 0 for label in OUTCOME_TRACKING_LABELS)
    if ready:
        return {
            "ready_for_outcome_tracking": True,
            "reason": "At least one outcome-trackable advice label was produced.",
            "recommended_next_sprint": "ME-EVAL01 - Advice outcome tracking and feedback loop",
        }
    if advice_counts.get("watchlist", 0) > 0 and sum(advice_counts.values()) == advice_counts.get(
        "watchlist",
        0,
    ):
        return {
            "ready_for_outcome_tracking": False,
            "reason": (
                "Only watchlist labels were produced; price/setup or portfolio "
                "context is needed before outcome tracking can evaluate advice "
                "quality."
            ),
            "recommended_next_sprint": "ME-DATA01 - Close highest-impact advice data coverage gaps",
        }
    return {
        "ready_for_outcome_tracking": False,
        "reason": (
            "Only watchlist or unable-to-advise labels were produced; price/setup "
            "or portfolio context is needed before outcome tracking can evaluate "
            "advice quality."
        ),
        "recommended_next_sprint": "ME-DATA01 - Close highest-impact advice data coverage gaps",
    }


def _manifest(batch: Mapping[str, Any]) -> dict[str, Any]:
    input_info = batch["input"]
    return {
        "schema_version": ADVICE_BATCH_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-deterministic-advice-batch-manifest",
        "run_id": batch.get("run_id"),
        "generated_at": batch.get("generated_at"),
        "input": {
            "ticker_status_index_path": input_info.get("ticker_status_index_path"),
            "ticker_status_index_run_id": input_info.get("ticker_status_index_run_id"),
            "target_universe_path": input_info.get("target_universe_path"),
            "target_size": input_info.get("target_size"),
        },
        "outputs": {
            "advice_index_json": "advice_index.json",
            "advice_index_md": "advice_index.md",
            "advice_summary_json": "advice_summary.json",
            "buy_candidates_md": "buy_candidates.md",
            "wait_for_price_md": "wait_for_price.md",
            "watchlist_md": "watchlist.md",
            "avoid_for_now_md": "avoid_for_now.md",
            "unable_to_advise_md": "unable_to_advise.md",
            "missing_data_report_md": "missing_data_report.md",
            "coverage_report_md": "coverage_report.md",
        },
        "baseline_guardrail": {
            "openai_api_required": False,
            "provider_invocation_allowed": False,
            "source_acquisition_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "advice_labels_produced": True,
        },
        "next_baseline_sprint": batch["summary"]["recommended_next_sprint"],
    }


def _coverage(
    *,
    target_size: int,
    status_ticker_count: int,
    advice_ticker_count: int,
    target_tickers: Sequence[str],
    status_tickers: Sequence[str],
) -> dict[str, Any]:
    status_set = set(status_tickers)
    target_set = set(target_tickers)
    if target_set:
        missing = len(target_set - status_set)
    else:
        missing = max(target_size - status_ticker_count, 0)
    coverage_percentage = (
        round((advice_ticker_count / target_size) * 100, 2) if target_size else 0.0
    )
    return {
        "target_tickers": target_size,
        "tickers_in_status_index": status_ticker_count,
        "tickers_with_advice": advice_ticker_count,
        "tickers_missing_artifact_or_status": missing,
        "coverage_percentage": coverage_percentage,
    }


def _target_size(
    *,
    target_size: int | None,
    target_universe_path: str | Path | None,
    status_ticker_count: int,
) -> int:
    if target_size is not None:
        return target_size
    target_tickers = _target_tickers(target_universe_path)
    if target_tickers:
        return len(target_tickers)
    return status_ticker_count


def _target_tickers(target_universe_path: str | Path | None) -> list[str]:
    if not target_universe_path:
        return []
    path = Path(target_universe_path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return sorted({_ticker_from_value(item) for item in data if _ticker_from_value(item)})
        if isinstance(data, dict):
            values = data.get("tickers") or data.get("universe") or []
            return sorted({_ticker_from_value(item) for item in values if _ticker_from_value(item)})
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tickers = [
            str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            for row in reader
        ]
    return sorted({ticker for ticker in tickers if ticker})


def _ticker_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip().upper() or None
    if isinstance(value, dict):
        ticker = value.get("ticker") or value.get("symbol")
        if isinstance(ticker, str):
            return ticker.strip().upper() or None
    return None


def _status_tickers(status_index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in status_index.get("tickers") or () if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )


def _write_selected_status_index_if_needed(
    input_path: Path,
    selected_status_index: Mapping[str, Any],
    *,
    max_tickers: int | None,
    run_id: str,
) -> Path:
    if max_tickers is None:
        return input_path
    tmp_path = Path("/private/tmp") / f"{run_id}-selected-ticker-status-index.json"
    _write_json(tmp_path, selected_status_index)
    return tmp_path


def _advice_rows(advice_index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in advice_index.get("tickers") or () if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )


def _missing_phrase(top_missing: Mapping[str, int]) -> str:
    if not top_missing:
        return "none"
    ordered = Counter(top_missing).most_common(3)
    return ", ".join(f"{name}: {count}" for name, count in ordered)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generated_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _md(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value).replace("|", "\\|")
