from __future__ import annotations

import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from market_engine.evaluation.advice_outcomes import (
    build_advice_outcome_evaluation,
    resolve_price_history_path,
    run_advice_outcome_evaluation,
)


def test_parses_advice_index_and_writes_all_output_files(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(80)])

    _, output_dir = run_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        output_root=tmp_path / "out",
        run_id="me-eval01-advice-outcomes-20260711T150000Z",
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "advice_outcome_index.json",
        "advice_outcome_report.md",
        "label_performance_summary.json",
        "manifest.json",
        "rule_feedback_report.md",
        "unresolved_outcomes.json",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["baseline_guardrail"]["openai_api_required"] is False


def test_resolves_local_csv_by_ticker(tmp_path: Path) -> None:
    path = tmp_path / "prices" / "nested" / "aaa.csv"
    _write_price_csv(path, [100])

    assert resolve_price_history_path(tmp_path / "prices", "AAA") == path


def test_uses_adjusted_close_over_close_when_available(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(
        tmp_path / "prices" / "AAA.csv",
        [100, 110, 120, 130, 140, 150],
        close_prices=[1, 1, 1, 1, 1, 1],
    )

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    row = evaluation["advice_outcome_index"]["tickers"][0]
    assert row["entry_price"] == 100
    assert row["outcomes"]["1w"]["end_price"] == 150
    assert row["outcomes"]["1w"]["return_pct"] == 50


def test_computes_5_21_63_trading_day_returns_correctly(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    prices = [100 + index for index in range(70)]
    _write_price_csv(tmp_path / "prices" / "AAA.csv", prices)

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )
    outcomes = evaluation["advice_outcome_index"]["tickers"][0]["outcomes"]

    assert outcomes["1w"]["return_pct"] == 5
    assert outcomes["1m"]["return_pct"] == 21
    assert outcomes["3m"]["return_pct"] == 63


def test_marks_unresolved_when_csv_missing(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    row = evaluation["advice_outcome_index"]["tickers"][0]
    assert row["outcomes"]["1w"] == {
        "status": "unresolved",
        "reason": "missing_price_history",
    }


def test_marks_unresolved_when_forward_data_insufficient(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100, 101, 102])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    row = evaluation["advice_outcome_index"]["tickers"][0]
    assert row["outcomes"]["1w"]["reason"] == "insufficient_forward_data"


def test_marks_unresolved_when_date_or_close_columns_missing(tmp_path: Path) -> None:
    missing_date_advice = _write_advice_index(tmp_path / "date", [_advice_row("AAA", "buy_candidate")])
    price_root = tmp_path / "date" / "prices"
    price_root.mkdir(parents=True)
    (price_root / "AAA.csv").write_text("close\n100\n", encoding="utf-8")

    invalid = build_advice_outcome_evaluation(
        missing_date_advice,
        price_data_root=price_root,
        run_id="run-20260711T150000Z",
    )
    assert invalid["advice_outcome_index"]["tickers"][0]["outcomes"]["1w"]["reason"] == "invalid_price_history"

    missing_close_advice = _write_advice_index(tmp_path / "close", [_advice_row("BBB", "buy_candidate")])
    close_root = tmp_path / "close" / "prices"
    close_root.mkdir(parents=True)
    (close_root / "BBB.csv").write_text("date,open\n2026-07-11,100\n", encoding="utf-8")

    missing = build_advice_outcome_evaluation(
        missing_close_advice,
        price_data_root=close_root,
        run_id="run-20260711T150000Z",
    )
    assert missing["advice_outcome_index"]["tickers"][0]["outcomes"]["1w"]["reason"] == "missing_close_price"


def test_buy_candidate_positive_return_is_supportive(tmp_path: Path) -> None:
    row = _evaluate_single(tmp_path, "buy_candidate", [100 + index for index in range(70)])

    assert row["label_interpretation"]["preliminary_outcome"] == "supportive"


def test_buy_candidate_negative_return_is_adverse(tmp_path: Path) -> None:
    row = _evaluate_single(tmp_path, "buy_candidate", [100 - index for index in range(70)])

    assert row["label_interpretation"]["preliminary_outcome"] == "adverse"


def test_wait_for_price_strong_up_is_possibly_too_conservative(tmp_path: Path) -> None:
    row = _evaluate_single(tmp_path, "wait_for_price", [100 + index for index in range(70)])

    assert row["label_interpretation"]["preliminary_outcome"] == "possibly_too_conservative"


def test_avoid_for_now_down_is_supportive(tmp_path: Path) -> None:
    row = _evaluate_single(tmp_path, "avoid_for_now", [100 - index for index in range(70)])

    assert row["label_interpretation"]["preliminary_outcome"] == "supportive"


def test_unresolved_outcomes_reason_counts_correct(tmp_path: Path) -> None:
    advice_index = _write_advice_index(
        tmp_path,
        [_advice_row("AAA", "buy_candidate"), _advice_row("BBB", "watchlist")],
    )

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    unresolved = evaluation["unresolved_outcomes"]
    assert unresolved["reason_counts"] == {"missing_price_history": 2}


def test_label_performance_summary_aggregates_by_label(tmp_path: Path) -> None:
    advice_index = _write_advice_index(
        tmp_path,
        [_advice_row("AAA", "buy_candidate"), _advice_row("BBB", "buy_candidate")],
    )
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(70)])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    label = evaluation["label_performance_summary"]["labels"]["buy_candidate"]
    assert label["count"] == 2
    assert label["resolved_count"] == 1
    assert label["unresolved_count"] == 1
    assert label["average_return_by_horizon"]["1w"] == 5


def test_markdown_report_contains_ticker_rows(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(70)])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    assert "| AAA | buy_candidate | medium | 2026-07-11 | 100.00 |" in evaluation["advice_outcome_report"]


def test_outcome_tracker_does_not_require_openai_env(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_" + "API_KEY", raising=False)
    monkeypatch.delenv("MARKET_ENGINE_" + "ADVISORY_MODEL", raising=False)
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(70)])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    assert evaluation["advice_outcome_index"]["summary"]["resolved_outcomes"] == 1


def test_outcome_tracker_does_not_touch_network(monkeypatch: Any, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ME-EVAL01 must not make network/provider calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(70)])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    assert evaluation["advice_outcome_index"]["summary"]["resolved_by_horizon"]["1w"] == 1


def test_manifest_has_no_broker_portfolio_watchlist_or_delivery_side_effects(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", "buy_candidate")])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", [100 + index for index in range(70)])

    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )

    assert evaluation["manifest"]["baseline_guardrail"] == {
        "broker_order_execution_performed": False,
        "live_source_acquisition_performed": False,
        "openai_api_required": False,
        "portfolio_watchlist_mutation_performed": False,
        "provider_invocation_allowed": False,
    }


def _evaluate_single(tmp_path: Path, label: str, prices: list[float]) -> dict[str, Any]:
    advice_index = _write_advice_index(tmp_path, [_advice_row("AAA", label)])
    _write_price_csv(tmp_path / "prices" / "AAA.csv", prices)
    evaluation = build_advice_outcome_evaluation(
        advice_index,
        price_data_root=tmp_path / "prices",
        run_id="run-20260711T150000Z",
    )
    return evaluation["advice_outcome_index"]["tickers"][0]


def _write_advice_index(root: Path, rows: list[dict[str, Any]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "advice_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-advice-batch-index-v1",
                "artifact_type": "market-engine-deterministic-advice-batch-index",
                "run_id": "advice-run-20260711T140000Z",
                "generated_at": "2026-07-11T14:00:00Z",
                "summary": {"tickers_total": len(rows)},
                "tickers": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def _advice_row(ticker: str, label: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "advice": label,
        "confidence": "medium",
        "primary_reason": "Test row.",
    }


def _write_price_csv(
    path: Path,
    adjusted_prices: list[float],
    *,
    close_prices: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    close_values = close_prices or adjusted_prices
    lines = ["Date,Adj Close,Close"]
    start = date(2026, 7, 11)
    for index, adjusted in enumerate(adjusted_prices):
        row_date = start + timedelta(days=index)
        lines.append(f"{row_date.isoformat()},{adjusted},{close_values[index]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
