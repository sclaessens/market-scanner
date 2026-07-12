from __future__ import annotations

import io
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from market_engine.evaluation.advice_outcome_command import run_command as run_evaluation_command
from market_engine.evaluation.advice_outcome_refresh import build_advice_outcome_refresh
from market_engine.evaluation.advice_outcome_refresh_command import (
    run_command as run_refresh_command,
)


def test_unresolved_outcome_is_resolved_with_newer_local_snapshot(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    outcome = refresh["refresh_index"]["outcomes"][0]
    assert outcome["previous_blocker"] == "insufficient_forward_data"
    assert outcome["new_status"] == "resolved"
    assert outcome["resolved"] is True
    assert outcome["outcome_metrics"]["1w"]["return_pct"] == 5


def test_outcome_remains_insufficient_forward_data(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100, 101, 102, 103])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    outcome = refresh["refresh_index"]["outcomes"][0]
    assert outcome["new_status"] == "unresolved"
    assert outcome["new_blocker"] == "insufficient_forward_data"


def test_ticker_without_local_csv_is_missing_price_history(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [], write_initial_price=False)

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    outcome = refresh["refresh_index"]["outcomes"][0]
    assert outcome["new_blocker"] == "missing_price_history"
    assert refresh["missing_price_history"]["tickers"] == ["AAA"]


def test_missing_price_history_report_covers_known_me_eval01_missing_tickers(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation_for_rows(
        tmp_path,
        [
            _advice_row("CLS", "watchlist"),
            _advice_row("CRDO", "watchlist"),
            _advice_row("IREN", "watchlist"),
            _advice_row("VRT", "watchlist"),
        ],
        initial_prices={},
    )

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["refresh_index"]["summary"]["missing_price_history"] == 4
    assert refresh["refresh_index"]["summary"]["missing_price_history_tickers"] == [
        "CLS",
        "CRDO",
        "IREN",
        "VRT",
    ]


def test_invalid_csv_fails_closed(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    refreshed = tmp_path / "refreshed"
    refreshed.mkdir()
    (refreshed / "AAA.csv").write_text("close\n100\n", encoding="utf-8")

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=refreshed,
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    outcome = refresh["refresh_index"]["outcomes"][0]
    assert outcome["new_status"] == "unresolved"
    assert outcome["new_blocker"] == "invalid_price_history"


def test_missing_evaluation_context_fails_closed_per_ticker(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    data = json.loads(previous_artifact.read_text())
    data["input"]["advice_index_path"] = (tmp_path / "missing" / "advice_index.json").as_posix()
    previous_artifact.write_text(json.dumps(data), encoding="utf-8")
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    outcome = refresh["refresh_index"]["outcomes"][0]
    assert outcome["new_status"] == "blocked"
    assert outcome["new_blocker"] == "missing_evaluation_context"


def test_refresh_does_not_touch_network(monkeypatch: Any, tmp_path: Path) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ME-EVAL02 must not make live/provider calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["refresh_index"]["summary"]["resolved"] == 1


def test_mixed_refresh_run_summarizes_resolved_insufficient_and_missing(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation_for_rows(
        tmp_path,
        [
            _advice_row("AAA", "buy_candidate"),
            _advice_row("BBB", "wait_for_price"),
            _advice_row("CCC", "watchlist"),
        ],
        initial_prices={"AAA": [100, 101], "BBB": [100, 101]},
    )
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])
    _write_price_csv(tmp_path / "refreshed" / "BBB.csv", [100, 101, 102])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["refresh_index"]["summary"]["selected_outcomes"] == 3
    assert refresh["refresh_index"]["summary"]["resolved"] == 1
    assert refresh["refresh_index"]["summary"]["insufficient_forward_data"] == 1
    assert refresh["refresh_index"]["summary"]["missing_price_history"] == 1


def test_idempotent_rerun_with_same_inputs(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    first = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )
    second = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert first["refresh_index"] == second["refresh_index"]


def test_refresh_schema_is_deterministic(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["refresh_index"]["schema_version"] == "market-engine-advice-outcome-refresh-run-v1"
    assert refresh["manifest"]["schema_version"] == "market-engine-advice-outcome-refresh-manifest-v1"
    assert sorted(refresh) == ["manifest", "missing_price_history", "refresh_index", "refresh_report"]


def test_original_advice_remains_unchanged(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102], advice="wait_for_price")
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["refresh_index"]["outcomes"][0]["advice"] == "wait_for_price"


def test_manifest_has_no_portfolio_broker_or_scheduler_side_effects(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])

    refresh = build_advice_outcome_refresh(
        previous_artifact,
        price_history_root=tmp_path / "refreshed",
        run_id="me-eval02-refresh-20260712T130000Z",
    )

    assert refresh["manifest"]["baseline_guardrail"] == {
        "broker_order_execution_performed": False,
        "live_source_acquisition_performed": False,
        "openai_api_required": False,
        "portfolio_watchlist_mutation_performed": False,
        "provider_invocation_allowed": False,
        "scheduler_implemented": False,
    }


def test_refresh_command_writes_artifacts_and_supports_ticker_filter(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation_for_rows(
        tmp_path,
        [_advice_row("AAA", "buy_candidate"), _advice_row("BBB", "watchlist")],
        initial_prices={"AAA": [100, 101], "BBB": [100, 101]},
    )
    _write_price_csv(tmp_path / "refreshed" / "AAA.csv", [100 + index for index in range(70)])
    _write_price_csv(tmp_path / "refreshed" / "BBB.csv", [100 + index for index in range(70)])
    stdout = io.StringIO()

    exit_code = run_refresh_command(
        [
            "--evaluation-artifact",
            previous_artifact.as_posix(),
            "--price-history-root",
            (tmp_path / "refreshed").as_posix(),
            "--output-root",
            (tmp_path / "refresh-runs").as_posix(),
            "--run-id",
            "refresh-run",
            "--tickers",
            "AAA",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["summary"]["selected_outcomes"] == 1
    output_dir = tmp_path / "refresh-runs" / "refresh-run"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "manifest.json",
        "missing_price_history.json",
        "refresh_outcome_index.json",
        "refresh_report.md",
    ]


def test_refresh_command_rejects_existing_output_without_allow_overwrite(tmp_path: Path) -> None:
    previous_artifact = _previous_evaluation(tmp_path, "AAA", [100, 101, 102])
    output_dir = tmp_path / "refresh-runs" / "refresh-run"
    output_dir.mkdir(parents=True)
    stderr = io.StringIO()

    exit_code = run_refresh_command(
        [
            "--evaluation-artifact",
            previous_artifact.as_posix(),
            "--price-history-root",
            (tmp_path / "refreshed").as_posix(),
            "--output-root",
            (tmp_path / "refresh-runs").as_posix(),
            "--run-id",
            "refresh-run",
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "output directory already exists" in stderr.getvalue()


def _previous_evaluation(
    tmp_path: Path,
    ticker: str,
    initial_prices: list[float],
    *,
    advice: str = "buy_candidate",
    write_initial_price: bool = True,
) -> Path:
    return _previous_evaluation_for_rows(
        tmp_path,
        [_advice_row(ticker, advice)],
        initial_prices={ticker: initial_prices} if write_initial_price else {},
    )


def _previous_evaluation_for_rows(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    initial_prices: dict[str, list[float]],
) -> Path:
    advice_index = tmp_path / "advice_index.json"
    advice_index.write_text(
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
    for ticker, prices in initial_prices.items():
        _write_price_csv(tmp_path / "initial" / f"{ticker}.csv", prices)
    exit_code = run_evaluation_command(
        [
            "--advice-index",
            advice_index.as_posix(),
            "--price-data-root",
            (tmp_path / "initial").as_posix(),
            "--output-root",
            (tmp_path / "evaluations").as_posix(),
            "--run-id",
            "previous-eval-20260712T120000Z",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == 0
    return tmp_path / "evaluations" / "previous-eval-20260712T120000Z" / "advice_outcome_index.json"


def _advice_row(ticker: str, advice: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "advice": advice,
        "confidence": "medium",
        "primary_reason": "Test row.",
    }


def _write_price_csv(path: Path, adjusted_prices: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = date(2026, 7, 11)
    lines = ["Date,Adj Close,Close"]
    for index, adjusted in enumerate(adjusted_prices):
        row_date = start + timedelta(days=index)
        lines.append(f"{row_date.isoformat()},{adjusted},{adjusted}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
