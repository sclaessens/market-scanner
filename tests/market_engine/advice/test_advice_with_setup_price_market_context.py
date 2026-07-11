from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from market_engine.advice.deterministic_advice import build_advice_index


def test_missing_context_remains_watchlist_for_valid_partial_artifact(
    tmp_path: Path,
) -> None:
    index_path = _write_status_index(tmp_path, [_status_row(tmp_path, "AAA")])

    advice = build_advice_index(
        index_path,
        run_id="run",
        generated_at="2026-07-11T00:00:00Z",
    )

    row = advice["tickers"][0]
    assert row["advice"] == "watchlist"
    assert row["setup_price_market_context"]["context_status"] == "missing"
    assert "setup_price_market_context" in row["missing_for_buy_candidate"]


def test_uptrend_pullback_near_entry_becomes_buy_candidate(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context=_setup_context(
                    trend_state="uptrend",
                    setup_state="pullback_watch",
                    price_position="near_entry_zone",
                    risk_state="normal",
                ),
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "buy_candidate"
    assert advice["tickers"][0]["confidence"] == "medium"


def test_uptrend_breakout_above_entry_becomes_wait_for_price(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context=_setup_context(
                    trend_state="uptrend",
                    setup_state="breakout_candidate",
                    price_position="above_preferred_entry",
                    risk_state="normal",
                ),
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "wait_for_price"


def test_downtrend_becomes_avoid_for_now(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context=_setup_context(
                    trend_state="downtrend",
                    setup_state="pullback_watch",
                    price_position="fair_zone",
                    risk_state="normal",
                ),
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "avoid_for_now"


def test_weak_setup_becomes_avoid_for_now(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context=_setup_context(
                    trend_state="sideways",
                    setup_state="weak_setup",
                    price_position="fair_zone",
                    risk_state="normal",
                ),
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "avoid_for_now"


def test_high_risk_becomes_avoid_for_now(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context=_setup_context(
                    trend_state="uptrend",
                    setup_state="pullback_watch",
                    price_position="near_entry_zone",
                    risk_state="high",
                ),
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "avoid_for_now"


def test_invalid_setup_price_market_context_becomes_unable_to_advise(
    tmp_path: Path,
) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row(
                tmp_path,
                "AAA",
                setup_context={
                    "context_status": "invalid",
                    "missing": ["valid_setup_price_market_context"],
                    "blocked_reasons": ["setup_price_market_context_invalid"],
                },
            )
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "unable_to_advise"
    assert "valid_setup_price_market_context" in advice["tickers"][0]["missing_for_buy_candidate"]


def test_context_evidence_is_included_in_advice_row(tmp_path: Path) -> None:
    context = _setup_context(
        trend_state="uptrend",
        setup_state="pullback_watch",
        price_position="near_entry_zone",
        risk_state="normal",
    )
    index_path = _write_status_index(
        tmp_path,
        [_status_row(tmp_path, "AAA", setup_context=context)],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    evidence = advice["tickers"][0]["setup_price_market_context"]["evidence"]
    assert evidence == context["evidence"]


def _write_status_index(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "ticker_status_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-ticker-status-index-v1",
                "artifact_type": "market-engine-ticker-status-index",
                "run_id": "status-run",
                "generated_at": "2026-07-11T00:00:00Z",
                "summary": {"tickers_total": len(rows)},
                "tickers": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _status_row(
    tmp_path: Path,
    ticker: str,
    *,
    setup_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": "blocked",
        "readiness_level": "partial_analysis",
        "context_stale": False,
        "actionable_review_allowed": False,
        "decision_engine_ready": False,
        "blocked_stage": "portfolio_review",
        "blocked_reasons": ["Stage preserves an upstream blocked state."],
        "readiness_blocked_reasons": ["missing_setup_or_price_context"],
        "missing_data_summary": ["portfolio_context"],
        "evidence_families_missing": ["setup_price_market"],
        "artifact_path": _write_dry_run(tmp_path, ticker, setup_context=setup_context).as_posix(),
        "artifact_sha256": "sha",
    }


def _write_dry_run(
    tmp_path: Path,
    ticker: str,
    *,
    setup_context: dict[str, Any] | None,
) -> Path:
    path = tmp_path / ticker / "dry_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ticker": ticker,
        "stage_results": [
            {
                "stage_name": "fundamental_observations",
                "status": "completed",
            }
        ],
        "provenance_summary": {
            "fundamental_observations": {
                "fundamental_observations_run_id": f"{ticker.lower()}-fundamental"
            }
        },
        "available_context_families": [],
    }
    if setup_context is not None:
        payload["setup_price_market_context"] = setup_context
    path.write_text(json.dumps({"payload": payload}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _setup_context(
    *,
    trend_state: str,
    setup_state: str,
    price_position: str,
    risk_state: str,
) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-setup-price-market-context-v1",
        "context_status": "partial",
        "price_context_available": True,
        "setup_context_available": True,
        "market_context_available": False,
        "trend_state": trend_state,
        "setup_state": setup_state,
        "price_position": price_position,
        "risk_state": risk_state,
        "evidence": [
            {
                "field": "setup_detection",
                "source_path": "fixture/dry_run.json",
                "source_family": "setup_detection",
            }
        ],
        "missing": ["market_context"],
        "blocked_reasons": [],
    }
