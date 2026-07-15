from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from market_engine.run import full_canonical_universe_analysis as run30


def test_full_universe_analysis_attempts_every_instrument_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    _write_price_csv(price_root / "BBB.csv", _trend_rows("2025-10-24", 260, first_close=90, step=-0.2))
    _write_price_csv(price_root / "SHORT.csv", _trend_rows("2026-07-01", 10, first_close=10, step=0.1))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe())

    artifacts, output_dir = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    summary = artifacts["universe_analysis_index"]["summary"]
    assert summary["total_canonical_instruments"] == 4
    assert summary["attempted_instruments"] == 4
    assert summary["eligible_analyzed"] == 2
    assert summary["blocked_insufficient_history"] == 1
    assert summary["blocked_missing_history"] == 1
    assert summary["blocked_instruments"] == 2
    assert summary["failed"] == 0
    assert set(summary["output_label_counts"]) == set(run30.SCREENING_LABELS)
    assert "advice_label_counts" not in summary
    assert {row["symbol"] for row in artifacts["universe_analysis_index"]["instruments"]} == {
        "AAA",
        "BBB",
        "MISS",
        "SHORT",
    }
    assert artifacts["manifest"]["guardrails"]["deterministic_advice_labels_produced"] is False
    assert artifacts["manifest"]["guardrails"]["technical_screening_labels_produced"] is True
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "analysis_outcome_distribution.json",
        "blocker_report.json",
        "candidate_ranking.json",
        "candidate_ranking.md",
        "manifest.json",
        "setup_detection_summary.json",
        "throughput_report.json",
        "top_candidates.md",
        "unable_to_analyse.md",
        "universe_analysis_index.json",
        "universe_analysis_summary.md",
    ]


def test_me_run30_reuses_canonical_setup_context_extractor(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    class Context:
        def to_payload(self) -> dict[str, Any]:
            return {
                "context_status": "partial",
                "trend_state": "uptrend",
                "setup_state": "pullback_watch",
                "price_position": "near_entry_zone",
                "risk_state": "normal",
                "missing": ["market_context"],
                "blocked_reasons": [],
            }

    def capture(row: dict[str, Any], payload: dict[str, Any], *, local_price_root: Path) -> Context:
        calls.append({"row": row, "payload": payload, "local_price_root": local_price_root})
        return Context()

    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA",)))
    monkeypatch.setattr(run30, "extract_setup_price_market_context", capture)

    artifacts, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    assert calls
    assert calls[0]["payload"]["setup_price_market_context"]["context_status"] == "partial"
    row = artifacts["universe_analysis_index"]["instruments"][0]
    assert row["setup_price_market_context"]["setup_state"] == "pullback_watch"
    assert row["output_label"] in run30.SCREENING_LABELS


def test_canonical_advice_labels_are_not_used_for_technical_screening(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA",)))

    artifacts, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    canonical_labels = {"buy_candidate", "wait_for_price", "watchlist", "avoid_for_now", "unable_to_advise"}
    labels = {row["output_label"] for row in artifacts["universe_analysis_index"]["instruments"]}
    assert labels.isdisjoint(canonical_labels)
    assert artifacts["candidate_ranking"]["ranking_policy"]["ranking_scope"] == "technical_setup_screening"
    assert artifacts["candidate_ranking"]["ranking_policy"]["full_advice_ready"] is False


def test_breakdown_uses_prior_support_window_excluding_latest_bar(tmp_path: Path) -> None:
    rows = _flat_rows("2026-01-01", 30, close=100)
    rows[-1]["Close"] = 95
    rows[-1]["Low"] = 94
    frame = run30._read_price_frame(_write_price_csv(tmp_path / "AAA.csv", rows, return_path=True))

    setup = run30._detect_technical_setup(frame)

    assert setup["prior_twenty_day_low"] == 99
    assert setup["support_window_excludes_latest_bar"] is True
    assert setup["support_break_pct"] < -0.02
    assert setup["price_position"] == "below_support_or_breakdown"


def test_intraday_low_below_support_without_close_break_is_not_breakdown(tmp_path: Path) -> None:
    rows = _flat_rows("2026-01-01", 30, close=100)
    rows[-1]["Close"] = 100
    rows[-1]["Low"] = 90
    frame = run30._read_price_frame(_write_price_csv(tmp_path / "AAA.csv", rows, return_path=True))

    setup = run30._detect_technical_setup(frame)

    assert setup["support_break_pct"] >= 0
    assert setup["price_position"] != "below_support_or_breakdown"


def test_technical_setup_rejects_insufficient_or_invalid_bars(tmp_path: Path) -> None:
    short = run30._read_price_frame(_write_price_csv(tmp_path / "SHORT.csv", _flat_rows("2026-01-01", 20, close=100), return_path=True))
    with pytest.raises(ValueError, match="at least 21"):
        run30._detect_technical_setup(short)

    invalid_rows = _flat_rows("2026-01-01", 30, close=100)
    invalid_rows[-1]["Close"] = -1
    invalid = run30._read_price_frame(_write_price_csv(tmp_path / "BAD.csv", invalid_rows, return_path=True))
    with pytest.raises(ValueError, match="positive"):
        run30._detect_technical_setup(invalid)


@pytest.mark.parametrize(
    ("snapshot_status", "expected"),
    [
        ("insufficient_history", "blocked_insufficient_history"),
        ("insufficient_forward_data", "blocked_stale_history"),
        ("stale_snapshot", "blocked_stale_history"),
        ("missing_price_history", "blocked_missing_history"),
        ("validation_failed", "blocked_invalid_history"),
        ("unsupported_symbol_mapping", "blocked_unsupported_mapping"),
    ],
)
def test_processing_status_mapping_is_specific(snapshot_status: str, expected: str) -> None:
    assert run30._processing_status(snapshot_status) == expected


def test_missing_evidence_never_adds_positive_score(tmp_path: Path) -> None:
    inspection = {"row_count": 252, "artifactpath": "prices/AAA.csv", "start_date": "2025-01-01", "end_date": "2026-07-10"}
    setup = {
        "trend_state": "uptrend",
        "setup_state": "pullback_watch",
        "price_position": "near_entry_zone",
        "risk_state": "normal",
        "prior_twenty_day_low": 99,
        "support_break_pct": 0.05,
        "support_window_excludes_latest_bar": True,
    }
    screening = {
        "label": "technical_setup_candidate",
        "blockers": [],
        "missing_evidence": ["fundamental_context", "portfolio_context", "market_context"],
    }

    result = run30._candidate_score(setup, setup, inspection, screening)

    assert result["positive_components"]["history_depth"] == 10
    assert result["penalties"]["missing_evidence"] == -15
    assert result["raw_score"] == sum(result["positive_components"].values()) + sum(result["penalties"].values())
    assert result["score_components"]["candidate_score"] == result["score"]


def test_blocked_and_failed_entries_are_never_ranking_eligible() -> None:
    ranked = run30._rank_candidates(
        [
            {
                "instrument_id": "equity:aaa",
                "symbol": "AAA",
                "source_symbol": "AAA",
                "candidate_score": 90,
                "output_label": "technical_setup_candidate",
                "confidence": "medium",
                "ranking_eligible": True,
                "full_advice_ready": False,
                "setup_detection": {"setup_state": "pullback_watch", "trend_state": "uptrend", "price_position": "near_entry_zone", "risk_state": "normal"},
                "missing_evidence": ["fundamental_context"],
                "blockers": [],
            },
            {
                "instrument_id": "equity:bbb",
                "symbol": "BBB",
                "source_symbol": "BBB",
                "candidate_score": None,
                "output_label": "unable_to_analyse",
                "confidence": "low",
                "ranking_eligible": False,
                "full_advice_ready": False,
                "setup_detection": {},
                "missing_evidence": ["valid_analysis_input"],
                "blockers": ["failed"],
            },
        ]
    )

    assert [row["symbol"] for row in ranked] == ["AAA"]


def test_candidate_ranking_is_deterministic_for_equal_scores(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    rows = _trend_rows("2025-10-24", 260, first_close=50, step=0.4)
    _write_price_csv(price_root / "BBB.csv", rows)
    _write_price_csv(price_root / "AAA.csv", rows)
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("BBB", "AAA")))

    first, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
        allow_overwrite=True,
    )
    second, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
        allow_overwrite=True,
    )

    first_symbols = [row["symbol"] for row in first["candidate_ranking"]["candidates"]]
    second_symbols = [row["symbol"] for row in second["candidate_ranking"]["candidates"]]
    assert first_symbols == second_symbols == ["AAA", "BBB"]


def test_default_overwrite_is_rejected_and_flag_allows_rewrite(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA",)))

    kwargs = {
        "run_id": "me-run30-test-20260714T100000Z",
        "universe_path": tmp_path / "canonical_universe.json",
        "price_history_root": price_root,
        "output_root": tmp_path / "runs",
        "cutoff_date": "2026-07-10",
    }
    run30.run_full_canonical_universe_analysis(**kwargs)
    with pytest.raises(FileExistsError):
        run30.run_full_canonical_universe_analysis(**kwargs)
    run30.run_full_canonical_universe_analysis(**kwargs, allow_overwrite=True)


def test_partial_existing_artifact_directory_is_not_silently_mixed(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    partial = output_root / "run"
    partial.mkdir(parents=True)
    (partial / "stale.txt").write_text("old", encoding="utf-8")
    artifacts = _minimal_artifacts()

    with pytest.raises(FileExistsError):
        run30._write_artifacts(artifacts, output_root=output_root, run_id="run", allow_overwrite=False)

    output_dir = run30._write_artifacts(artifacts, output_root=output_root, run_id="run", allow_overwrite=True)

    assert not (output_dir / "stale.txt").exists()
    assert (output_dir / "manifest.json").exists()


def test_throughput_aggregation_reports_distribution_and_rates() -> None:
    entries = [
        _runtime_row("BBB", 0.0, "eligible_analyzed"),
        _runtime_row("AAA", 0.2, "blocked_missing_history"),
        _runtime_row("CCC", 0.1, "failed"),
        _runtime_row("DDD", 0.4, "eligible_analyzed"),
    ]

    report = run30._aggregate_throughput(
        entries,
        1.0,
        start_time="2026-07-14T10:00:00Z",
        end_time="2026-07-14T10:00:01Z",
    )

    assert report["attempted_instruments"] == 4
    assert report["analysed_instruments"] == 2
    assert report["blocked_instruments"] == 1
    assert report["failed_instruments"] == 1
    assert report["median_runtime_seconds_per_ticker"] == 0.1
    assert report["p95_runtime_seconds_per_ticker"] == 0.4
    assert report["minimum_runtime_seconds_per_ticker"] == 0.0
    assert report["maximum_runtime_seconds_per_ticker"] == 0.4
    assert report["tickers_per_second"] == 4
    assert report["tickers_per_minute"] == 240
    assert report["successful_analysis_rate"] == 0.5
    assert report["failure_rate"] == 0.25
    assert [row["symbol"] for row in report["slowest_tickers"][:3]] == ["DDD", "AAA", "CCC"]


def test_artifact_contract_counts_are_internally_consistent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    _write_price_csv(price_root / "BBB.csv", _trend_rows("2025-10-24", 260, first_close=90, step=-0.2))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA", "BBB", "MISS")))

    artifacts, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    summary = artifacts["universe_analysis_index"]["summary"]
    assert sum(summary["final_processing_status_counts"].values()) == summary["total_canonical_instruments"]
    assert sum(summary["output_label_counts"].values()) == summary["total_canonical_instruments"]
    assert artifacts["candidate_ranking"]["candidate_count"] == len(artifacts["candidate_ranking"]["candidates"])
    assert "technical setup screening output" in artifacts["top_candidates"]


def test_run_command_rejects_existing_output_without_allow_overwrite(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA",)))
    args = [
        "--run-id",
        "me-run30-test-20260714T100000Z",
        "--universe",
        str(tmp_path / "canonical_universe.json"),
        "--price-history-root",
        str(price_root),
        "--output-root",
        str(tmp_path / "runs"),
        "--cutoff-date",
        "2026-07-10",
    ]

    assert run30.run_command(args, stdout=_Sink(), stderr=_Sink()) == 0
    assert run30.run_command(args, stdout=_Sink(), stderr=_Sink()) == 2
    assert run30.run_command([*args, "--allow-overwrite"], stdout=_Sink(), stderr=_Sink()) == 0


def test_governance_guardrails_are_false_in_manifest(tmp_path: Path) -> None:
    artifacts = run30._artifacts(
        run_id="me-run30-test-20260714T100000Z",
        universe=_universe(symbols=()),
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=tmp_path / "prices",
        cutoff_date="2026-07-10",
        entries=[],
        ranked=[],
        runtime_seconds=0.0,
        start_time="2026-07-14T10:00:00Z",
        end_time="2026-07-14T10:00:00Z",
        top_candidate_limit=25,
    )
    guardrails = artifacts["manifest"]["guardrails"]
    assert guardrails["provider_invocation_performed"] is False
    assert guardrails["provider_or_model_advice_generation_performed"] is False
    assert guardrails["broker_order_execution_performed"] is False
    assert guardrails["portfolio_watchlist_mutation_performed"] is False
    assert guardrails["telegram_delivery_performed"] is False
    assert guardrails["synthetic_forward_data_used"] is False


def _universe(symbols: tuple[str, ...] = ("AAA", "BBB", "MISS", "SHORT")) -> dict[str, Any]:
    return {
        "universe_version": "test-canonical-universe-v1",
        "summary": {
            "total_instruments": len(symbols),
            "layer_counts": {"test": len(symbols)},
            "unique_equities": len(symbols),
            "etf_count": 0,
            "context_count": 0,
        },
        "instruments": [
            {
                "instrument_id": f"equity:{symbol.lower()}",
                "symbol": symbol,
                "source_symbol": symbol,
                "asset_type": "equity",
                "universe_memberships": ["test"],
                "source_mapping_status": "mapped",
            }
            for symbol in symbols
        ],
    }


def _trend_rows(start_date: str, count: int, *, first_close: float, step: float) -> list[dict[str, Any]]:
    cursor = date.fromisoformat(start_date)
    rows = []
    for index in range(count):
        close = first_close + step * index
        rows.append(
            {
                "Date": (cursor + timedelta(days=index)).isoformat(),
                "Adj Close": close,
                "Close": close,
                "High": close + 1,
                "Low": close - 1,
                "Open": close - 0.25,
                "Volume": 1000 + index,
            }
        )
    return rows


def _flat_rows(start_date: str, count: int, *, close: float) -> list[dict[str, Any]]:
    cursor = date.fromisoformat(start_date)
    return [
        {
            "Date": (cursor + timedelta(days=index)).isoformat(),
            "Adj Close": close,
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Open": close,
            "Volume": 1000 + index,
        }
        for index in range(count)
    ]


def _write_price_csv(path: Path, rows: list[dict[str, Any]], *, return_path: bool = False) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
        writer.writeheader()
        writer.writerows(rows)
    return path if return_path else None


def _minimal_artifacts() -> dict[str, Any]:
    return {
        "manifest": {"guardrails": {}, "outputs": {}},
        "universe_analysis_index": {"summary": {}, "instruments": []},
        "universe_analysis_summary": "# Summary\n",
        "throughput_report": {},
        "setup_detection_summary": {},
        "analysis_outcome_distribution": {},
        "blocker_report": {},
        "candidate_ranking": {},
        "candidate_ranking_markdown": "# Ranking\n",
        "top_candidates": "# Top\n",
        "unable_to_analyse": "# Unable\n",
    }


def _runtime_row(symbol: str, runtime: float, status: str) -> dict[str, Any]:
    return {
        "instrument_id": f"equity:{symbol.lower()}",
        "symbol": symbol,
        "runtime_seconds": runtime,
        "final_processing_status": status,
    }


class _Sink:
    def write(self, _text: str) -> int:
        return len(_text)

    def flush(self) -> None:
        return None
