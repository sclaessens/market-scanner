from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

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
    assert summary["insufficient_history"] == 1
    assert summary["blocked"] == 1
    assert {row["symbol"] for row in artifacts["universe_analysis_index"]["instruments"]} == {
        "AAA",
        "BBB",
        "MISS",
        "SHORT",
    }
    assert (output_dir / "manifest.json").exists()
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


def test_candidate_ranking_excludes_missing_and_blocked_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    price_root = tmp_path / "prices"
    _write_price_csv(price_root / "AAA.csv", _trend_rows("2025-10-24", 260, first_close=50, step=0.4))
    _write_price_csv(price_root / "SHORT.csv", _trend_rows("2026-07-01", 10, first_close=10, step=0.1))
    monkeypatch.setattr(run30, "build_universe_snapshot", lambda _path, *, price_history_root: _universe(symbols=("AAA", "MISS", "SHORT")))

    artifacts, _output_dir = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    ranking = artifacts["candidate_ranking"]["candidates"]
    assert [row["symbol"] for row in ranking] == ["AAA"]
    blocked = {
        row["symbol"]: row
        for row in artifacts["universe_analysis_index"]["instruments"]
        if row["symbol"] in {"MISS", "SHORT"}
    }
    assert blocked["MISS"]["candidate_score"] is None
    assert blocked["SHORT"]["candidate_score"] is None
    assert blocked["MISS"]["ranking_eligible"] is False
    assert blocked["SHORT"]["ranking_eligible"] is False


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
    )
    second, _ = run30.run_full_canonical_universe_analysis(
        run_id="me-run30-test-20260714T100000Z",
        universe_path=tmp_path / "canonical_universe.json",
        price_history_root=price_root,
        output_root=tmp_path / "runs",
        cutoff_date="2026-07-10",
    )

    first_symbols = [row["symbol"] for row in first["candidate_ranking"]["candidates"]]
    second_symbols = [row["symbol"] for row in second["candidate_ranking"]["candidates"]]
    assert first_symbols == second_symbols == ["AAA", "BBB"]


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


def _write_price_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
        writer.writeheader()
        writer.writerows(rows)
