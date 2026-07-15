from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from market_engine.run import broad_non_price_evidence_advice_readiness as run31
from market_engine.run.local_portfolio_context_fixture import LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION


def test_run31_attempts_every_selected_instrument_and_writes_required_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch, symbols=("AAA", "BBB"))
    technical_dir = _write_technical_artifact(tmp_path, [_technical("AAA", score=90), _technical("BBB", score=60)])
    fundamental_path = _write_fundamentals(tmp_path, {"AAA": "SUFFICIENT_DATA", "BBB": {"quality_state": "INSUFFICIENT_DATA", "quality_metadata_status": "row_missing"}})
    market_path = _write_market(tmp_path)
    portfolio_path = _write_portfolio(tmp_path, positions={"AAA": _position()})

    artifacts, output_dir = run31.run_broad_non_price_evidence_advice_readiness(
        run_id="me-run31-test",
        canonical_universe=tmp_path / "canonical.json",
        price_history_root=tmp_path / "prices",
        technical_screening_artifact=technical_dir,
        output_root=tmp_path / "runs",
        fundamental_evidence_path=fundamental_path,
        market_context_path=market_path,
        portfolio_context_path=portfolio_path,
    )

    summary = artifacts["evidence_coverage_summary"]["summary"]
    assert summary["attempted_instruments"] == 2
    assert summary["technical_analysed"] == 2
    assert summary["fundamental_counts"]["available"] == 1
    assert summary["fundamental_counts"]["missing"] == 1
    assert summary["market_counts"]["available"] == 2
    assert summary["portfolio_applicable"] == 1
    assert summary["advice_completed"] == 2
    assert set(row["symbol"] for row in artifacts["evidence_coverage_index"]["instruments"]) == {"AAA", "BBB"}
    assert _output_names(output_dir) == sorted(run31._required_outputs().values())
    assert artifacts["manifest"]["guardrails"]["parallel_advice_rules_added"] is False
    assert artifacts["manifest"]["guardrails"]["broker_order_execution_performed"] is False


def test_fundamental_statuses_cover_available_partial_missing_stale_and_invalid(tmp_path: Path) -> None:
    rows = {
        "AAA": {"quality_state": "SUFFICIENT_DATA", "source_timestamp": "2026-07-01T00:00:00Z"},
        "BBB": {"quality_state": "PARTIAL_DATA", "source_timestamp": "2026-07-01T00:00:00Z"},
        "CCC": {"quality_state": "INSUFFICIENT_DATA", "quality_metadata_status": "row_missing", "source_timestamp": "2026-07-01T00:00:00Z"},
        "DDD": {"quality_state": "SUFFICIENT_DATA", "source_timestamp": "2025-01-01T00:00:00Z", "source_last_updated": "2025-01-01"},
        "EEE": {"quality_state": "BROKEN", "source_timestamp": "2026-07-01T00:00:00Z"},
    }
    path = _write_fundamentals(tmp_path, rows)
    loaded = run31._load_fundamental_rows(path)

    assert run31._fundamental_context("AAA", loaded, source_path=path)["status"] == "available"
    assert run31._fundamental_context("BBB", loaded, source_path=path)["status"] == "partial"
    assert run31._fundamental_context("CCC", loaded, source_path=path)["status"] == "missing"
    assert run31._fundamental_context("DDD", loaded, source_path=path)["status"] == "stale"
    assert run31._fundamental_context("EEE", loaded, source_path=path)["status"] == "invalid"
    missing = run31._fundamental_context("ZZZ", loaded, source_path=path)
    assert missing["status"] == "missing"
    assert missing["source_checksum"] == run31._sha256(path)


def test_market_context_statuses_cover_available_missing_stale_and_invalid(tmp_path: Path) -> None:
    available = run31._load_market_context(_write_market(tmp_path, date_value="2026-07-01"))
    assert available["status"] == "available"

    assert run31._load_market_context(tmp_path / "missing.csv")["status"] == "missing"
    assert run31._load_market_context(_write_market(tmp_path, date_value="2025-01-01", name="stale.csv"))["status"] == "stale"

    invalid = tmp_path / "invalid_market.csv"
    invalid.write_text("date\n2026-07-01\n", encoding="utf-8")
    assert run31._load_market_context(invalid)["status"] == "invalid"


def test_portfolio_context_is_available_only_for_existing_positions(tmp_path: Path) -> None:
    portfolio = run31._load_portfolio_context(_write_portfolio(tmp_path, positions={"AAA": _position()}))

    held = run31._portfolio_context_for_instrument("AAA", portfolio, generated_at="2026-07-10T00:00:00Z", run_id="run")
    absent = run31._portfolio_context_for_instrument("BBB", portfolio, generated_at="2026-07-10T00:00:00Z", run_id="run")

    assert held["status"] == "available"
    assert held["values"]["context_provenance"]["portfolio_write_authority"] is False
    assert held["values"]["context_provenance"]["no_portfolio_or_watchlist_mutation"] is True
    assert absent["status"] == "not_applicable"


def test_portfolio_context_reports_invalid_missing_and_stale_applicable_context(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_portfolio.json"
    invalid_path.write_text(json.dumps({"portfolio_context_batch_format_version": "wrong", "positions_by_ticker": {"AAA": _position()}}), encoding="utf-8")
    invalid = run31._load_portfolio_context(invalid_path)
    assert run31._portfolio_context_for_instrument("AAA", invalid, generated_at="2026-07-10T00:00:00Z", run_id="run")["status"] == "invalid"

    missing_path = _write_portfolio(tmp_path, positions={"AAA": {"position_state": "held"}}, name="missing_portfolio.json")
    missing = run31._load_portfolio_context(missing_path)
    assert run31._portfolio_context_for_instrument("AAA", missing, generated_at="2026-07-10T00:00:00Z", run_id="run")["status"] == "missing"

    stale_path = _write_portfolio(tmp_path, positions={"AAA": _position()}, snapshot="2025-01-01T00:00:00Z", name="stale_portfolio.json")
    stale = run31._load_portfolio_context(stale_path)
    assert run31._portfolio_context_for_instrument("AAA", stale, generated_at="2026-07-10T00:00:00Z", run_id="run")["status"] == "stale"


def test_canonical_advice_engine_is_called_without_parallel_labels(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch)
    technical_dir = _write_technical_artifact(tmp_path, [_technical("AAA", score=80)])
    fundamental_path = _write_fundamentals(tmp_path, {"AAA": "SUFFICIENT_DATA"})
    market_path = _write_market(tmp_path)
    captured: dict[str, Any] = {}

    def fake_build_advice_index(status_index_path: Path, *, run_id: str, generated_at: str) -> dict[str, Any]:
        captured["path"] = status_index_path
        captured["status_index"] = json.loads(status_index_path.read_text(encoding="utf-8"))
        return {
            "schema_version": "market-engine-advice-index-v1",
            "run_id": run_id,
            "generated_at": generated_at,
            "tickers": [
                {
                    "ticker": "AAA",
                    "advice": "buy_candidate",
                    "confidence": "medium",
                    "advice_readiness": "ready",
                    "blockers": [],
                    "missing_for_buy_candidate": [],
                }
            ],
        }

    monkeypatch.setattr(run31, "build_advice_index", fake_build_advice_index)

    artifacts, _ = run31.run_broad_non_price_evidence_advice_readiness(
        run_id="me-run31-canonical-call",
        canonical_universe=tmp_path / "canonical.json",
        price_history_root=tmp_path / "prices",
        technical_screening_artifact=technical_dir,
        output_root=tmp_path / "runs",
        fundamental_evidence_path=fundamental_path,
        market_context_path=market_path,
        portfolio_context_path=_write_portfolio(tmp_path, positions={}),
    )

    row = artifacts["evidence_coverage_index"]["instruments"][0]
    assert captured["status_index"]["tickers"][0]["ticker"] == "AAA"
    assert row["canonical_advice_label"] == "buy_candidate"
    assert row["full_advice_ready"] is True
    assert artifacts["full_advice_ranking"]["candidates"][0]["canonical_advice_label"] == "buy_candidate"
    assert row["technical_screening_label"] == "technical_setup_candidate"


def test_missing_fundamental_evidence_fails_closed_to_unable_to_advise(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch)
    technical_dir = _write_technical_artifact(tmp_path, [_technical("AAA", score=90)])

    artifacts, _ = run31.run_broad_non_price_evidence_advice_readiness(
        run_id="me-run31-fail-closed",
        canonical_universe=tmp_path / "canonical.json",
        price_history_root=tmp_path / "prices",
        technical_screening_artifact=technical_dir,
        output_root=tmp_path / "runs",
        fundamental_evidence_path=tmp_path / "missing_fundamentals.csv",
        market_context_path=_write_market(tmp_path),
        portfolio_context_path=_write_portfolio(tmp_path, positions={}),
    )

    row = artifacts["evidence_coverage_index"]["instruments"][0]
    assert row["canonical_advice_label"] == "unable_to_advise"
    assert row["full_advice_ready"] is False
    assert "fundamental_context" in row["missing_evidence"]


def test_technical_ranking_stays_separate_from_full_advice_ranking(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch, symbols=("AAA", "BBB"))
    technical_dir = _write_technical_artifact(
        tmp_path,
        [
            _technical("AAA", score=90, ranking_eligible=True),
            _technical("BBB", score=95, ranking_eligible=False),
        ],
    )
    fundamental_path = _write_fundamentals(tmp_path, {"AAA": "INSUFFICIENT_DATA", "BBB": "INSUFFICIENT_DATA"})

    artifacts, _ = run31.run_broad_non_price_evidence_advice_readiness(
        run_id="me-run31-ranking-separation",
        canonical_universe=tmp_path / "canonical.json",
        price_history_root=tmp_path / "prices",
        technical_screening_artifact=technical_dir,
        output_root=tmp_path / "runs",
        fundamental_evidence_path=fundamental_path,
        market_context_path=_write_market(tmp_path),
        portfolio_context_path=_write_portfolio(tmp_path, positions={}),
    )

    assert [row["symbol"] for row in artifacts["technical_ranking"]["candidates"]] == ["AAA"]
    assert artifacts["full_advice_ranking"]["candidates"] == []
    assert artifacts["evidence_coverage_summary"]["summary"]["technical_ranking_eligible"] == 1


def test_full_advice_ranking_uses_stable_tie_breakers() -> None:
    ranked = run31._full_advice_ranking(
        [
            _entry_for_ranking("BBB", label="buy_candidate", score=80),
            _entry_for_ranking("AAA", label="buy_candidate", score=80),
            _entry_for_ranking("CCC", label="wait_for_price", score=99),
        ]
    )

    assert [row["symbol"] for row in ranked] == ["AAA", "BBB", "CCC"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_source_lineage_contains_paths_checksums_dates_and_statuses(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch)
    fundamental_path = _write_fundamentals(tmp_path, {"AAA": "SUFFICIENT_DATA"})
    artifacts, _ = run31.run_broad_non_price_evidence_advice_readiness(
        run_id="me-run31-lineage",
        canonical_universe=tmp_path / "canonical.json",
        price_history_root=tmp_path / "prices",
        technical_screening_artifact=_write_technical_artifact(tmp_path, [_technical("AAA")]),
        output_root=tmp_path / "runs",
        fundamental_evidence_path=fundamental_path,
        market_context_path=_write_market(tmp_path),
        portfolio_context_path=_write_portfolio(tmp_path, positions={}),
    )

    lineage = artifacts["source_lineage"]["entries"][0]["source_lineage"]
    fundamental = next(row for row in lineage if row["family"] == "fundamental_context")
    market = next(row for row in lineage if row["family"] == "market_context")
    assert fundamental["source_path"] == fundamental_path.as_posix()
    assert fundamental["source_checksum"] == run31._sha256(fundamental_path)
    assert fundamental["source_date"].startswith("2026-07-01")
    assert market["status"] == "available"


def test_output_directory_is_not_reused_without_explicit_overwrite(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch)
    kwargs = _minimal_run_kwargs(tmp_path)
    run31.run_broad_non_price_evidence_advice_readiness(run_id="me-run31-overwrite", **kwargs)

    with pytest.raises(FileExistsError):
        run31.run_broad_non_price_evidence_advice_readiness(run_id="me-run31-overwrite", **kwargs)

    artifacts, _ = run31.run_broad_non_price_evidence_advice_readiness(run_id="me-run31-overwrite", allow_overwrite=True, **kwargs)
    assert artifacts["manifest"]["run_id"] == "me-run31-overwrite"


def test_prepare_output_dir_uses_temp_directory_until_validation(tmp_path: Path) -> None:
    temp_dir = run31._prepare_output_dir(output_root=tmp_path, run_id="run", allow_overwrite=False)

    assert temp_dir.name == ".run.tmp"
    assert not (tmp_path / "run").exists()


def test_validate_required_outputs_rejects_partial_artifact_set(tmp_path: Path) -> None:
    temp_dir = tmp_path / ".run.tmp"
    temp_dir.mkdir()
    (temp_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact set incomplete"):
        run31._validate_required_outputs(temp_dir)


def test_command_prints_final_output_directory(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    _patch_universe(monkeypatch)
    kwargs = _minimal_run_kwargs(tmp_path)
    args = [
        "--run-id",
        "me-run31-cli",
        "--canonical-universe",
        str(kwargs["canonical_universe"]),
        "--price-history-root",
        str(kwargs["price_history_root"]),
        "--technical-screening-artifact",
        str(kwargs["technical_screening_artifact"]),
        "--output-root",
        str(kwargs["output_root"]),
        "--fundamental-evidence-path",
        str(kwargs["fundamental_evidence_path"]),
        "--market-context-path",
        str(kwargs["market_context_path"]),
        "--portfolio-context-path",
        str(kwargs["portfolio_context_path"]),
    ]

    assert run31.main(args) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["output_dir"].endswith("/me-run31-cli")
    assert Path(payload["output_dir"]).exists()


def test_persisted_advice_artifact_paths_use_final_directory(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_universe(monkeypatch)
    kwargs = _minimal_run_kwargs(tmp_path)

    _, output_dir = run31.run_broad_non_price_evidence_advice_readiness(run_id="me-run31-final-paths", **kwargs)

    input_index = json.loads((output_dir / "canonical_advice_input_index.json").read_text(encoding="utf-8"))
    output_index = json.loads((output_dir / "canonical_advice_output_index.json").read_text(encoding="utf-8"))
    assert ".me-run31-final-paths.tmp" not in json.dumps(input_index)
    assert ".me-run31-final-paths.tmp" not in json.dumps(output_index)
    assert input_index["tickers"][0]["artifact_path"].startswith(output_dir.as_posix())
    assert Path(input_index["tickers"][0]["artifact_path"]).exists()


def _minimal_run_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "canonical_universe": tmp_path / "canonical.json",
        "price_history_root": tmp_path / "prices",
        "technical_screening_artifact": _write_technical_artifact(tmp_path, [_technical("AAA")]),
        "output_root": tmp_path / "runs",
        "fundamental_evidence_path": _write_fundamentals(tmp_path, {"AAA": "SUFFICIENT_DATA"}),
        "market_context_path": _write_market(tmp_path),
        "portfolio_context_path": _write_portfolio(tmp_path, positions={}),
    }


def _patch_universe(monkeypatch: Any, symbols: tuple[str, ...] = ("AAA",)) -> None:
    monkeypatch.setattr(
        run31,
        "build_universe_snapshot",
        lambda _path, *, price_history_root: _universe(symbols),
    )


def _universe(symbols: tuple[str, ...] = ("AAA",)) -> dict[str, Any]:
    instruments = [
        {
            "instrument_id": f"equity:{symbol.lower()}",
            "symbol": symbol,
            "asset_type": "equity",
            "name": symbol,
            "exchange": "XNYS",
            "country": "US",
            "currency": "USD",
            "sector": "Technology",
            "industry": "Software",
            "universe_memberships": ["local_price_history_covered", "sp500"],
            "source_symbol": symbol,
        }
        for symbol in symbols
    ]
    return {"universe_version": "test-universe", "summary": {"total_instruments": len(instruments)}, "instruments": instruments}


def _write_technical_artifact(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    artifact_dir = tmp_path / "technical"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / "manifest.json").write_text(json.dumps({"run_id": "me-run30-test"}, indent=2), encoding="utf-8")
    payload = {
        "schema_version": "market-engine-run30-universe-analysis-index-v1",
        "run_id": "me-run30-test",
        "summary": {"attempted_instruments": len(rows)},
        "instruments": rows,
    }
    (artifact_dir / "universe_analysis_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_dir


def _technical(symbol: str, *, score: int = 80, ranking_eligible: bool = True) -> dict[str, Any]:
    return {
        "instrument_id": f"equity:{symbol.lower()}",
        "symbol": symbol,
        "asset_type": "equity",
        "final_processing_status": "eligible_analyzed",
        "output_label": "technical_setup_candidate",
        "candidate_score": score,
        "ranking_eligible": ranking_eligible,
        "ranking_scope": run31.TECHNICAL_RANKING_SCOPE,
        "price_history": {"end_date": "2026-07-10"},
        "setup_detection": {
            "trend_state": "uptrend",
            "setup_state": "pullback_watch",
            "price_position": "near_entry_zone",
            "risk_state": "normal",
        },
        "setup_price_market_context": {
            "context_status": "partial",
            "trend_state": "uptrend",
            "setup_state": "pullback_watch",
            "price_position": "near_entry_zone",
            "risk_state": "normal",
            "missing": ["market_context"],
            "blocked_reasons": [],
            "evidence": [],
        },
        "blockers": [],
        "missing_evidence": [],
    }


def _write_fundamentals(tmp_path: Path, states: dict[str, str | dict[str, str]]) -> Path:
    path = tmp_path / "fundamentals.csv"
    fieldnames = [
        "ticker",
        "quality_state",
        "source_data_status",
        "source_timestamp",
        "source_last_updated",
        "source_name",
        "generated_at",
        "quality_reason",
        "quality_metadata_status",
        "missing_required_fields",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ticker, value in sorted(states.items()):
            row = {
                "ticker": ticker,
                "quality_state": value if isinstance(value, str) else value.get("quality_state", ""),
                "source_data_status": "complete",
                "source_timestamp": "2026-07-01T00:00:00Z",
                "source_last_updated": "2026-07-01",
                "source_name": "test-source",
                "generated_at": "2026-07-10T00:00:00Z",
                "quality_reason": "test fixture",
                "quality_metadata_status": "",
                "missing_required_fields": "",
            }
            if isinstance(value, dict):
                row.update(value)
            writer.writerow(row)
    return path


def _write_market(tmp_path: Path, *, date_value: str = "2026-07-01", name: str = "market.csv") -> Path:
    path = tmp_path / name
    path.write_text("date,regime,spy_close,qqq_close\n" f"{date_value},BULLISH,600,520\n", encoding="utf-8")
    return path


def _write_portfolio(
    tmp_path: Path,
    *,
    positions: dict[str, dict[str, Any]],
    snapshot: str = "2026-07-01T00:00:00Z",
    name: str = "portfolio.json",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "portfolio_context_batch_format_version": LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION,
                "portfolio_context_format_version": "market-engine-portfolio-context-v1",
                "portfolio_snapshot_timestamp": snapshot,
                "portfolio_base_currency": "EUR",
                "portfolio_total_value": 100000,
                "positions_by_ticker": positions,
                "concentration_thresholds": {},
                "policy_constraints": {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _position() -> dict[str, Any]:
    return {
        "position_state": "held",
        "current_quantity": 10,
        "current_market_value": 1000,
        "current_ticker_exposure_pct": 1.0,
    }


def _entry_for_ranking(symbol: str, *, label: str, score: int) -> dict[str, Any]:
    return {
        "instrument_id": f"equity:{symbol.lower()}",
        "symbol": symbol,
        "canonical_advice_label": label,
        "advice_confidence": "medium",
        "advice_readiness": "ready",
        "technical_candidate_score": score,
        "full_advice_ready": True,
        "missing_evidence": [],
        "full_advice_blockers": [],
    }


def _output_names(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())
