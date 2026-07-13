from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_engine.data.local_market_data_universe import (
    MarketDataUniverseError,
    build_data_run,
    build_universe_snapshot,
    inspect_price_history,
    run_command,
    validate_price_history_csv,
)


def test_canonical_universe_loads_and_preserves_memberships(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    _write_price_csv(tmp_path / "prices" / "NVDA.csv")

    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    nvda = _instrument(snapshot, "NVDA")
    assert "local_price_history_covered" in nvda["universe_memberships"]
    assert "explicit_supplemental_watch" in nvda["universe_memberships"]
    assert nvda["instrument_id"] == "equity:nvda"


def test_overlapping_layer_entries_are_deduplicated(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    _write_price_csv(tmp_path / "prices" / "NVDA.csv")

    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    assert [entry["symbol"] for entry in snapshot["instruments"]].count("NVDA") == 1
    assert snapshot["summary"]["duplicate_attempts_before_deduplication"] >= 1
    assert snapshot["summary"]["duplicates_after_deduplication"] == 0


def test_duplicate_canonical_symbol_fails_closed(tmp_path: Path) -> None:
    config = _write_config(tmp_path, duplicate_symbol=True)

    with pytest.raises(MarketDataUniverseError, match="ambiguous source mapping"):
        build_universe_snapshot(config, price_history_root=tmp_path / "prices")


def test_ambiguous_source_mapping_fails_closed(tmp_path: Path) -> None:
    config = _write_config(tmp_path, ambiguous_mapping=True)

    with pytest.raises(MarketDataUniverseError, match="duplicate acquisition symbol"):
        build_universe_snapshot(config, price_history_root=tmp_path / "prices")


def test_context_only_instrument_is_not_advice_eligible(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    vix = _instrument(snapshot, "VIX")
    assert vix["context_only"] is True
    assert vix["advice_eligible"] is False


def test_supplemental_missing_eval_tickers_are_included(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    for ticker in ("CLS", "CRDO", "IREN", "VRT"):
        assert _instrument(snapshot, ticker)["symbol"] == ticker


def test_symbol_overrides_support_class_share_mapping(tmp_path: Path) -> None:
    config = _write_config(tmp_path, include_class_share=True)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    brk = _instrument(snapshot, "BRK.B")
    assert brk["source_symbol"] == "BRK-B"


def test_european_suffix_and_etf_mapping_are_preserved(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    asml = _instrument(snapshot, "ASML")
    spy = _instrument(snapshot, "SPY")
    assert asml["exchange"] == "EURONEXT"
    assert spy["asset_type"] == "etf"


def test_unsupported_mapping_is_reported(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")

    rhm = _instrument(snapshot, "RHM")
    assert rhm["source_mapping_status"] == "unsupported"


def test_price_history_missing_valid_insufficient_and_forward_statuses(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    snapshot = build_universe_snapshot(config, price_history_root=tmp_path / "prices")
    nvda = _instrument(snapshot, "NVDA")

    missing = inspect_price_history(nvda, price_history_root=tmp_path / "prices")
    assert missing["snapshotstatus"] == "missing_price_history"

    _write_price_csv(tmp_path / "prices" / "NVDA.csv", rows=300, start=date(2026, 1, 1))
    valid = inspect_price_history(nvda, price_history_root=tmp_path / "prices")
    assert valid["snapshotstatus"] == "valid_current_snapshot"

    _write_price_csv(tmp_path / "prices" / "NVDA.csv", rows=10, start=date(2026, 1, 1))
    insufficient = inspect_price_history(nvda, price_history_root=tmp_path / "prices")
    assert insufficient["snapshotstatus"] == "insufficient_history"

    _write_price_csv(tmp_path / "prices" / "NVDA.csv", rows=252, start=date(2025, 1, 1))
    forward = inspect_price_history(nvda, price_history_root=tmp_path / "prices")
    assert forward["snapshotstatus"] == "insufficient_forward_data"


def test_invalid_csv_duplicate_dates_non_monotone_missing_ohlc_and_empty_fail(tmp_path: Path) -> None:
    missing_ohlc = tmp_path / "missing.csv"
    missing_ohlc.write_text("Date,Close\n2026-01-01,1\n", encoding="utf-8")
    assert validate_price_history_csv(missing_ohlc)["status"] == "validation_failed"

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("Date,Open,High,Low,Close\n2026-01-01,1,1,1,1\n2026-01-01,1,1,1,1\n", encoding="utf-8")
    assert "Duplicate" in validate_price_history_csv(duplicate)["note"]

    non_monotone = tmp_path / "non_monotone.csv"
    non_monotone.write_text("Date,Open,High,Low,Close\n2026-01-02,1,1,1,1\n2026-01-01,1,1,1,1\n", encoding="utf-8")
    assert "monotonic" in validate_price_history_csv(non_monotone)["note"]

    empty = tmp_path / "empty.csv"
    empty.write_text("Date,Open,High,Low,Close\n", encoding="utf-8")
    assert "empty" in validate_price_history_csv(empty)["note"]


def test_incremental_run_imports_missing_and_skips_valid(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    _write_price_csv(tmp_path / "prices" / "NVDA.csv", rows=300, start=date(2026, 1, 1))
    _write_price_csv(tmp_path / "import" / "CLS.csv", rows=300, start=date(2026, 1, 1))

    run, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="run",
        import_root=tmp_path / "import",
        tickers=("NVDA", "CLS"),
        skip_valid=True,
    )

    statuses = {row["symbol"]: row["snapshotstatus"] for row in run["instrument_results"]["results"]}
    assert statuses["NVDA"] == "skipped"
    assert statuses["CLS"] == "imported"


def test_report_only_does_not_import_and_force_refresh_refreshes(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    _write_price_csv(tmp_path / "import" / "CLS.csv", rows=300, start=date(2026, 1, 1))

    report, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="report",
        import_root=tmp_path / "import",
        tickers=("CLS",),
        report_only=True,
    )
    assert report["instrument_results"]["results"][0]["snapshotstatus"] == "missing_price_history"

    imported, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="import",
        import_root=tmp_path / "import",
        tickers=("CLS",),
    )
    assert imported["instrument_results"]["results"][0]["snapshotstatus"] == "imported"

    refreshed, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="refresh",
        import_root=tmp_path / "import",
        tickers=("CLS",),
        force_refresh=True,
    )
    assert refreshed["instrument_results"]["results"][0]["snapshotstatus"] == "refreshed"


def test_layer_filter_ticker_filter_limit_and_idempotent_rerun(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    first, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="run-a",
        layer="explicit_supplemental_watch",
        tickers=("CLS", "CRDO"),
        limit=1,
        report_only=True,
    )
    second, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="run-a",
        layer="explicit_supplemental_watch",
        tickers=("CLS", "CRDO"),
        limit=1,
        report_only=True,
    )
    assert first["instrument_results"] == second["instrument_results"]
    assert len(first["instrument_results"]["results"]) == 1
    assert first["instrument_results"]["results"][0]["symbol"] == "CLS"


def test_me_eval02_unresolved_readiness_and_no_advice_generation(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run, _ = build_data_run(
        universe_path=config,
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "runs",
        run_id="run",
        tickers=("CLS", "CRDO", "IREN", "VRT"),
        report_only=True,
    )

    readiness = run["unresolved_outcome_readiness"]
    assert readiness["covered_critical_tickers"] == ["CLS", "CRDO", "IREN", "VRT"]
    assert readiness["not_ready_count"] == 4
    assert run["manifest"]["baseline_guardrail"]["advice_generation_performed"] is False
    assert run["manifest"]["baseline_guardrail"]["broker_order_execution_performed"] is False


def test_command_writes_artifacts(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    stdout = io.StringIO()

    exit_code = run_command(
        [
            "--universe",
            config.as_posix(),
            "--output-root",
            (tmp_path / "prices").as_posix(),
            "--artifact-root",
            (tmp_path / "runs").as_posix(),
            "--run-id",
            "run",
            "--tickers",
            "CLS",
            "--report-only",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert (tmp_path / "runs" / "run" / "manifest.json").exists()
    assert json.loads(stdout.getvalue())["summary"]["missing"] == 1


def _instrument(snapshot: dict[str, object], symbol: str) -> dict[str, object]:
    matches = [entry for entry in snapshot["instruments"] if entry["symbol"] == symbol]
    assert len(matches) == 1
    return matches[0]


def _write_config(
    tmp_path: Path,
    *,
    duplicate_symbol: bool = False,
    ambiguous_mapping: bool = False,
    include_class_share: bool = False,
) -> Path:
    instruments = [
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "CLS", "name": "Celestica Inc."},
        {"symbol": "CRDO", "name": "Credo Technology Group"},
        {"symbol": "IREN", "name": "IREN Limited"},
        {"symbol": "VRT", "name": "Vertiv Holdings"},
        {"symbol": "ASML", "name": "ASML Holding", "exchange": "EURONEXT", "country": "NL", "currency": "EUR"},
        {"symbol": "RHM", "name": "Rheinmetall", "source_notes": "Future source mapping required."},
    ]
    if duplicate_symbol:
        instruments.append({"symbol": "NVDA", "source_symbol": "NVDA2", "name": "NVIDIA duplicate"})
    if ambiguous_mapping:
        instruments.append({"symbol": "NVDA2", "source_symbol": "NVDA", "name": "Ambiguous NVIDIA"})
    if include_class_share:
        instruments.append({"symbol": "BRK.B", "name": "Berkshire Hathaway Class B"})
    payload = {
        "schema_version": "market-engine-canonical-local-market-data-universe-config-v1",
        "universe_version": "test-universe-v1",
        "snapshot_date": "2026-07-12",
        "provenance": ["test"],
        "point_in_time_note": "test",
        "layers": [
            {
                "layer_id": "local_price_history_covered",
                "source_type": "local_price_history_directory",
                "membership": "local_price_history_covered",
            },
            {
                "layer_id": "explicit_supplemental_watch",
                "source_type": "explicit_instruments",
                "membership": "explicit_supplemental_watch",
                "instruments": instruments,
            },
            {
                "layer_id": "market_context",
                "source_type": "explicit_instruments",
                "membership": "market_context",
                "asset_type": "index",
                "context_only": True,
                "advice_eligible": False,
                "instruments": [{"symbol": "VIX", "name": "Volatility Index"}],
            },
            {
                "layer_id": "etf_context",
                "source_type": "explicit_instruments",
                "membership": "etf_context",
                "asset_type": "etf",
                "instruments": [{"symbol": "SPY", "name": "SPY"}],
            },
        ],
        "symbol_overrides": [
            {"canonical_symbol": "BRK.B", "source_symbol": "BRK-B", "reason": "test"}
        ],
    }
    path = tmp_path / "canonical_universe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_price_csv(path: Path, *, rows: int = 300, start: date = date(2026, 1, 1)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Date,Adj Close,Close,High,Low,Open,Volume"]
    for index in range(rows):
        row_date = start + timedelta(days=index)
        price = 100 + index
        lines.append(f"{row_date.isoformat()},{price},{price},{price + 1},{price - 1},{price},1000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
