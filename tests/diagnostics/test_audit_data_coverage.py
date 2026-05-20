from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.diagnostics import audit_data_coverage as audit_module


FORBIDDEN_FIELD_TERMS = {
    "ranking",
    "scoring",
    "tradeability",
    "tradeable",
    "allocation",
    "urgency",
    "conviction",
}


def _paths(tmp_path: Path) -> audit_module.AuditPaths:
    return audit_module.AuditPaths(
        portfolio_positions=tmp_path / "data" / "portfolio" / "portfolio_positions.csv",
        watchlist_active=tmp_path / "data" / "watchlist" / "watchlist_active.csv",
        scanner_ranked=tmp_path / "data" / "processed" / "scanner_ranked.csv",
        portfolio_metadata=tmp_path / "data" / "portfolio" / "portfolio_metadata.csv",
        fundamentals=tmp_path / "data" / "raw" / "fundamentals.csv",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _metadata_row(ticker: str = "AAA", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "sector": "Technology",
        "industry": "Software",
        "asset_class": "Equity",
        "currency": "USD",
        "metadata_source": "manual",
        "metadata_last_updated": "2026-05-01",
    }
    row.update(overrides)
    return row


def _fundamental_row(ticker: str = "AAA", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "as_of_date": "2026-05-01",
        "source_name": "manual",
        "source_last_updated": "2026-05-01",
        "report_period": "2026Q1",
        "currency": "USD",
        "revenue_growth_yoy": "0.10",
        "eps_growth_yoy": "0.08",
        "gross_margin": "0.55",
        "operating_margin": "0.22",
        "debt_to_equity": "0.30",
        "free_cash_flow_positive": "true",
    }
    row.update(overrides)
    return row


def _audit_explicit(tmp_path: Path, tickers: str = "AAA") -> dict[str, Any]:
    return audit_module.run_coverage_audit(
        target_mode="explicit",
        tickers=tickers,
        as_of_date="2026-05-20",
        paths=_paths(tmp_path),
    )


def _flatten_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value.keys())
        for item in value.values():
            keys |= _flatten_keys(item)
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys |= _flatten_keys(item)
        return keys
    return set()


def test_portfolio_metadata_complete_coverage(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_metadata, [_metadata_row("AAA", metadata_last_updated="2026-05-20")])

    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["metadata_complete_count"] == 1
    assert metadata["metadata_invalid_count"] == 0
    assert metadata["metadata_coverage_percentage"] == 100.0
    assert metadata["metadata_freshness_distribution"]["fresh"] == 1


def test_portfolio_metadata_updated_after_target_date_remains_complete_and_reported(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_metadata, [_metadata_row("AAA", metadata_last_updated="2026-06-01")])

    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["metadata_complete_count"] == 1
    assert metadata["metadata_invalid_count"] == 0
    assert metadata["metadata_coverage_percentage"] == 100.0
    assert metadata["metadata_last_updated_after_target_date_count"] == 1
    assert metadata["metadata_freshness_distribution"]["updated_after_target_date"] == 1


def test_malformed_metadata_last_updated_remains_invalid(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_metadata, [_metadata_row("AAA", metadata_last_updated="not-a-date")])

    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["metadata_complete_count"] == 0
    assert metadata["metadata_invalid_count"] == 1
    assert metadata["metadata_coverage_percentage"] == 0.0
    assert metadata["metadata_freshness_distribution"]["invalid"] == 1


def test_portfolio_metadata_missing_coverage(tmp_path: Path):
    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["metadata_missing_count"] == 1
    assert metadata["source_status"] == "missing"


def test_portfolio_metadata_partial_coverage(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_metadata, [_metadata_row("AAA", industry="")])

    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["metadata_partial_count"] == 1
    assert metadata["missing_industry_count"] == 1


def test_fundamentals_sufficient_coverage(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(
        paths.fundamentals,
        [_fundamental_row("AAA", as_of_date="2026-05-20", source_last_updated="2026-05-20")],
    )

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["fundamentals_sufficient_count"] == 1
    assert fundamentals["ticker_date_match_success_count"] == 1
    assert fundamentals["date_mismatch_count"] == 0
    assert fundamentals["diagnostics"][0]["date_match_status"] == "exact_ticker_date_match"
    assert fundamentals["fundamentals_coverage_percentage"] == 100.0


def test_fundamentals_partial_coverage(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("AAA", eps_growth_yoy="", operating_margin="")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["fundamentals_partial_count"] == 1
    assert fundamentals["fundamentals_sufficient_count"] == 0
    assert fundamentals["missing_eps_growth_yoy_count"] == 1
    assert fundamentals["missing_operating_margin_count"] == 1


def test_fundamentals_missing_source_rows(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("BBB")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["source_row_missing_count"] == 1
    assert fundamentals["ticker_date_match_failure_count"] == 1
    assert fundamentals["diagnostics"][0]["match_failure_reason"] == "no source row on or before opportunity date"


def test_fundamentals_date_mismatch_is_reported(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("AAA", as_of_date="2026-05-01")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["ticker_date_match_success_count"] == 1
    assert fundamentals["date_mismatch_count"] == 1
    assert fundamentals["diagnostics"][0]["source_as_of_date"] == "2026-05-01"
    assert fundamentals["diagnostics"][0]["target_date"] == "2026-05-20"
    assert fundamentals["diagnostics"][0]["date_match_status"] == "source_as_of_before_target_date"


def test_fundamentals_future_as_of_date_is_match_failure(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("AAA", as_of_date="2026-06-01")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["source_row_missing_count"] == 1
    assert fundamentals["diagnostics"][0]["match_failure_reason"] == "no source row on or before opportunity date"


def test_invalid_source_freshness_date_is_reported(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("AAA", source_last_updated="2026-06-01")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["fundamentals_invalid_count"] == 1
    assert fundamentals["diagnostics"][0]["match_failure_reason"] == "source_last_updated after opportunity date"


def test_duplicate_metadata_ticker_handling(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_metadata, [_metadata_row("AAA"), _metadata_row(" aaa ")])

    result = _audit_explicit(tmp_path)

    metadata = result["portfolio_metadata"]
    assert metadata["duplicate_metadata_ticker_count"] == 1
    assert metadata["metadata_invalid_count"] == 1


def test_duplicate_ticker_date_handling(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.fundamentals, [_fundamental_row("AAA"), _fundamental_row(" aaa ")])

    result = _audit_explicit(tmp_path)

    fundamentals = result["fundamentals"]
    assert fundamentals["duplicate_ticker_date_identity_count"] == 1
    assert fundamentals["fundamentals_invalid_count"] == 1


def test_explicit_ticker_list_target_universe(tmp_path: Path):
    result = _audit_explicit(tmp_path, tickers=" aaa , BBB ")

    assert result["target_total_tickers"] == 2
    assert result["target_total_ticker_date_rows"] == 2
    assert result["explicit_target_date_source"] == "operator_provided"


def test_explicit_mode_without_target_date_fails_safely(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --target-date"):
        audit_module.run_coverage_audit(target_mode="explicit", tickers="AAA", paths=_paths(tmp_path))


def test_cli_explicit_mode_without_target_date_reports_clear_failure(capsys):
    result = audit_module.main(["--target-mode", "explicit", "--tickers", "AAA"])

    output = capsys.readouterr().out
    assert result == 1
    assert "explicit target mode requires --target-date" in output


def test_scanner_output_target_universe(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(
        paths.scanner_ranked,
        [
            {"ticker": "AAA", "date": "2026-05-20", "grade": "A"},
            {"ticker": "BBB", "date": "2026-05-20", "grade": "C"},
        ],
    )

    result = audit_module.run_coverage_audit(target_mode="scanner", as_of_date="2026-05-20", paths=paths)

    assert result["target_total_tickers"] == 2
    assert result["target_total_ticker_date_rows"] == 2


def test_scanner_ab_target_universe(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(
        paths.scanner_ranked,
        [
            {"ticker": "AAA", "date": "2026-05-20", "grade": "A"},
            {"ticker": "BBB", "date": "2026-05-20", "grade": "B"},
            {"ticker": "CCC", "date": "2026-05-20", "grade": "C"},
        ],
    )

    result = audit_module.run_coverage_audit(target_mode="scanner-ab", as_of_date="2026-05-20", paths=paths)

    assert result["target_total_tickers"] == 2
    assert result["target_total_ticker_date_rows"] == 2


def test_portfolio_watchlist_target_universe(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.portfolio_positions, [{"ticker": "AAA", "quantity": "1", "status": "OPEN"}])
    _write_csv(paths.watchlist_active, [{"ticker": "BBB", "is_active": "TRUE"}])

    result = audit_module.run_coverage_audit(target_mode="portfolio-watchlist", as_of_date="2026-05-20", paths=paths)

    assert result["target_total_tickers"] == 2


def test_no_decision_engine_reporting_or_telegram_dependencies():
    source = inspect.getsource(audit_module)

    assert "decision_engine" not in source
    assert "scripts.reporting" not in source
    assert "scripts.telegram" not in source
    assert "build_telegram_summary" not in source


def test_no_generated_processed_artifacts_are_written(tmp_path: Path):
    paths = _paths(tmp_path)

    _audit_explicit(tmp_path)

    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "reports").exists()
    assert not paths.fundamentals.exists()


def test_no_forbidden_fields_are_emitted(tmp_path: Path):
    result = _audit_explicit(tmp_path)
    emitted_keys = {key.lower() for key in _flatten_keys(result)}

    for term in FORBIDDEN_FIELD_TERMS:
        assert all(term not in key for key in emitted_keys)


def test_deterministic_malformed_input_handling(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_csv(paths.scanner_ranked, [{"date": "2026-05-20"}])

    with pytest.raises(ValueError, match="missing required columns"):
        audit_module.run_coverage_audit(target_mode="scanner", as_of_date="2026-05-20", paths=paths)


def test_deterministic_empty_target_universe_handling(tmp_path: Path):
    with pytest.raises(ValueError, match="explicit target mode requires"):
        audit_module.run_coverage_audit(target_mode="explicit", tickers="", as_of_date="2026-05-20", paths=_paths(tmp_path))
