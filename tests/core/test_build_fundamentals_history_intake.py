from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.core.build_fundamentals_history_intake import (
    REQUIRED_COLUMNS,
    validate_fundamentals_history,
)


def _valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "fiscal_period": "FY",
        "period_end_date": "2025-06-30",
        "report_date": "2025-07-30",
        "currency": "USD",
        "revenue": "245122000000",
        "gross_profit": "171008000000",
        "operating_income": "109433000000",
        "net_income": "88136000000",
        "diluted_eps": "11.80",
        "total_debt": "67127000000",
        "total_equity": "268477000000",
        "free_cash_flow": "74071000000",
        "source_name": "company filing",
        "source_reference": "FY2025 Form 10-K",
        "source_freshness_date": "2026-05-27",
        "extraction_date": "2026-05-27",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_valid_minimal_raw_history_fixture_passes(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row()])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "VALID"
    assert result["row_count"] == 1
    assert result["missing_required_columns"] == []
    assert result["forbidden_columns"] == []
    assert result["duplicate_key_count"] == 0
    assert result["issue_count"] == 0


def test_missing_required_column_fails_fast(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    row = _valid_row()
    row.pop("source_reference")
    _write_fixture(input_path, [row])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["missing_required_columns"] == ["source_reference"]
    assert result["forbidden_columns"] == []
    assert result["duplicate_key_count"] == 0
    assert result["issue_count"] == 1


def test_duplicate_ticker_year_period_fails_fast(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(
        input_path,
        [
            _valid_row(),
            _valid_row(revenue="250000000000"),
        ],
    )

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["duplicate_key_count"] == 2
    assert result["issue_count"] == 2
    assert result["invalid_date_columns"] == {}
    assert result["invalid_numeric_columns"] == {}


def test_forbidden_semantic_column_fails_fast(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    row = _valid_row()
    row["tradeability"] = "high"
    _write_fixture(input_path, [row])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["forbidden_columns"] == ["tradeability"]
    assert result["duplicate_key_count"] == 0


def test_invalid_dates_fail_when_present(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row(report_date="not-a-date", extraction_date="2026-13-40")])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["invalid_date_columns"] == {"report_date": [2], "extraction_date": [2]}


def test_invalid_numeric_fields_fail(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row(revenue="not-a-number", total_debt="unknown")])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["invalid_numeric_columns"] == {"revenue": [2], "total_debt": [2]}


def test_missing_numeric_values_are_allowed(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row(revenue="", free_cash_flow="")])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "VALID"
    assert result["invalid_numeric_columns"] == {}


def test_zero_and_negative_numeric_values_are_accepted(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row(revenue="0", net_income="-125000000", free_cash_flow="-1.5")])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "VALID"
    assert result["invalid_numeric_columns"] == {}


def test_missing_required_identity_and_source_fields_fail(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(
        input_path,
        [
            _valid_row(
                ticker="",
                fiscal_year="",
                fiscal_period="",
                currency="",
                source_name="",
                source_reference="",
            )
        ],
    )

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["missing_required_value_columns"] == {
        "ticker": [2],
        "fiscal_year": [2],
        "fiscal_period": [2],
        "currency": [2],
        "source_name": [2],
        "source_reference": [2],
    }


def test_invalid_fiscal_year_and_period_fail(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row(fiscal_year="twenty-five", fiscal_period="H1")])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "INVALID"
    assert result["invalid_fiscal_year_rows"] == [2]
    assert result["invalid_fiscal_period_rows"] == [2]


def test_no_category_or_metric_outputs_are_required_or_calculated(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_fixture(input_path, [_valid_row()])

    result = validate_fundamentals_history(input_path)

    assert result["status"] == "VALID"
    assert "category" not in result
    assert "gross_margin" not in result
    assert set(REQUIRED_COLUMNS).isdisjoint(result.keys())
