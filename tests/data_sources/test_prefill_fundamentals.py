from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.data_sources import prefill_fundamentals


FORBIDDEN_TERMS = {
    "allocation",
    "ranking",
    "score",
    "tradeable",
    "urgency",
    "conviction",
    "final_action",
}


def _fundamental_row(ticker: str = " aaa ", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "as_of_date": "2026-05-01",
        "source_name": "local_export",
        "source_last_updated": "2026-05-01",
        "report_period": "2026Q1",
        "currency": "USD",
        "revenue_growth_yoy": "0.12",
        "eps_growth_yoy": "0.08",
        "gross_margin": "0.55",
        "operating_margin": "0.22",
        "debt_to_equity": "0.40",
        "free_cash_flow_positive": "true",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_successful_fundamentals_prefill_writes_schema_and_normalizes_tickers(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    output_path = tmp_path / "data" / "raw" / "fundamentals.csv"
    _write_input(input_path, [_fundamental_row()])

    result = prefill_fundamentals.prepare_fundamentals_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=True,
    )

    written = pd.read_csv(output_path)
    assert list(written.columns) == prefill_fundamentals.OUTPUT_COLUMNS
    assert written.loc[0, "ticker"] == "AAA"
    assert result.audit.validation_status == "VALIDATED"
    assert result.audit.written_row_count == 1


def test_fundamentals_dry_run_does_not_write_output(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    output_path = tmp_path / "data" / "raw" / "fundamentals.csv"
    _write_input(input_path, [_fundamental_row()])

    result = prefill_fundamentals.prepare_fundamentals_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=False,
    )

    assert not output_path.exists()
    assert result.audit.dry_run is True
    assert result.audit.written_row_count == 0


def test_fundamentals_missing_required_columns_fail_fast(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    row = _fundamental_row()
    row.pop("source_name")
    _write_input(input_path, [row])

    with pytest.raises(ValueError, match="missing required columns"):
        prefill_fundamentals.prepare_fundamentals_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_fundamentals_duplicate_row_identity_fails_fast(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    _write_input(input_path, [_fundamental_row("AAA"), _fundamental_row(" aaa ")])

    with pytest.raises(ValueError, match="duplicate row identity"):
        prefill_fundamentals.prepare_fundamentals_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_fundamentals_malformed_dates_fail_fast(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    _write_input(input_path, [_fundamental_row(as_of_date="2026/05/01")])

    with pytest.raises(ValueError, match="as_of_date must be an ISO date"):
        prefill_fundamentals.prepare_fundamentals_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_fundamentals_partial_and_stale_records_are_reported(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    _write_input(
        input_path,
        [
            _fundamental_row("AAA", eps_growth_yoy=""),
            _fundamental_row("BBB", as_of_date="2025-01-01", source_last_updated="2025-01-01"),
        ],
    )

    result = prefill_fundamentals.prepare_fundamentals_prefill(input_path=input_path, as_of_date="2026-05-20")

    assert result.audit.validation_status == "VALIDATED_WITH_DIAGNOSTICS"
    assert result.audit.partial_row_count == 1
    assert result.audit.stale_row_count == 1


def test_fundamentals_invalid_records_are_not_written(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    output_path = tmp_path / "data" / "raw" / "fundamentals.csv"
    _write_input(input_path, [_fundamental_row(gross_margin="not-a-number")])

    result = prefill_fundamentals.prepare_fundamentals_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=False,
    )

    assert result.audit.validation_status == "FAILED"
    assert result.audit.invalid_row_count == 1
    with pytest.raises(ValueError, match="invalid records"):
        prefill_fundamentals.prepare_fundamentals_prefill(
            input_path=input_path,
            output_path=output_path,
            as_of_date="2026-05-20",
            write=True,
        )
    assert not output_path.exists()


def test_fundamentals_existing_artifact_requires_overwrite_flag(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    output_path = tmp_path / "data" / "raw" / "fundamentals.csv"
    _write_input(input_path, [_fundamental_row()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("ticker\nOLD\n")

    with pytest.raises(FileExistsError, match="allow-overwrite"):
        prefill_fundamentals.prepare_fundamentals_prefill(
            input_path=input_path,
            output_path=output_path,
            as_of_date="2026-05-20",
            write=True,
        )

    assert output_path.read_text() == "ticker\nOLD\n"


def test_fundamentals_output_has_no_forbidden_semantic_columns(tmp_path: Path):
    input_path = tmp_path / "fundamentals_export.csv"
    _write_input(input_path, [_fundamental_row()])

    result = prefill_fundamentals.prepare_fundamentals_prefill(input_path=input_path, as_of_date="2026-05-20")
    columns = " ".join(result.output_df.columns).lower()

    for term in FORBIDDEN_TERMS:
        assert term not in columns
