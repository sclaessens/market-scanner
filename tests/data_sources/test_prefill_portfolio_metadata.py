from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.data_sources import prefill_portfolio_metadata


FORBIDDEN_TERMS = {
    "allocation",
    "ranking",
    "score",
    "tradeable",
    "urgency",
    "conviction",
    "final_action",
}


def _metadata_row(ticker: str = " aaa ", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "sector": "Technology",
        "industry": "Software",
        "asset_class": "Equity",
        "currency": "USD",
        "metadata_source": "local_export",
        "metadata_last_updated": "2026-05-01",
        "sector_taxonomy": "manual",
        "industry_group": "",
        "country": "US",
        "region": "North America",
        "exchange": "NYSE",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_successful_portfolio_metadata_prefill_writes_schema_and_normalizes_tickers(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    output_path = tmp_path / "data" / "portfolio" / "portfolio_metadata.csv"
    _write_input(input_path, [_metadata_row()])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=True,
    )

    written = pd.read_csv(output_path)
    assert list(written.columns) == prefill_portfolio_metadata.OUTPUT_COLUMNS
    assert written.loc[0, "ticker"] == "AAA"
    assert result.audit.validation_status == "VALIDATED"
    assert result.audit.written_row_count == 1


def test_portfolio_metadata_dry_run_does_not_write_output(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    output_path = tmp_path / "data" / "portfolio" / "portfolio_metadata.csv"
    _write_input(input_path, [_metadata_row()])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=False,
    )

    assert not output_path.exists()
    assert result.audit.dry_run is True
    assert result.audit.written_row_count == 0


def test_portfolio_metadata_missing_required_columns_fail_fast(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    row = _metadata_row()
    row.pop("metadata_source")
    _write_input(input_path, [row])

    with pytest.raises(ValueError, match="missing required columns"):
        prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_portfolio_metadata_duplicate_tickers_fail_fast(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row("AAA"), _metadata_row(" aaa ")])

    with pytest.raises(ValueError, match="duplicate row identity"):
        prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_portfolio_metadata_missing_required_values_are_reported_as_partial(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row(industry="")])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")

    assert result.audit.validation_status == "VALIDATED_WITH_DIAGNOSTICS"
    assert result.audit.partial_row_count == 1


def test_portfolio_metadata_stale_records_are_reported(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row(metadata_last_updated="2025-01-01")])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")

    assert result.audit.validation_status == "VALIDATED_WITH_DIAGNOSTICS"
    assert result.audit.stale_row_count == 1


def test_portfolio_metadata_invalid_asset_class_is_not_written(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    output_path = tmp_path / "data" / "portfolio" / "portfolio_metadata.csv"
    _write_input(input_path, [_metadata_row(asset_class="Crypto")])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(
        input_path=input_path,
        output_path=output_path,
        as_of_date="2026-05-20",
        write=False,
    )

    assert result.audit.validation_status == "FAILED"
    assert result.audit.invalid_row_count == 1
    with pytest.raises(ValueError, match="invalid records"):
        prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(
            input_path=input_path,
            output_path=output_path,
            as_of_date="2026-05-20",
            write=True,
        )
    assert not output_path.exists()


def test_portfolio_metadata_malformed_date_fails_fast(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row(metadata_last_updated="2026/05/01")])

    with pytest.raises(ValueError, match="metadata_last_updated must be an ISO date"):
        prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")


def test_portfolio_metadata_only_ticker_does_not_imply_universe_expansion(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row("AAA"), _metadata_row("METADATAONLY")])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")

    assert result.output_df["ticker"].tolist() == ["AAA", "METADATAONLY"]
    assert result.audit.refresh_mode == "provider_assisted_prefill"


def test_portfolio_metadata_existing_artifact_requires_overwrite_flag(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    output_path = tmp_path / "data" / "portfolio" / "portfolio_metadata.csv"
    _write_input(input_path, [_metadata_row()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("ticker\nOLD\n")

    with pytest.raises(FileExistsError, match="allow-overwrite"):
        prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(
            input_path=input_path,
            output_path=output_path,
            as_of_date="2026-05-20",
            write=True,
        )

    assert output_path.read_text() == "ticker\nOLD\n"


def test_portfolio_metadata_output_has_no_forbidden_semantic_columns(tmp_path: Path):
    input_path = tmp_path / "metadata_export.csv"
    _write_input(input_path, [_metadata_row()])

    result = prefill_portfolio_metadata.prepare_portfolio_metadata_prefill(input_path=input_path, as_of_date="2026-05-20")
    columns = " ".join(result.output_df.columns).lower()

    for term in FORBIDDEN_TERMS:
        assert term not in columns
