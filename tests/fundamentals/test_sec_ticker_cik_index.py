from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from scripts.fundamentals import sec_ticker_cik_index as cik_index


def _write_sec_source(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(index): row for index, row in enumerate(rows)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_project_tickers(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)


def test_cik_integer_normalization_to_ten_digits() -> None:
    assert cik_index.normalize_cik(320193) == "0000320193"


def test_cik_string_normalization_to_ten_digits() -> None:
    assert cik_index.normalize_cik("789019") == "0000789019"


def test_invalid_cik_handling() -> None:
    for value in ["", "ABC", "12345678901", "0"]:
        try:
            cik_index.normalize_cik(value)
        except ValueError:
            continue
        raise AssertionError(f"invalid CIK was accepted: {value}")


def test_ticker_normalization_including_lowercase_input() -> None:
    assert cik_index.normalize_ticker(" aapl ") == "AAPL"


def test_exact_ticker_match(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    _write_sec_source(
        source_path,
        [{"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc.", "exchange": "Nasdaq"}],
    )

    index_df = cik_index.build_ticker_cik_index(source_path)
    coverage_df = cik_index.build_cik_coverage(["AAPL"], index_df)

    row = coverage_df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["cik"] == "320193"
    assert row["cik_padded"] == "0000320193"
    assert row["company_name"] == "Apple Inc."
    assert row["mapping_status"] == "CIK_MATCHED"
    assert row["review_required"] == "false"


def test_missing_ticker_preserved_as_cik_missing(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    _write_sec_source(source_path, [{"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}])

    coverage_df = cik_index.build_cik_coverage(["MISSING"], cik_index.build_ticker_cik_index(source_path))

    row = coverage_df.iloc[0]
    assert row["ticker"] == "MISSING"
    assert row["mapping_status"] == "CIK_MISSING"
    assert row["review_required"] == "true"
    assert row["cik_padded"] == ""


def test_duplicate_ticker_mapping_is_ambiguous(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    _write_sec_source(
        source_path,
        [
            {"ticker": "DUPL", "cik_str": 111111, "title": "First Duplicate"},
            {"ticker": "DUPL", "cik_str": 222222, "title": "Second Duplicate"},
        ],
    )

    coverage_df = cik_index.build_cik_coverage(["DUPL"], cik_index.build_ticker_cik_index(source_path))

    row = coverage_df.iloc[0]
    assert row["mapping_status"] == "CIK_AMBIGUOUS"
    assert row["review_required"] == "true"
    assert row["cik_padded"] == "0000111111|0000222222"


def test_one_output_row_per_requested_ticker(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    _write_sec_source(source_path, [{"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}])

    requested = ["AAPL", "MISSING", "aapl"]
    coverage_df = cik_index.build_cik_coverage(requested, cik_index.build_ticker_cik_index(source_path))

    assert len(coverage_df) == len(requested)
    assert list(coverage_df["ticker"]) == ["AAPL", "MISSING", "AAPL"]


def test_missing_mappings_are_preserved_in_file_based_coverage_output(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    project_path = tmp_path / "project_tickers.csv"
    _write_sec_source(source_path, [{"ticker": "NVDA", "cik_str": "1045810", "title": "NVIDIA CORP"}])
    _write_project_tickers(project_path, ["NVDA", "NOTSEC"])

    coverage_df = cik_index.build_coverage_from_files(
        ticker_source_path=source_path,
        project_tickers_path=project_path,
    )

    assert len(coverage_df) == 2
    missing = coverage_df.loc[coverage_df["ticker"] == "NOTSEC"].iloc[0]
    assert missing["mapping_status"] == "CIK_MISSING"
    assert missing["review_required"] == "true"


def test_generated_coverage_report_writes_only_to_provided_temp_path(tmp_path: Path) -> None:
    source_path = tmp_path / "company_tickers.json"
    project_path = tmp_path / "project_tickers.csv"
    output_path = tmp_path / "generated" / "coverage.csv"
    _write_sec_source(source_path, [{"ticker": "TWLO", "cik_str": 1447669, "title": "Twilio Inc."}])
    _write_project_tickers(project_path, ["TWLO"])

    coverage_df = cik_index.build_coverage_from_files(
        ticker_source_path=source_path,
        project_tickers_path=project_path,
        output_path=output_path,
    )

    written_df = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.is_relative_to(tmp_path)
    assert written_df.to_dict(orient="records") == coverage_df.to_dict(orient="records")


def test_no_network_call_required_on_import(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("SEC-3 import must not use network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    importlib.reload(cik_index)


def test_no_sec_download_or_pipeline_integration_is_exposed() -> None:
    assert not hasattr(cik_index, "download_companyfacts_bulk_zip")
    assert not hasattr(cik_index, "build_fundamental_metrics")
    assert not hasattr(cik_index, "build_fundamental_layer")
    assert not hasattr(cik_index, "build_fundamental_analysis")
