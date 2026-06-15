from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from market_engine.source_intake import manual_smoke
from market_engine.source_intake.fake_provider import FakeProviderScenario, FakeSourceProvider
from market_engine.source_intake.runner import run_source_intake
from market_engine.source_intake.smoke_artifacts import (
    SEC_COMPANYFACTS_SMOKE_ROOT,
    write_sec_companyfacts_smoke_artifacts,
)


REQUIRED_FIELDS = ("revenue", "net_income", "operating_cash_flow", "capital_expenditures")


def test_manual_smoke_artifact_writing_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manual_smoke.main([])

    assert not Path("data").exists()


def test_artifact_writer_requires_market_engine_sec_path(tmp_path):
    summary = _summary_with_missing_numeric()

    with pytest.raises(ValueError):
        write_sec_companyfacts_smoke_artifacts(
            summary=summary,
            tickers=["NVDA"],
            max_tickers=1,
            run_id="test-run",
            artifact_dir=tmp_path / "reports" / "bad",
        )


def test_artifact_writer_creates_expected_files_under_market_engine_path(tmp_path):
    summary = _summary_with_missing_numeric()
    artifact_dir = tmp_path / SEC_COMPANYFACTS_SMOKE_ROOT / "test-run"

    written_dir = write_sec_companyfacts_smoke_artifacts(
        summary=summary,
        tickers=["NVDA"],
        max_tickers=1,
        run_id="test-run",
        artifact_dir=artifact_dir,
    )

    assert written_dir == artifact_dir
    assert sorted(path.name for path in written_dir.iterdir()) == [
        "coverage_summary.csv",
        "missing_fields.csv",
        "provider_errors.csv",
        "smoke_metadata.json",
        "ticker_results.csv",
    ]


def test_artifact_writer_does_not_overwrite_existing_directory(tmp_path):
    summary = _summary_with_missing_numeric()
    artifact_dir = tmp_path / SEC_COMPANYFACTS_SMOKE_ROOT / "test-run"
    artifact_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        write_sec_companyfacts_smoke_artifacts(
            summary=summary,
            tickers=["NVDA"],
            max_tickers=1,
            run_id="test-run",
            artifact_dir=artifact_dir,
        )


def test_coverage_summary_csv_has_expected_columns(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)

    assert _headers(artifact_dir / "coverage_summary.csv") == [
        "provider_name",
        "ticker_count",
        "readiness_status",
        "count",
        "disclaimer",
    ]


def test_ticker_results_csv_has_expected_columns(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)

    assert _headers(artifact_dir / "ticker_results.csv") == [
        "ticker",
        "provider_name",
        "readiness_status",
        "available_fields",
        "missing_fields",
        "raw_evidence_present",
        "raw_evidence_summary",
        "provider_error_type",
        "provider_error_message",
        "intake_success",
    ]


def test_missing_fields_csv_has_expected_columns(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)

    assert _headers(artifact_dir / "missing_fields.csv") == [
        "field_name",
        "missing_count",
        "disclaimer",
    ]


def test_provider_errors_csv_has_expected_columns(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)

    assert _headers(artifact_dir / "provider_errors.csv") == [
        "ticker",
        "provider_error_type",
        "provider_error_message",
        "disclaimer",
    ]


def test_smoke_metadata_json_has_expected_fields(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)
    metadata = json.loads((artifact_dir / "smoke_metadata.json").read_text(encoding="utf-8"))

    assert metadata["provider_name"] == "fake-sec-provider"
    assert metadata["run_id"] == "test-run"
    assert metadata["ticker_count"] == 1
    assert metadata["max_tickers"] == 1
    assert metadata["required_fields"] == list(REQUIRED_FIELDS)
    assert metadata["disclaimer"] == "source coverage evidence only; not analysis; not source truth"


def test_artifact_content_does_not_include_forbidden_authority_fields(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)
    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in artifact_dir.iterdir())
    forbidden_terms = (
        "BUY",
        "SELL",
        "HOLD",
        "recommendation",
        "allocation",
        "ranking",
        "score",
        "conviction",
        "urgency",
        "tradeability",
        "position sizing",
        "execution",
    )

    assert all(term not in combined_text for term in forbidden_terms)


def test_artifact_preserves_missing_numeric_values_without_zero_fill(tmp_path):
    artifact_dir = _write_test_artifacts(tmp_path)
    ticker_rows = list(csv.DictReader((artifact_dir / "ticker_results.csv").open(encoding="utf-8")))

    assert ticker_rows[0]["missing_fields"] == "operating_cash_flow"
    assert ticker_rows[0]["available_fields"] == "revenue|net_income|capital_expenditures"


def test_artifact_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _summary_with_missing_numeric():
    return run_source_intake(
        tickers=["NVDA"],
        provider=FakeSourceProvider(
            name_value="fake-sec-provider",
            scenarios={
                "NVDA": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "net_income": 20,
                        "operating_cash_flow": None,
                        "capital_expenditures": 0,
                    },
                    raw_evidence={"fixture": "NVDA"},
                    raw_evidence_summary="test fixture",
                )
            },
        ),
        required_fields=REQUIRED_FIELDS,
    )


def _write_test_artifacts(tmp_path) -> Path:
    artifact_dir = tmp_path / SEC_COMPANYFACTS_SMOKE_ROOT / "test-run"
    return write_sec_companyfacts_smoke_artifacts(
        summary=_summary_with_missing_numeric(),
        tickers=["NVDA"],
        max_tickers=1,
        run_id="test-run",
        artifact_dir=artifact_dir,
    )


def _headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return next(csv.reader(handle))
