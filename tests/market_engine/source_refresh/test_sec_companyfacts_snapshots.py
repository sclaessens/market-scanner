from __future__ import annotations

import json

import pytest

from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.runner import run_source_intake
from market_engine.source_intake.sec_companyfacts_provider import (
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    SecCompanyFactsProvider,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    SEC_COMPANYFACTS_SOURCE_NAME,
    SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION,
    SecCompanyFactsSnapshotJsonError,
    SecCompanyFactsSnapshotMetadataError,
    SecCompanyFactsSnapshotMissingError,
    SecCompanyFactsSnapshotMismatchError,
    SecCompanyFactsSnapshotUnsupportedFormatError,
    load_latest_sec_companyfacts_raw_snapshot,
    load_sec_companyfacts_raw_snapshot,
    persist_sec_companyfacts_provider_error,
    persist_sec_companyfacts_raw_snapshot,
)


def test_raw_sec_companyfacts_payload_can_be_persisted_and_loaded(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="nvda",
        cik="1045810",
        raw_payload=_companyfacts_payload(revenue=100),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    loaded = load_sec_companyfacts_raw_snapshot(
        snapshot_path,
        expected_ticker="NVDA",
        expected_cik="0001045810",
    )

    assert snapshot_path == tmp_path / "20260615T120000Z" / "raw" / "NVDA_companyfacts.json"
    assert loaded.ticker == "NVDA"
    assert loaded.cik == "0001045810"
    assert loaded.source_name == SEC_COMPANYFACTS_SOURCE_NAME
    assert loaded.payload_format_version == SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION
    assert loaded.raw_payload == _companyfacts_payload(revenue=100)
    assert (tmp_path / "20260615T120000Z" / "snapshot_metadata.json").exists()
    assert (tmp_path / "20260615T120000Z" / "ticker_manifest.csv").exists()


def test_cached_sec_companyfacts_snapshot_loads_without_provider_call(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(revenue=100),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    def provider_call_should_not_happen(url: str):
        raise AssertionError("cached loading must not call provider")

    provider = SecCompanyFactsProvider(
        cached_snapshot_path=snapshot_path,
        fetch_json=provider_call_should_not_happen,
    )
    summary = run_source_intake(
        tickers=("NVDA",),
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.AVAILABLE
    assert result.raw_evidence_present is True
    assert result.raw_evidence_summary == "Cached SEC CompanyFacts snapshot NVDA_companyfacts CIK0001045810"
    assert result.normalized_data["revenue"] == 100


def test_latest_cached_snapshot_selection_is_deterministic(tmp_path):
    persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T110000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(revenue=90),
        fetched_at="2026-06-15T11:00:00+00:00",
    )
    persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(revenue=100),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    loaded = load_latest_sec_companyfacts_raw_snapshot(root_dir=tmp_path, ticker="NVDA")

    assert loaded.snapshot_id == "NVDA_companyfacts"
    assert loaded.raw_payload == _companyfacts_payload(revenue=100)


def test_provider_errors_are_persisted_separately_from_raw_payloads(tmp_path):
    error_path = persist_sec_companyfacts_provider_error(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="NOPE",
        cik=None,
        error_type="UnsupportedTickerError",
        error_message="no supported CIK mapping",
    )

    text = error_path.read_text(encoding="utf-8")
    assert "UnsupportedTickerError" in text
    assert "no supported CIK mapping" in text
    assert not (tmp_path / "20260615T120000Z" / "raw").exists()


def test_missing_cache_file_fails_explicitly(tmp_path):
    with pytest.raises(SecCompanyFactsSnapshotMissingError):
        load_sec_companyfacts_raw_snapshot(tmp_path / "missing.json")


def test_invalid_snapshot_json_fails_explicitly(tmp_path):
    snapshot_path = tmp_path / "bad.json"
    snapshot_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SecCompanyFactsSnapshotJsonError):
        load_sec_companyfacts_raw_snapshot(snapshot_path)


def test_missing_required_snapshot_metadata_fails_explicitly(tmp_path):
    snapshot_path = tmp_path / "missing_metadata.json"
    snapshot_path.write_text(
        json.dumps({"metadata": {"ticker": "NVDA"}, "raw_payload": _companyfacts_payload(revenue=100)}),
        encoding="utf-8",
    )

    with pytest.raises(SecCompanyFactsSnapshotMetadataError):
        load_sec_companyfacts_raw_snapshot(snapshot_path)


def test_unsupported_snapshot_format_fails_explicitly(tmp_path):
    snapshot_path = tmp_path / "unsupported_format.json"
    envelope = {
        "metadata": {
            "ticker": "NVDA",
            "cik": "0001045810",
            "source_name": SEC_COMPANYFACTS_SOURCE_NAME,
            "fetched_at": "2026-06-15T12:00:00+00:00",
            "snapshot_id": "NVDA_companyfacts",
            "payload_format_version": "unsupported",
        },
        "raw_payload": _companyfacts_payload(revenue=100),
    }
    snapshot_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(SecCompanyFactsSnapshotUnsupportedFormatError):
        load_sec_companyfacts_raw_snapshot(snapshot_path)


def test_snapshot_entity_mismatch_fails_explicitly(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(revenue=100),
    )

    with pytest.raises(SecCompanyFactsSnapshotMismatchError):
        load_sec_companyfacts_raw_snapshot(snapshot_path, expected_ticker="AMD")


def test_cached_loading_does_not_create_downstream_context_or_outputs(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        root_dir=tmp_path,
        run_id="20260615T120000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(revenue=100),
    )
    loaded = load_sec_companyfacts_raw_snapshot(snapshot_path)

    assert loaded.raw_payload == _companyfacts_payload(revenue=100)
    assert not hasattr(loaded, "observations")
    assert not hasattr(loaded, "recommendation")
    assert not hasattr(loaded, "portfolio")
    assert not hasattr(loaded, "delivery")
    assert not hasattr(loaded, "telegram")
    assert not hasattr(loaded, "decision_engine")


def test_source_refresh_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _companyfacts_payload(*, revenue: int) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(revenue, "2025-12-31")]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(20, "2025-12-31")]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(30, "2025-12-31")]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(5, "2025-12-31")]}
                },
            }
        }
    }


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }
