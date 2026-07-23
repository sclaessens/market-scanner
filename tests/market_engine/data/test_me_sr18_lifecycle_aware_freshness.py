from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_engine.data import scheduled_canonical_price_refresh as scheduled
from market_engine.data.instrument_lifecycle import (
    DEFAULT_LIFECYCLE_REGISTRY,
    LIFECYCLE_SCHEMA_VERSION,
    InstrumentLifecycleError,
    apply_lifecycle_registry,
    load_lifecycle_registry,
    record_provenance_checksum,
)


RUN_AT = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
SOURCE_SHA = "8" * 40


def test_governed_canary_records_reconcile_to_official_dates() -> None:
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    by_ticker = {row["ticker"]: row for row in registry["records"]}

    assert set(by_ticker) == {"BLD", "JHG", "GTLS", "FDXF", "HONA", "Q", "SOLS"}
    assert {
        ticker: (
            by_ticker[ticker]["delisting_end_date"],
            by_ticker[ticker]["status_effective_date"],
        )
        for ticker in ("BLD", "JHG", "GTLS")
    } == {
        "BLD": ("2026-06-30", "2026-07-01"),
        "JHG": ("2026-06-30", "2026-07-01"),
        "GTLS": ("2026-07-16", "2026-07-17"),
    }
    assert {
        ticker: (
            by_ticker[ticker]["listing_start_date"],
            by_ticker[ticker]["regular_way_listing_date"],
        )
        for ticker in ("FDXF", "HONA", "Q", "SOLS")
    } == {
        "FDXF": ("2026-05-27", "2026-06-01"),
        "HONA": ("2026-06-15", "2026-06-29"),
        "Q": ("2025-10-27", "2025-11-03"),
        "SOLS": ("2025-10-20", "2025-10-30"),
    }
    assert all(
        evidence["source_url"].startswith("https://")
        for record in registry["records"]
        for evidence in record["evidence"]
    )


def test_repository_universe_becomes_949_active_and_retains_three_inactive() -> None:
    universe = scheduled.load_authoritative_universe(
        scheduled.DEFAULT_UNIVERSE_SNAPSHOT
    )
    governed = apply_lifecycle_registry(
        universe["instruments"],
        load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY),
        as_of=date(2026, 7, 23),
    )

    assert governed["active_universe_size"] == 949
    assert governed["inactive_retained_instrument_count"] == 3
    assert {
        row["symbol"] for row in governed["inactive_instruments"]
    } == {"BLD", "JHG", "GTLS"}


def test_inactive_effective_date_is_not_applied_early_and_checksum_changes() -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    registry = _normalized_registry([record])
    before = apply_lifecycle_registry(
        [instrument],
        registry,
        as_of=date(2026, 7, 14),
    )
    after = apply_lifecycle_registry(
        [instrument],
        registry,
        as_of=date(2026, 7, 15),
    )

    assert before["active_instruments"][0]["symbol"] == "OLD"
    assert after["inactive_instruments"][0]["symbol"] == "OLD"
    assert before["active_universe_checksum"] != after["active_universe_checksum"]


def test_future_listing_is_pending_and_not_refreshable(tmp_path: Path) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-20",
        listing_start_date="2026-07-20",
        regular_way_listing_date="2026-07-22",
    )
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("pending instrument must not call provider")
        ),
    )

    row = report["tickers"][0]
    assert row["lifecycle_status"] == "pending"
    assert row["freshness_status"] == "not_expected"
    assert row["reason_code"] == "PRE_LISTING_NOT_EXPECTED"
    assert report["run_status"] == "completed"


def test_inactive_history_is_retained_and_recent_listing_is_limited(
    tmp_path: Path,
) -> None:
    old = _instrument("OLD")
    new = _instrument("NEW")
    plain = _instrument("PLAIN")
    records = [
        _record(
            old,
            lifecycle_status="inactive",
            status_effective_date="2026-07-11",
            delisting_end_date="2026-07-10",
        ),
        _record(
            new,
            lifecycle_status="active",
            status_effective_date="2026-07-05",
            listing_start_date="2026-07-05",
            regular_way_listing_date="2026-07-06",
        ),
    ]
    fixture = _fixture(
        tmp_path,
        [old, new, plain],
        records,
        histories={
            "OLD": ("2026-07-01", "2026-07-10"),
            "NEW": ("2026-07-05", "2026-07-14"),
            "PLAIN": ("2025-11-05", "2026-07-14"),
        },
    )
    old_bytes = (fixture["published"] / "data/processed/OLD.csv").read_bytes()
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("all active histories are current")
        ),
    )
    rows = {row["ticker"]: row for row in report["tickers"]}

    assert rows["OLD"]["freshness_status"] == "not_expected"
    assert rows["OLD"]["history_coverage_status"] == "retained_inactive"
    assert rows["NEW"]["freshness_status"] == "already_current"
    assert rows["NEW"]["history_coverage_status"] == "limited_history"
    assert rows["PLAIN"]["history_coverage_status"] == "sufficient"
    assert report["run_status"] == "completed"
    assert (
        fixture["stage"] / "data/processed/OLD.csv"
    ).read_bytes() == old_bytes

    validation = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
    )
    assert validation["validated"] is True

    calls: list[dict[str, Any]] = []
    consumed = scheduled.run_validated_analysis(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
        analysis_runner=lambda **kwargs: calls.append(kwargs),
        analysis_kwargs={},
    )
    assert consumed["analysis_executed"] is True
    analysed = calls[0]["universe_snapshot"]["instruments"]
    assert {row["symbol"] for row in analysed} == {"NEW", "PLAIN"}
    assert next(
        row for row in analysed if row["symbol"] == "NEW"
    )["history_coverage_status"] == "limited_history"


def test_unexplained_short_history_remains_degraded(tmp_path: Path) -> None:
    instrument = _instrument("PLAIN")
    fixture = _fixture(
        tmp_path,
        [instrument],
        [],
        histories={"PLAIN": ("2026-07-05", "2026-07-14")},
    )
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )

    assert report["tickers"][0]["freshness_status"] == "already_current"
    assert (
        report["tickers"][0]["history_coverage_status"]
        == "insufficient_unexplained"
    )
    assert report["run_status"] == "degraded"


def test_listing_start_after_first_observation_fails_closed(
    tmp_path: Path,
) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-10",
        listing_start_date="2026-07-10",
        regular_way_listing_date="2026-07-11",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"NEW": ("2026-07-05", "2026-07-14")},
    )
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )

    assert report["tickers"][0]["freshness_status"] == "failed"
    assert (
        report["tickers"][0]["reason_code"]
        == "LISTING_START_AFTER_FIRST_OBSERVATION"
    )
    assert report["run_status"] == "degraded"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(lifecycle_status="unknown"),
        lambda row: row.update(provenance_checksum=None),
        lambda row: row.update(
            status_effective_date="2026-07-10",
            delisting_end_date="2026-07-10",
        ),
    ],
)
def test_malformed_lifecycle_records_are_blocked(
    tmp_path: Path,
    mutation: Any,
) -> None:
    instrument = _instrument("OLD")
    row = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    mutation(row)
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "records": [row],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InstrumentLifecycleError):
        load_lifecycle_registry(path)


def test_lifecycle_and_manifest_tampering_are_detected(tmp_path: Path) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-05",
        listing_start_date="2026-07-05",
        regular_way_listing_date="2026-07-06",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"NEW": ("2026-07-05", "2026-07-14")},
    )
    _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["records"][0]["listing_start_date"] = "2026-07-06"
    fixture["registry"].write_text(json.dumps(registry), encoding="utf-8")
    blocked_registry = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
    )
    assert "AUTHORITATIVE_UNIVERSE_INVALID" in blocked_registry["reason_codes"]

    fixture["registry"].write_text(
        json.dumps(
            {
                "schema_version": "unknown-lifecycle-v99",
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    unknown = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
    )
    assert "AUTHORITATIVE_UNIVERSE_INVALID" in unknown["reason_codes"]


def test_unknown_manifest_schema_is_blocked(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    fixture = _fixture(
        tmp_path,
        [instrument],
        [],
        histories={"AAA": ("2025-11-05", "2026-07-14")},
    )
    _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )
    path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unknown-v99"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
    )
    assert "PUBLISHED_MANIFEST_SCHEMA_MISMATCH" in result["reason_codes"]


def test_identical_v2_input_requires_no_empty_publication_commit(
    tmp_path: Path,
) -> None:
    instrument = _instrument("AAA")
    fixture = _fixture(
        tmp_path,
        [instrument],
        [],
        histories={"AAA": ("2025-11-05", "2026-07-14")},
    )
    _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )
    second_stage = tmp_path / "second-stage"
    second = scheduled.run_scheduled_refresh(
        run_id="me-sr18-test-repeat-20260715T100000Z",
        source_main_sha=SOURCE_SHA,
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        published_root=fixture["stage"],
        staging_root=second_stage,
        report_output=tmp_path / "second-report.json",
        run_at=RUN_AT,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )

    assert second["publication"]["publication_required"] is False
    assert second["publication"]["manifest_change_required"] is False
    assert second["publication"]["empty_commit_required"] is False
    assert not (second_stage / scheduled.LATEST_MANIFEST).exists()


def _instrument(symbol: str) -> dict[str, Any]:
    return {
        "active": True,
        "analysis_eligible": True,
        "asset_type": "equity",
        "country": "US",
        "exchange": "NYSE",
        "instrument_id": f"equity:{symbol.lower()}",
        "source_mapping_status": "mapped",
        "source_symbol": symbol,
        "symbol": symbol,
        "universe_memberships": ["test"],
    }


def _record(
    instrument: dict[str, Any],
    *,
    lifecycle_status: str,
    status_effective_date: str,
    listing_start_date: str | None = None,
    regular_way_listing_date: str | None = None,
    delisting_end_date: str | None = None,
) -> dict[str, Any]:
    row = {
        "corporate_action_type": (
            "spin_off_listing"
            if lifecycle_status == "active"
            else "cash_acquisition"
        ),
        "delisting_end_date": delisting_end_date,
        "evidence": [
            {
                "evidence_retrieved_at": "2026-07-15T09:00:00Z",
                "source_authority": "sec",
                "source_publication_date": "2026-07-01",
                "source_type": "form_8_k",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1/test.htm"
                ),
            }
        ],
        "exchange": "NYSE",
        "instrument_id": instrument["instrument_id"],
        "issuer_name": f"{instrument['symbol']} Corporation",
        "lifecycle_reason": (
            "active_recent_listing"
            if lifecycle_status == "active"
            else "inactive_after_completed_corporate_action"
        ),
        "lifecycle_status": lifecycle_status,
        "listing_start_date": listing_start_date,
        "provenance_checksum": None,
        "regular_way_listing_date": regular_way_listing_date,
        "status_effective_date": status_effective_date,
        "successor_or_acquirer": (
            None
            if lifecycle_status == "active"
            else {"name": "Acquirer Corporation", "ticker": "BUYR"}
        ),
        "ticker": instrument["symbol"],
    }
    row["provenance_checksum"] = record_provenance_checksum(row)
    return row


def _normalized_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "records": records,
    }
    return {
        **payload,
        "registry_checksum": scheduled._canonical_checksum(payload),
        "records_by_instrument_id": {
            row["instrument_id"]: row for row in records
        },
    }


def _fixture(
    root: Path,
    instruments: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    histories: dict[str, tuple[str, str]],
) -> dict[str, Path]:
    published = root / "published"
    for symbol, (start, end) in histories.items():
        _write_history(
            published / "data/processed" / f"{symbol}.csv",
            start=start,
            end=end,
        )
    universe = root / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "schema_version": scheduled.UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
                "universe_version": "me-sr18-test-universe-v1",
                "instruments": instruments,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry = root / "lifecycle.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "records": records,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "published": published,
        "stage": root / "stage",
        "report": root / "report.json",
        "universe": universe,
        "registry": registry,
    }


def _run(fixture: dict[str, Path], *, provider: Any) -> dict[str, Any]:
    return scheduled.run_scheduled_refresh(
        run_id="me-sr18-test-20260715T100000Z",
        source_main_sha=SOURCE_SHA,
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        published_root=fixture["published"],
        staging_root=fixture["stage"],
        report_output=fixture["report"],
        run_at=RUN_AT,
        provider=provider,
        sleeper=lambda _seconds: None,
    )


def _write_history(path: Path, *, start: str, end: str) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    days = (end_date - start_date).days
    dates = [
        start_date + timedelta(days=index)
        for index in range(days + 1)
    ]
    if len(dates) >= 252:
        dates = dates[-252:]
    frame = pd.DataFrame(
        [
            {
                "Date": day.isoformat(),
                "Adj Close": 100 + index,
                "Close": 100 + index,
                "High": 101 + index,
                "Low": 99 + index,
                "Open": 99.5 + index,
                "Volume": 1000,
            }
            for index, day in enumerate(dates)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
