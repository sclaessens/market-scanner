from __future__ import annotations

import copy
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pytest

from market_engine.data import scheduled_canonical_price_refresh as scheduled
from market_engine.data import observation_receipts as receipt_contract
from market_engine.data.instrument_lifecycle import (
    DEFAULT_LIFECYCLE_REGISTRY,
    LEGACY_LIFECYCLE_SCHEMA_VERSION,
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

    assert set(by_ticker) == {
        "BLD",
        "JHG",
        "GTLS",
        "FDXF",
        "HONA",
        "Q",
        "SOLS",
        "EA",
        "NSA",
        "TMHC",
    }
    assert {
        ticker: (
            by_ticker[ticker]["delisting_end_date"],
            by_ticker[ticker]["status_effective_date"],
        )
        for ticker in ("BLD", "JHG", "GTLS", "EA", "NSA", "TMHC")
    } == {
        "BLD": ("2026-06-30", "2026-07-01"),
        "JHG": ("2026-06-30", "2026-07-01"),
        "GTLS": ("2026-07-16", "2026-07-17"),
        "EA": ("2026-08-04", "2026-08-05"),
        "NSA": ("2026-07-21", "2026-07-22"),
        "TMHC": ("2026-07-24", "2026-07-25"),
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
        as_of=date(2026, 7, 24),
    )

    assert governed["active_universe_size"] == 948
    assert governed["inactive_retained_instrument_count"] == 4
    assert {
        row["symbol"] for row in governed["inactive_instruments"]
    } == {"BLD", "JHG", "GTLS", "NSA"}


def test_tmhc_becomes_inactive_after_proven_final_regular_way_session() -> None:
    universe = scheduled.load_authoritative_universe(
        scheduled.DEFAULT_UNIVERSE_SNAPSHOT
    )
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    on_final_session = apply_lifecycle_registry(
        universe["instruments"],
        registry,
        as_of=date(2026, 7, 24),
    )
    after_final_session = apply_lifecycle_registry(
        universe["instruments"],
        registry,
        as_of=date(2026, 7, 25),
    )

    assert "TMHC" in {
        row["symbol"] for row in on_final_session["active_instruments"]
    }
    tmhc = next(
        row
        for row in after_final_session["inactive_instruments"]
        if row["symbol"] == "TMHC"
    )
    assert tmhc["last_trading_session"] == "2026-07-24"
    assert tmhc["canonical_ohlcv_last_observed_session"] == "2026-07-23"
    assert (
        tmhc["terminal_session_daily_ohlcv_status"]
        == "no_valid_daily_ohlcv_bar_from_provider_as_of"
    )
    assert tmhc["observation_evidence"]["provider_identity"] == (
        "Yahoo Finance via yfinance"
    )
    assert tmhc["transaction_closing_date"] == "2026-07-24"
    assert tmhc["lifecycle_status_effective_date"] == "2026-07-25"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(observation_evidence=None),
        lambda row: row["observation_evidence"].update(
            relevant_session="2026-07-23"
        ),
    ],
    ids=["missing", "contradictory"],
)
def test_tmhc_observation_provenance_fails_closed(
    tmp_path: Path,
    mutation: Any,
) -> None:
    registry = json.loads(
        Path(DEFAULT_LIFECYCLE_REGISTRY).read_text(encoding="utf-8")
    )
    record = next(row for row in registry["records"] if row["ticker"] == "TMHC")
    mutation(record)
    record["provenance_checksum"] = record_provenance_checksum(record)
    path = tmp_path / "tmhc-observation.json"
    path.write_text(
        json.dumps({"schema_version": LIFECYCLE_SCHEMA_VERSION, "records": [record]}),
        encoding="utf-8",
    )

    with pytest.raises(InstrumentLifecycleError, match="observation evidence"):
        load_lifecycle_registry(path)


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
    assert report["run_status"] == "completed", report["tickers"]


def test_scheduled_listing_requires_completion_before_later_active_projection(
    tmp_path: Path,
) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-20",
        listing_start_date="2026-07-20",
        regular_way_listing_date="2026-07-22",
    )
    registry_path = _write_registry(tmp_path, [record])
    registry = load_lifecycle_registry(registry_path)
    evidence = registry["records"][0]["evidence"][0]

    assert evidence["source_publication_date"] < "2026-07-22"
    assert evidence["transition_support"] == ["listing_schedule"]

    pending = apply_lifecycle_registry(
        [instrument],
        registry,
        as_of=date(2026, 7, 15),
    )
    assert pending["pending_instruments"][0]["symbol"] == "NEW"

    with pytest.raises(
        InstrumentLifecycleError,
        match="completion evidence is required before active projection",
    ):
        apply_lifecycle_registry(
            [instrument],
            registry,
            as_of=date(2026, 7, 23),
        )


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


def test_review_repro_post_delisting_flat_rows_are_not_retained_healthy(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2026-07-01", "2026-07-12")},
    )
    path = fixture["published"] / "data/processed/OLD.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["Date"] > "2026-07-10", "Volume"] = 0
    frame.loc[frame["Date"] > "2026-07-10", [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
    ]] = 110
    frame.to_csv(path, index=False)

    original = path.read_bytes()
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("inactive instrument must not call provider")
        ),
    )

    row = report["tickers"][0]
    assert row["freshness_status"] == "failed"
    assert row["reason_code"] == "RETAINED_HISTORY_EXTENDS_AFTER_DELISTING"
    assert report["run_status"] == "failed"
    assert report["publication"]["publication_set_valid"] is False
    assert report["publication"]["publication_required"] is False
    assert (fixture["stage"] / "data/processed/OLD.csv").read_bytes() == original


def test_review_repro_limited_history_requires_complete_sessions_since_listing(
    tmp_path: Path,
) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-01",
        listing_start_date="2026-06-01",
        regular_way_listing_date="2026-06-08",
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

    row = report["tickers"][0]
    assert row["freshness_status"] == "already_current"
    assert row["history_coverage_status"] == "insufficient_unexplained"
    assert row["history_coverage_reason_code"] == (
        "HISTORY_START_TOO_LATE_AFTER_LISTING"
    )
    assert report["run_status"] == "degraded"


def test_review_repro_checksum_valid_sec_evidence_on_other_host_is_blocked(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    row = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    row["evidence"][0]["source_url"] = (
        "https://example.com/official-looking-form-8-k"
    )
    row["provenance_checksum"] = record_provenance_checksum(row)
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

    with pytest.raises(
        InstrumentLifecycleError,
        match="evidence host",
    ):
        load_lifecycle_registry(path)


@pytest.mark.parametrize(
    ("history_end", "expected_status", "expected_reason"),
    [
        (
            "2026-07-09",
            "ends_before",
            "RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION",
        ),
        (
            "2026-07-10",
            "aligned",
            "RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION",
        ),
        (
            "2026-07-11",
            "extends_after",
            "RETAINED_HISTORY_EXTENDS_AFTER_DELISTING",
        ),
    ],
)
def test_retained_history_requires_exact_delisting_boundary(
    tmp_path: Path,
    history_end: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    instrument = _instrument("ARBITRARY")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"ARBITRARY": ("2026-07-01", history_end)},
    )

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("inactive instrument must not call provider")
        ),
    )
    row = report["tickers"][0]

    assert row["retained_history_boundary_status"] == expected_status
    assert row["retained_history_boundary_reason_code"] == expected_reason
    assert row["freshness_status"] == (
        "not_expected" if expected_status == "aligned" else "failed"
    )
    assert report["publication"]["publication_set_valid"] is (
        expected_status == "aligned"
    )


@pytest.mark.parametrize("ticker", ["BLD", "JHG"])
def test_real_post_delisting_tail_shape_is_blocked_generically(
    tmp_path: Path,
    ticker: str,
) -> None:
    instrument = _instrument(ticker)
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-01",
        delisting_end_date="2026-06-30",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={},
    )
    path = fixture["published"] / "data/processed" / f"{ticker}.csv"
    _write_rows(
        path,
        [
            ("2026-06-30", 100.0, 1000),
            ("2026-07-01", 100.0, 0),
            ("2026-07-02", 100.0, 0),
        ],
    )

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("inactive instrument must not call provider")
        ),
    )

    assert report["tickers"][0]["reason_code"] == (
        "RETAINED_HISTORY_EXTENDS_AFTER_DELISTING"
    )


@pytest.mark.parametrize("ticker", ["BLD", "JHG"])
def test_bld_jhg_boundary_remediation_preserves_valid_history(
    tmp_path: Path,
    ticker: str,
) -> None:
    instrument = _instrument(ticker)
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-01",
        delisting_end_date="2026-06-30",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={},
    )
    path = fixture["published"] / "data/processed" / f"{ticker}.csv"
    _write_rows(path, [("2026-06-30", 100.0, 1000)])
    retained_bytes = path.read_bytes()

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("inactive instrument must not call provider")
        ),
    )
    row = report["tickers"][0]

    assert row["retained_history_boundary_status"] == "aligned"
    assert row["retained_history_boundary_reason_code"] == (
        "RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION"
    )
    assert row["freshness_status"] == "not_expected"
    assert row["history_coverage_status"] == "retained_inactive"
    assert report["run_status"] == "completed"
    assert report["publication"]["publication_set_valid"] is True
    assert (
        fixture["stage"] / "data/processed" / f"{ticker}.csv"
    ).read_bytes() == retained_bytes


def test_consumer_recomputes_retained_history_date_boundary(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2026-07-01", "2026-07-10")},
    )
    _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("inactive instrument must not call provider")
        ),
    )
    price_path = fixture["stage"] / "data/processed/OLD.csv"
    frame = pd.read_csv(price_path)
    extra = frame.iloc[[-1]].copy()
    extra["Date"] = "2026-07-11"
    extra["Volume"] = 0
    pd.concat([frame, extra], ignore_index=True).to_csv(price_path, index=False)

    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["tickers"][0]
    entry["persisted_file_checksum"] = scheduled._sha256_file(price_path)
    entry["resulting_last_observation"] = "2026-07-11"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
    )

    assert "PUBLISHED_RETAINED_HISTORY_BOUNDARY_INVALID" in result[
        "reason_codes"
    ]


def test_recent_listing_coverage_uses_when_issued_sessions_and_holidays(
    tmp_path: Path,
) -> None:
    instrument = _instrument("RECENT")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-29",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={},
    )
    sessions = _us_sessions("2026-06-15", "2026-07-14")
    _write_rows(
        fixture["published"] / "data/processed/RECENT.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions)],
    )

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )
    row = report["tickers"][0]

    assert "2026-06-19" not in sessions
    assert "2026-07-03" not in sessions
    assert all(date.fromisoformat(day).weekday() < 5 for day in sessions)
    assert row["freshness_status"] == "already_current"
    assert row["history_coverage_status"] == "limited_history"
    assert row["history_listing_boundary_type"] == "when_issued_start"
    assert row["history_expected_session_count"] == len(sessions)
    assert len(_us_sessions("2026-06-15", "2026-06-28")) > 0
    assert row["history_initial_session_lag"] == 0
    assert row["history_missing_session_count"] == 0
    assert row["history_session_coverage_ratio"] == 1.0
    assert report["run_status"] == "completed"


def test_one_session_listing_start_tolerance_is_explicit_and_bounded(
    tmp_path: Path,
) -> None:
    instrument = _instrument("LAGGED")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-15",
    )
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    sessions = _us_sessions("2026-06-15", "2026-07-14")[1:]
    _write_rows(
        fixture["published"] / "data/processed/LAGGED.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions)],
    )

    row = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )["tickers"][0]

    assert scheduled.MAX_LISTING_START_SESSION_LAG == 1
    assert row["history_coverage_status"] == "limited_history"
    assert row["history_initial_session_lag"] == 1
    assert row["history_session_coverage_ratio"] < 1.0
    assert row["history_bounded_session_coverage_ratio"] == 1.0
    assert row["history_required_session_coverage_ratio"] == 1.0


def test_internal_listing_history_gap_is_unexplained(
    tmp_path: Path,
) -> None:
    instrument = _instrument("GAPPED")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-15",
    )
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    sessions = _us_sessions("2026-06-15", "2026-07-14")
    del sessions[5]
    _write_rows(
        fixture["published"] / "data/processed/GAPPED.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions)],
    )

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )
    row = report["tickers"][0]

    assert row["history_coverage_status"] == "insufficient_unexplained"
    assert row["history_coverage_reason_code"] == (
        "HISTORY_SESSION_GAPS_AFTER_LISTING"
    )
    assert report["run_status"] == "degraded"


@pytest.mark.parametrize(
    ("listing_start", "regular_way", "history_start"),
    [
        ("2026-06-01", "2026-06-08", "2026-07-01"),
        ("2025-07-01", "2025-07-01", "2026-07-01"),
    ],
)
def test_late_short_history_is_never_explained_by_listing(
    tmp_path: Path,
    listing_start: str,
    regular_way: str,
    history_start: str,
) -> None:
    instrument = _instrument("LATE")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date=listing_start,
        listing_start_date=listing_start,
        regular_way_listing_date=regular_way,
    )
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    sessions = _us_sessions(history_start, "2026-07-14")
    _write_rows(
        fixture["published"] / "data/processed/LATE.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions[-10:])],
    )

    row = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )["tickers"][0]

    assert row["history_coverage_status"] == "insufficient_unexplained"
    assert row["history_coverage_reason_code"] == (
        "HISTORY_START_TOO_LATE_AFTER_LISTING"
    )


@pytest.mark.parametrize("ticker", ["FDXF", "HONA", "Q", "SOLS"])
def test_governed_recent_listings_require_proven_session_coverage(
    tmp_path: Path,
    ticker: str,
) -> None:
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    record = next(row for row in registry["records"] if row["ticker"] == ticker)
    instrument = _instrument(ticker)
    instrument["exchange"] = record["exchange"]
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    sessions = _us_sessions(record["listing_start_date"], "2026-07-14")
    _write_rows(
        fixture["published"] / "data/processed" / f"{ticker}.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions)],
    )

    row = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("current history must not call provider")
        ),
    )["tickers"][0]

    assert row["history_coverage_status"] == "limited_history"
    assert row["history_missing_session_count"] == 0


def test_stale_freshness_remains_independent_from_listing_coverage(
    tmp_path: Path,
) -> None:
    instrument = _instrument("STALE")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-15",
    )
    fixture = _fixture(tmp_path, [instrument], [record], histories={})
    sessions = _us_sessions("2026-06-15", "2026-07-13")
    _write_rows(
        fixture["published"] / "data/processed/STALE.csv",
        [(day, 100.0 + index, 1000) for index, day in enumerate(sessions)],
    )

    row = _run(fixture, provider=lambda *_args: pd.DataFrame())["tickers"][0]

    assert row["freshness_status"] == "failed"
    assert row["reason_code"] == "EXPECTED_SESSION_COVERAGE_INCOMPLETE"
    assert row["history_coverage_status"] == "insufficient_unexplained"
    assert row["history_coverage_reason_code"] == (
        "HISTORY_END_BEFORE_EXPECTED_SESSION"
    )


def test_governed_official_issuer_and_acquirer_hosts_are_accepted() -> None:
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    evidence = [
        item
        for record in registry["records"]
        for item in record["evidence"]
    ]

    assert any(item["source_authority"] == "issuer" for item in evidence)
    assert any(item["source_authority"] == "acquirer" for item in evidence)
    assert all(item["source_host"] in item["source_url"] for item in evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.update(lifecycle_reason="arbitrary_reason"),
            "lifecycle reason",
        ),
        (
            lambda row: row.update(corporate_action_type="arbitrary_action"),
            "corporate action type",
        ),
        (
            lambda row: row["evidence"][0].update(
                source_type="arbitrary_source"
            ),
            "source type",
        ),
        (
            lambda row: row["evidence"][0].update(
                source_publication_date="2026-07-16"
            ),
            "publication date",
        ),
        (
            lambda row: row["evidence"][0].update(
                transition_support=["trading_termination"]
            ),
            "completion and trading termination",
        ),
        (
            lambda row: row["evidence"][0].update(
                subject_ticker="IMPOSTOR"
            ),
            "identity mismatch",
        ),
        (
            lambda row: row.update(
                delisting_end_date=None,
                last_trading_session=None,
            ),
            "requires a last trading session",
        ),
    ],
)
def test_checksum_valid_semantic_evidence_mutations_fail_closed(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    instrument = _instrument("OLD")
    row = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    mutation(row)
    row["provenance_checksum"] = record_provenance_checksum(row)
    path = _write_registry(tmp_path, [row])

    with pytest.raises(InstrumentLifecycleError, match=message):
        load_lifecycle_registry(path)


def test_inactive_evidence_cannot_postdate_effective_transition(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    row = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-11",
        delisting_end_date="2026-07-10",
    )
    row["evidence"][0].update(
        source_publication_date="2026-07-12",
        evidence_retrieved_at="2026-07-20T09:00:00Z",
    )
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(
        InstrumentLifecycleError,
        match="inactive transition evidence date",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


def test_issuer_authority_rejects_ungoverned_https_host(
    tmp_path: Path,
) -> None:
    instrument = _instrument("FUTURE")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-20",
        listing_start_date="2026-07-20",
        regular_way_listing_date="2026-07-22",
    )
    row["official_source_hosts"] = {"issuer": ["official.example.com"]}
    row["evidence"][0].update(
        source_authority="issuer",
        source_type="distribution_timing_release",
        source_url="https://untrusted.example.net/listing",
        source_host="untrusted.example.net",
        transition_support=["listing_schedule"],
    )
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(InstrumentLifecycleError, match="evidence host"):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


@pytest.mark.parametrize(
    ("source_url", "accepted"),
    [
        ("https://www.nyse.com/listings/future", True),
        ("https://exchange.example.com/listings/future", False),
    ],
)
def test_exchange_authority_is_bound_to_exchange_host(
    tmp_path: Path,
    source_url: str,
    accepted: bool,
) -> None:
    instrument = _instrument("FUTURE")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-20",
        listing_start_date="2026-07-20",
        regular_way_listing_date="2026-07-22",
    )
    source_host = source_url.split("/")[2]
    row["evidence"][0].update(
        source_authority="exchange",
        source_type="exchange_notice",
        source_url=source_url,
        source_host=source_host,
        transition_support=["listing_schedule"],
    )
    row["provenance_checksum"] = record_provenance_checksum(row)
    path = _write_registry(tmp_path, [row])

    if accepted:
        assert load_lifecycle_registry(path)["records"][0]["ticker"] == (
            "FUTURE"
        )
    else:
        with pytest.raises(InstrumentLifecycleError, match="evidence host"):
            load_lifecycle_registry(path)


def test_announcement_only_cannot_prove_completed_listing(
    tmp_path: Path,
) -> None:
    instrument = _instrument("NEW")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-01",
        listing_start_date="2026-07-01",
        regular_way_listing_date="2026-07-02",
    )
    row["evidence"][0]["transition_support"] = ["listing_schedule"]
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(
        InstrumentLifecycleError,
        match="listing completion evidence",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


def test_listing_completion_between_when_issued_and_regular_way_is_rejected(
    tmp_path: Path,
) -> None:
    instrument = _instrument("GENERIC")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-29",
    )
    row["evidence"][0].update(
        source_publication_date="2026-06-16",
        transition_support=["listing_completion", "listing_schedule"],
    )
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(
        InstrumentLifecycleError,
        match="LISTING_COMPLETION_BEFORE_REGULAR_WAY",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


def test_listing_completion_before_when_issued_is_rejected(
    tmp_path: Path,
) -> None:
    instrument = _instrument("PRESTART")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-29",
    )
    _set_active_listing_evidence(
        row,
        schedule_date="2026-06-01",
        completion_date="2026-06-14",
    )
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(
        InstrumentLifecycleError,
        match="LISTING_COMPLETION_BEFORE_REGULAR_WAY",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


@pytest.mark.parametrize(
    "completion_date",
    ["2026-06-29", "2026-06-30"],
)
def test_listing_completion_on_or_after_regular_way_is_accepted(
    tmp_path: Path,
    completion_date: str,
) -> None:
    instrument = _instrument("BOUNDARY")
    row = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-06-15",
        listing_start_date="2026-06-15",
        regular_way_listing_date="2026-06-29",
    )
    _set_active_listing_evidence(
        row,
        schedule_date="2026-06-01",
        completion_date=completion_date,
    )
    row["provenance_checksum"] = record_provenance_checksum(row)

    registry = load_lifecycle_registry(_write_registry(tmp_path, [row]))

    assert registry["records"][0]["ticker"] == "BOUNDARY"


@pytest.mark.parametrize("ticker", ["FDXF", "HONA", "Q", "SOLS"])
def test_governed_recent_listing_completion_is_bound_to_regular_way(
    tmp_path: Path,
    ticker: str,
) -> None:
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    row = json.loads(
        json.dumps(
            next(
                record
                for record in registry["records"]
                if record["ticker"] == ticker
            )
        )
    )
    regular_way = date.fromisoformat(row["regular_way_listing_date"])
    completion_evidence = [
        evidence
        for evidence in row["evidence"]
        if "listing_completion" in evidence["transition_support"]
    ]
    assert completion_evidence
    assert all(
        date.fromisoformat(evidence["source_publication_date"]) >= regular_way
        for evidence in completion_evidence
    )
    premature_date = (regular_way - timedelta(days=1)).isoformat()
    for evidence in row["evidence"]:
        if "listing_completion" in evidence["transition_support"]:
            evidence["source_publication_date"] = premature_date
    row["provenance_checksum"] = record_provenance_checksum(row)

    with pytest.raises(
        InstrumentLifecycleError,
        match="LISTING_COMPLETION_BEFORE_REGULAR_WAY",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [row]))


def test_lifecycle_v1_is_explicitly_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "market-engine-instrument-lifecycle-registry-v1"
                ),
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InstrumentLifecycleError, match="unsupported"):
        load_lifecycle_registry(path)


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


def test_identical_v3_input_requires_no_empty_publication_commit(
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
        "currency": "USD",
        "exchange": "NYSE",
        "instrument_id": f"equity:{symbol.lower()}",
        "source_mapping_status": "mapped",
        "source_symbol": symbol,
        "symbol": symbol,
        "universe_memberships": ["test"],
    }


def _set_active_listing_evidence(
    row: dict[str, Any],
    *,
    schedule_date: str,
    completion_date: str,
) -> None:
    schedule = dict(row["evidence"][0])
    schedule.update(
        source_publication_date=schedule_date,
        transition_support=["listing_schedule"],
    )
    completion = dict(schedule)
    completion.update(
        source_publication_date=completion_date,
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1/completion.htm"
        ),
        transition_support=["listing_completion"],
    )
    row["evidence"] = [schedule, completion]


def test_transaction_closing_date_never_extends_freshness_cutoff() -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record["transaction_closing_date"] = "2026-07-15"
    record["provenance_checksum"] = record_provenance_checksum(record)
    governed = apply_lifecycle_registry(
        [instrument],
        _normalized_registry([record]),
        as_of=date(2026, 7, 15),
    )

    _profile, expected = scheduled.expected_completed_session(
        governed["inactive_instruments"][0],
        datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
    )

    assert expected == date(2026, 7, 14)


def test_inactive_backfill_quarantines_post_cutoff_bar_but_blocks_without_evidence(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-13")},
    )

    def provider(_symbol: str, _start: str, _end: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Date": day,
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.0,
                    "Adj Close": 100.0,
                    "Volume": 1000,
                }
                for day in ("2026-07-14", "2026-07-15")
            ]
        )

    report = _run(fixture, provider=provider)
    row = report["tickers"][0]
    staged = pd.read_csv(fixture["stage"] / "data/processed/OLD.csv")

    assert row["freshness_status"] == "not_expected"
    assert row["validation_status"] == "valid"
    assert row["resulting_last_observation"] == "2026-07-14"
    assert staged["Date"].max() == "2026-07-14"
    assert row["rejected_bar_diagnostics"] == [
        {
            "ticker": "OLD",
            "session_date": "2026-07-15",
            "cutoff_date": "2026-07-14",
            "last_trading_session": "2026-07-14",
                "canonical_ohlcv_last_observed_session": "2026-07-14",
            "lifecycle_event": "inactive_after_completed_corporate_action",
            "lifecycle_provenance_checksum": record["provenance_checksum"],
            "provider": scheduled.PROVIDER_IDENTITY,
            "retry_number": 1,
            "final_reason_code": "PROVIDER_BAR_AFTER_LIFECYCLE_CUTOFF",
            "disposition": "quarantined_not_persisted",
        }
    ]
    assert row["mutation_evidence_status"] == "invalid"
    assert report["run_status"] == "failed"


def test_contradictory_inactive_effective_date_fails_closed(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record["inactive_effective_date"] = "2026-07-16"
    record["provenance_checksum"] = record_provenance_checksum(record)

    with pytest.raises(
        InstrumentLifecycleError,
        match="inactive status and inactive effective dates must match",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [record]))


@pytest.mark.parametrize("present_field", ["legacy", "canonical", "both"])
def test_lifecycle_alias_compatibility_inputs_normalize(
    tmp_path: Path,
    present_field: str,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    if present_field == "legacy":
        record.pop("last_trading_session")
    elif present_field == "canonical":
        record.pop("delisting_end_date")
    record["provenance_checksum"] = record_provenance_checksum(record)

    loaded = load_lifecycle_registry(_write_registry(tmp_path, [record]))
    normalized = loaded["records"][0]

    assert normalized["delisting_end_date"] == "2026-07-14"
    assert normalized["last_trading_session"] == "2026-07-14"


def test_lifecycle_alias_conflict_fails_closed_with_values(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record["last_trading_session"] = "2026-07-13"
    record["provenance_checksum"] = record_provenance_checksum(record)

    with pytest.raises(
        InstrumentLifecycleError,
        match="semantic alias conflict.*OLD.*last_trading_session.*delisting_end_date",
    ):
        load_lifecycle_registry(_write_registry(tmp_path, [record]))


@pytest.mark.parametrize("present_field", ["legacy", "canonical", "both"])
def test_observation_alias_boundary_projects_to_canonical_model(
    tmp_path: Path,
    present_field: str,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    if present_field in {"legacy", "both"}:
        record["price_observation_end_session"] = "2026-07-14T00:00:00Z"
        record["final_session_observation_status"] = "observed"
    if present_field == "legacy":
        record.pop("canonical_ohlcv_last_observed_session")
        record.pop("terminal_session_daily_ohlcv_status")
    record["provenance_checksum"] = record_provenance_checksum(record)

    normalized = load_lifecycle_registry(
        _write_registry(tmp_path, [record])
    )["records"][0]

    assert normalized["canonical_ohlcv_last_observed_session"] == "2026-07-14"
    assert normalized["terminal_session_daily_ohlcv_status"] == "observed_daily_ohlcv"
    assert "price_observation_end_session" not in normalized
    assert "final_session_observation_status" not in normalized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price_observation_end_session", "2026-07-13"),
        ("final_session_observation_status", "no_valid_price_observation"),
    ],
)
@pytest.mark.parametrize("include_batch_peer", [False, True])
def test_observation_alias_conflicts_fail_closed_with_diagnostics(
    tmp_path: Path,
    field: str,
    value: str,
    include_batch_peer: bool,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record[field] = value
    record["provenance_checksum"] = record_provenance_checksum(record)
    records = [record]
    if include_batch_peer:
        peer = _record(
            _instrument("PEER"),
            lifecycle_status="inactive",
            status_effective_date="2026-07-15",
            delisting_end_date="2026-07-14",
        )
        records.insert(0, peer)

    with pytest.raises(InstrumentLifecycleError) as captured:
        load_lifecycle_registry(_write_registry(tmp_path, records))

    message = str(captured.value)
    assert "OLD" in message
    assert field in message
    assert "contract_version" in message
    assert "canonical_raw" in message
    assert "legacy_raw" in message
    assert "canonical_normalized" in message
    assert "legacy_normalized" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price_observation_end_session", "not-a-date"),
        ("final_session_observation_status", "unknown-status"),
    ],
)
def test_invalid_observation_alias_values_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record[field] = value
    record["provenance_checksum"] = record_provenance_checksum(record)

    with pytest.raises(InstrumentLifecycleError) as captured:
        load_lifecycle_registry(_write_registry(tmp_path, [record]))

    message = str(captured.value)
    assert "semantic alias invalid" in message
    assert "OLD" in message
    assert field in message
    assert repr(value) in message
    assert "canonical_normalized" in message
    assert "legacy_normalized" in message


def test_v2_lifecycle_record_remains_compatible(tmp_path: Path) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    for field in (
        "last_trading_session",
        "transaction_closing_date",
        "trading_suspension_effective_date",
        "inactive_effective_date",
            "canonical_ohlcv_last_observed_session",
            "terminal_session_daily_ohlcv_status",
            "observation_status_as_of",
            "observation_evidence",
        "trading_suspension_effective_timing",
    ):
        record.pop(field)
    record["provenance_checksum"] = record_provenance_checksum(record)
    path = tmp_path / "v2-registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_LIFECYCLE_SCHEMA_VERSION,
                "records": [record],
            }
        ),
        encoding="utf-8",
    )

    normalized = load_lifecycle_registry(path)["records"][0]

    assert normalized["last_trading_session"] == "2026-07-14"
    assert normalized["canonical_ohlcv_last_observed_session"] == "2026-07-14"
    assert normalized["terminal_session_daily_ohlcv_status"] == (
        "observed_daily_ohlcv"
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"transaction_closing_date": "2026-07-13"},
            "transaction closing cannot precede",
        ),
        (
            {
                "trading_suspension_effective_date": "2026-07-14",
                "trading_suspension_effective_timing": "before_open",
            },
            "trading suspension chronology",
        ),
        (
            {"canonical_ohlcv_last_observed_session": "2026-07-15"},
            "price observation end cannot follow",
        ),
    ],
)
def test_invalid_lifecycle_chronology_fails_closed(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    record.update(updates)
    record["provenance_checksum"] = record_provenance_checksum(record)

    with pytest.raises(InstrumentLifecycleError, match=message):
        load_lifecycle_registry(_write_registry(tmp_path, [record]))


def test_after_close_suspension_on_final_session_is_valid(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-14",
        delisting_end_date="2026-07-14",
    )
    record.update(
        transaction_closing_date="2026-07-14",
        trading_suspension_effective_date="2026-07-14",
        trading_suspension_effective_timing="after_close",
        inactive_effective_date="2026-07-14",
    )
    record["provenance_checksum"] = record_provenance_checksum(record)

    normalized = load_lifecycle_registry(_write_registry(tmp_path, [record]))[
        "records"
    ][0]

    assert normalized["trading_suspension_effective_timing"] == "after_close"


def test_changed_price_files_include_inactive_bounded_backfill(
    tmp_path: Path,
) -> None:
    active = _instrument("ACTIVE")
    inactive = _instrument("INACTIVE")
    record = _record(
        inactive,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [active, inactive],
        [record],
        histories={
            "ACTIVE": ("2025-07-01", "2026-07-13"),
            "INACTIVE": ("2025-07-01", "2026-07-13"),
        },
    )

    report = _run(
        fixture,
        provider=lambda *_args: pd.DataFrame(
            [
                {
                    "Date": "2026-07-14",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.0,
                    "Adj Close": 100.0,
                    "Volume": 1000,
                }
            ]
        ),
    )
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert by_ticker["ACTIVE"]["freshness_status"] == "updated"
    assert by_ticker["INACTIVE"]["freshness_status"] == "not_expected"
    assert report["publication"]["changed_price_file_count"] == 2
    assert report["publication"]["changed_price_files"] == [
        "data/processed/ACTIVE.csv",
        "data/processed/INACTIVE.csv",
    ]


def test_aligned_inactive_history_is_not_a_changed_file(
    tmp_path: Path,
) -> None:
    inactive = _instrument("INACTIVE")
    record = _record(
        inactive,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [inactive],
        [record],
        histories={"INACTIVE": ("2025-07-01", "2026-07-14")},
    )

    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(
            AssertionError("aligned inactive history must not call provider")
        ),
    )

    assert report["publication"]["changed_price_file_count"] == 0
    assert report["publication"]["changed_price_files"] == []


def test_changed_price_manifest_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publication"]["changed_price_file_count"] = 0
    manifest["publication"]["changed_price_files"] = []
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
    )

    assert validation["validated"] is False
    assert "PUBLISHED_CHANGED_PRICE_FILE_SET_MISMATCH" in validation[
        "reason_codes"
    ]


def test_changed_price_manifest_cannot_forge_unchanged_baseline(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tickers"][0]["previous_file_checksum"] = manifest["tickers"][0][
        "persisted_file_checksum"
    ]
    manifest["publication"]["changed_price_file_count"] = 0
    manifest["publication"]["changed_price_files"] = []
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
    )

    assert validation["validated"] is False
    assert "PUBLISHED_CHANGED_PRICE_BASELINE_MISMATCH" in validation[
        "reason_codes"
    ]


def test_manifest_cannot_forge_previous_observation_metadata(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tickers"][0]["previous_last_observation"] = "2026-07-14"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        run_at=RUN_AT,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
    )

    assert validation["validated"] is False
    assert "PUBLISHED_OBSERVATION_MUTATION_METADATA_INVALID" in validation[
        "reason_codes"
    ]
    assert "PUBLISHED_OBSERVATION_BASELINE_MISMATCH" in validation[
        "reason_codes"
    ]


def test_singleton_revalidation_preserves_lifecycle_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inactive = _instrument("INACTIVE")
    record = _record(
        inactive,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [inactive],
        [record],
        histories={"INACTIVE": ("2025-07-01", "2026-07-13")},
    )
    singleton_frame = pd.DataFrame(
        [
            {
                "Date": day,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Adj Close": 100.0,
                "Volume": 1000,
            }
            for day in ("2026-07-14", "2026-07-15")
        ]
    )
    monkeypatch.setattr(
        scheduled,
        "download_yfinance_batch",
        lambda symbols, _start, _end: {
            symbol: pd.DataFrame(
                [
                    {
                        "Date": "2026-07-14",
                        "Open": 100.0,
                        "High": 90.0,
                        "Low": 99.0,
                        "Close": 100.0,
                        "Adj Close": 100.0,
                        "Volume": 1000,
                    }
                ]
            )
            for symbol in symbols
        },
    )
    monkeypatch.setattr(
        scheduled,
        "_download_yfinance_history",
        lambda *_args: singleton_frame.copy(),
    )

    report = _run(fixture, provider=None)
    row = report["tickers"][0]
    staged = pd.read_csv(fixture["stage"] / "data/processed/INACTIVE.csv")
    quarantine = [
        item
        for item in row["rejected_bar_diagnostics"]
        if item.get("final_reason_code")
        == "PROVIDER_BAR_AFTER_LIFECYCLE_CUTOFF"
    ]
    lifecycle_context = apply_lifecycle_registry(
        [inactive],
        _normalized_registry([record]),
        as_of=date(2026, 7, 15),
    )["inactive_instruments"][0]
    batch_validated = scheduled._validate_provider_frame(
        singleton_frame,
        date(2026, 7, 14),
        provider_symbol="INACTIVE",
        retry_number=1,
        lifecycle_context=lifecycle_context,
    )
    batch_quarantine = batch_validated.attrs["rejected_bar_diagnostics"]

    assert row["freshness_status"] == "not_expected"
    assert row["resulting_last_observation"] == "2026-07-14"
    assert staged["Date"].max() == "2026-07-14"
    assert quarantine[0]["session_date"] == "2026-07-15"
    assert quarantine[0]["canonical_ohlcv_last_observed_session"] == (
        "2026-07-14"
    )
    assert quarantine[0]["disposition"] == batch_quarantine[0]["disposition"]
    assert quarantine[0]["final_reason_code"] == batch_quarantine[0][
        "final_reason_code"
    ]
    assert pd.to_datetime(batch_validated["Date"]).dt.date.max() == date(
        2026, 7, 14
    )
    assert row["provider_retrieval"][-1]["classification"] == (
        "single_ticker_refetch_valid"
    )


def test_inactive_backfill_requires_every_expected_exchange_session(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-10")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(
            ["2026-07-13", "2026-07-14"]
        ),
    )["tickers"][0]
    staged = pd.read_csv(fixture["stage"] / "data/processed/OLD.csv")

    assert row["freshness_status"] == "not_expected"
    assert row["expected_backfill_sessions"] == ["2026-07-13", "2026-07-14"]
    assert row["observed_backfill_sessions"] == ["2026-07-13", "2026-07-14"]
    assert staged["Date"].tolist()[-2:] == ["2026-07-13", "2026-07-14"]


def test_active_recent_listing_completeness_starts_after_canonical_end(
    tmp_path: Path,
) -> None:
    instrument = _instrument("NEW")
    record = _record(
        instrument,
        lifecycle_status="active",
        status_effective_date="2026-07-01",
        listing_start_date="2026-07-01",
        regular_way_listing_date="2026-07-02",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"NEW": ("2026-07-01", "2026-07-10")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(
            ["2026-07-13", "2026-07-14"]
        ),
    )["tickers"][0]

    assert row["freshness_status"] == "already_current"
    assert row["previous_last_observation"] == "2026-07-10"
    assert row["resulting_last_observation"] == "2026-07-14"
    assert row["expected_backfill_sessions"] == [
        "2026-07-13",
        "2026-07-14",
    ]


@pytest.mark.parametrize(
    "received",
    [["2026-07-14"], ["2026-07-13"]],
    ids=["terminal-only", "internal-session-only"],
)
def test_incomplete_inactive_backfill_fails_closed(
    tmp_path: Path,
    received: list[str],
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-10")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(received),
    )["tickers"][0]

    assert row["freshness_status"] == "failed"
    assert row["rows_added"] == 0
    assert pd.read_csv(fixture["stage"] / "data/processed/OLD.csv")[
        "Date"
    ].max() == "2026-07-10"


def test_bounded_refetch_can_fill_one_missing_session(tmp_path: Path) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-10")},
    )
    responses = iter(
        [_provider_rows(["2026-07-14"]), _provider_rows(["2026-07-13"])]
    )
    requests: list[tuple[str, str]] = []

    def approved_provider(_symbol: str, start: str, end: str) -> pd.DataFrame:
        requests.append((start, end))
        return next(responses)

    approved_provider.provider_id = "approved-primary"
    policy_path = tmp_path / "market-price-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": receipt_contract.POLICY_SCHEMA_VERSION,
                "providers": [
                    {
                        "provider_id": "approved-primary",
                        "approval_id": "approved-primary-v1",
                        "data_type": "daily_ohlcv",
                        "approved_for_acquisition": True,
                        "approved_for_raw_storage": True,
                        "approved_for_replay": True,
                        "approved_for_canonical_publication": True,
                        "acquisition_routes": ["primary_replay"],
                        "exchanges": ["NYSE"],
                        "retention_classification": "immutable_test_evidence",
                        "redistribution_classification": "test_only",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = _run(
        fixture,
        provider=approved_provider,
        source_policy_path=policy_path,
    )["tickers"][0]

    assert row["freshness_status"] == "not_expected"
    assert row["rows_added"] == 2
    assert row["previous_last_observation"] == "2026-07-10"
    assert row["resulting_last_observation"] == "2026-07-14"
    assert requests == [
        ("2026-07-11", "2026-07-15"),
        ("2026-07-12", "2026-07-15"),
    ]


def test_gap_directed_replay_window_is_bounded_and_order_independent() -> None:
    expected = ("2026-07-12", "2026-07-16")
    assert scheduled.gap_directed_replay_window(
        [date(2026, 7, 15), date(2026, 7, 13)],
        request_start="2026-07-12",
        request_end_exclusive="2026-07-16",
    ) == expected
    assert scheduled.gap_directed_replay_window(
        [date(2026, 7, 13), date(2026, 7, 15)],
        request_start="2026-07-12",
        request_end_exclusive="2026-07-16",
    ) == expected


@pytest.mark.parametrize(
    "missing,start,end",
    [
        ([], "2026-07-12", "2026-07-16"),
        ([date(2026, 7, 11)], "2026-07-12", "2026-07-16"),
        ([date(2026, 7, 16)], "2026-07-12", "2026-07-16"),
    ],
)
def test_gap_directed_replay_window_rejects_unbounded_inputs(
    missing: list[date], start: str, end: str
) -> None:
    with pytest.raises(ValueError, match="gap-directed replay"):
        scheduled.gap_directed_replay_window(
            missing,
            request_start=start,
            request_end_exclusive=end,
        )


def test_exchange_holiday_is_not_an_expected_missing_session(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-07",
        delisting_end_date="2026-07-06",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-02")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(["2026-07-06"]),
        run_at=datetime(2026, 7, 7, 21, 0, tzinfo=UTC),
    )["tickers"][0]

    assert row["expected_backfill_sessions"] == ["2026-07-06"]
    assert row["freshness_status"] == "not_expected"


def test_ea_terminal_only_primary_blocks_without_approved_gap_fill(
    tmp_path: Path,
) -> None:
    instrument = _instrument("EA")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-08-05",
        delisting_end_date="2026-08-04",
    )
    record["evidence"][0]["evidence_retrieved_at"] = "2026-08-05T09:00:00Z"
    record["provenance_checksum"] = record_provenance_checksum(record)
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"EA": ("2025-07-01", "2026-07-23")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(["2026-08-04"]),
        run_at=datetime(2026, 8, 5, 21, 0, tzinfo=UTC),
    )["tickers"][0]

    assert row["previous_last_observation"] == "2026-07-23"
    assert row["freshness_status"] == "failed"
    assert row["reason_code"] == "RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION"
    assert row["resulting_last_observation"] == "2026-07-23"
    assert row["rows_added"] == 0
    assert row["previous_row_count"] == row["resulting_row_count"]
    assert row["expected_backfill_sessions"] == [
        "2026-07-24",
        "2026-07-27",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-03",
        "2026-08-04",
    ]
    assert row["observation_receipts"] == []
    assert row["fallback_candidate_sessions"] == [
        "2026-07-24",
        "2026-07-27",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-03",
        "2026-08-04",
    ]


def _approved_gap_fill_run(
    tmp_path: Path,
    *,
    receipt_session: str = "2026-07-13",
    provider_sessions: Sequence[str] = ("2026-07-14",),
    receipt_request_end: str = "2026-07-15",
    expect_completed: bool = True,
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-15",
        delisting_end_date="2026-07-14",
    )
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-10")},
    )
    policy_path = tmp_path / "market-price-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": receipt_contract.POLICY_SCHEMA_VERSION,
                "providers": [
                    {
                        "provider_id": "approved-test-fallback",
                        "approval_id": "approval-test-fallback-v1",
                        "data_type": "daily_ohlcv",
                        "approved_for_acquisition": True,
                        "approved_for_raw_storage": True,
                        "approved_for_replay": True,
                        "approved_for_canonical_publication": True,
                        "acquisition_routes": [
                            "primary",
                            "primary_replay",
                            "fallback",
                        ],
                        "exchanges": ["NYSE"],
                        "retention_classification": "immutable_test_evidence",
                        "redistribution_classification": "test_only",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.dumps(
        {
            "bars": [
                {
                    "session_date": receipt_session,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                    "adj_close": "100",
                    "volume": 1000,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw = receipt_contract.preserve_raw_artifact(
        payload,
        artifact_root=fixture["published"],
        provider_id="approved-test-fallback",
        content_type="application/json",
    )
    policy = receipt_contract.load_source_policy(policy_path)
    receipts = receipt_contract.build_observation_receipts(
        payload,
        policy=policy,
        provider_id="approved-test-fallback",
        provider_symbol="OLD",
        acquisition_route="fallback",
        instrument_id=instrument["instrument_id"],
        ticker="OLD",
        exchange="NYSE",
        currency="USD",
        retrieved_at="2026-07-15T09:00:00Z",
        request_start="2026-07-13",
        request_end_exclusive=receipt_request_end,
        raw_artifact_locator=raw["raw_artifact_locator"],
        raw_artifact_sha256=raw["raw_artifact_sha256"],
        response_status=200,
        content_type="application/json",
    )
    primary_receipts: list[dict[str, Any]] = []
    if provider_sessions:
        primary_payload = json.dumps(
            {
                "bars": [
                    {
                        "session_date": session,
                        "open": "100",
                        "high": "101",
                        "low": "99",
                        "close": "100",
                        "adj_close": "100",
                        "volume": 1000,
                    }
                    for session in provider_sessions
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        primary_raw = receipt_contract.preserve_raw_artifact(
            primary_payload,
            artifact_root=fixture["published"],
            provider_id="approved-test-fallback",
            content_type="application/json",
        )
        primary_receipts = receipt_contract.build_observation_receipts(
            primary_payload,
            policy=policy,
            provider_id="approved-test-fallback",
            provider_symbol="OLD",
            acquisition_route="primary",
            instrument_id=instrument["instrument_id"],
            ticker="OLD",
            exchange="NYSE",
            currency="USD",
            retrieved_at="2026-07-15T09:00:00Z",
            request_start="2026-07-13",
            request_end_exclusive="2026-07-15",
            raw_artifact_locator=primary_raw["raw_artifact_locator"],
            raw_artifact_sha256=primary_raw["raw_artifact_sha256"],
            response_status=200,
            content_type="application/json",
        )
    report = _run(
        fixture,
        provider=lambda *_args: _provider_rows(provider_sessions),
        source_policy_path=policy_path,
        observation_receipts={
            instrument["instrument_id"]: receipts + primary_receipts
        },
    )
    if expect_completed:
        assert report["run_status"] == "completed", report["tickers"]
    else:
        assert report["run_status"] == "failed", report["tickers"]
    return fixture, policy_path, report


def test_primary_and_fallback_conflict_fails_closed(tmp_path: Path) -> None:
    _, _, report = _approved_gap_fill_run(
        tmp_path,
        receipt_session="2026-07-14",
        provider_sessions=("2026-07-13", "2026-07-14"),
        expect_completed=False,
    )
    assert report["tickers"][0]["freshness_status"] == "failed"
    assert any(
        diagnostic["reason_code"]
        == "OBSERVATION_RECEIPT_INVALID"
        for diagnostic in report["tickers"][0]["rejected_bar_diagnostics"]
    )


def test_fallback_post_cutoff_observation_fails_closed(tmp_path: Path) -> None:
    _, _, report = _approved_gap_fill_run(
        tmp_path,
        receipt_session="2026-07-15",
        provider_sessions=("2026-07-13", "2026-07-14"),
        receipt_request_end="2026-07-16",
        expect_completed=False,
    )
    assert report["tickers"][0]["freshness_status"] == "failed"
    assert any(
        diagnostic["reason_code"]
        == "OBSERVATION_RECEIPT_IDENTITY_MISMATCH"
        for diagnostic in report["tickers"][0]["rejected_bar_diagnostics"]
    )


def test_trusted_publisher_replays_raw_receipt_to_canonical_row(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is True, result

    installed_snapshot_result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert installed_snapshot_result["validated"] is True, (
        installed_snapshot_result
    )


def test_primary_observed_label_cannot_replace_replayable_receipt(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["tickers"][0]
    entry["primary_observed_sessions"] = ["2026-07-14"]
    entry["observation_receipts"] = [
        receipt
        for receipt in entry["observation_receipts"]
        if receipt["session_date"] != "2026-07-14"
    ]
    entry["observation_receipt_root"] = receipt_contract.observation_receipt_root(
        entry["observation_receipts"]
    )
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is False
    assert "PUBLISHED_MUTATION_EVIDENCE_RECONCILIATION_INVALID" in result[
        "reason_codes"
    ]


def test_primary_observed_diagnostic_relabel_does_not_change_evidence(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tickers"][0]["primary_observed_sessions"] = [
        "2026-07-13",
        "2026-07-14",
    ]
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is True, result


def test_publisher_rejects_forged_publication_mutation_root(
    tmp_path: Path,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mutation_evidence_summary"]["publication_mutation_root"] = "f" * 64
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is False
    assert "PUBLISHED_MUTATION_EVIDENCE_ROOT_MISMATCH" in result["reason_codes"]


@pytest.mark.parametrize(
    "mutation",
    [
        "close",
        "volume",
        "adj_close",
        "missing_receipt",
        "duplicate_receipt",
        "extra_receipt_without_row",
        "wrong_ticker",
        "wrong_exchange",
        "wrong_session",
        "post_cutoff_session",
        "unknown_provider",
        "wrong_approval_id",
        "unknown_parser",
        "unsuccessful_response",
        "wrong_raw_checksum",
        "missing_raw_artifact",
        "missing_root_leaf",
        "extra_root_leaf",
    ],
)
def test_trusted_publisher_rejects_receipt_and_csv_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["tickers"][0]
    if mutation in {"close", "volume", "adj_close"}:
        csv_path = fixture["stage"] / entry["persisted_file_path"]
        frame = pd.read_csv(csv_path)
        column = {"close": "Close", "volume": "Volume", "adj_close": "Adj Close"}[mutation]
        frame.loc[frame["Date"] == "2026-07-13", column] += 1
        frame.to_csv(csv_path, index=False)
        entry["persisted_file_checksum"] = scheduled._sha256_file(csv_path)
    elif mutation == "missing_receipt":
        entry["observation_receipts"] = []
    elif mutation == "duplicate_receipt":
        entry["observation_receipts"].append(copy.deepcopy(entry["observation_receipts"][0]))
    elif mutation == "extra_receipt_without_row":
        extra = copy.deepcopy(entry["observation_receipts"][0])
        extra["session_date"] = "2026-07-12"
        entry["observation_receipts"].append(extra)
    elif mutation == "wrong_ticker":
        entry["observation_receipts"][0]["ticker"] = "OTHER"
    elif mutation == "wrong_exchange":
        entry["observation_receipts"][0]["exchange"] = "NASDAQ"
    elif mutation == "wrong_session":
        entry["observation_receipts"][0]["session_date"] = "2026-07-14"
    elif mutation == "post_cutoff_session":
        entry["observation_receipts"][0]["session_date"] = "2026-07-15"
    elif mutation == "unknown_provider":
        entry["observation_receipts"][0]["provider_id"] = "reachable-unapproved"
    elif mutation == "wrong_approval_id":
        entry["observation_receipts"][0]["source_approval_id"] = "not-approved"
    elif mutation == "unknown_parser":
        entry["observation_receipts"][0]["parser_version"] = "unknown"
    elif mutation == "unsuccessful_response":
        entry["observation_receipts"][0]["response_status"] = 500
    elif mutation == "wrong_raw_checksum":
        entry["observation_receipts"][0]["raw_artifact_sha256"] = "0" * 64
    elif mutation == "missing_raw_artifact":
        raw_path = fixture["stage"] / entry["observation_receipts"][0][
            "raw_artifact_locator"
        ]
        raw_path.unlink()
    elif mutation == "missing_root_leaf":
        entry["observation_receipt_root"] = receipt_contract.observation_receipt_root([])
    else:
        entry["observation_receipt_root"] = "f" * 64
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is False
    assert any("OBSERVATION_RECEIPT" in code for code in result["reason_codes"])


def test_trusted_publisher_rejects_unbound_raw_artifact(tmp_path: Path) -> None:
    fixture, policy_path, _ = _approved_gap_fill_run(tmp_path)
    extra = fixture["stage"] / "evidence/market_price/unbound/extra.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('{"bars": []}', encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        expected_source_main_sha=SOURCE_SHA,
        baseline_publication_root=fixture["published"],
        source_policy_path=policy_path,
        run_at=RUN_AT,
    )

    assert result["validated"] is False
    assert "PUBLISHED_RAW_OBSERVATION_ARTIFACT_SET_MISMATCH" in result[
        "reason_codes"
    ]


def test_terminal_status_without_replayable_attestation_remains_unresolved(
    tmp_path: Path,
) -> None:
    universe = scheduled.load_authoritative_universe(
        scheduled.DEFAULT_UNIVERSE_SNAPSHOT
    )
    instrument = next(
        row for row in universe["instruments"] if row["symbol"] == "TMHC"
    )
    registry = load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    record = next(row for row in registry["records"] if row["ticker"] == "TMHC")
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"TMHC": ("2025-07-01", "2026-07-23")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: pd.DataFrame(),
        run_at=datetime(2026, 8, 9, 18, 0, tzinfo=UTC),
    )["tickers"][0]

    assert row["expected_backfill_sessions"] == ["2026-07-24"]
    assert row["explained_missing_sessions"] == ["2026-07-24"]
    assert row["fallback_candidate_sessions"] == ["2026-07-24"]
    assert row["observation_receipts"] == []
    assert row["observation_receipt_root"] is None
    assert row["session_resolution"]["unresolved_sessions"] == ["2026-07-24"]


def test_later_terminal_daily_ohlcv_replaces_temporary_absence_status(
    tmp_path: Path,
) -> None:
    instrument = _instrument("OLD")
    record = _record(
        instrument,
        lifecycle_status="inactive",
        status_effective_date="2026-07-25",
        delisting_end_date="2026-07-24",
    )
    record.update(
        canonical_ohlcv_last_observed_session="2026-07-23",
        terminal_session_daily_ohlcv_status=(
            "no_valid_daily_ohlcv_bar_from_provider_as_of"
        ),
        observation_status_as_of="2026-08-09T00:00:00Z",
        observation_evidence={
            "provider_identity": scheduled.PROVIDER_IDENTITY,
            "retrieved_at": "2026-08-09T00:00:00Z",
            "as_of_date": "2026-08-09",
            "request_start": "2026-07-24",
            "request_end_exclusive": "2026-07-25",
            "response_outcome": "empty_provider_response",
            "relevant_session": "2026-07-24",
            "daily_ohlcv_validation_status": "no_valid_daily_ohlcv_bar_returned",
            "evidence_locator": "provider-request:test:OLD",
            "response_checksum": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        },
        transaction_closing_date="2026-07-24",
        trading_suspension_effective_date="2026-07-24",
        trading_suspension_effective_timing="after_close",
    )
    record["evidence"][0]["evidence_retrieved_at"] = "2026-07-25T09:00:00Z"
    record["provenance_checksum"] = record_provenance_checksum(record)
    fixture = _fixture(
        tmp_path,
        [instrument],
        [record],
        histories={"OLD": ("2025-07-01", "2026-07-23")},
    )

    row = _run(
        fixture,
        provider=lambda *_args: _provider_rows(
            ["2026-07-24", "2026-07-25"]
        ),
        run_at=datetime(2026, 7, 25, 21, 0, tzinfo=UTC),
    )["tickers"][0]

    assert row["resulting_last_observation"] == "2026-07-24"
    assert row["terminal_session_daily_ohlcv_status"] == "observed_daily_ohlcv"
    assert row["observation_status_as_of"] is None
    assert row["observation_evidence"] is None
    assert row["rejected_bar_diagnostics"][0]["session_date"] == "2026-07-25"


def _provider_rows(sessions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": session,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Adj Close": 100.0,
                "Volume": 1000,
            }
            for session in sessions
        ]
    )


def _record(
    instrument: dict[str, Any],
    *,
    lifecycle_status: str,
    status_effective_date: str,
    listing_start_date: str | None = None,
    regular_way_listing_date: str | None = None,
    delisting_end_date: str | None = None,
) -> dict[str, Any]:
    if lifecycle_status == "active":
        assert listing_start_date is not None
        assert regular_way_listing_date is not None
        regular_way = date.fromisoformat(regular_way_listing_date)
        evidence_publication_date = min(
            regular_way,
            date(2026, 7, 15),
        ).isoformat()
        transition_support = ["listing_schedule"]
        if regular_way <= date(2026, 7, 15):
            transition_support.append("listing_completion")
        transition_support.sort()
    else:
        assert delisting_end_date is not None
        evidence_publication_date = delisting_end_date
        transition_support = [
            "corporate_action_completion",
            "trading_termination",
        ]
    row = {
        "corporate_action_type": (
            "spin_off_listing"
            if lifecycle_status == "active"
            else "cash_acquisition"
        ),
        "delisting_end_date": delisting_end_date,
        "last_trading_session": delisting_end_date,
        "transaction_closing_date": (
            status_effective_date if lifecycle_status == "inactive" else None
        ),
        "trading_suspension_effective_date": (
            status_effective_date if lifecycle_status == "inactive" else None
        ),
        "inactive_effective_date": (
            status_effective_date if lifecycle_status == "inactive" else None
        ),
        "canonical_ohlcv_last_observed_session": (
            delisting_end_date if lifecycle_status == "inactive" else None
        ),
        "terminal_session_daily_ohlcv_status": (
            "observed_daily_ohlcv" if lifecycle_status == "inactive" else None
        ),
        "observation_status_as_of": None,
        "observation_evidence": None,
        "trading_suspension_effective_timing": (
            "before_open" if lifecycle_status == "inactive" else None
        ),
        "evidence": [
            {
                "evidence_retrieved_at": "2026-07-15T09:00:00Z",
                "source_authority": "sec",
                "source_publication_date": evidence_publication_date,
                "source_host": "www.sec.gov",
                "source_type": "form_8_k",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/1/test.htm"
                ),
                "subject_exchange": "NYSE",
                "subject_instrument_id": instrument["instrument_id"],
                "subject_ticker": instrument["symbol"],
                "transition_support": transition_support,
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
        "official_source_hosts": {},
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


def _run(
    fixture: dict[str, Path],
    *,
    provider: Any,
    run_at: datetime = RUN_AT,
    source_policy_path: Path = scheduled.DEFAULT_SOURCE_POLICY,
    observation_receipts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    absence_attestations: Mapping[
        str, Sequence[Mapping[str, Any]]
    ] | None = None,
) -> dict[str, Any]:
    return scheduled.run_scheduled_refresh(
        run_id="me-sr18-test-20260715T100000Z",
        source_main_sha=SOURCE_SHA,
        universe_snapshot_path=fixture["universe"],
        lifecycle_registry_path=fixture["registry"],
        published_root=fixture["published"],
        staging_root=fixture["stage"],
        report_output=fixture["report"],
        run_at=run_at,
        provider=provider,
        sleeper=lambda _seconds: None,
        source_policy_path=source_policy_path,
        observation_receipts=observation_receipts,
        absence_attestations=absence_attestations,
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


def _write_rows(
    path: Path,
    rows: list[tuple[str, float, int]],
) -> None:
    frame = pd.DataFrame(
        [
            {
                "Date": day,
                "Adj Close": close,
                "Close": close,
                "High": close + 1,
                "Low": close - 1,
                "Open": close - 0.5,
                "Volume": volume,
            }
            for day, close, volume in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _us_sessions(start: str, end: str) -> list[str]:
    cursor = date.fromisoformat(start)
    boundary = date.fromisoformat(end)
    sessions: list[str] = []
    while cursor <= boundary:
        if scheduled._is_trading_session(cursor, "us_equities"):
            sessions.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return sessions


def _write_registry(
    root: Path,
    records: list[dict[str, Any]],
) -> Path:
    path = root / "semantic-registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path
