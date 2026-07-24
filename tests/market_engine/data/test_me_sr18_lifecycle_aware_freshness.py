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
    assert report["publication"]["publication_set_valid"] is False
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

    assert row["freshness_status"] == "stale"
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
            lambda row: row.update(delisting_end_date=None),
            "requires a delisting end date",
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
