from __future__ import annotations

import copy
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from market_engine.data import scheduled_canonical_price_refresh as scheduled


SOURCE_SHA = "8" * 40
RUN_AT = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)


def _instrument(
    symbol: str,
    *,
    exchange: str = "US",
    country: str = "US",
) -> dict[str, Any]:
    return {
        "active": True,
        "instrument_id": f"equity:{symbol.lower()}",
        "symbol": symbol,
        "source_symbol": symbol,
        "source_mapping_status": "mapped",
        "exchange": exchange,
        "country": country,
    }


def test_normal_update_adds_one_valid_completed_session(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    calls: list[tuple[str, str, str]] = []

    def provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append((symbol, start, end))
        return _frame([("2026-07-14", 500)])

    report = _run(fixture, provider=provider)
    row = report["tickers"][0]

    assert row["freshness_status"] == "updated"
    assert row["reason_code"] == "VALIDATED_UPDATE_PERSISTED"
    assert row["rows_added"] == 1
    assert row["resulting_last_observation"] == "2026-07-14"
    assert report["run_status"] == "completed"
    assert report["publication"]["publication_set_valid"] is True
    assert report["publication"]["publication_required"] is True
    assert (fixture["stage"] / scheduled.LATEST_MANIFEST).is_file()
    assert calls == [("AAA", "2026-07-14", "2026-07-15")]


def test_multiple_missing_sessions_are_appended(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-09")
    report = _run(
        fixture,
        provider=lambda *_args: _frame(
            [("2026-07-10", 500), ("2026-07-13", 501), ("2026-07-14", 502)]
        ),
    )

    assert report["tickers"][0]["rows_added"] == 3
    assert report["tickers"][0]["freshness_status"] == "updated"


def test_missing_local_history_creates_a_valid_snapshot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    (fixture["published"] / "data/processed/AAA.csv").unlink()
    rows = [
        ((date(2026, 7, 14) - timedelta(days=251 - index)).isoformat(), 100 + index)
        for index in range(252)
    ]
    report = _run(fixture, provider=lambda *_args: _frame(rows))
    assert report["tickers"][0]["freshness_status"] == "updated"
    assert report["tickers"][0]["previous_last_observation"] is None
    assert (fixture["stage"] / "data/processed/AAA.csv").is_file()


def test_short_current_history_has_separate_unexplained_coverage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-14")
    path = fixture["published"] / "data/processed/AAA.csv"
    _frame(
        [((date(2026, 7, 14) - timedelta(days=9 - index)).isoformat(), 100 + index) for index in range(10)]
    ).to_csv(path, index=False)
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )
    assert report["tickers"][0]["freshness_status"] == "already_current"
    assert (
        report["tickers"][0]["history_coverage_status"]
        == "insufficient_unexplained"
    )
    assert report["run_status"] == "degraded"


def test_default_provider_is_called_in_bounded_multi_symbol_batches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    calls: list[tuple[list[str], str, str]] = []

    def batch(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        calls.append((symbols, start, end))
        return {symbol: _frame([("2026-07-14", 500)]) for symbol in symbols}

    monkeypatch.setattr(scheduled, "download_yfinance_batch", batch)
    report = scheduled.run_scheduled_refresh(
        run_id="me-sr17-test-20260715T100000Z",
        source_main_sha=SOURCE_SHA,
        universe_snapshot_path=fixture["universe"],
        published_root=fixture["published"],
        staging_root=fixture["stage"],
        report_output=fixture["report"],
        run_at=RUN_AT,
        batch_size=25,
        sleeper=lambda _seconds: None,
    )
    assert calls == [(["AAA", "BBB"], "2026-07-14", "2026-07-15")]
    assert report["status_counts"]["updated"] == 2
    assert report["provider_configuration"]["request_mode"] == "bounded_multi_symbol_batches"


def test_already_current_and_weekend_runs_do_not_call_provider_or_request_commit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-17")
    report = _run(
        fixture,
        run_at=datetime(2026, 7, 19, 10, 0, tzinfo=UTC),
        provider=lambda *_args: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    assert report["tickers"][0]["freshness_status"] == "already_current"
    assert report["tickers"][0]["reason_code"] == "NO_NEW_SESSION_EXPECTED"
    assert report["tickers"][0]["expected_completed_session"] == "2026-07-17"
    assert report["publication"]["publication_required"] is True
    assert report["publication"]["manifest_change_required"] is True
    assert (fixture["stage"] / scheduled.LATEST_MANIFEST).is_file()


@pytest.mark.parametrize(
    ("instrument", "run_at", "expected"),
    [
        (_instrument("USA", exchange="US", country="US"), datetime(2026, 7, 6, 5, 30, tzinfo=UTC), "2026-07-02"),
        (_instrument("AMS", exchange="XAMS", country="NL"), datetime(2026, 4, 7, 5, 30, tzinfo=UTC), "2026-04-02"),
        (_instrument("PAR", exchange="XPAR", country="FR"), datetime(2026, 5, 2, 12, 0, tzinfo=UTC), "2026-04-30"),
        (_instrument("LON", exchange="XLON", country="GB"), datetime(2026, 12, 29, 8, 0, tzinfo=UTC), "2026-12-24"),
    ],
)
def test_us_and_european_holidays_are_exchange_aware(
    instrument: dict[str, Any], run_at: datetime, expected: str
) -> None:
    _profile, session = scheduled.expected_completed_session(instrument, run_at)
    assert session is not None
    assert session.isoformat() == expected


def test_exchange_timezone_boundary_uses_same_day_only_after_close() -> None:
    instrument = _instrument("AMS", exchange="XAMS", country="NL")
    _profile, before_close = scheduled.expected_completed_session(
        instrument, datetime(2026, 7, 14, 15, 0, tzinfo=UTC)
    )
    _profile, after_close = scheduled.expected_completed_session(
        instrument, datetime(2026, 7, 14, 16, 0, tzinfo=UTC)
    )
    assert before_close == date(2026, 7, 13)
    assert after_close == date(2026, 7, 14)


def test_unknown_exchange_is_unsupported_without_provider_call(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA", exchange="UNKNOWN", country="ZZ")],
        end="2026-07-13",
    )
    report = _run(
        fixture,
        provider=lambda *_args: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )
    assert report["tickers"][0]["freshness_status"] == "unsupported"
    assert report["tickers"][0]["reason_code"] == "UNSUPPORTED_EXCHANGE"


def test_missing_provider_mapping_is_unsupported(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    instrument["source_mapping_status"] = "unsupported"
    fixture = _fixture(tmp_path, [instrument], end="2026-07-13")
    report = _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    assert report["tickers"][0]["reason_code"] == "PROVIDER_MAPPING_MISSING"
    assert report["tickers"][0]["provider_identity"] is None


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (TimeoutError("slow"), "PROVIDER_TIMEOUT"),
        (RuntimeError("429 rate limit"), "PROVIDER_RATE_LIMITED"),
        (RuntimeError("provider down"), "PROVIDER_ERROR"),
    ],
)
def test_provider_errors_use_bounded_retries_and_preserve_history(
    tmp_path: Path, failure: Exception, reason: str
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    original = (fixture["published"] / "data/processed/AAA.csv").read_bytes()
    calls = 0
    sleeps: list[float] = []

    def provider(*_args: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise failure

    report = _run(fixture, provider=provider, sleeper=sleeps.append)
    staged = fixture["stage"] / "data/processed/AAA.csv"

    assert calls == 3
    assert sleeps == [1.0, 2.0]
    assert report["tickers"][0]["reason_code"] == reason
    assert report["tickers"][0]["freshness_status"] == "failed"
    assert staged.read_bytes() == original


@pytest.mark.parametrize(
    ("provider", "reason"),
    [
        (lambda *_args: pd.DataFrame({"Date": ["2026-07-14"]}), "PROVIDER_PAYLOAD_SCHEMA_INVALID"),
        (
            lambda *_args: _frame([("2026-07-14", 500), ("2026-07-14", 501)]),
            "PROVIDER_DUPLICATE_TIMESTAMP",
        ),
        (lambda *_args: _frame([("2026-07-15", 500)]), "PROVIDER_FUTURE_DATED_BAR"),
        (lambda *_args: _frame([("2026-07-14", 500)], high=490), "PROVIDER_OHLC_INVALID"),
        (
            lambda *_args: _frame([("2026-07-14", 500), ("2026-07-13", 499)]),
            "PROVIDER_PAYLOAD_NOT_CHRONOLOGICAL",
        ),
    ],
)
def test_malformed_provider_payloads_fail_closed(
    tmp_path: Path, provider: Any, reason: str
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    original = (fixture["published"] / "data/processed/AAA.csv").read_bytes()
    report = _run(fixture, provider=provider)
    assert report["tickers"][0]["reason_code"] == reason
    assert (fixture["stage"] / "data/processed/AAA.csv").read_bytes() == original


def test_empty_provider_response_is_unproven_and_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    original = (fixture["published"] / "data/processed/AAA.csv").read_bytes()
    report = _run(fixture, provider=lambda *_args: pd.DataFrame())
    assert report["tickers"][0]["freshness_status"] == "failed"
    assert report["tickers"][0]["reason_code"] == (
        "EXPECTED_SESSION_COVERAGE_INCOMPLETE"
    )
    assert (fixture["stage"] / "data/processed/AAA.csv").read_bytes() == original


def test_rejected_ohlc_bar_records_safe_exact_diagnostics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    report = _run(
        fixture,
        provider=lambda *_args: _frame(
            [("2026-07-14", 500)],
            high=490,
        ),
    )

    row = report["tickers"][0]
    diagnostic = row["rejected_bar_diagnostics"][0]
    assert row["reason_code"] == "PROVIDER_OHLC_INVALID"
    assert diagnostic["ticker"] == "AAA"
    assert diagnostic["session_date"] == "2026-07-14"
    assert diagnostic["provider"] == scheduled.PROVIDER_IDENTITY
    assert diagnostic["raw_ohlcv"]["open"] == 499.5
    assert diagnostic["canonicalized_ohlcv"]["open"] == 499.5
    assert diagnostic["violations"] == [
        {
            "relation": "HIGH_BELOW_OPEN",
            "left_value": 490.0,
            "right_value": 499.5,
            "absolute_deviation": 9.5,
            "relative_deviation": 9.5 / 499.5,
        },
        {
            "relation": "HIGH_BELOW_CLOSE",
            "left_value": 490.0,
            "right_value": 500.0,
            "absolute_deviation": 10.0,
            "relative_deviation": 0.02,
        },
        {
            "relation": "HIGH_BELOW_LOW",
            "left_value": 490.0,
            "right_value": 499.0,
            "absolute_deviation": 9.0,
            "relative_deviation": 9.0 / 499.0,
        },
    ]
    assert "token" not in json.dumps(diagnostic).lower()


def test_material_open_below_low_remains_blocked() -> None:
    frame = _frame([("2026-07-14", 100)])
    frame.loc[0, "Low"] = 110
    with pytest.raises(scheduled.ProviderBoundaryError) as exc_info:
        scheduled._validate_provider_frame(
            frame,
            date(2026, 7, 14),
            provider_symbol="AAA",
            retry_number=1,
        )
    assert exc_info.value.reason_code == "PROVIDER_OHLC_INVALID"
    assert exc_info.value.diagnostic["violations"][0]["absolute_deviation"] == 10.5


def test_empty_batch_retries_splits_and_uses_single_ticker_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def empty_batch(
        symbols: Sequence[str],
        _start: str,
        _end: str,
    ) -> dict[str, pd.DataFrame]:
        calls.append(tuple(symbols))
        return {symbol: pd.DataFrame() for symbol in symbols}

    def single(symbol: str, *_args: Any) -> pd.DataFrame:
        return _frame([("2026-07-14", 500 if symbol == "AAA" else 600)])

    monkeypatch.setattr(scheduled, "download_yfinance_batch", empty_batch)
    monkeypatch.setattr(scheduled, "_download_yfinance_history", single)
    report = _run(
        fixture,
        provider=None,
        sleeper=sleeps.append,
        max_attempts=2,
    )

    assert calls == [
        ("AAA", "BBB"),
        ("AAA", "BBB"),
        ("AAA",),
        ("AAA",),
        ("BBB",),
        ("BBB",),
    ]
    by_ticker = {row["ticker"]: row for row in report["tickers"]}
    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["freshness_status"] == "updated"
    assert by_ticker["AAA"]["provider_retrieval"][-1]["request_mode"] == "single_ticker"
    assert report["publication"]["publication_set_valid"] is True


def test_invalid_batch_bar_is_revalidated_by_bounded_single_ticker_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    single_calls: list[str] = []

    monkeypatch.setattr(
        scheduled,
        "download_yfinance_batch",
        lambda symbols, _start, _end: {
            symbol: _frame([("2026-07-14", 500)], high=490)
            for symbol in symbols
        },
    )

    def valid_single(symbol: str, *_args: Any) -> pd.DataFrame:
        single_calls.append(symbol)
        return _frame([("2026-07-14", 500)])

    monkeypatch.setattr(scheduled, "_download_yfinance_history", valid_single)
    report = _run(fixture, provider=None, max_attempts=2)

    row = report["tickers"][0]
    assert single_calls == ["AAA"]
    assert row["freshness_status"] == "updated"
    assert row["reason_code"] == "VALIDATED_UPDATE_PERSISTED"
    assert len(row["rejected_bar_diagnostics"]) == 1
    assert [
        item["classification"] for item in row["provider_retrieval"][-2:]
    ] == ["original_invalid_bar", "single_ticker_refetch_valid"]


@pytest.mark.parametrize(
    ("single_result", "expected_reason", "terminal_classification"),
    [
        (
            lambda *_args: _frame([("2026-07-14", 500)], high=490),
            "PROVIDER_OHLC_INVALID",
            "single_ticker_refetch_invalid",
        ),
        (
            lambda *_args: pd.DataFrame(),
            "EXPECTED_SESSION_NOT_AVAILABLE",
            "single_ticker_refetch_missing",
        ),
        (
            lambda *_args: (_ for _ in ()).throw(TimeoutError("bounded")),
            "PROVIDER_TIMEOUT",
            "single_ticker_refetch_provider_failure",
        ),
    ],
)
def test_invalid_batch_bar_remains_blocked_when_single_ticker_revalidation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_result: Any,
    expected_reason: str,
    terminal_classification: str,
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    original = (fixture["published"] / "data/processed/AAA.csv").read_bytes()
    single_calls = 0

    monkeypatch.setattr(
        scheduled,
        "download_yfinance_batch",
        lambda symbols, _start, _end: {
            symbol: _frame([("2026-07-14", 500)], high=490)
            for symbol in symbols
        },
    )

    def bounded_single(*args: Any) -> pd.DataFrame:
        nonlocal single_calls
        single_calls += 1
        return single_result(*args)

    monkeypatch.setattr(scheduled, "_download_yfinance_history", bounded_single)
    report = _run(fixture, provider=None, max_attempts=2)
    row = report["tickers"][0]

    assert single_calls == 2
    assert row["reason_code"] == expected_reason
    assert row["freshness_status"] == "failed"
    assert row["provider_retrieval"][-1]["classification"] == terminal_classification
    assert (fixture["stage"] / "data/processed/AAA.csv").read_bytes() == original
    assert "secret" not in json.dumps(row["provider_retrieval"]).lower()


def test_ohlc_revalidation_is_ticker_local_and_revalidates_full_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    single_calls: list[str] = []

    def batch(
        symbols: Sequence[str],
        _start: str,
        _end: str,
    ) -> dict[str, pd.DataFrame]:
        return {
            symbol: (
                _frame([("2026-07-14", 500)], high=490)
                if symbol == "AAA"
                else _frame([("2026-07-14", 600)])
            )
            for symbol in symbols
        }

    def future_single(symbol: str, *_args: Any) -> pd.DataFrame:
        single_calls.append(symbol)
        return _frame([("2026-07-15", 500)])

    monkeypatch.setattr(scheduled, "download_yfinance_batch", batch)
    monkeypatch.setattr(scheduled, "_download_yfinance_history", future_single)
    report = _run(fixture, provider=None, max_attempts=1)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert single_calls == ["AAA"]
    assert by_ticker["AAA"]["reason_code"] == "PROVIDER_FUTURE_DATED_BAR"
    assert by_ticker["BBB"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["provider_retrieval"] == [
        {
            "request_mode": "batch",
            "batch_symbols": ["AAA", "BBB"],
            "attempt": 1,
            "split_depth": 0,
            "classification": "PROVIDER_BATCH_COMPLETE",
            "received_bar_count": 1,
        }
    ]
    assert report["publication"]["publication_required"] is False


def test_partial_batch_retries_only_missing_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    calls: list[tuple[str, ...]] = []

    def partial_batch(
        symbols: Sequence[str],
        _start: str,
        _end: str,
    ) -> dict[str, pd.DataFrame]:
        calls.append(tuple(symbols))
        return {
            symbol: (
                _frame([("2026-07-14", 500)])
                if symbol == "AAA"
                else pd.DataFrame()
            )
            for symbol in symbols
        }

    monkeypatch.setattr(scheduled, "download_yfinance_batch", partial_batch)
    monkeypatch.setattr(
        scheduled,
        "_download_yfinance_history",
        lambda *_args: pd.DataFrame(),
    )
    report = _run(fixture, provider=None, sleeper=lambda _delay: None, max_attempts=2)

    assert calls == [("AAA", "BBB"), ("BBB",)]
    by_ticker = {row["ticker"]: row for row in report["tickers"]}
    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["reason_code"] == (
        "EXPECTED_SESSION_COVERAGE_INCOMPLETE"
    )
    assert report["publication"]["publication_required"] is False


def test_partial_success_is_terminal_and_later_timeout_has_attempt_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    batch_calls: list[tuple[str, ...]] = []

    def partial_then_timeout(
        symbols: Sequence[str],
        _start: str,
        _end: str,
    ) -> dict[str, pd.DataFrame]:
        batch_calls.append(tuple(symbols))
        if len(batch_calls) == 1:
            return {
                "AAA": _frame([("2026-07-14", 500)]),
                "BBB": pd.DataFrame(),
            }
        raise TimeoutError("later attempt")

    monkeypatch.setattr(scheduled, "download_yfinance_batch", partial_then_timeout)
    monkeypatch.setattr(
        scheduled,
        "_download_yfinance_history",
        lambda *_args: (_ for _ in ()).throw(TimeoutError("singleton")),
    )
    report = _run(fixture, provider=None, max_attempts=2)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert batch_calls == [("AAA", "BBB"), ("BBB",)]
    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert len(by_ticker["AAA"]["provider_retrieval"]) == 1
    assert by_ticker["BBB"]["reason_code"] == "PROVIDER_TIMEOUT"
    assert [
        item["received_bar_count"]
        for item in by_ticker["BBB"]["provider_retrieval"][:2]
    ] == [0, 0]
    assert by_ticker["BBB"]["provider_retrieval"][1]["classification"] == "PROVIDER_TIMEOUT"


def test_terminal_singleton_failures_are_isolated_per_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )

    def batch(
        symbols: Sequence[str],
        _start: str,
        _end: str,
    ) -> dict[str, pd.DataFrame]:
        if len(symbols) > 1:
            return {symbol: pd.DataFrame() for symbol in symbols}
        if symbols[0] == "AAA":
            raise TimeoutError("aaa")
        raise RuntimeError("429 rate limit")

    def single(symbol: str, *_args: Any) -> pd.DataFrame:
        if symbol == "AAA":
            raise TimeoutError("aaa")
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(scheduled, "download_yfinance_batch", batch)
    monkeypatch.setattr(scheduled, "_download_yfinance_history", single)
    report = _run(fixture, provider=None, max_attempts=1)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert by_ticker["AAA"]["reason_code"] == "PROVIDER_TIMEOUT"
    assert by_ticker["BBB"]["reason_code"] == "PROVIDER_RATE_LIMITED"
    assert by_ticker["AAA"]["provider_retrieval"][-1]["classification"] == "PROVIDER_TIMEOUT"
    assert by_ticker["BBB"]["provider_retrieval"][-1]["classification"] == "PROVIDER_RATE_LIMITED"
    assert report["publication"]["publication_required"] is False


def test_allow_degraded_still_reconciles_declared_freshness(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    manifest = _run(fixture, provider=lambda *_args: pd.DataFrame())
    path = fixture["stage"] / scheduled.LATEST_MANIFEST
    assert not path.exists()
    manifest["publication"]["publication_required"] = True
    manifest["tickers"][0]["freshness_status"] = "already_current"
    manifest["status_counts"]["already_current"] = 1
    manifest["status_counts"]["stale"] = 0
    manifest["run_status"] = "completed"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=RUN_AT,
        allow_degraded=True,
    )
    assert (
        "PUBLISHED_FRESHNESS_CLASSIFICATION_INVALID"
        in validation["reason_codes"]
    )


def test_default_publication_validation_rejects_degraded_manifest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )

    def provider(symbol: str, *_args: Any) -> pd.DataFrame:
        if symbol == "BBB":
            return pd.DataFrame()
        return _frame([("2026-07-14", 500)])

    manifest = _run(fixture, provider=provider)
    assert manifest["run_status"] == "degraded"
    assert manifest["publication"]["publication_required"] is False

    manifest["publication"]["publication_required"] = True
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    diagnostic = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=RUN_AT,
        allow_degraded=True,
    )
    publication = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=RUN_AT,
    )

    assert diagnostic["validated"] is False
    assert "PUBLISHED_EXPECTED_SESSION_COMPLETENESS_INVALID" in diagnostic[
        "reason_codes"
    ]
    assert publication["validated"] is False
    assert "PUBLISHED_DATASET_DEGRADED" in publication["reason_codes"]
    assert "PUBLISHED_DATASET_STALE" in publication["reason_codes"]


def test_historical_rewrite_is_restored_byte_for_byte(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    source = fixture["published"] / "data/processed/AAA.csv"
    original = source.read_bytes()
    report = _run(
        fixture,
        provider=lambda *_args: _frame([("2026-07-13", 999), ("2026-07-14", 500)]),
    )
    row = report["tickers"][0]
    assert row["freshness_status"] == "failed"
    assert row["reason_code"] == "HISTORICAL_VALUE_REWRITE_BLOCKED"
    assert (fixture["stage"] / "data/processed/AAA.csv").read_bytes() == original


def test_history_truncation_detection_is_explicit() -> None:
    before = _frame([("2026-07-11", 100), ("2026-07-12", 101)])
    after = _frame([("2026-07-12", 101)])
    assert scheduled._historical_conflict(before, after) == "HISTORY_TRUNCATION_BLOCKED"


def test_one_ticker_failure_does_not_discard_another_valid_update(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )
    old_bbb = (fixture["published"] / "data/processed/BBB.csv").read_bytes()

    def provider(symbol: str, *_args: Any) -> pd.DataFrame:
        if symbol == "BBB":
            raise TimeoutError("slow")
        return _frame([("2026-07-14", 500)])

    report = _run(fixture, provider=provider)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}
    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["freshness_status"] == "failed"
    assert report["run_status"] == "degraded"
    assert report["publication"]["publication_set_valid"] is True
    assert report["publication"]["changed_price_file_count"] == 1
    assert report["publication"]["publication_required"] is False
    assert not (fixture["stage"] / scheduled.LATEST_MANIFEST).exists()
    assert (fixture["stage"] / "data/processed/BBB.csv").read_bytes() == old_bbb


def test_one_stale_ticker_blocks_another_valid_update_from_publication(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )

    def provider(symbol: str, *_args: Any) -> pd.DataFrame:
        if symbol == "BBB":
            return pd.DataFrame()
        return _frame([("2026-07-14", 500)])

    report = _run(fixture, provider=provider)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["freshness_status"] == "failed"
    assert by_ticker["BBB"]["reason_code"] == (
        "EXPECTED_SESSION_COVERAGE_INCOMPLETE"
    )
    assert report["run_status"] == "degraded"
    assert report["publication"]["publication_set_valid"] is True
    assert report["publication"]["changed_price_file_count"] == 1
    assert report["publication"]["publication_required"] is False
    assert not (fixture["stage"] / scheduled.LATEST_MANIFEST).exists()


def test_one_validation_failure_blocks_another_valid_update_from_publication(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("AAA"), _instrument("BBB")],
        end="2026-07-13",
    )

    def provider(symbol: str, *_args: Any) -> pd.DataFrame:
        if symbol == "BBB":
            return pd.DataFrame({"Date": ["2026-07-14"]})
        return _frame([("2026-07-14", 500)])

    report = _run(fixture, provider=provider)
    by_ticker = {row["ticker"]: row for row in report["tickers"]}

    assert by_ticker["AAA"]["freshness_status"] == "updated"
    assert by_ticker["BBB"]["freshness_status"] == "failed"
    assert by_ticker["BBB"]["reason_code"] == (
        "PROVIDER_PAYLOAD_SCHEMA_INVALID"
    )
    assert report["run_status"] == "degraded"
    assert report["publication"]["publication_set_valid"] is True
    assert report["publication"]["changed_price_file_count"] == 1
    assert report["publication"]["publication_required"] is False
    assert not (fixture["stage"] / scheduled.LATEST_MANIFEST).exists()


def test_atomic_writes_leave_no_temporary_files_and_manifest_is_stably_ordered(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        [_instrument("BBB"), _instrument("AAA")],
        end="2026-07-13",
    )
    report = _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    assert [row["ticker"] for row in report["tickers"]] == ["AAA", "BBB"]
    assert not list(fixture["stage"].rglob("*.tmp"))
    assert report["manifest_checksum"] == scheduled._manifest_checksum(report)


def test_published_file_checksums_validate_and_file_tampering_blocks(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    accepted = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert accepted["validated"] is True

    path = fixture["stage"] / "data/processed/AAA.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    blocked = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_FILE_CHECKSUM_MISMATCH" in blocked["reason_codes"]


def test_manifest_tampering_and_universe_mismatch_are_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_main_sha"] = "9" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_MANIFEST_CHECKSUM_MISMATCH" in tampered["reason_codes"]

    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_mismatch = scheduled.validate_published_dataset(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=RUN_AT,
        expected_source_main_sha=SOURCE_SHA,
    )
    assert "PUBLISHED_SOURCE_MAIN_SHA_MISMATCH" in source_mismatch["reason_codes"]

    other_universe = json.loads(fixture["universe"].read_text(encoding="utf-8"))
    other_universe["universe_version"] = "other-universe"
    other_path = tmp_path / "other-universe.json"
    other_path.write_text(json.dumps(other_universe), encoding="utf-8")
    mismatch = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=other_path, run_at=RUN_AT
    )
    assert "PUBLISHED_UNIVERSE_BINDING_MISMATCH" in mismatch["reason_codes"]


def test_manifest_identity_and_observation_must_reconcile_with_canonical_data(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    manifest_path = fixture["stage"] / scheduled.LATEST_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tickers"][0]["ticker"] = "IMPOSTOR"
    manifest["tickers"][0]["resulting_last_observation"] = "2099-01-01"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_TICKER_IDENTITY_MISMATCH" in result["reason_codes"]

    manifest["tickers"][0]["ticker"] = "AAA"
    manifest["manifest_checksum"] = scheduled._manifest_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_LAST_OBSERVATION_MISMATCH" in result["reason_codes"]


def test_publication_rejects_files_outside_the_exact_data_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    extra = fixture["stage"] / "manifests/extra.json"
    extra.write_text("{}\n", encoding="utf-8")

    result = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_DATA_BRANCH_CONTENT_INVALID" in result["reason_codes"]


def test_unbound_loose_price_file_is_not_authoritative(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    _write_history(fixture["stage"] / "data/processed/LOOSE.csv", end="2026-07-14", start_close=50)
    result = scheduled.validate_published_dataset(
        fixture["stage"], universe_snapshot_path=fixture["universe"], run_at=RUN_AT
    )
    assert "PUBLISHED_UNBOUND_PRICE_FILE_SET" in result["reason_codes"]


@pytest.mark.parametrize("content", [None, "{", "[]"])
def test_missing_or_malformed_manifest_returns_controlled_block(
    tmp_path: Path, content: str | None
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    if content is not None:
        path = root / scheduled.LATEST_MANIFEST
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")
    result = scheduled.validate_published_dataset(root, universe_snapshot_path=tmp_path / "none")
    assert result["validated"] is False
    assert result["reason_codes"] == ["PUBLISHED_MANIFEST_MISSING_OR_MALFORMED"]


def test_identical_inputs_produce_deterministic_report_content(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first", [_instrument("AAA")], end="2026-07-13")
    second = _fixture(tmp_path / "second", [_instrument("AAA")], end="2026-07-13")
    provider = lambda *_args: _frame([("2026-07-14", 500)])
    assert _run(first, provider=provider) == _run(second, provider=provider)


def test_analysis_consumption_runs_only_after_manifest_validation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    result = scheduled.run_validated_analysis(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=RUN_AT,
        analysis_runner=runner,
        analysis_kwargs={"run_id": "analysis-1"},
    )
    assert result["analysis_executed"] is True
    assert calls[0]["price_history_root"] == fixture["stage"] / "data/processed"
    assert [row["symbol"] for row in calls[0]["universe_snapshot"]["instruments"]] == [
        "AAA"
    ]


def test_analysis_consumption_blocks_stale_or_invalid_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    calls = 0

    def runner(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1

    result = scheduled.run_validated_analysis(
        fixture["stage"],
        universe_snapshot_path=fixture["universe"],
        run_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        analysis_runner=runner,
        analysis_kwargs={},
    )
    assert result["analysis_executed"] is False
    assert "PUBLISHED_DATASET_STALE" in result["validation"]["reason_codes"]
    assert calls == 0


def test_no_fundamental_operator_approval_is_generated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_instrument("AAA")], end="2026-07-13")
    report = _run(fixture, provider=lambda *_args: _frame([("2026-07-14", 500)]))
    assert report["fundamental_evidence"] == {
        "status": "not_evaluated",
        "reason_code": "NO_RELIABLE_AUTOMATED_FUNDAMENTAL_FRESHNESS_CONTRACT",
        "approval_required": False,
        "approval_generated": False,
    }
    assert not list(fixture["stage"].rglob("*approval*"))


def test_authoritative_repository_universe_has_952_mapped_instruments() -> None:
    universe = scheduled.load_authoritative_universe(scheduled.DEFAULT_UNIVERSE_SNAPSHOT)
    assert len(universe["instruments"]) == 952
    assert {row["source_mapping_status"] for row in universe["instruments"]} == {"mapped"}


def _fixture(
    root: Path,
    instruments: list[dict[str, Any]],
    *,
    end: str,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    published = root / "published"
    for index, instrument in enumerate(instruments):
        _write_history(
            published / "data/processed" / f"{instrument['source_symbol']}.csv",
            end=end,
            start_close=100 + index,
        )
    universe = root / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "schema_version": scheduled.UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
                "universe_version": "test-universe-v1",
                "instruments": instruments,
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
    }


def _run(
    fixture: dict[str, Path],
    *,
    provider: Any,
    run_at: datetime = RUN_AT,
    sleeper: Any = lambda _seconds: None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    return scheduled.run_scheduled_refresh(
        run_id="me-sr17-test-20260715T100000Z",
        source_main_sha=SOURCE_SHA,
        universe_snapshot_path=fixture["universe"],
        published_root=fixture["published"],
        staging_root=fixture["stage"],
        report_output=fixture["report"],
        run_at=run_at,
        workflow_run_id="123",
        provider=provider,
        max_attempts=max_attempts,
        sleeper=sleeper,
    )


def _write_history(path: Path, *, end: str, start_close: float) -> None:
    end_date = date.fromisoformat(end)
    rows = [
        ((end_date - timedelta(days=251 - index)).isoformat(), start_close + index)
        for index in range(252)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(rows).to_csv(path, index=False)


def _frame(
    rows: list[tuple[str, float]],
    *,
    high: float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": day,
                "Adj Close": close,
                "Close": close,
                "High": close + 1 if high is None else high,
                "Low": close - 1,
                "Open": close - 0.5,
                "Volume": 1000,
            }
            for day, close in rows
        ]
    )
