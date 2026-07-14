from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_engine.data import incremental_market_data_refresh as refresh_module
from market_engine.data.incremental_market_data_refresh import (
    determine_safe_cutoff_date,
    refresh_one_instrument,
    run_incremental_refresh,
)


def test_incremental_refresh_appends_missing_rows_and_replaces_overlap(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    _write_price_csv(history_path, _rows("2025-10-17", 260))
    original = history_path.read_text(encoding="utf-8")

    provider_calls: list[tuple[str, str, str]] = []

    def provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        provider_calls.append((symbol, start, end))
        return _frame(
            [
                ("2026-07-01", 999),
                ("2026-07-02", 1000),
                ("2026-07-03", 1001),
                ("2026-07-04", 1002),
                ("2026-07-05", 1003),
                ("2026-07-06", 1004),
                ("2026-07-07", 1005),
                ("2026-07-08", 1006),
                ("2026-07-09", 1007),
                ("2026-07-10", 1008),
            ]
        )

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=provider,
    )

    assert result["status"] == "incrementally_updated"
    assert result["requested_download_start"] == "2026-06-26"
    assert result["rows_added"] == 7
    assert result["rows_replaced_within_overlap"] == 3
    assert result["file_changed"] is True
    assert provider_calls == [("AAA", "2026-06-26", "2026-07-11")]
    refreshed = pd.read_csv(history_path)
    assert len(refreshed) == 267
    assert refreshed.iloc[-1]["Date"] == "2026-07-10"
    assert float(refreshed.loc[refreshed["Date"] == "2026-07-02", "Close"].iloc[0]) == 1000
    assert history_path.read_text(encoding="utf-8") != original


def test_already_current_history_does_not_call_provider_or_rewrite(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    _write_price_csv(history_path, _rows("2025-10-25", 260))
    original = history_path.read_text(encoding="utf-8")

    def provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        raise AssertionError("provider should not be called")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-11",
        overlap_calendar_days=7,
        provider=provider,
    )

    assert result["status"] == "already_current"
    assert result["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == original


def test_second_run_after_incremental_update_is_idempotent(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    _write_price_csv(history_path, _rows("2025-10-17", 260))

    first = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: _frame(_rows("2026-07-01", 10, first_close=900)),
    )
    after_first = history_path.read_text(encoding="utf-8")

    second = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: (_ for _ in ()).throw(AssertionError("provider should not be called")),
    )

    assert first["status"] == "incrementally_updated"
    assert second["status"] == "already_current"
    assert second["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == after_first


def test_download_failure_preserves_existing_history(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    _write_price_csv(history_path, _rows("2025-10-17", 260))
    original = history_path.read_text(encoding="utf-8")

    def provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        raise RuntimeError("network unavailable")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=provider,
    )

    assert result["status"] == "download_failed"
    assert result["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == original


def test_validation_failure_preserves_existing_history(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    _write_price_csv(history_path, _rows("2025-10-17", 260))
    original = history_path.read_text(encoding="utf-8")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: _frame([("2026-07-04", 100)], high=90, low=110),
    )

    assert result["status"] == "validation_failed"
    assert result["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == original


def test_provider_overlap_without_new_rows_marks_stale_after_update(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    existing_rows = _rows("2025-10-17", 260)
    _write_price_csv(history_path, existing_rows)
    original = history_path.read_text(encoding="utf-8")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: _frame(existing_rows[-6:]),
    )

    assert result["status"] == "stale_after_update"
    assert result["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == original


def test_missing_history_creates_new_snapshot(tmp_path: Path) -> None:
    instrument = _instrument("AAA")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: _frame(_rows("2025-10-24", 260)),
    )

    assert result["status"] == "new_snapshot_created"
    assert result["file_changed"] is True
    assert (tmp_path / "AAA.csv").exists()


def test_invalid_existing_history_uses_full_rebuild_path(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    (tmp_path / "AAA.csv").write_text("close\n1\n", encoding="utf-8")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: _frame(_rows("2025-10-24", 260)),
    )

    assert result["status"] == "full_rebuild_completed"
    assert result["requested_download_start"] == "2025-01-01"
    assert result["file_changed"] is True


def test_short_current_history_is_marked_insufficient_without_unnecessary_rewrite(tmp_path: Path) -> None:
    instrument = _instrument("AAA")
    history_path = tmp_path / "AAA.csv"
    rows = _rows("2026-07-01", 10)
    _write_price_csv(history_path, rows)
    original = history_path.read_text(encoding="utf-8")

    result = refresh_one_instrument(
        instrument,
        price_history_root=tmp_path,
        cutoff_date="2026-07-10",
        overlap_calendar_days=7,
        provider=lambda _symbol, _start, _end: (_ for _ in ()).throw(AssertionError("provider should not be called")),
    )

    assert result["status"] == "insufficient_history"
    assert result["file_changed"] is False
    assert history_path.read_text(encoding="utf-8") == original


def test_safe_cutoff_skips_weekends() -> None:
    assert determine_safe_cutoff_date(today=date(2026, 7, 13)) == "2026-07-10"
    assert determine_safe_cutoff_date(today=date(2026, 7, 14)) == "2026-07-13"


def test_run_invokes_evaluation_before_and_after_refresh(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        refresh_module,
        "build_universe_snapshot",
        lambda _path, *, price_history_root: {
            "universe_version": "test-universe-v1",
            "instruments": [_instrument("AAA")],
        },
    )

    def fake_coverage(*, run_id: str, **_kwargs: Any) -> tuple[dict[str, Any], Path]:
        calls.append(run_id)
        return (
            {
                "coverage_summary": {
                    "summary": {
                        "total_canonical_instruments": 1,
                        "valid": 1,
                        "insufficient": 0,
                        "missing": 0,
                        "invalid": 0,
                        "unsupported": 0,
                    }
                }
            },
            tmp_path / "artifacts" / run_id,
        )

    def fake_refresh(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("refresh")
        return {
            "schema_version": "test-refresh-v1",
            "summary": {
                "histories_checked": 1,
                "already_current": 1,
                "incrementally_updated": 0,
                "new_snapshot_created": 0,
                "full_rebuild_required": 0,
                "full_rebuild_completed": 0,
                "download_failed": 0,
                "empty_provider_response": 0,
                "merge_failed": 0,
                "validation_failed": 0,
                "stale_after_update": 0,
                "insufficient_history": 0,
                "unsupported_mapping": 0,
                "rows_downloaded": 0,
                "rows_added": 0,
                "rows_replaced_within_overlap": 0,
                "files_rewritten": 0,
                "files_unchanged": 1,
            },
            "per_ticker_status": [{"status": "already_current"}],
        }

    def fake_evaluation(_artifact: Path, *, run_id: str, **_kwargs: Any) -> tuple[dict[str, Any], Path]:
        calls.append(run_id)
        return (
            {
                "refresh_index": {
                    "summary": {
                        "selected_outcomes": 1,
                        "resolved": 0,
                        "still_unresolved": 1,
                        "blocker_counts": {"insufficient_forward_data": 1},
                    }
                }
            },
            tmp_path / "eval" / run_id,
        )

    monkeypatch.setattr(refresh_module, "build_data_run", fake_coverage)
    monkeypatch.setattr(refresh_module, "refresh_price_histories", fake_refresh)
    monkeypatch.setattr(refresh_module, "run_advice_outcome_refresh", fake_evaluation)

    artifacts, _run_dir = run_incremental_refresh(
        run_id="me-data05-test-20260713T140000Z",
        universe_path=tmp_path / "universe.json",
        price_history_root=tmp_path / "prices",
        artifact_root=tmp_path / "artifacts",
        evaluation_artifact=tmp_path / "evaluation.json",
        evaluation_output_root=tmp_path / "eval",
        cutoff_date="2026-07-10",
    )

    assert calls == [
        "me-data05-test-20260713T140000Z-coverage-before",
        "me-data05-test-20260713T140000Z-evaluation-before",
        "refresh",
        "me-data05-test-20260713T140000Z-coverage-after",
        "me-data05-test-20260713T140000Z-evaluation-after",
    ]
    assert artifacts["before_after_comparison"]["newly_resolved"] == 0


def test_refresh_universe_request_fails_closed_before_provider_or_manifest(tmp_path: Path) -> None:
    def provider(_symbol: str, _start: str, _end: str) -> pd.DataFrame:
        raise AssertionError("provider should not be called")

    with pytest.raises(ValueError, match="universe refresh requested"):
        run_incremental_refresh(
            run_id="me-data05-test-universe-refresh-20260713T140000Z",
            universe_path=tmp_path / "universe.json",
            price_history_root=tmp_path / "prices",
            artifact_root=tmp_path / "artifacts",
            evaluation_artifact=tmp_path / "evaluation.json",
            evaluation_output_root=tmp_path / "eval",
            cutoff_date="2026-07-10",
            refresh_universe=True,
            provider=provider,
        )

    assert not (tmp_path / "artifacts" / "me-data05-test-universe-refresh-20260713T140000Z").exists()


def test_compact_artifact_payloads_store_full_ticker_list_once(tmp_path: Path) -> None:
    rows = [
        {"instrument_id": "equity:aaa", "symbol": "AAA", "status": "already_current"},
        {"instrument_id": "equity:bbb", "symbol": "BBB", "status": "incrementally_updated"},
        {"instrument_id": "equity:ccc", "symbol": "CCC", "status": "new_snapshot_created"},
        {"instrument_id": "equity:ddd", "symbol": "DDD", "status": "full_rebuild_completed"},
        {"instrument_id": "equity:eee", "symbol": "EEE", "status": "stale_after_update"},
    ]
    artifacts = refresh_module._artifact_payloads(
        manifest=_manifest(tmp_path),
        refresh={
            "schema_version": "market-engine-data05-price-refresh-summary-v1",
            "summary": {
                "histories_checked": 5,
                "already_current": 1,
                "incrementally_updated": 1,
                "new_snapshot_created": 1,
                "full_rebuild_required": 0,
                "full_rebuild_completed": 1,
                "download_failed": 0,
                "empty_provider_response": 0,
                "merge_failed": 0,
                "validation_failed": 0,
                "stale_after_update": 1,
                "insufficient_history": 0,
                "unsupported_mapping": 0,
                "rows_downloaded": 10,
                "rows_added": 2,
                "rows_replaced_within_overlap": 1,
                "files_rewritten": 3,
                "files_unchanged": 2,
            },
            "per_ticker_status": rows,
        },
        coverage_before=_coverage(valid=4, insufficient=1),
        coverage_after=_coverage(valid=4, insufficient=1),
        evaluation_before=_evaluation(resolved=0),
        evaluation_after=_evaluation(resolved=0),
        acceptance={"schema_version": "test-acceptance-v1", "status": "incremental_refresh_operational", "checks": {}},
    )

    assert set(artifacts["refresh_summary"]) == {"schema_version", "summary"}
    assert "per_ticker_status" not in artifacts["refresh_summary"]
    assert "entries" not in artifacts["refresh_summary"]
    assert len(artifacts["per_ticker_status"]["entries"]) == 5
    assert "already_current" not in artifacts
    assert {row["status"] for row in artifacts["incremental_updates"]["entries"]} == {
        "incrementally_updated",
        "new_snapshot_created",
        "full_rebuild_completed",
    }
    assert [row["status"] for row in artifacts["failed_updates"]["entries"]] == ["stale_after_update"]
    assert artifacts["acceptance_result"]["status"] == "incremental_refresh_operational"
    assert "Histories checked" in artifacts["report"]

    output_dir = tmp_path / "run"
    refresh_module._write_refresh_artifacts(output_dir, artifacts)
    written = sorted(path.name for path in output_dir.iterdir())
    assert written == [
        "acceptance_result.json",
        "before_after_comparison.json",
        "coverage_after.json",
        "coverage_before.json",
        "evaluation_after.json",
        "evaluation_before.json",
        "failed_updates.json",
        "full_rebuilds.json",
        "incremental_updates.json",
        "manifest.json",
        "new_snapshots.json",
        "per_ticker_status.json",
        "refresh_summary.json",
        "report.md",
        "validation_summary.json",
    ]


def _instrument(symbol: str) -> dict[str, Any]:
    return {
        "instrument_id": f"equity:{symbol}",
        "symbol": symbol,
        "source_symbol": symbol,
        "source_mapping_status": "mapped",
    }


def _rows(start: str, count: int, *, first_close: float = 100.0) -> list[tuple[str, float]]:
    cursor = date.fromisoformat(start)
    return [
        ((cursor + timedelta(days=index)).isoformat(), first_close + index)
        for index in range(count)
    ]


def _frame(rows: list[tuple[str, float]], *, high: float | None = None, low: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": day,
                "Adj Close": close,
                "Close": close,
                "High": close + 1 if high is None else high,
                "Low": close - 1 if low is None else low,
                "Open": close - 0.5,
                "Volume": 1000,
            }
            for day, close in rows
        ]
    )


def _write_price_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(rows).to_csv(path, index=False)


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return {
        "run_id": "me-data05-test-20260713T140000Z",
        "cutoff_date": "2026-07-10",
        "overlap_policy": {"overlap_calendar_days": 7},
        "universe_refresh_requested": False,
        "universe_refresh_performed": False,
        "universe_refresh_status": "not_requested",
        "coverage_before_artifact": (tmp_path / "run" / "coverage_before.json").as_posix(),
        "coverage_after_artifact": (tmp_path / "run" / "coverage_after.json").as_posix(),
    }


def _coverage(*, valid: int, insufficient: int) -> dict[str, Any]:
    return {
        "coverage_summary": {
            "summary": {
                "total_canonical_instruments": valid + insufficient,
                "valid": valid,
                "insufficient": insufficient,
                "missing": 0,
                "invalid": 0,
                "unsupported": 0,
            }
        }
    }


def _evaluation(*, resolved: int) -> dict[str, Any]:
    return {
        "refresh_index": {
            "summary": {
                "selected_outcomes": 1,
                "resolved": resolved,
                "still_unresolved": 1 - resolved,
                "blocker_counts": {"insufficient_forward_data": 1 - resolved},
            }
        }
    }
