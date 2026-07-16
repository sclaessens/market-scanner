from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from market_engine.data import fundamental_evidence_coverage as data06
from market_engine.source_refresh.sec_companyfacts_snapshots import persist_sec_companyfacts_raw_snapshot


def test_manual_inventory_and_complete_partial_missing_classification(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(
        tmp_path,
        [
            _manual("AAA"),
            _manual("BBB", eps_growth_yoy=""),
        ],
    )
    artifacts, output_dir = _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)
    summary = artifacts["fundamental_coverage_summary"]

    assert summary["overall_counts"]["complete"] == 1
    assert summary["overall_counts"]["partial"] == 1
    assert summary["overall_counts"]["missing"] == 1
    assert (output_dir / "normalized_fundamental_quality.csv").exists()


def test_canonical_ticker_matching_rejects_unknown_manual_rows(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA"), _manual("ZZZ")])
    artifacts, _ = _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)
    inventory = artifacts["evidence_source_inventory"]["sources"]
    manual = next(row for row in inventory if row["evidence_family"] == "manual_mvp_fundamentals")

    assert manual["tickers_found"] == 2
    assert manual["canonical_matches"] == 1
    assert manual["unsupported_tickers"] == ["ZZZ"]


def test_duplicate_same_priority_conflicts_fail_closed(tmp_path: Path) -> None:
    rows = [_manual("AAA", revenue_growth_yoy="0.10"), _manual("AAA", revenue_growth_yoy="0.20")]
    source = data06._load_manual_fundamentals(_write_raw_fundamentals(tmp_path, rows), data06._parse_date("2026-07-10", field_name="as_of_date"))
    status = data06._classify_ticker(_instrument("AAA"), source["candidates"], parsed_as_of=data06._parse_date("2026-07-10", field_name="as_of_date"), generated_at="2026-07-10T00:00:00Z")

    assert status["overall_fundamental_status"] == "conflicting"
    assert status["fundamental_quality_row"]["quality_metadata_status"] == "invalid"


def test_identical_duplicate_rows_are_deterministic(tmp_path: Path) -> None:
    rows = [_manual("AAA"), _manual("AAA")]
    source = data06._load_manual_fundamentals(_write_raw_fundamentals(tmp_path, rows), data06._parse_date("2026-07-10", field_name="as_of_date"))
    status = data06._classify_ticker(_instrument("AAA"), source["candidates"], parsed_as_of=data06._parse_date("2026-07-10", field_name="as_of_date"), generated_at="2026-07-10T00:00:00Z")

    assert status["overall_fundamental_status"] == "complete"


def test_invalid_future_and_stale_dates_are_classified(tmp_path: Path) -> None:
    source = data06._load_manual_fundamentals(
        _write_raw_fundamentals(
            tmp_path,
            [
                _manual("AAA", source_freshness_date="not-a-date"),
                _manual("BBB", source_freshness_date="2026-07-11"),
                _manual("CCC", source_freshness_date="2025-01-01"),
            ],
        ),
        data06._parse_date("2026-07-10", field_name="as_of_date"),
    )
    by_ticker = {row["ticker"]: row for row in source["candidates"]}

    assert by_ticker["AAA"]["coverage_status"] == "invalid"
    assert by_ticker["BBB"]["coverage_status"] == "invalid"
    assert by_ticker["CCC"]["coverage_status"] == "stale"


def test_missing_provenance_and_unsupported_schema_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,source_name\nAAA,manual\n", encoding="utf-8")

    loaded = data06._load_manual_fundamentals(bad, data06._parse_date("2026-07-10", field_name="as_of_date"))

    assert loaded["status"] == "unsupported_schema"
    assert "source_reference" in loaded["missing_columns"]


def test_malformed_sec_payload_is_inventory_error(tmp_path: Path) -> None:
    root = tmp_path / "sec"
    bad = root / "run" / "raw" / "AAA_companyfacts.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not-json", encoding="utf-8")

    loaded = data06._load_sec_companyfacts_sources(root, data06._parse_date("2026-07-10", field_name="as_of_date"))

    assert loaded["errors"]
    assert loaded["candidates"] == []


def test_sec_companyfacts_source_context_is_partial_and_preserves_units(tmp_path: Path) -> None:
    root = tmp_path / "sec"
    persist_sec_companyfacts_raw_snapshot(
        raw_payload=_companyfacts_payload(),
        ticker="AAA",
        cik="0000000001",
        run_id="sec-run",
        fetched_at="2026-06-15T00:00:00Z",
        root_dir=root,
    )
    loaded = data06._load_sec_companyfacts_sources(root, data06._parse_date("2026-07-10", field_name="as_of_date"))
    row = loaded["candidates"][0]

    assert row["coverage_status"] == "partial"
    assert row["units"]["revenue"] == "USD"
    assert row["metrics"]["revenue"] == 100
    assert row["missing_metrics"] == list(data06.MVP_METRIC_FIELDS)


def test_null_numeric_values_are_not_converted_to_zero(tmp_path: Path) -> None:
    source = data06._load_manual_fundamentals(_write_raw_fundamentals(tmp_path, [_manual("AAA", gross_margin="")]), data06._parse_date("2026-07-10", field_name="as_of_date"))
    row = source["candidates"][0]

    assert row["metrics"]["gross_margin"] is None
    assert "gross_margin" in row["missing_metrics"]


def test_source_priority_selects_manual_over_sec(tmp_path: Path) -> None:
    manual = data06._manual_row_to_candidate(_manual("AAA"), tmp_path / "raw.csv", data06._parse_date("2026-07-10", field_name="as_of_date"))
    sec = {**manual, "source_family": "sec_companyfacts_source_context", "priority": 60, "coverage_status": "partial"}

    selected, conflict = data06._select_candidate([sec, manual])

    assert not conflict
    assert selected["source_family"] == "manual_mvp_fundamentals"


def test_stable_ordering_and_deterministic_output(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("BBB"), _manual("AAA")])
    first, _ = _run_small(tmp_path / "first", monkeypatch, raw_fundamentals_path=raw)
    second, _ = _run_small(tmp_path / "second", monkeypatch, raw_fundamentals_path=raw)

    assert [row["ticker"] for row in first["per_ticker_fundamental_status"]["tickers"]] == ["AAA", "BBB", "CCC"]
    assert first["fundamental_coverage_summary"]["overall_counts"] == second["fundamental_coverage_summary"]["overall_counts"]


def test_run_writes_compact_nonduplicative_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA")])
    _, output_dir = _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)
    names = {path.name for path in output_dir.iterdir()}

    assert {
        "manifest.json",
        "evidence_source_inventory.json",
        "fundamental_coverage_summary.json",
        "per_ticker_fundamental_status.json",
        "missing_fundamental_evidence.json",
        "partial_fundamental_evidence.json",
        "invalid_or_stale_evidence.json",
        "coverage_report.md",
        "normalized_fundamental_quality.csv",
    }.issubset(names)


def test_downstream_me_run31_consumes_normalized_csv(tmp_path: Path, monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_run31(**kwargs: Any) -> tuple[dict[str, Any], Path]:
        seen.update(kwargs)
        return _fake_run31_artifacts(kwargs["run_id"]), tmp_path / "run31"

    fake_run31._data06_test_patch = True  # type: ignore[attr-defined]
    monkeypatch.setattr(data06, "run_broad_non_price_evidence_advice_readiness", fake_run31)
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA")])
    _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)

    assert Path(seen["fundamental_evidence_path"]).name == "normalized_fundamental_quality.csv"


def test_cli_fails_closed_on_invalid_input(tmp_path: Path) -> None:
    rc = data06.run_command(["--run-id", "run", "--as-of-date", "bad-date"], stdout=_Sink(), stderr=_Sink())

    assert rc == 1


def test_report_has_no_recommendation_or_allocation_authority(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA")])
    artifacts, _ = _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)
    manifest = artifacts["manifest"]

    assert manifest["side_effects"]["allocation_performed"] is False
    assert manifest["side_effects"]["decision_engine_authority_changed"] is False
    assert manifest["side_effects"]["recommendation_rules_changed"] is False


def _run_small(tmp_path: Path, monkeypatch: Any, *, raw_fundamentals_path: Path) -> tuple[dict[str, Any], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data06, "build_universe_snapshot", lambda *args, **kwargs: _universe())
    if not getattr(data06.run_broad_non_price_evidence_advice_readiness, "_data06_test_patch", False):
        fake = lambda **kwargs: (_fake_run31_artifacts(kwargs["run_id"]), tmp_path / "run31")
        fake._data06_test_patch = True  # type: ignore[attr-defined]
        monkeypatch.setattr(data06, "run_broad_non_price_evidence_advice_readiness", fake)
    existing = _write_existing(tmp_path)
    return data06.run_fundamental_evidence_coverage(
        run_id="me-data06-test",
        output_root=tmp_path / "artifacts",
        existing_fundamental_path=existing,
        raw_fundamentals_path=raw_fundamentals_path,
        intake_fundamentals_path=tmp_path / "missing_intake.csv",
        sec_companyfacts_root=tmp_path / "missing_sec",
        company_profile_root=tmp_path / "missing_profiles",
        run31_output_root=tmp_path / "run31_runs",
        run31_compact_evidence_root=tmp_path / "run31_evidence",
        allow_overwrite=True,
    )


def _write_raw_fundamentals(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "raw_fundamentals.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "as_of_date", "source_name", "source_reference", "source_freshness_date", "currency", *data06.MVP_METRIC_FIELDS, "fundamental_notes"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_existing(tmp_path: Path) -> Path:
    path = tmp_path / "existing.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=data06.FUNDAMENTAL_QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "ticker": "CCC",
                "date": "2026-07-10",
                "quality_state": "INSUFFICIENT_DATA",
                "quality_metadata_status": "row_missing",
                "source_data_status": "row_missing",
                "generated_at": "2026-07-10T00:00:00Z",
            }
        )
    return path


def _manual(ticker: str, **overrides: str) -> dict[str, str]:
    row = {
        "ticker": ticker,
        "as_of_date": "2026-05-24",
        "source_name": "manual_official_annual_report_standard_batch_1",
        "source_reference": "local:test",
        "source_freshness_date": "2026-05-24",
        "currency": "USD",
        "revenue_growth_yoy": "0.10",
        "eps_growth_yoy": "0.20",
        "gross_margin": "0.30",
        "operating_margin": "0.15",
        "debt_to_equity": "0.40",
        "fundamental_notes": "fixture",
    }
    row.update(overrides)
    return row


def _instrument(ticker: str) -> dict[str, Any]:
    return {"instrument_id": f"equity:{ticker}", "symbol": ticker, "asset_type": "equity", "name": ticker}


def _universe() -> dict[str, Any]:
    return {
        "universe_version": "test-universe-v1",
        "snapshot_date": "2026-07-10",
        "instruments": [_instrument("AAA"), _instrument("BBB"), _instrument("CCC")],
    }


def _fake_run31_artifacts(run_id: str) -> dict[str, Any]:
    return {
        "manifest": {"run_id": run_id},
        "compact_evidence_dir": "artifacts/run31/compact",
        "evidence_coverage_summary": {
            "summary": {
                "attempted_instruments": 3,
                "canonical_advice_input_ready": 1,
                "full_advice_ready": 0,
                "advice_engine_completed": 3,
                "advice_counts": {"unable_to_advise": 2},
            }
        },
        "evidence_coverage_index": {"instruments": [{"symbol": "AAA", "canonical_advice_input_ready": True}]},
    }


def _companyfacts_payload() -> dict[str, Any]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(100)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(10)]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact(12)]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fact(3)]}},
            }
        }
    }


def _fact(value: int) -> dict[str, Any]:
    return {
        "val": value,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "accn": "0000000000-26-000001",
        "frame": "CY2025",
    }


class _Sink:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        return None
