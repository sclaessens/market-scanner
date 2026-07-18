from __future__ import annotations

import csv
import json
from collections import Counter
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
        "per_ticker_fundamental_transitions.json",
    }.issubset(names)


def test_downstream_me_run31_consumes_normalized_csv(tmp_path: Path, monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_run31(**kwargs: Any) -> tuple[dict[str, Any], Path]:
        seen.update(kwargs)
        return _fake_run31_artifacts(kwargs["run_id"], Path(kwargs["fundamental_evidence_path"])), tmp_path / "run31"

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


def test_explicit_baseline_path_and_checksum_are_recorded(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA")])
    artifacts, _ = _run_small(tmp_path, monkeypatch, raw_fundamentals_path=raw)
    manifest = artifacts["manifest"]

    assert manifest["input_paths"]["baseline_run_evidence"].endswith("/baseline")
    assert manifest["baseline_artifact_path"].endswith("baseline/evidence_coverage_index.json")
    assert manifest["input_checksums"][manifest["baseline_artifact_path"]] == manifest["baseline_checksum"]


def test_default_baseline_is_used_when_present(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _write_raw_fundamentals(tmp_path, [_manual("AAA")])
    artifacts, _ = _run_small(
        tmp_path,
        monkeypatch,
        raw_fundamentals_path=raw,
        use_default_baseline=True,
    )

    assert artifacts["manifest"]["baseline_run_id"] == "baseline-review-fixture"


def test_missing_baseline_path_fails_closed() -> None:
    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="does not exist"):
        data06._load_before_baseline(
            "missing-baseline",
            canonical_symbols=["AAA"],
            canonical_universe_version="test-universe-v1",
        )


def test_malformed_baseline_json_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "evidence_coverage_index.json").write_text("{bad", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="not parseable"):
        data06._load_before_baseline(
            root,
            canonical_symbols=["AAA"],
            canonical_universe_version="test-universe-v1",
        )


def test_unsupported_baseline_artifact_type_fails_closed(tmp_path: Path) -> None:
    root = _write_baseline(tmp_path / "baseline", {"AAA": "missing"}, artifact_type="unsupported")

    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="artifact type"):
        _load_test_baseline(root, ["AAA"])


def test_baseline_universe_size_and_version_mismatches_fail_closed(tmp_path: Path) -> None:
    size_root = _write_baseline(tmp_path / "size", {"AAA": "missing"})
    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="size mismatch"):
        _load_test_baseline(size_root, ["AAA", "BBB"])

    version_root = _write_baseline(
        tmp_path / "version",
        {"AAA": "missing"},
        universe_version="different-universe",
    )
    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="version mismatch"):
        _load_test_baseline(version_root, ["AAA"])


def test_duplicate_baseline_tickers_fail_closed(tmp_path: Path) -> None:
    root = _write_baseline(tmp_path / "baseline", {"AAA": "missing", "BBB": "missing"})
    index_path = root / "evidence_coverage_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["instruments"][1]["symbol"] = "AAA"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="duplicate baseline ticker"):
        _load_test_baseline(root, ["AAA", "BBB"])


def test_baseline_count_per_ticker_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _write_baseline(tmp_path / "baseline", {"AAA": "missing"})
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["coverage_summary"]["fundamental_counts"]["missing"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(data06.FundamentalEvidenceCoverageValidationError, match="count mismatch"):
        _load_test_baseline(root, ["AAA"])


def test_readiness_transitions_are_derived_from_baseline_and_sorted() -> None:
    comparison, _ = _comparison_fixture(
        before_statuses={"AAA": "complete", "BBB": "missing", "CCC": "missing"},
        after_statuses={"AAA": "complete", "BBB": "complete", "CCC": "complete"},
        before_ready={"AAA"},
        after_ready={"BBB", "CCC"},
        before_full={"AAA"},
        after_full={"BBB", "CCC"},
    )

    assert comparison["improvement_tickers"]["newly_advice_input_ready"] == ["BBB", "CCC"]
    assert comparison["regression_tickers"]["lost_advice_input_readiness"] == ["AAA"]
    assert comparison["improvement_tickers"]["newly_full_advice_ready"] == ["BBB", "CCC"]
    assert comparison["regression_tickers"]["lost_full_advice_readiness"] == ["AAA"]


@pytest.mark.parametrize(
    ("before_status", "after_status", "expected"),
    [
        ("complete", "partial", "complete_to_partial"),
        ("complete", "missing", "complete_to_missing"),
        ("partial", "missing", "partial_to_missing"),
        ("complete", "stale", "became_stale"),
        ("complete", "invalid", "became_invalid"),
        ("complete", "conflicting", "became_conflicting"),
    ],
)
def test_fundamental_regression_categories(before_status: str, after_status: str, expected: str) -> None:
    comparison, detail = _comparison_fixture(
        before_statuses={"AAA": before_status},
        after_statuses={"AAA": after_status},
    )

    assert comparison["run_status"] == data06.RUN_STATUS_COMPLETED_WITH_REGRESSIONS
    assert comparison["regression_tickers"][expected] == ["AAA"]
    assert expected in detail["tickers"][0]["transition_types"]


def test_no_regressions_and_transition_reconciliation() -> None:
    comparison, detail = _comparison_fixture(
        before_statuses={"AAA": "missing", "BBB": "partial"},
        after_statuses={"AAA": "partial", "BBB": "complete"},
        after_ready={"BBB"},
    )

    assert comparison["run_status"] == data06.RUN_STATUS_COMPLETED
    assert comparison["regression_ticker_count"] == 0
    assert comparison["improvement_ticker_count"] == 2
    assert comparison["transition_reconciliation"] == {
        "ticker_count": 2,
        "before_unique_tickers": 2,
        "after_unique_tickers": 2,
        "counts_reconciled": True,
    }
    assert detail["ticker_count"] == 2


def test_multiple_transition_types_for_one_ticker() -> None:
    comparison, detail = _comparison_fixture(
        before_statuses={"AAA": "complete"},
        after_statuses={"AAA": "stale"},
        before_ready={"AAA"},
        before_full={"AAA"},
    )
    transitions = detail["tickers"][0]["transition_types"]

    assert transitions == [
        "complete_to_stale",
        "became_stale",
        "lost_advice_input_readiness",
        "lost_full_advice_readiness",
    ]
    assert comparison["regression_ticker_count"] == 1


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["2026-07-01", "2026-06-01"], "current"),
        (["2025-01-01", "2025-02-01"], "stale"),
        (["2026-07-01", "2025-01-01"], "mixed"),
        (["bad-date"], "invalid"),
        (["2026-07-11"], "invalid"),
        ([""], "unknown"),
    ],
)
def test_inventory_freshness_classification(dates: list[str], expected: str) -> None:
    result = data06._classify_inventory_freshness(
        [{"source_date": value} for value in dates],
        as_of=data06._parse_date("2026-07-10", field_name="as_of_date"),
        date_fields=("source_date",),
    )

    assert result["freshness_status"] == expected


def test_inventory_freshness_not_assessed_and_rejected_missing_date() -> None:
    as_of = data06._parse_date("2026-07-10", field_name="as_of_date")
    not_assessed = data06._classify_inventory_freshness(
        [{}], as_of=as_of, date_fields=(), applicable=False
    )
    rejected = data06._classify_inventory_freshness(
        [{}], as_of=as_of, date_fields=("fundamentals_last_updated",)
    )

    assert not_assessed["freshness_status"] == "not_assessed"
    assert rejected["freshness_status"] == "unknown"
    assert rejected["freshness_status"] != "current"


def test_sec_inventory_uses_each_actual_record_date() -> None:
    result = data06._classify_inventory_freshness(
        [{"source_date": "2026-06-19"}, {"source_date": "2025-01-01"}],
        as_of=data06._parse_date("2026-07-10", field_name="as_of_date"),
        date_fields=("source_date",),
    )

    assert result["freshness_status"] == "mixed"
    assert result["freshness_counts"] == {"current": 1, "stale": 1}


def _run_small(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    raw_fundamentals_path: Path,
    use_default_baseline: bool = False,
) -> tuple[dict[str, Any], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data06, "build_universe_snapshot", lambda *args, **kwargs: _universe())
    if not getattr(data06.run_broad_non_price_evidence_advice_readiness, "_data06_test_patch", False):
        fake = lambda **kwargs: (
            _fake_run31_artifacts(kwargs["run_id"], Path(kwargs["fundamental_evidence_path"])),
            tmp_path / "run31",
        )
        fake._data06_test_patch = True  # type: ignore[attr-defined]
        monkeypatch.setattr(data06, "run_broad_non_price_evidence_advice_readiness", fake)
    existing = _write_existing(tmp_path)
    baseline = _write_baseline(tmp_path / "baseline", {ticker: "missing" for ticker in ("AAA", "BBB", "CCC")})
    if use_default_baseline:
        monkeypatch.setattr(data06, "DEFAULT_BASELINE_RUN_EVIDENCE", baseline)
    return data06.run_fundamental_evidence_coverage(
        run_id="me-data06-test",
        output_root=tmp_path / "artifacts",
        existing_fundamental_path=existing,
        raw_fundamentals_path=raw_fundamentals_path,
        intake_fundamentals_path=tmp_path / "missing_intake.csv",
        sec_companyfacts_root=tmp_path / "missing_sec",
        company_profile_root=tmp_path / "missing_profiles",
        baseline_run_evidence=None if use_default_baseline else baseline,
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


def _fake_run31_artifacts(run_id: str, fundamental_evidence_path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(fundamental_evidence_path.open(newline="", encoding="utf-8")))
    instruments = []
    for row in rows:
        metadata_status = row["quality_metadata_status"]
        status = {
            "complete": "available",
            "partial": "partial",
            "stale": "stale",
            "invalid": "invalid",
            "row_missing": "missing",
        }[metadata_status]
        ready = status == "available"
        instruments.append(
            {
                "symbol": row["ticker"],
                "canonical_advice_input_ready": ready,
                "full_advice_ready": False,
                "canonical_advice_label": "wait_for_price" if ready else "unable_to_advise",
                "fundamental_context": {
                    "status": status,
                    "blockers": [row["invalid_data_reason"]] if row["invalid_data_reason"] else [],
                },
            }
        )
    ready_count = sum(1 for row in instruments if row["canonical_advice_input_ready"])
    fundamental_counts = Counter(row["fundamental_context"]["status"] for row in instruments)
    return {
        "manifest": {"run_id": run_id},
        "compact_evidence_dir": "artifacts/run31/compact",
        "evidence_coverage_summary": {
            "summary": {
                "attempted_instruments": 3,
                "canonical_instruments": 3,
                "canonical_advice_input_ready": ready_count,
                "full_advice_ready": 0,
                "advice_engine_completed": 3,
                "advice_counts": {"unable_to_advise": 3 - ready_count},
                "fundamental_counts": {
                    key: fundamental_counts.get(key, 0)
                    for key in ("available", "partial", "missing", "stale", "invalid", "blocked", "not_applicable")
                },
            }
        },
        "evidence_coverage_index": {
            "schema_version": data06.BASELINE_INDEX_SCHEMA_VERSION,
            "run_id": run_id,
            "instruments": instruments,
        },
    }


def _write_baseline(
    root: Path,
    statuses: dict[str, str],
    *,
    advice_ready: set[str] | None = None,
    full_ready: set[str] | None = None,
    universe_version: str = "test-universe-v1",
    artifact_type: str = data06.BASELINE_MANIFEST_ARTIFACT_TYPE,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    advice_ready = advice_ready or set()
    full_ready = full_ready or set()
    raw_status = {
        "complete": "available",
        "partial": "partial",
        "missing": "missing",
        "stale": "stale",
        "invalid": "invalid",
        "conflicting": "invalid",
    }
    instruments = []
    for ticker, status in statuses.items():
        blockers = ["duplicate_fundamental_rows_conflict"] if status == "conflicting" else []
        instruments.append(
            {
                "symbol": ticker,
                "canonical_advice_input_ready": ticker in advice_ready,
                "full_advice_ready": ticker in full_ready,
                "canonical_advice_label": "wait_for_price" if ticker in advice_ready else "unable_to_advise",
                "fundamental_context": {"status": raw_status[status], "blockers": blockers},
            }
        )
    counts = Counter(item["fundamental_context"]["status"] for item in instruments)
    run_id = "baseline-review-fixture"
    manifest = {
        "artifact_type": artifact_type,
        "schema_version": data06.BASELINE_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "canonical_universe_version": universe_version,
        "input_artifacts": {"fundamental_evidence_path": "data/processed/fundamental_quality.csv"},
        "coverage_summary": {
            "canonical_instruments": len(instruments),
            "attempted_instruments": len(instruments),
            "canonical_advice_input_ready": len(advice_ready),
            "full_advice_ready": len(full_ready),
            "fundamental_counts": {
                key: counts.get(key, 0)
                for key in ("available", "partial", "missing", "stale", "invalid", "blocked", "not_applicable")
            },
            "advice_counts": {"unable_to_advise": len(instruments) - len(advice_ready)},
        },
    }
    index = {
        "schema_version": data06.BASELINE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "instruments": instruments,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "evidence_coverage_index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def _load_test_baseline(root: Path, tickers: list[str]) -> dict[str, Any]:
    return data06._load_before_baseline(
        root,
        canonical_symbols=tickers,
        canonical_universe_version="test-universe-v1",
    )


def _comparison_fixture(
    *,
    before_statuses: dict[str, str],
    after_statuses: dict[str, str],
    before_ready: set[str] | None = None,
    after_ready: set[str] | None = None,
    before_full: set[str] | None = None,
    after_full: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_ready = before_ready or set()
    after_ready = after_ready or set()
    before_full = before_full or set()
    after_full = after_full or set()
    tickers = sorted(before_statuses)
    assert tickers == sorted(after_statuses)
    before_records = [
        {
            "ticker": ticker,
            "fundamental_status": before_statuses[ticker],
            "advice_input_ready": ticker in before_ready,
            "full_advice_ready": ticker in before_full,
            "unable_to_advise": ticker not in before_ready,
        }
        for ticker in tickers
    ]
    before_counts = Counter(before_statuses.values())
    baseline = {
        "run_id": "baseline-review-fixture",
        "artifact_path": "baseline/evidence_coverage_index.json",
        "artifact_type": data06.BASELINE_INDEX_ARTIFACT_TYPE,
        "schema_version": data06.BASELINE_INDEX_SCHEMA_VERSION,
        "checksum": "fixture-checksum",
        "canonical_universe_version": "test-universe-v1",
        "canonical_universe_size": len(tickers),
        "records": before_records,
        "counts": {
            "fundamental_complete": before_counts.get("complete", 0),
            "fundamental_partial": before_counts.get("partial", 0),
            "fundamental_missing": before_counts.get("missing", 0),
            "invalid_stale_conflicting": sum(
                before_counts.get(status, 0) for status in ("invalid", "stale", "conflicting")
            ),
            "canonical_advice_input_ready": len(before_ready),
            "full_advice_ready": len(before_full),
            "unable_to_advise": len(tickers) - len(before_ready),
            "source_run_id": "baseline-review-fixture",
        },
    }
    raw_status = {
        "complete": "available",
        "partial": "partial",
        "missing": "missing",
        "stale": "stale",
        "invalid": "invalid",
        "conflicting": "invalid",
    }
    per_ticker = [
        {"ticker": ticker, "overall_fundamental_status": after_statuses[ticker]}
        for ticker in tickers
    ]
    instruments = [
        {
            "symbol": ticker,
            "fundamental_context": {
                "status": raw_status[after_statuses[ticker]],
                "blockers": (
                    ["duplicate_fundamental_rows_conflict"]
                    if after_statuses[ticker] == "conflicting"
                    else []
                ),
            },
            "canonical_advice_input_ready": ticker in after_ready,
            "full_advice_ready": ticker in after_full,
            "canonical_advice_label": "wait_for_price" if ticker in after_ready else "unable_to_advise",
        }
        for ticker in tickers
    ]
    after_counts = Counter(after_statuses.values())
    after = {
        "fundamental_complete": after_counts.get("complete", 0),
        "fundamental_partial": after_counts.get("partial", 0),
        "fundamental_missing": after_counts.get("missing", 0),
        "invalid_stale_conflicting": sum(
            after_counts.get(status, 0) for status in ("invalid", "stale", "conflicting")
        ),
        "canonical_advice_input_ready": len(after_ready),
        "full_advice_ready": len(after_full),
        "unable_to_advise": len(tickers) - len(after_ready),
    }
    run31 = {
        "evidence_coverage_index": {
            "schema_version": data06.BASELINE_INDEX_SCHEMA_VERSION,
            "run_id": "after-review-fixture",
            "instruments": instruments,
        },
        "evidence_coverage_summary": {
            "summary": {
                "attempted_instruments": len(tickers),
                "canonical_instruments": len(tickers),
                "canonical_advice_input_ready": len(after_ready),
                "full_advice_ready": len(after_full),
                "advice_counts": {"unable_to_advise": len(tickers) - len(after_ready)},
                "fundamental_counts": {
                    key: Counter(item["fundamental_context"]["status"] for item in instruments).get(key, 0)
                    for key in ("available", "partial", "missing", "stale", "invalid", "blocked", "not_applicable")
                },
            }
        },
    }
    return data06._before_after_comparison(baseline, after, per_ticker, run31)


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
