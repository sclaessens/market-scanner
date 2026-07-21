from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from market_engine.data import validated_fundamental_metric_sourcing as sourcing
from market_engine.data import operator_source_approval as source_approval
from market_engine.data.operator_fundamental_metric_package import INPUT_SCHEMA_VERSION, REPORT_SCHEMA_VERSION, VALIDATOR_VERSION


METRICS = {
    "revenue_growth_yoy": {"value": 12.5, "unit": "percent", "raw_source_field": "revenueGrowth"},
    "eps_growth_yoy": {"value": -0.04, "unit": "ratio", "raw_source_field": "epsGrowth"},
    "gross_margin": {"value": 48.0, "unit": "percent"},
    "operating_margin": {"value": 0.17, "unit": "ratio"},
    "debt_to_equity": {"value": 0.42, "unit": "ratio"},
}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _instrument(
    ticker: str,
    *,
    asset_type: str = "equity",
    technical: bool = False,
    **extra: object,
) -> dict[str, object]:
    return {
        "symbol": ticker,
        "instrument_id": f"{asset_type}:{ticker.lower()}",
        "name": ticker,
        "asset_type": asset_type,
        "exchange": "XNAS",
        "country": "US",
        "technical": technical,
        **extra,
    }


def _operator_record(ticker: str = "AAA", *, instrument_id: str | None = None) -> dict[str, object]:
    return {
        "ticker": ticker,
        "instrument_id": instrument_id or f"equity:{ticker.lower()}",
        "provider_symbol": ticker,
        "provider": "operator-primary-source-package",
        "source_date": "2026-07-01",
        "reporting_period": "2026-Q2",
        "source_reference": f"local-evidence://{ticker}/2026-Q2",
        "parser_version": "fixture-parser-v1",
        "snapshot_id": "fixture-snapshot",
        "acquired_at": "2026-07-02T10:00:00Z",
        "metrics": copy.deepcopy(METRICS),
    }


def _operator_package(path: Path, records: list[dict[str, object]]) -> Path:
    return _write_json(
        path,
        {"schema_version": sourcing.OPERATOR_IMPORT_SCHEMA_VERSION, "records": records},
    )


def _approval_for_package(tmp: Path, package_path: Path, *, decision: str = "approved") -> Path:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package.setdefault("package_id", "fixture-package")
    package.setdefault("package_schema_version", INPUT_SCHEMA_VERSION)
    package_path.write_text(json.dumps(package), encoding="utf-8")
    package_id = package["package_id"]
    input_path = _write_json(tmp / f"{package_path.stem}-input.json", {"schema_version": INPUT_SCHEMA_VERSION, "package_id": package_id})
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report_path = _write_json(tmp / f"{package_path.stem}-report.json", {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "package_id": package_id,
        "status": "accepted",
        "downstream_consumability": "structurally_valid_for_explicit_source_approval_review",
        "input_sha256": input_sha,
    })
    source_path = tmp / f"{package_path.stem}-source.html"
    source_path.write_text("official fixture source", encoding="utf-8")
    metrics = sorted({metric for row in package["records"] for metric in row.get("metrics", {})})
    reviews = {name: {"status": "approved"} for name in source_approval.REQUIRED_REVIEW_DIMENSIONS}
    return _write_json(tmp / f"{package_path.stem}-approval.json", {
        "schema_version": source_approval.DECISION_SCHEMA_VERSION,
        "decision_id": "fixture-decision",
        "decision": decision,
        "scope": source_approval.APPROVED_SCOPE,
        "approved_tickers": sorted({row["ticker"] for row in package["records"]}),
        "reviewer_roles": list(source_approval.REQUIRED_REVIEWER_ROLES),
        "package_id": package_id,
        "artifact_bindings": {
            "input_path": input_path.as_posix(),
            "input_sha256": input_sha,
            "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
            "validation_report_path": report_path.as_posix(),
            "validation_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "source_documents": [{"relative_path": source_path.name, "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}],
        "reviews": reviews,
        "approved_metrics": metrics,
        "explicitly_missing_metrics": sorted(set(sourcing.MVP_METRIC_FIELDS) - set(metrics)),
    })


@pytest.fixture()
def run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    instruments = [
        _instrument("AAA", technical=True),
        _instrument("BBB"),
        _instrument("CCC"),
        _instrument("ETF1", asset_type="etf"),
    ]
    universe = {"universe_version": "fixture-universe-v1", "instruments": instruments}
    monkeypatch.setattr(sourcing, "build_universe_snapshot", lambda *_args, **_kwargs: universe)

    canonical = _write_json(
        tmp_path / "canonical.json",
        {"universe_version": "fixture-universe-v1", "symbol_overrides": []},
    )
    baseline = tmp_path / "baseline-data06"
    rows = [
        {
            "ticker": "AAA",
            "overall_fundamental_status": "partial",
            "missing_metrics": ["eps_growth_yoy"],
            "selected_source_family": "local_csv",
            "source_date": "2026-07-01",
        },
        {
            "ticker": "BBB",
            "overall_fundamental_status": "missing",
            "missing_metrics": list(sourcing.MVP_METRIC_FIELDS),
            "selected_source_family": None,
            "source_date": None,
        },
        {
            "ticker": "CCC",
            "overall_fundamental_status": "complete",
            "missing_metrics": [],
            "selected_source_family": "local_csv",
            "source_date": "2026-07-01",
        },
        {
            "ticker": "ETF1",
            "overall_fundamental_status": "missing",
            "missing_metrics": list(sourcing.MVP_METRIC_FIELDS),
            "selected_source_family": None,
            "source_date": None,
        },
    ]
    _write_json(
        baseline / "manifest.json",
        {
            "canonical_universe": {
                "universe_version": "fixture-universe-v1",
                "total_instruments": 4,
            }
        },
    )
    _write_json(baseline / "per_ticker_fundamental_status.json", {"tickers": rows})
    _write_json(
        baseline / "fundamental_coverage_summary.json",
        {
            "after": {
                "canonical_advice_input_ready": 1,
                "full_advice_ready": 0,
                "unable_to_advise": 3,
            }
        },
    )
    (baseline / "normalized_fundamental_quality.csv").write_text(
        "ticker,revenue_growth_yoy\nCCC,0.1\n", encoding="utf-8"
    )

    run31 = tmp_path / "run31"
    readiness_rows = []
    for instrument in instruments:
        ready = instrument["symbol"] == "CCC"
        readiness_rows.append(
            {
                "symbol": instrument["symbol"],
                "canonical_advice_input_ready": ready,
                "technical_screening": {
                    "ranking_eligible": bool(instrument["technical"]),
                    "status": "available",
                    "setup_price_market_context": {"context_status": "available"},
                },
                "market_context": {"status": "available"},
                "portfolio_context": {"status": "not_applicable"},
            }
        )
    _write_json(run31 / "evidence_coverage_index.json", {"instruments": readiness_rows})
    _write_json(run31 / "manifest.json", {"canonical_universe_version": "fixture-universe-v1"})
    return {
        "tmp": tmp_path,
        "instruments": instruments,
        "canonical": canonical,
        "baseline": baseline,
        "run31": run31,
    }


def _run(run_fixture: dict[str, object], **overrides: object):
    tmp = run_fixture["tmp"]
    values: dict[str, object] = {
        "run_id": "fixture-run",
        "source_mode": "operator_import",
        "batch_tier": "pilot",
        "canonical_universe": run_fixture["canonical"],
        "price_history_root": tmp / "prices",
        "baseline_data06_run": run_fixture["baseline"],
        "baseline_run31_evidence": run_fixture["run31"],
        "operator_import_path": tmp / "missing.json",
        "output_root": tmp / "outputs",
        "raw_snapshot_root": tmp / "snapshots",
    }
    values.update(overrides)
    package_path = Path(values["operator_import_path"])
    if package_path.is_file() and "source_approval_decision_path" not in overrides:
        values["source_approval_decision_path"] = _approval_for_package(Path(tmp), package_path)
    if package_path.is_file() and "source_document_root" not in overrides:
        values["source_document_root"] = tmp
    return sourcing.run_validated_fundamental_metric_sourcing(**values)


def test_gap_analysis_is_unique_deterministic_and_prioritized(run_fixture: dict[str, object]) -> None:
    artifacts, _ = _run(run_fixture)

    rows = artifacts["per_ticker_sourcing_status"]["tickers"]
    assert [row["ticker"] for row in rows] == ["AAA", "BBB", "CCC", "ETF1"]
    assert len({row["ticker"] for row in rows}) == 4
    assert {row["ticker"]: row["sourcing_tier"] for row in rows} == {
        "AAA": "tier_1",
        "BBB": "tier_3",
        "CCC": "not_selected",
        "ETF1": "not_selected",
    }
    assert artifacts["metric_gap_analysis"]["missing_metric_counts"]["eps_growth_yoy"] == 3
    assert artifacts["sourcing_plan"]["pilot_tickers"] == ["AAA", "BBB"]


def test_missing_operator_package_fails_closed_without_provider_calls(run_fixture: dict[str, object]) -> None:
    artifacts, output = _run(run_fixture)

    batch = artifacts["batch_execution_summary"]
    blocker = artifacts["blocker_report"]
    assert artifacts["manifest"]["run_status"] == "blocked_external_source_requirement"
    assert batch["provider_calls_performed"] == 0
    assert batch["requests_attempted"] == 0
    assert batch["input_presence_checks"] == 1
    assert batch["imports_attempted"] == 0
    assert {
        "selected_count": batch["selected_count"],
        "success_count": batch["success_count"],
        "blocked_count": batch["blocked_count"],
        "failed_count": batch["failed_count"],
        "pending_count": batch["pending_count"],
        "not_selected_count": batch["not_selected_count"],
    } == {
        "selected_count": 2,
        "success_count": 0,
        "blocked_count": 2,
        "failed_count": 0,
        "pending_count": 0,
        "not_selected_count": 2,
    }
    assert batch["reconciliation"]["reconciled"] is True
    assert batch["selected_status_counts"] == {"blocked_no_source": 2}
    assert blocker["selected_status_counts"] == batch["selected_status_counts"]
    assert blocker["selected_blocked_count"] == batch["blocked_count"]
    assert artifacts["manifest"]["sourcing_status_reconciliation"]["reconciliation"] == batch["reconciliation"]
    assert artifacts["coverage_before_after"]["coverage_claim"] == "no_coverage_change_claimed"
    assert output.joinpath("blocker_report.json").exists()
    report = output.joinpath("coverage_report.md").read_text(encoding="utf-8")
    assert "- blocked: 2" in report
    assert "- reconciled: true" in report
    assert not Path(run_fixture["tmp"], "snapshots", "fixture-run").exists()


def test_missing_package_reconciles_twelve_selected_across_952_tickers() -> None:
    rows = [
        {
            "ticker": f"T{index:03d}",
            "sourcing_status": "blocked_no_source" if index < 12 else "not_selected",
        }
        for index in range(952)
    ]
    selected = [row["ticker"] for row in rows[:12]]

    result = sourcing._reconcile_sourcing_status_counts(
        rows,
        selected,
        run_status="blocked_external_source_requirement",
    )

    assert result == {
        "selected_count": 12,
        "success_count": 0,
        "blocked_count": 12,
        "failed_count": 0,
        "pending_count": 0,
        "not_selected_count": 940,
        "selected_status_counts": {"blocked_no_source": 12},
        "global_status_counts": {"blocked_no_source": 12, "not_selected": 940},
        "reconciliation": {
            "selected_count": 12,
            "terminal_success_count": 0,
            "terminal_blocked_count": 12,
            "terminal_failed_count": 0,
            "pending_count": 0,
            "reconciled": True,
        },
    }


def test_inventory_and_unapproved_acquisition_never_call_a_provider(run_fixture: dict[str, object]) -> None:
    for index, source_mode in enumerate(("inventory_only", "approved_acquisition")):
        artifacts, _ = _run(run_fixture, run_id=f"mode-{index}", source_mode=source_mode)
        assert artifacts["manifest"]["guardrails"]["provider_calls_performed"] is False
        assert artifacts["batch_execution_summary"]["requests_attempted"] == 0
        assert artifacts["source_approval_decision"]["approval_status"] == "blocked_no_approved_source"


def test_source_approval_records_zero_budget_and_never_persists_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUNDAMENTAL_PROVIDER_API_KEY", "do-not-persist-this-value")
    approval = sourcing._source_approval_decision("approved_acquisition", None)

    assert approval["credentials_status"] == "configured"
    assert approval["rate_limit"] == {"requests": 0, "retries": 0, "timeout_seconds": 0, "budget": 0}
    assert approval["secret_names_detected"] == []
    assert "do-not-persist-this-value" not in json.dumps(approval)
    decisions = {route["decision"] for route in approval["routes_inspected"]}
    assert {
        "approved_existing_local_acquisition",
        "blocked_missing_credentials",
        "blocked_unsupported_region",
        "blocked_unsupported_metric_contract",
    } <= decisions


@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        (_instrument("AAA"), "mapped"),
        (_instrument("ETF", asset_type="etf"), "unsupported_asset_type"),
        (_instrument("DUP", duplicate_listing=True), "rejected_duplicate_listing"),
        (_instrument("AMB", provider_symbol_candidates=["AMB-A", "AMB-B"]), "ambiguous"),
        (_instrument("MISS", provider_symbol_required=True), "missing_provider_symbol"),
    ],
)
def test_symbol_mapping_statuses(tmp_path: Path, instrument: dict[str, object], expected: str) -> None:
    config = _write_json(tmp_path / "canonical.json", {"symbol_overrides": []})
    assert sourcing._build_symbol_mappings([instrument], config)[0]["mapping_status"] == expected


def test_symbol_mapping_handles_alias_and_unsupported_exchange(tmp_path: Path) -> None:
    config = _write_json(
        tmp_path / "canonical.json",
        {
            "symbol_overrides": [
                {"canonical_symbol": "BRK.B", "source_symbol": "BRK-B"},
                {"canonical_symbol": "RHM", "source_symbol": "RHM.DE", "mapping_status": "unsupported"},
            ]
        },
    )
    mappings = sourcing._build_symbol_mappings([_instrument("BRK.B"), _instrument("RHM")], config)

    assert [(row["ticker"], row["provider_symbol"], row["mapping_status"]) for row in mappings] == [
        ("BRK.B", "BRK-B", "mapped_with_explicit_alias"),
        ("RHM", "RHM.DE", "unsupported_exchange"),
    ]
    assert mappings[0]["share_class"] == "class_share"


def test_symbol_mapping_retains_adr_listing_metadata(tmp_path: Path) -> None:
    config = _write_json(tmp_path / "canonical.json", {"symbol_overrides": []})
    mapping = sourcing._build_symbol_mappings([_instrument("ASML")], config)[0]

    assert mapping["listing_form"] == "adr_or_cross_listed"


def test_operator_import_normalizes_all_metrics_and_retains_lineage(tmp_path: Path) -> None:
    path = _operator_package(tmp_path / "input.json", [_operator_record()])
    records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": _instrument("AAA")},
        as_of=date(2026, 7, 18),
    )

    assert summary["validation_status"] == "passed"
    assert records[0]["metrics"] == {
        "revenue_growth_yoy": 0.125,
        "eps_growth_yoy": -0.04,
        "gross_margin": 0.48,
        "operating_margin": 0.17,
        "debt_to_equity": 0.42,
    }
    assert len(records[0]["metric_lineage"]) == 5
    assert records[0]["metric_lineage"][0]["transformation"] == "percent_to_ratio"
    assert records[0]["source_package_checksum"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert records[0]["metric_lineage"][0]["source_package_checksum"] == records[0]["source_package_checksum"]
    normalized_checksum = records[0].pop("normalized_record_checksum")
    assert normalized_checksum == hashlib.sha256(
        json.dumps(records[0], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_operator_import_preserves_nulls_and_partial_status(tmp_path: Path) -> None:
    record = _operator_record()
    record["metrics"]["gross_margin"]["value"] = None
    path = _operator_package(tmp_path / "input.json", [record])
    records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": _instrument("AAA")},
        as_of=date(2026, 7, 18),
    )

    assert summary["validation_status"] == "passed"
    assert records[0]["metrics"]["gross_margin"] is None
    assert records[0]["coverage_status"] == "partial"
    assert records[0]["missing_metrics"] == ["gross_margin"]


@pytest.mark.parametrize(
    ("mutator", "issue"),
    [
        (lambda row: row.update(source_date="bad-date"), "AAA_invalid_source_date"),
        (lambda row: row.update(source_date="2027-01-01"), "AAA_future_dated"),
        (lambda row: row.update(source_date="2025-01-01"), "AAA_stale"),
        (lambda row: row.update(provider_symbol="WRONG"), "AAA_provider_symbol_mismatch"),
        (lambda row: row.update(instrument_id="equity:wrong"), "AAA_instrument_identity_mismatch"),
        (lambda row: row["metrics"]["gross_margin"].update(value="48"), "AAA_gross_margin_invalid_numeric"),
        (lambda row: row["metrics"]["gross_margin"].update(unit="currency"), "AAA_gross_margin_invalid_unit"),
        (
            lambda row: row["metrics"]["gross_margin"].update(reporting_period="2026-Q1"),
            "AAA_gross_margin_reporting_period_mismatch",
        ),
    ],
)
def test_operator_import_rejects_invalid_evidence(tmp_path: Path, mutator, issue: str) -> None:
    record = _operator_record()
    mutator(record)
    path = _operator_package(tmp_path / "input.json", [record])
    records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": _instrument("AAA")},
        as_of=date(2026, 7, 18),
    )

    assert summary["validation_status"] == "failed"
    assert issue in summary["issues"]
    assert records == []


def test_operator_import_deduplicates_identical_records(tmp_path: Path) -> None:
    record = _operator_record()
    path = _operator_package(tmp_path / "input.json", [record, copy.deepcopy(record)])
    records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": _instrument("AAA")},
        as_of=date(2026, 7, 18),
    )

    assert summary["validation_status"] == "passed"
    assert summary["duplicate_records_deduplicated"] == 1
    assert len(records) == 1


def test_operator_import_rejects_conflicting_duplicate_period(tmp_path: Path) -> None:
    first = _operator_record()
    second = copy.deepcopy(first)
    second["metrics"]["gross_margin"]["value"] = 41.0
    path = _operator_package(tmp_path / "input.json", [first, second])
    _records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": _instrument("AAA")},
        as_of=date(2026, 7, 18),
    )

    assert summary["validation_status"] == "failed"
    assert "AAA_conflicting_duplicate_period" in summary["issues"]


def test_operator_import_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(sourcing.ValidatedFundamentalMetricSourcingError):
        sourcing._load_and_validate_operator_import(
            path,
            mappings={},
            instruments={},
            as_of=date(2026, 7, 18),
        )


def test_batch_scope_excludes_unselected_records_without_failing(tmp_path: Path) -> None:
    path = _operator_package(tmp_path / "input.json", [_operator_record("AAA"), _operator_record("BBB")])
    records, summary = sourcing._load_and_validate_operator_import(
        path,
        mappings={
            "AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"},
            "BBB": {"mapping_status": "mapped", "provider_symbol": "BBB"},
        },
        instruments={"AAA": _instrument("AAA"), "BBB": _instrument("BBB")},
        as_of=date(2026, 7, 18),
        allowed_tickers={"AAA"},
    )

    assert [row["ticker"] for row in records] == ["AAA"]
    assert summary["excluded_unselected_count"] == 1
    assert summary["validation_status"] == "passed"


def test_partial_operator_package_reconciles_success_and_blocked_tickers(run_fixture: dict[str, object]) -> None:
    tmp = run_fixture["tmp"]
    package = _operator_package(tmp / "operator.json", [_operator_record("AAA")])
    artifacts, output = _run(run_fixture, operator_import_path=package)

    assert artifacts["manifest"]["run_status"] == "completed_import_without_downstream"
    assert artifacts["batch_execution_summary"]["imported_count"] == 1
    assert artifacts["batch_execution_summary"]["success_count"] == 1
    assert artifacts["batch_execution_summary"]["blocked_count"] == 1
    assert artifacts["batch_execution_summary"]["failed_count"] == 0
    assert artifacts["batch_execution_summary"]["pending_count"] == 0
    assert artifacts["batch_execution_summary"]["reconciliation"]["reconciled"] is True
    statuses = {row["ticker"]: row["sourcing_status"] for row in artifacts["per_ticker_sourcing_status"]["tickers"]}
    assert statuses["AAA"] == "complete"
    assert statuses["BBB"] == "blocked_provider_coverage"
    snapshot = artifacts["manifest"]["raw_snapshot"]
    assert snapshot["raw_checksum"] == hashlib.sha256(package.read_bytes()).hexdigest()
    assert Path(snapshot["raw_path"]).exists()
    assert output.joinpath("normalized_fundamental_metrics.csv").exists()
    assert artifacts["coverage_before_after"]["downstream_executed"] is False


def test_missing_concrete_approval_blocks_before_import_and_snapshot(run_fixture: dict[str, object]) -> None:
    tmp = Path(run_fixture["tmp"])
    package = _operator_package(tmp / "operator-no-approval.json", [_operator_record("AAA")])
    artifacts, _ = _run(run_fixture, operator_import_path=package, source_approval_decision_path=None)

    assert artifacts["manifest"]["run_status"] == "blocked_external_source_requirement"
    assert artifacts["batch_execution_summary"]["imports_attempted"] == 0
    assert artifacts["manifest"]["raw_snapshot"] is None
    assert artifacts["manifest"]["downstream"] is None
    assert artifacts["concrete_source_approval_validation"]["reason_codes"] == ["SOURCE_APPROVAL_DECISION_MISSING"]


def test_malformed_approval_list_blocks_before_parser_snapshot_and_downstream(
    run_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = Path(run_fixture["tmp"])
    package = _operator_package(tmp / "operator-malformed-list.json", [_operator_record("AAA")])
    decision_path = _approval_for_package(tmp, package)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["approved_tickers"] = ["AAA", {"unexpected": "object"}]
    _write_json(decision_path, decision)
    calls = {"parser": 0, "snapshot": 0, "downstream": 0}

    def fail_parser(*_args, **_kwargs):
        calls["parser"] += 1
        raise AssertionError("package parser must not run")

    def fail_snapshot(*_args, **_kwargs):
        calls["snapshot"] += 1
        raise AssertionError("snapshot persistence must not run")

    def fail_downstream(*_args, **_kwargs):
        calls["downstream"] += 1
        raise AssertionError("downstream DATA06/RUN31 must not run")

    monkeypatch.setattr(sourcing, "_load_and_validate_operator_import", fail_parser)
    monkeypatch.setattr(sourcing, "_persist_operator_snapshot", fail_snapshot)
    monkeypatch.setattr(sourcing, "run_fundamental_evidence_coverage", fail_downstream)
    artifacts, output = _run(
        run_fixture,
        run_id="malformed-list",
        operator_import_path=package,
        source_approval_decision_path=decision_path,
        execute_downstream=True,
        data06_run_id="must-not-run-data06",
        run31_run_id="must-not-run-run31",
    )

    assert calls == {"parser": 0, "snapshot": 0, "downstream": 0}
    assert artifacts["manifest"]["run_status"] == "blocked_external_source_requirement"
    assert artifacts["batch_execution_summary"]["execution_reason"] == "concrete_source_approval_failed"
    assert artifacts["batch_execution_summary"]["imports_attempted"] == 0
    assert artifacts["batch_execution_summary"]["normalized_count"] == 0
    assert artifacts["batch_execution_summary"]["provider_calls_performed"] == 0
    assert artifacts["normalized_metric_evidence"]["record_count"] == 0
    assert artifacts["manifest"]["raw_snapshot"] is None
    assert artifacts["manifest"]["downstream"] is None
    assert artifacts["manifest"]["guardrails"]["network_access_performed"] is False
    assert artifacts["blocker_report"]["primary_blocker"] == "concrete_source_approval_failed"
    assert artifacts["batch_execution_summary"]["reconciliation"]["reconciled"] is True
    assert "APPROVED_TICKERS_INVALID" in artifacts["concrete_source_approval_validation"]["reason_codes"]
    assert output.joinpath("concrete_source_approval_validation.json").is_file()


def test_governed_v2_route_uses_strong_reconciliation_before_parser_snapshot_and_downstream(
    run_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = Path(run_fixture["tmp"])
    package = _write_json(
        tmp / "tampered-governed-v2.json",
        {
            "schema_version": sourcing.DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION,
            "package_id": "tampered-governed-v2",
            "records": [],
        },
    )
    decision = _write_json(tmp / "derivation-approval.json", {"decision": "approved"})
    calls = {"strong_gate": 0, "parser": 0, "snapshot": 0, "downstream": 0}

    def blocked_gate(*_args, **_kwargs):
        calls["strong_gate"] += 1
        return {
            "schema_version": "market-engine-data10-derivation-approval-validation-v2",
            "validation_status": "blocked",
            "concrete_package_source_approved": False,
            "reason_codes": ["GOVERNED_PACKAGE_REPLAY_MISMATCH"],
            "issues": [],
        }

    def fail_parser(*_args, **_kwargs):
        calls["parser"] += 1
        raise AssertionError("parser must not run before strong reconciliation passes")

    def fail_snapshot(*_args, **_kwargs):
        calls["snapshot"] += 1
        raise AssertionError("snapshot must not be written after reconciliation failure")

    def fail_downstream(*_args, **_kwargs):
        calls["downstream"] += 1
        raise AssertionError("DATA06/RUN31 must not run after reconciliation failure")

    monkeypatch.setattr(sourcing, "validate_derivation_approval_decision", blocked_gate)
    monkeypatch.setattr(sourcing, "_load_and_validate_operator_import", fail_parser)
    monkeypatch.setattr(sourcing, "_persist_operator_snapshot", fail_snapshot)
    monkeypatch.setattr(sourcing, "run_fundamental_evidence_coverage", fail_downstream)
    artifacts, _ = _run(
        run_fixture,
        run_id="strong-gate-block",
        operator_import_path=package,
        source_approval_decision_path=decision,
        source_document_root=tmp,
        execute_downstream=True,
        data06_run_id="must-not-run-data06",
        run31_run_id="must-not-run-run31",
    )

    assert calls == {"strong_gate": 1, "parser": 0, "snapshot": 0, "downstream": 0}
    assert artifacts["manifest"]["raw_snapshot"] is None
    assert artifacts["manifest"]["downstream"] is None
    assert artifacts["batch_execution_summary"]["imports_attempted"] == 0
    assert artifacts["concrete_source_approval_validation"]["reason_codes"] == [
        "GOVERNED_PACKAGE_REPLAY_MISMATCH"
    ]


@pytest.mark.parametrize("decision", ["blocked", "rejected", "unknown"])
def test_non_approved_decision_blocks_before_parser(run_fixture: dict[str, object], decision: str) -> None:
    tmp = Path(run_fixture["tmp"])
    package = _operator_package(tmp / f"operator-{decision}.json", [_operator_record("AAA")])
    approval = _approval_for_package(tmp, package, decision=decision)
    artifacts, _ = _run(run_fixture, run_id=f"run-{decision}", operator_import_path=package, source_approval_decision_path=approval)

    assert artifacts["batch_execution_summary"]["imports_attempted"] == 0
    assert artifacts["normalized_metric_evidence"]["record_count"] == 0
    assert artifacts["manifest"]["raw_snapshot"] is None


def test_one_byte_package_change_invalidates_approval(run_fixture: dict[str, object]) -> None:
    tmp = Path(run_fixture["tmp"])
    package = _operator_package(tmp / "operator-tampered.json", [_operator_record("AAA")])
    approval = _approval_for_package(tmp, package)
    package.write_bytes(package.read_bytes() + b" ")
    artifacts, _ = _run(run_fixture, run_id="tampered", operator_import_path=package, source_approval_decision_path=approval)

    validation = artifacts["concrete_source_approval_validation"]
    assert validation["concrete_package_source_approved"] is False
    assert "PACKAGE_CHECKSUM_MISMATCH" in validation["reason_codes"]
    assert artifacts["batch_execution_summary"]["imports_attempted"] == 0


def test_multi_ticker_operator_package_is_blocked_by_bounded_approval(run_fixture: dict[str, object]) -> None:
    tmp = run_fixture["tmp"]
    package = _operator_package(
        tmp / "operator-full.json",
        [_operator_record("AAA"), _operator_record("BBB")],
    )
    artifacts, _ = _run(run_fixture, run_id="full-import", operator_import_path=package)

    batch = artifacts["batch_execution_summary"]
    assert batch["selected_count"] == 2
    assert batch["success_count"] == 0
    assert batch["complete_count"] == 0
    assert batch["partial_count"] == 0
    assert batch["blocked_count"] == 2
    assert batch["failed_count"] == 0
    assert batch["pending_count"] == 0
    assert batch["not_selected_count"] == 2
    assert batch["selected_status_counts"] == {"blocked_no_source": 2}
    assert "BOUNDED_PILOT_TICKER_LIMIT_EXCEEDED" in artifacts["concrete_source_approval_validation"]["reason_codes"]
    assert batch["reconciliation"]["reconciled"] is True


def test_existing_output_and_snapshot_are_not_overwritten_by_default(run_fixture: dict[str, object]) -> None:
    tmp = run_fixture["tmp"]
    package = _operator_package(tmp / "operator.json", [_operator_record("AAA")])
    _run(run_fixture, operator_import_path=package)

    with pytest.raises(FileExistsError):
        _run(run_fixture, operator_import_path=package)


def test_invalid_package_creates_no_raw_snapshot(run_fixture: dict[str, object]) -> None:
    tmp = run_fixture["tmp"]
    record = _operator_record("AAA")
    record["source_date"] = "2030-01-01"
    package = _operator_package(tmp / "operator.json", [record])
    artifacts, _ = _run(run_fixture, operator_import_path=package)

    assert artifacts["manifest"]["run_status"] == "failed_validation"
    assert artifacts["manifest"]["raw_snapshot"] is None
    assert not Path(tmp, "snapshots", "fixture-run").exists()


def test_validation_issues_are_not_counted_as_failed_tickers(run_fixture: dict[str, object]) -> None:
    tmp = run_fixture["tmp"]
    record = _operator_record("AAA")
    record["metrics"]["gross_margin"]["value"] = "invalid"
    record["metrics"]["operating_margin"]["value"] = "also-invalid"
    package = _operator_package(tmp / "operator-invalid.json", [record])
    artifacts, _ = _run(run_fixture, run_id="invalid-counts", operator_import_path=package)

    batch = artifacts["batch_execution_summary"]
    assert batch["success_count"] == 0
    assert batch["blocked_count"] == 1
    assert batch["failed_count"] == 1
    assert batch["pending_count"] == 0
    assert batch["validation_issue_count"] == 2
    assert batch["validation_failed_record_count"] == 1
    assert batch["package_validation_failed"] is True
    assert batch["selected_status_counts"] == {
        "blocked_invalid_payload": 1,
        "failed_validation": 1,
    }
    assert batch["reconciliation"]["reconciled"] is True


def test_unknown_selected_status_fails_reconciliation_closed() -> None:
    with pytest.raises(
        sourcing.ValidatedFundamentalMetricSourcingError,
        match="selected sourcing statuses are not classified: unknown_status",
    ):
        sourcing._reconcile_sourcing_status_counts(
            [{"ticker": "AAA", "sourcing_status": "unknown_status"}],
            ["AAA"],
            run_status="blocked_external_source_requirement",
        )


def test_pending_selected_status_is_rejected_for_terminal_run() -> None:
    with pytest.raises(
        sourcing.ValidatedFundamentalMetricSourcingError,
        match="terminal run status blocked_external_source_requirement contains 1 pending selected tickers",
    ):
        sourcing._reconcile_sourcing_status_counts(
            [{"ticker": "AAA", "sourcing_status": "selected"}],
            ["AAA"],
            run_status="blocked_external_source_requirement",
        )


def test_downstream_uses_real_normalized_artifact_when_explicitly_enabled(
    run_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = run_fixture["tmp"]
    package = _operator_package(tmp / "operator.json", [_operator_record("AAA")])
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return (
            {
                "manifest": {"run_status": "completed"},
                "fundamental_coverage_summary": {"after": {
                    "fundamental_complete": 2,
                    "fundamental_partial": 1,
                    "fundamental_missing": 1,
                    "invalid_stale_conflicting": 0,
                    "canonical_advice_input_ready": 1,
                    "full_advice_ready": 0,
                    "unable_to_advise": 3,
                }},
                "before_after_comparison": {"improvement_counts": {"missing_to_complete": 1}},
                "per_ticker_fundamental_status": {"tickers": [{
                    "ticker": "AAA",
                    "overall_fundamental_status": "complete",
                    "canonical_advice_input_ready": True,
                }]},
            },
            tmp / "downstream-data06",
        )

    monkeypatch.setattr(sourcing, "run_fundamental_evidence_coverage", fake_run)
    artifacts, _ = _run(
        run_fixture,
        operator_import_path=package,
        execute_downstream=True,
        data06_run_id="data06-after",
        run31_run_id="run31-after",
    )

    assert len(calls) == 1
    assert Path(calls[0]["raw_fundamentals_path"]).name == "normalized_fundamental_metrics.csv"
    assert calls[0]["baseline_run_evidence"] == run_fixture["run31"]
    assert artifacts["coverage_before_after"]["coverage_claim"] == "measured_downstream_result"
    assert artifacts["manifest"]["run_status"] == "completed_with_coverage_measurement"


def test_artifacts_are_deterministically_ordered_and_contain_safety_guardrails(run_fixture: dict[str, object]) -> None:
    first, _ = _run(run_fixture, run_id="first")
    second, _ = _run(run_fixture, run_id="second")

    assert first["metric_gap_analysis"] == second["metric_gap_analysis"]
    assert first["sourcing_plan"] == second["sourcing_plan"]
    assert first["fundamental_source_symbol_mapping"] == second["fundamental_source_symbol_mapping"]
    guardrails = first["manifest"]["guardrails"]
    assert all(value is False for value in guardrails.values())
    serialized = json.dumps(first)
    assert "position_size" not in serialized
    assert '"allocation_performed": true' not in serialized
