from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import market_engine.data.data11_execution as data11_execution
from market_engine.data.data11_execution import load_downstream_after_authority
from market_engine.data.data11_governance import (
    DEFAULT_DOWNSTREAM_AUTHORITY,
    DEFAULT_RUN30_AUTHORITY,
    Data11GovernanceError,
    build_downstream_measurement,
    effective_freshness,
    execute_approved_candidate,
    load_downstream_prestate,
    metric_comparability,
    persist_approval_bundle,
    select_duration_facts,
    validate_approval_decision,
    validate_authoritative_run30,
    validate_temporal_boundary,
)
from market_engine.data.primary_source_metric_derivation import derive_primary_source_metrics
from market_engine.data.targeted_diversified_fundamental_derivation import build_fact_package


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_runner(tmp_path: Path, stage: str, calls: list[str], artifact_bindings: dict | None = None):
    def runner(**kwargs):
        calls.append(stage)
        run_id = f"synthetic-{stage}"
        if stage == "data07":
            governed_path = Path(kwargs["operator_import_path"])
            decision_path = Path(kwargs["source_approval_decision_path"])
            governed = json.loads(governed_path.read_text(encoding="utf-8"))
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            expected_input = {
                "decision_sha256": _sha(decision_path),
                "governed_package_path": governed_path.resolve().as_posix(),
                "governed_package_sha256": _sha(governed_path),
                "governed_package_id": governed["package_id"],
                "decision_id": decision["decision_id"],
                "ticker": decision["ticker"],
                "instrument_id": decision["instrument_id"],
                "approved_metrics": decision["approved_derived_metrics"],
                "calculation_checksums": decision["approved_calculation_checksums"],
                "approval_artifact_bindings": {
                    name: {
                        "path": (decision_path.parent / Path(binding["path"]).name).resolve().as_posix(),
                        "sha256": binding["sha256"],
                    }
                    for name, binding in decision["artifact_bindings"].items()
                },
            }
            schema = "market-engine-data11-data07-import-output-v1"
        elif stage == "data06":
            expected_input = {key: kwargs[key] for key in (
                "data07_output_path", "data07_output_sha256", "data07_receipt_path", "data07_receipt_sha256"
            )}
            schema = "market-engine-data11-data06-refresh-output-v1"
        else:
            expected_input = {key: kwargs[key] for key in (
                "data06_output_path", "data06_output_sha256", "data06_receipt_path", "data06_receipt_sha256"
            )}
            schema = "market-engine-data11-run31-refresh-output-v1"
        output_path = tmp_path / f"{stage}-output.json"
        receipt_path = tmp_path / f"{stage}-receipt.json"
        output = {
            "schema_version": schema, "stage": stage, "run_id": run_id, "status": "completed",
            "input_bindings": expected_input, "artifact_bindings": artifact_bindings or {},
        }
        _write_json(output_path, output)
        receipt = {
            "schema_version": "market-engine-data11-stage-receipt-v1", "stage": stage,
            "run_id": run_id, "status": "completed", "input_bindings": expected_input,
            "output_path": output_path.resolve().as_posix(), "output_sha256": _sha(output_path),
        }
        _write_json(receipt_path, receipt)
        return {
            "schema_version": "market-engine-data11-stage-result-v1", "stage": stage,
            "run_id": run_id, "status": "completed", "output_path": output_path.resolve().as_posix(),
            "output_sha256": _sha(output_path), "receipt_path": receipt_path.resolve().as_posix(),
            "receipt_sha256": _sha(receipt_path),
        }
    return runner


def _bound_after_state(tmp_path: Path, decision_path: Path, payload_mutator=None):
    root = tmp_path
    evidence = root / "after"
    state = load_downstream_prestate()
    version = "me-data04-complete-canonical-local-market-dataset-v1"
    fundamental_rows = [copy.deepcopy(row) for row in state["by_ticker"].values()]
    advice_ready = {row["ticker"] for row in fundamental_rows if row["overall_fundamental_status"] == "complete"}
    readiness_rows = [
        {
            "symbol": row["ticker"], "instrument_id": row["instrument_id"],
            "fundamental_context": {"status": row["overall_fundamental_status"]},
            "canonical_advice_input_ready": row["ticker"] in advice_ready,
            "full_advice_ready": False, "canonical_advice_label": (
                "ready_for_advice" if row["ticker"] in advice_ready else "unable_to_advise"
            ),
        }
        for row in fundamental_rows
    ]
    counts = state["before"]
    payloads = {
        "data06_manifest": {"run_id": "synthetic-data06", "canonical_universe": {"universe_version": version, "total_instruments": 952}},
        "data06_summary": {"after": {key: counts[key] for key in (
            "fundamental_complete", "fundamental_partial", "fundamental_missing", "invalid_stale_conflicting"
        )}},
        "data06_per_ticker": {"tickers": fundamental_rows},
        "run31_compact_index": {"run_id": "synthetic-run31", "canonical_universe_version": version},
        "run31_evidence_summary": {"summary": {
            "canonical_advice_input_ready": 6, "full_advice_ready": 0, "unable_to_advise": 946,
        }},
        "run31_per_ticker_index": {"instruments": readiness_rows},
    }
    if payload_mutator is not None:
        payload_mutator(payloads)
    bindings = {}
    for name, payload in payloads.items():
        path = evidence / f"{name}.json"
        _write_json(path, payload)
        bindings[name] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
    top = {
        Path(bindings["run31_evidence_summary"]["path"]).name: bindings["run31_evidence_summary"]["sha256"],
        Path(bindings["run31_per_ticker_index"]["path"]).name: bindings["run31_per_ticker_index"]["sha256"],
    }
    top_path = evidence / "run31_top_level_checksums.json"
    _write_json(top_path, top)
    bindings["run31_top_level_checksums"] = {"path": top_path.relative_to(root).as_posix(), "sha256": _sha(top_path)}
    calls = []
    result = execute_approved_candidate(
        decision_path,
        data07_runner=_stage_runner(root, "data07", calls),
        data06_runner=_stage_runner(root, "data06", calls, {name: bindings[name] for name in (
            "data06_manifest", "data06_summary", "data06_per_ticker"
        )}),
        run31_runner=_stage_runner(root, "run31", calls, {name: bindings[name] for name in (
            "run31_compact_index", "run31_evidence_summary", "run31_per_ticker_index", "run31_top_level_checksums"
        )}),
    )
    assert result["status"] == "completed"
    for stage in ("data07", "data06", "run31"):
        for kind in ("output", "receipt"):
            path = Path(result["validated_execution_proof"].stage_bindings[stage][f"{kind}_path"])
            bindings[f"{stage}_{kind}"] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
    authority = {
        "schema_version": "market-engine-data11-downstream-after-authority-v1",
        "canonical_universe_version": version, "canonical_universe_size": 952,
        "lineage": {"data07_run_id": "synthetic-data07", "data06_run_id": "synthetic-data06", "run31_run_id": "synthetic-run31"},
        "trusted_artifacts": bindings,
    }
    authority_path = root / "after-authority.json"
    _write_json(authority_path, authority)
    return authority_path, result["validated_execution_proof"], authority, payloads


def _materialize_authority(tmp_path: Path, authority_path: Path) -> tuple[Path, dict, dict[str, dict]]:
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    payloads = {}
    for name, binding in authority["trusted_artifacts"].items():
        source = Path(binding["path"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        target = tmp_path / source
        _write_json(target, payload)
        binding["sha256"] = _sha(target)
        payloads[name] = payload
    target_authority = tmp_path / authority_path
    _write_json(target_authority, authority)
    return target_authority, authority, payloads


def _refresh_binding(tmp_path: Path, authority: dict, name: str, payload: dict) -> None:
    target = tmp_path / authority["trusted_artifacts"][name]["path"]
    _write_json(target, payload)
    authority["trusted_artifacts"][name]["sha256"] = _sha(target)
    _write_json(tmp_path / DEFAULT_RUN30_AUTHORITY, authority)


def test_run30_authority_accepts_the_deterministic_bound_top_25() -> None:
    authority = validate_authoritative_run30()
    assert len(authority["top_candidates"]) == 25
    assert [row["rank"] for row in authority["top_candidates"]] == list(range(1, 26))
    assert authority["top_candidates"][0]["symbol"] == "ASB"


def test_changed_ranking_content_fails_the_tracked_checksum(tmp_path: Path) -> None:
    _, authority, payloads = _materialize_authority(tmp_path, DEFAULT_RUN30_AUTHORITY)
    payloads["run30_ranking"]["candidates"][0]["symbol"] = "FORGED"
    target = tmp_path / authority["trusted_artifacts"]["run30_ranking"]["path"]
    _write_json(target, payloads["run30_ranking"])
    with pytest.raises(Data11GovernanceError, match="checksum mismatch"):
        validate_authoritative_run30(repository_root=tmp_path)


@pytest.mark.parametrize("artifact", ["run30_manifest", "canonical_universe"])
def test_changed_bound_manifest_or_universe_fails_the_tracked_checksum(tmp_path: Path, artifact: str) -> None:
    _, authority, payloads = _materialize_authority(tmp_path, DEFAULT_RUN30_AUTHORITY)
    payloads[artifact]["tampered"] = True
    _write_json(tmp_path / authority["trusted_artifacts"][artifact]["path"], payloads[artifact])
    with pytest.raises(Data11GovernanceError, match="checksum mismatch"):
        validate_authoritative_run30(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ranking, index: ranking["candidates"][0].update(symbol="FORGED", instrument_id="equity:forged", source_symbol="FORGED", traceability={"price_history_path": "data/processed/FORGED.csv"}), "not present"),
        (lambda ranking, index: ranking["candidates"][0].update(source_symbol="WRONG"), "source_symbol"),
        (lambda ranking, index: index["instruments"][[row["instrument_id"] for row in index["instruments"]].index(ranking["candidates"][0]["instrument_id"])].update(asset_type="etf"), "asset type"),
        (lambda ranking, index: ranking["candidates"][1].update(instrument_id=ranking["candidates"][0]["instrument_id"]), "duplicate"),
        (lambda ranking, index: index["instruments"].append(copy.deepcopy(index["instruments"][0])), "size"),
        (lambda ranking, index: index["instruments"][1].update(symbol=index["instruments"][0]["symbol"]), "ambiguous"),
        (lambda ranking, index: index["instruments"][[row["instrument_id"] for row in index["instruments"]].index(ranking["candidates"][0]["instrument_id"])].update(price_history={}), "price-history"),
    ],
)
def test_semantically_invalid_but_rebound_run30_inputs_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    _, authority, payloads = _materialize_authority(tmp_path, DEFAULT_RUN30_AUTHORITY)
    mutation(payloads["run30_ranking"], payloads["run30_universe_index"])
    _refresh_binding(tmp_path, authority, "run30_ranking", payloads["run30_ranking"])
    _refresh_binding(tmp_path, authority, "run30_universe_index", payloads["run30_universe_index"])
    with pytest.raises(Data11GovernanceError, match=message):
        validate_authoritative_run30(repository_root=tmp_path)


def _duration(*, start: str, end: str, value: float, fp: str, filed: str = "2026-08-01", accn: str = "a") -> dict:
    return {"start": start, "end": end, "val": value, "fy": 2026, "fp": fp, "filed": filed, "accn": accn}


@pytest.mark.parametrize("reverse", [False, True])
def test_qtd_wins_over_ytd_independent_of_source_order(reverse: bool) -> None:
    qtd = ("Revenue", _duration(start="2026-04-01", end="2026-06-30", value=25, fp="Q2"))
    ytd = ("Revenue", _duration(start="2026-01-01", end="2026-06-30", value=55, fp="Q2"))
    rows = [qtd, ytd]
    if reverse:
        rows.reverse()
    aligned, selection = select_duration_facts({"revenue": rows})
    assert aligned["revenue"][1]["val"] == 25
    assert selection["duration_class"] == "discrete_quarter"


@pytest.mark.parametrize(
    ("fp", "start", "end", "expected"),
    [
        ("Q1", "2026-01-01", "2026-03-31", "discrete_quarter"),
        ("Q2", "2026-04-01", "2026-06-30", "discrete_quarter"),
        ("Q3", "2026-07-01", "2026-09-30", "discrete_quarter"),
        ("FY", "2026-01-01", "2026-12-31", "annual"),
    ],
)
def test_period_policy_covers_quarters_and_fiscal_year(fp: str, start: str, end: str, expected: str) -> None:
    _, selection = select_duration_facts({"revenue": [("Revenue", _duration(start=start, end=end, value=1, fp=fp))]})
    assert selection["duration_class"] == expected


def test_conflicting_semantic_duplicates_are_blocked_across_raw_tags() -> None:
    first = _duration(start="2026-04-01", end="2026-06-30", value=25, fp="Q2")
    second = {**first, "val": 26}
    with pytest.raises(Data11GovernanceError, match="conflicting duration facts"):
        select_duration_facts({"revenue": [("Revenue", first), ("SalesRevenueNet", second)]})


def test_numerator_must_match_the_selected_denominator_period_exactly() -> None:
    revenue = _duration(start="2026-04-01", end="2026-06-30", value=100, fp="Q2")
    gross_ytd = _duration(start="2026-01-01", end="2026-06-30", value=40, fp="Q2")
    aligned, _ = select_duration_facts({"revenue": [("Revenue", revenue)], "gross_profit": [("GrossProfit", gross_ytd)]})
    assert set(aligned) == {"revenue"}


def test_comparability_separates_duration_freshness_framework_and_calendar() -> None:
    base = {"value": "0.5", "duration_class": "discrete_quarter", "fiscal_period": "Q2", "period_start": "2026-04-01", "period_end": "2026-06-30", "freshness_status": "current"}
    rows = [
        {"ticker": "A", "accounting_framework": "us_gaap", "metrics": {"gross_margin": dict(base)}},
        {"ticker": "B", "accounting_framework": "ifrs", "metrics": {"gross_margin": {**base, "period_start": "2026-05-01", "period_end": "2026-07-31"}}},
        {"ticker": "C", "accounting_framework": "us_gaap", "metrics": {"gross_margin": {**base, "duration_class": "year_to_date"}}},
        {"ticker": "D", "accounting_framework": "us_gaap", "metrics": {"gross_margin": {**base, "freshness_status": "stale"}}},
    ]
    result = metric_comparability(rows)
    assert result[("A", "gross_margin")][0] == "comparable_limited"
    assert "ACCOUNTING_FRAMEWORK_DIFFERENCE" in result[("A", "gross_margin")][1]
    assert "STALE_OR_INVALID_EVIDENCE" in result[("D", "gross_margin")][1]
    assert "DURATION_CLASS_MISMATCH" in result[("C", "gross_margin")][1]


def test_ash_q3_ytd_is_not_silently_equated_to_q2_ytd() -> None:
    evidence = {"value": "0.5", "duration_class": "year_to_date", "freshness_status": "current", "period_start": "2026-01-01"}
    rows = [
        {"ticker": "ASH", "accounting_framework": "us_gaap", "metrics": {"gross_margin": {**evidence, "fiscal_period": "Q3", "period_end": "2026-09-30"}}},
        {"ticker": "PEER", "accounting_framework": "us_gaap", "metrics": {"gross_margin": {**evidence, "fiscal_period": "Q2", "period_end": "2026-06-30"}}},
    ]
    status, reasons = metric_comparability(rows)[("ASH", "gross_margin")]
    assert status == "not_comparable"
    assert reasons == ["FISCAL_PERIOD_MISMATCH"]


def _companyfacts() -> dict:
    def fact(value: float) -> dict:
        return {"units": {"USD": [{"start": "2026-04-01", "end": "2026-06-30", "val": value, "fy": 2026, "fp": "Q2", "filed": "2026-08-01", "accn": "0001", "form": "10-Q"}]}}
    return {"entityName": "Issuer", "facts": {"us-gaap": {"Revenues": fact(100), "GrossProfit": fact(40), "OperatingIncomeLoss": fact(20)}}}


def _approval_bundle(tmp_path: Path) -> Path:
    fact, _ = build_fact_package(
        {"ticker": "AAA", "instrument_id": "equity:aaa", "rank": 1}, payload=_companyfacts(),
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json", source_checksum="a" * 64,
        generated_at="2026-08-13T12:00:00Z", trusted_now="2026-08-13T12:00:00Z", run_id="test-run",
    )
    catalog = json.loads(Path("config/market_engine/data10_fundamental_metric_formula_catalog.json").read_text(encoding="utf-8"))
    derived, validation = derive_primary_source_metrics(fact, catalog)
    bundle = tmp_path / "AAA"
    persist_approval_bundle(
        bundle_dir=bundle, run_id="test-run", ticker="AAA", cik="0000000001",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
        full_source_snapshot_sha256="a" * 64, fact_package=fact, formula_catalog=catalog,
        derived_package=derived, derivation_validation=validation,
    )
    return bundle


def _approve(bundle: Path) -> Path:
    mapping_path = bundle / "mapping_review.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["decision"] = "approved"
    _write_json(mapping_path, mapping)
    decision_path = bundle / "approval_candidate.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "approved"
    decision["reviews"] = {name: {"status": "approved"} for name in decision["reviews"]}
    decision["artifact_bindings"]["mapping_review"]["sha256"] = _sha(mapping_path)
    _write_json(decision_path, decision)
    return decision_path


def test_persisted_approval_candidate_replays_from_its_bundle(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    result = validate_approval_decision(decision)
    assert result["validation_status"] == "approved"


@pytest.mark.parametrize(
    ("filename", "mutate", "reason"),
    [
        ("source_evidence.json", lambda value: value["observations"][0].update(value=999), "SOURCE_EVIDENCE_FACT_RECONCILIATION_FAILED"),
        ("fact_package.json", lambda value: value["facts"][0].update(value=999), "DERIVED_REPLAY_MISMATCH"),
        ("mapping_review.json", lambda value: value["mappings"][0].__setitem__(1, "wrong"), "CANONICAL_MAPPING_FACT_RECONCILIATION_FAILED"),
        ("formula_catalog.json", lambda value: value["formulas"][0].update(version="9.9.9"), "DERIVED_REPLAY_MISMATCH"),
        ("derived_package.json", lambda value: value["derivations"][0].update(calculation_result=999), "DERIVED_REPLAY_MISMATCH"),
        ("derived_package.json", lambda value: value["derivations"][0].update(calculation_checksum="0" * 64), "DERIVED_REPLAY_MISMATCH"),
    ],
)
def test_approval_rejects_checksum_rebound_semantic_tampering(tmp_path: Path, filename: str, mutate, reason: str) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    artifact = decision_path.parent / filename
    value = json.loads(artifact.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(artifact, value)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["artifact_bindings"][filename.removesuffix(".json")]["sha256"] = _sha(artifact)
    _write_json(decision_path, decision)
    assert reason in validate_approval_decision(decision_path)["reason_codes"]


@pytest.mark.parametrize("status", ["pending", "rejected", "blocked"])
def test_nonapproved_decisions_make_zero_downstream_calls(tmp_path: Path, status: str) -> None:
    decision_path = _approval_bundle(tmp_path) / "approval_candidate.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = status
    _write_json(decision_path, decision)
    calls = []
    result = execute_approved_candidate(decision_path, data07_runner=lambda: calls.append("data07"))
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


@pytest.mark.parametrize("ticker", ["ASH", "BIO", "CI"])
def test_committed_candidates_remain_pending_without_execution_authority(ticker: str) -> None:
    decision_path = Path(
        "artifacts/market_engine/run_evidence/"
        "me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z/"
        f"approval_candidates/{ticker}/approval_candidate.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    calls = []
    result = execute_approved_candidate(
        decision_path,
        data07_runner=lambda **kwargs: calls.append(kwargs),
    )
    assert decision["decision"] == "pending"
    assert result["approval_validation"]["validation_status"] == "blocked"
    assert "APPROVAL_PENDING" in result["approval_validation"]["reason_codes"]
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


def test_valid_synthetic_approval_executes_each_bounded_stage_once(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    result = execute_approved_candidate(
        decision, data07_runner=_stage_runner(tmp_path, "data07", calls),
        data06_runner=_stage_runner(tmp_path, "data06", calls),
        run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert result["calls"] == {"data07": 1, "data06": 1, "run31": 1}
    assert calls == ["data07", "data06", "run31"]


@pytest.mark.parametrize(
    "override",
    [
        {"operator_import_path": "BIO/governed_package_candidate.json"}, {"ticker": "BIO"},
        {"instrument_id": "equity:bio"}, {"decision_id": "forged"},
        {"source_approval_decision_path": "forged.json"}, {"execute_downstream": True},
        {"canonical_universe": "forged.json"}, {"price_history_root": "forged"},
        {"baseline_data06_run": "forged"}, {"baseline_run31_evidence": "forged"},
        {"raw_snapshot_root": "forged"}, {"as_of_date": "2099-01-01"},
        {"freshness_reference_date": "2099-01-01"},
    ],
)
def test_authority_carrying_data07_kwargs_are_rejected_before_any_call(tmp_path: Path, override: dict) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    result = execute_approved_candidate(
        decision, data07_runner=_stage_runner(tmp_path, "data07", calls),
        data07_operational_kwargs=override,
    )
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert result["stop_reason"].startswith("DATA07_AUTHORITY_OR_UNKNOWN_KWARGS_FORBIDDEN")
    assert calls == []


@pytest.mark.parametrize(
    ("stage", "override"),
    [
        ("data06", {"canonical_universe": "forged.json"}),
        ("data06", {"price_history_root": "forged"}),
        ("data06", {"baseline_data06_run": "forged"}),
        ("data06", {"as_of_date": "2099-01-01"}),
        ("run31", {"canonical_universe": "forged.json"}),
        ("run31", {"price_history_root": "forged"}),
        ("run31", {"baseline_run31_evidence": "forged"}),
        ("run31", {"compact_evidence_root": "forged"}),
        ("run31", {"freshness_reference_date": "2099-01-01"}),
    ],
)
def test_downstream_authority_kwargs_are_rejected_before_any_stage_call(
    tmp_path: Path, stage: str, override: dict
) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    kwargs = {f"{stage}_operational_kwargs": override}
    result = execute_approved_candidate(
        decision,
        data07_runner=_stage_runner(tmp_path, "data07", calls),
        data06_runner=_stage_runner(tmp_path, "data06", calls),
        run31_runner=_stage_runner(tmp_path, "run31", calls),
        **kwargs,
    )
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert result["stop_reason"].startswith(f"{stage.upper()}_AUTHORITY_OR_UNKNOWN_KWARGS_FORBIDDEN")
    assert calls == []


def test_public_binding_factory_and_stage_chain_are_not_supported_api() -> None:
    assert not hasattr(data11_execution, "make_validated_approval_binding")
    assert not hasattr(data11_execution, "execute_bound_stage_chain")


def test_plain_mapping_cannot_be_used_as_an_execution_context() -> None:
    calls = []
    result = execute_approved_candidate(
        {"validation_status": "approved", "execution_binding": {"ticker": "BIO"}},
        data07_runner=lambda **kwargs: calls.append(kwargs),
    )
    assert result["stop_reason"] == "APPROVAL_OR_EXECUTION_BINDING_INVALID"
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


def test_caller_cannot_supply_prebuilt_approval_validation(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    validation = validate_approval_decision(decision)
    with pytest.raises(TypeError, match="approval_validation"):
        execute_approved_candidate(
            decision,
            approval_validation=validation,
            data07_runner=lambda **kwargs: None,
        )


@pytest.mark.parametrize("mutated_artifact", ["approval_candidate.json", "governed_package_candidate.json"])
def test_runner_reads_only_validated_snapshot_when_original_bundle_changes(
    tmp_path: Path, mutated_artifact: str
) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    original_hashes = {
        name: _sha(decision_path.parent / name)
        for name in ("approval_candidate.json", "governed_package_candidate.json")
    }
    snapshot_paths: list[Path] = []
    calls: list[str] = []
    valid = _stage_runner(tmp_path, "data07", calls)

    def mutate_original_then_read_snapshot(**kwargs):
        original = decision_path.parent / mutated_artifact
        original.write_bytes(original.read_bytes() + b" ")
        snapshot_decision = Path(kwargs["source_approval_decision_path"])
        snapshot_package = Path(kwargs["operator_import_path"])
        snapshot_paths.extend((snapshot_decision, snapshot_package))
        assert snapshot_decision.parent == snapshot_package.parent
        assert snapshot_decision.parent != decision_path.parent
        assert _sha(snapshot_decision) == original_hashes["approval_candidate.json"]
        assert _sha(snapshot_package) == original_hashes["governed_package_candidate.json"]
        return valid(**kwargs)

    result = execute_approved_candidate(decision_path, data07_runner=mutate_original_then_read_snapshot)
    assert result["status"] == "completed"
    assert result["calls"] == {"data07": 1, "data06": 0, "run31": 0}
    assert calls == ["data07"]
    assert snapshot_paths and all(not path.exists() for path in snapshot_paths)


@pytest.mark.parametrize(
    ("mutated_artifact", "reason"),
    [
        ("approval_candidate.json", "APPROVAL_DECISION_CHECKSUM_MISMATCH"),
        ("governed_package_candidate.json", "GOVERNED_PACKAGE_CHECKSUM_MISMATCH"),
    ],
)
def test_checksum_change_during_snapshot_binding_blocks_data07(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated_artifact: str, reason: str
) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    original_materialize = data11_execution._materialize_execution_snapshot
    calls = []

    def mutate_before_materialize(context, root):
        artifact = decision_path.parent / mutated_artifact
        artifact.write_bytes(artifact.read_bytes() + b" ")
        return original_materialize(context, root)

    monkeypatch.setattr(data11_execution, "_materialize_execution_snapshot", mutate_before_materialize)
    result = execute_approved_candidate(
        decision_path,
        data07_runner=lambda **kwargs: calls.append(kwargs),
    )
    assert result["stop_reason"] == reason
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


@pytest.mark.parametrize(
    ("target", "mutation", "reason"),
    [
        ("output", "missing", "DATA07_OUTPUT_DECISION_CHECKSUM_MISMATCH"),
        ("receipt", "missing", "DATA07_RECEIPT_DECISION_CHECKSUM_MISMATCH"),
        ("output", "wrong", "DATA07_OUTPUT_DECISION_CHECKSUM_MISMATCH"),
        ("receipt", "wrong", "DATA07_RECEIPT_DECISION_CHECKSUM_MISMATCH"),
        ("output", "wrong_package", "DATA07_OUTPUT_GOVERNED_PACKAGE_CHECKSUM_MISMATCH"),
    ],
)
def test_data07_decision_and_package_checksum_contract_is_fail_closed(
    tmp_path: Path, target: str, mutation: str, reason: str
) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    calls: list[str] = []
    valid = _stage_runner(tmp_path, "data07", calls)

    def mutate_stage_evidence(**kwargs):
        result = valid(**kwargs)
        output_path = Path(result["output_path"])
        receipt_path = Path(result["receipt_path"])
        output = json.loads(output_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload = output if target == "output" else receipt
        field = "governed_package_sha256" if mutation == "wrong_package" else "decision_sha256"
        if mutation == "missing":
            payload["input_bindings"].pop(field)
        else:
            payload["input_bindings"][field] = "0" * 64
        if target == "output":
            _write_json(output_path, output)
            receipt["output_sha256"] = _sha(output_path)
            _write_json(receipt_path, receipt)
        else:
            _write_json(receipt_path, receipt)
        result.update(output_sha256=_sha(output_path), receipt_sha256=_sha(receipt_path))
        return result

    result = execute_approved_candidate(decision_path, data07_runner=mutate_stage_evidence)
    assert result["stop_reason"] == reason
    assert result["calls"] == {"data07": 1, "data06": 0, "run31": 0}


def test_safe_output_root_and_run_id_are_forwarded_as_non_authority_operationals(tmp_path: Path) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    relative_output = f"artifacts/market_engine/data11-safe-{tmp_path.name}"
    received = []
    valid = _stage_runner(tmp_path, "data07", [])

    def capture(**kwargs):
        received.append(kwargs)
        return valid(**kwargs)

    result = execute_approved_candidate(
        decision_path,
        data07_runner=capture,
        data07_operational_kwargs={"run_id": "data11-safe-run-1", "output_root": relative_output, "log_level": "INFO"},
    )
    assert result["status"] == "completed"
    assert received[0]["run_id"] == "data11-safe-run-1"
    assert received[0]["output_root"] == (Path(relative_output).resolve()).as_posix()
    assert received[0]["log_level"] == "INFO"


@pytest.mark.parametrize("run_id", ["", "contains space", "../escape", "x" * 129])
def test_invalid_run_id_blocks_before_data07(tmp_path: Path, run_id: str) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    calls = []
    result = execute_approved_candidate(
        decision_path,
        data07_runner=lambda **kwargs: calls.append(kwargs),
        data07_operational_kwargs={"run_id": run_id},
    )
    assert result["stop_reason"] == "DATA07_RUN_ID_INVALID"
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


def _temporary_execution_root(tmp_path: Path) -> Path:
    _materialize_authority(tmp_path, DEFAULT_RUN30_AUTHORITY)
    _materialize_authority(tmp_path, DEFAULT_DOWNSTREAM_AUTHORITY)
    return tmp_path


@pytest.mark.parametrize("output_root", ["../escape", "/tmp/escape", "artifacts/outside"])
def test_untrusted_output_roots_block_before_data07(tmp_path: Path, output_root: str) -> None:
    root = _temporary_execution_root(tmp_path)
    decision_path = _approve(_approval_bundle(root))
    calls = []
    result = execute_approved_candidate(
        decision_path,
        repository_root=root,
        data07_runner=lambda **kwargs: calls.append(kwargs),
        data07_operational_kwargs={"output_root": output_root},
    )
    assert result["calls"] == {"data07": 0, "data06": 0, "run31": 0}
    assert calls == []


def test_existing_output_target_blocks_before_data07(tmp_path: Path) -> None:
    root = _temporary_execution_root(tmp_path)
    target = root / "artifacts/market_engine/existing"
    target.mkdir(parents=True)
    decision_path = _approve(_approval_bundle(root))
    result = execute_approved_candidate(
        decision_path,
        repository_root=root,
        data07_runner=lambda **kwargs: None,
        data07_operational_kwargs={"output_root": "artifacts/market_engine/existing"},
    )
    assert result["stop_reason"] == "DATA07_OUTPUT_ROOT_ALREADY_EXISTS"
    assert result["calls"]["data07"] == 0


def test_output_root_symlink_escape_blocks_before_data07(tmp_path: Path) -> None:
    root = _temporary_execution_root(tmp_path)
    approved_root = root / "artifacts/market_engine"
    approved_root.mkdir(parents=True, exist_ok=True)
    outside = root / "outside"
    outside.mkdir()
    (approved_root / "escape-link").symlink_to(outside, target_is_directory=True)
    decision_path = _approve(_approval_bundle(root))
    result = execute_approved_candidate(
        decision_path,
        repository_root=root,
        data07_runner=lambda **kwargs: None,
        data07_operational_kwargs={"output_root": "artifacts/market_engine/escape-link/new"},
    )
    assert result["stop_reason"] == "DATA07_OUTPUT_ROOT_SYMLINK_ESCAPE"
    assert result["calls"]["data07"] == 0


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda value: value.update(package_id="forged-package"), "GOVERNED_PACKAGE_REPLAY_MISMATCH"),
        (lambda value: value["records"][0].update(ticker="BIO"), "GOVERNED_PACKAGE_IDENTITY_MISMATCH"),
        (lambda value: value["records"][0].update(instrument_id="equity:bio"), "GOVERNED_PACKAGE_IDENTITY_MISMATCH"),
        (lambda value: value["records"][0].update(approval_decision_reference="wrong"), "GOVERNED_PACKAGE_DECISION_REFERENCE_MISMATCH"),
        (lambda value: value["records"][0]["metrics"].pop(next(iter(value["records"][0]["metrics"]))), "GOVERNED_PACKAGE_METRIC_SET_MISMATCH"),
        (lambda value: next(iter(value["records"][0]["metrics"].values()))["derivation_lineage"].update(calculation_checksum="0" * 64), "GOVERNED_PACKAGE_CALCULATION_SET_MISMATCH"),
    ],
)
def test_rebound_governed_package_substitution_never_creates_execution_binding(tmp_path: Path, mutation, reason: str) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    governed_path = decision_path.parent / "governed_package_candidate.json"
    governed = json.loads(governed_path.read_text(encoding="utf-8"))
    mutation(governed)
    _write_json(governed_path, governed)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["artifact_bindings"]["governed_package_candidate"]["sha256"] = _sha(governed_path)
    _write_json(decision_path, decision)
    validation = validate_approval_decision(decision_path)
    assert validation["execution_binding"] is None
    assert reason in validation["reason_codes"]


@pytest.mark.parametrize("data07_result", [
    {"status": "blocked", "reason_codes": ["NO_IMPORT"]},
    {"status": "failed", "reason_codes": ["IMPORT_FAILED"]},
    {"status": "completed"},
])
def test_data07_nonvalidated_result_stops_data06_and_run31(tmp_path: Path, data07_result: dict) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    def data07(**kwargs):
        calls.append("data07")
        return data07_result
    result = execute_approved_candidate(
        decision, data07_runner=data07,
        data06_runner=_stage_runner(tmp_path, "data06", calls),
        run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert calls == ["data07"]
    assert result["stages"]["data06"]["status"] == "not_started"
    assert result["stages"]["run31"]["status"] == "not_started"


def test_data07_exception_stops_the_stage_chain(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    def data07(**kwargs):
        calls.append("data07")
        raise RuntimeError("synthetic")
    result = execute_approved_candidate(
        decision, data07_runner=data07,
        data06_runner=_stage_runner(tmp_path, "data06", calls),
        run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert result["status"] == "failed"
    assert result["calls"] == {"data07": 1, "data06": 0, "run31": 0}


def test_data07_checksum_mismatch_stops_the_stage_chain(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    valid = _stage_runner(tmp_path, "data07", calls)
    def bad_checksum(**kwargs):
        result = valid(**kwargs)
        result["output_sha256"] = "0" * 64
        return result
    result = execute_approved_candidate(
        decision, data07_runner=bad_checksum,
        data06_runner=_stage_runner(tmp_path, "data06", calls),
    )
    assert result["stop_reason"] == "DATA07_RESULT_CHECKSUM_MISMATCH"
    assert result["calls"]["data06"] == 0


@pytest.mark.parametrize("data06_result", [
    {"status": "blocked", "reason_codes": ["NO_REFRESH"]},
    {"status": "failed", "reason_codes": ["REFRESH_FAILED"]},
])
def test_data06_nonvalidated_result_stops_run31(tmp_path: Path, data06_result: dict) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    def data06(**kwargs):
        calls.append("data06")
        return data06_result
    result = execute_approved_candidate(
        decision, data07_runner=_stage_runner(tmp_path, "data07", calls),
        data06_runner=data06, run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert calls == ["data07", "data06"]
    assert result["stages"]["run31"]["status"] == "not_started"
    assert result["calls"]["run31"] == 0


def test_data06_cannot_rebind_itself_to_an_unrelated_data07_output(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    valid = _stage_runner(tmp_path, "data06", calls)
    def rebound(**kwargs):
        result = valid(**kwargs)
        output_path = Path(result["output_path"])
        receipt_path = Path(result["receipt_path"])
        output = json.loads(output_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        output["input_bindings"]["data07_output_sha256"] = "f" * 64
        receipt["input_bindings"]["data07_output_sha256"] = "f" * 64
        _write_json(output_path, output)
        receipt["output_sha256"] = _sha(output_path)
        _write_json(receipt_path, receipt)
        result.update(output_sha256=_sha(output_path), receipt_sha256=_sha(receipt_path))
        return result
    result = execute_approved_candidate(
        decision, data07_runner=_stage_runner(tmp_path, "data07", calls),
        data06_runner=rebound, run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert result["stop_reason"] == "DATA06_RECEIPT_INPUT_BINDING_MISMATCH"
    assert result["calls"]["run31"] == 0


def test_data06_exception_stops_run31(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    calls = []
    def data06(**kwargs):
        calls.append("data06")
        raise RuntimeError("synthetic")
    result = execute_approved_candidate(
        decision, data07_runner=_stage_runner(tmp_path, "data07", calls),
        data06_runner=data06, run31_runner=_stage_runner(tmp_path, "run31", calls),
    )
    assert result["stop_reason"] == "DATA06_EXCEPTION:RuntimeError"
    assert result["calls"]["run31"] == 0


def test_wrong_decision_reference_is_blocked(tmp_path: Path) -> None:
    decision_path = _approve(_approval_bundle(tmp_path))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision_id"] = "wrong-reference"
    _write_json(decision_path, decision)
    assert "DECISION_REFERENCE_MISMATCH" in validate_approval_decision(decision_path)["reason_codes"]


def test_real_downstream_prestate_is_nonzero_and_checksum_bound() -> None:
    state = load_downstream_prestate()
    assert state["measurement_status"] == "measured"
    assert state["before"] == {
        "fundamental_complete": 6, "fundamental_partial": 39, "fundamental_missing": 907,
        "invalid_stale_conflicting": 0, "advice_input_ready": 6, "full_advice_ready": 0,
        "unable_to_advise": 946,
    }


def test_missing_or_checksum_invalid_downstream_prestate_is_unknown(tmp_path: Path) -> None:
    assert load_downstream_prestate(repository_root=tmp_path)["measurement_status"] == "unknown_not_measured"
    _, authority, _ = _materialize_authority(tmp_path, DEFAULT_DOWNSTREAM_AUTHORITY)
    authority["trusted_artifacts"]["data06_summary"]["sha256"] = "0" * 64
    _write_json(tmp_path / DEFAULT_DOWNSTREAM_AUTHORITY, authority)
    result = load_downstream_prestate(repository_root=tmp_path)
    assert result["measurement_status"] == "unknown_not_measured"


def test_no_downstream_run_preserves_authoritative_state_and_proves_outside_zero() -> None:
    state = load_downstream_prestate()
    tickers = ["ASB", "ASH"]
    cohort = {"selected_tickers": tickers}
    results = {"instruments": [{"ticker": ticker, "status": "pending_approval", "reason_codes": ["APPROVAL_REQUIRED"]} for ticker in tickers]}
    delta = build_downstream_measurement(cohort, results, state, downstream_executed=False)
    assert delta["after_authoritative"] == delta["before"]
    assert delta["regressions_outside_selected_cohort"] == 0
    assert all(row["candidate_only_status"] == "candidate_partial_pending_approval" for row in delta["rows"])


def test_caller_mapping_cannot_claim_authoritative_after_state() -> None:
    before = {
        "measurement_status": "measured", "reason_codes": [], "before": {
            "fundamental_complete": 1, "fundamental_partial": 1, "fundamental_missing": 1,
            "invalid_stale_conflicting": 0, "advice_input_ready": 1, "full_advice_ready": 0,
            "unable_to_advise": 2,
        },
        "by_ticker": {
            "A": {"instrument_id": "equity:a", "overall_fundamental_status": "missing", "canonical_advice_input_ready": False},
            "B": {"instrument_id": "equity:b", "overall_fundamental_status": "complete", "canonical_advice_input_ready": True},
            "C": {"instrument_id": "equity:c", "overall_fundamental_status": "partial", "canonical_advice_input_ready": False},
        },
        "data06_run_id": "before-data06", "run31_run_id": "before-run31", "authority_path": "before.json", "authority_sha256": "a" * 64, "artifact_bindings": {},
    }
    after = copy.deepcopy(before)
    after["before"].update(fundamental_complete=0, fundamental_partial=3, fundamental_missing=0, advice_input_ready=0, unable_to_advise=3)
    after["by_ticker"]["A"].update(overall_fundamental_status="partial")
    after["by_ticker"]["B"].update(overall_fundamental_status="partial", canonical_advice_input_ready=False)
    delta = build_downstream_measurement(
        {"selected_tickers": ["A"]},
        {"instruments": [{"ticker": "A", "status": "pending_approval", "reason_codes": []}]},
        before,
        downstream_executed=True,
        authoritative_after=after,
    )
    assert delta["measurement_status"] == "unknown_not_measured"
    assert delta["reason_codes"] == ["AUTHORITATIVE_AFTER_STATE_UNKNOWN"]
    assert delta["rows"] == []


def test_checksum_bound_952_row_after_state_is_accepted(tmp_path: Path) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    authority_path, proof, _, _ = _bound_after_state(tmp_path, decision)
    after = load_downstream_after_authority(authority_path.name, execution_proof=proof, repository_root=tmp_path)
    assert after is not None
    delta = build_downstream_measurement(
        {"selected_tickers": ["ASH"]},
        {"instruments": [{"ticker": "ASH", "status": "pending_approval", "reason_codes": []}]},
        load_downstream_prestate(), downstream_executed=True, authoritative_after=after,
    )
    assert delta["measurement_status"] == "measured"
    assert delta["after_authoritative"] == load_downstream_prestate()["before"]
    assert delta["regressions_outside_selected_cohort"] == 0


@pytest.mark.parametrize("artifact", ["data06_per_ticker", "run31_per_ticker_index"])
def test_rebound_after_artifact_cannot_escape_stage_output_binding(tmp_path: Path, artifact: str) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    authority_path, proof, authority, payloads = _bound_after_state(tmp_path, decision)
    if artifact == "data06_per_ticker":
        payloads[artifact]["tickers"][0]["overall_fundamental_status"] = "complete"
    else:
        payloads[artifact]["instruments"][0]["canonical_advice_input_ready"] = True
    path = tmp_path / authority["trusted_artifacts"][artifact]["path"]
    _write_json(path, payloads[artifact])
    authority["trusted_artifacts"][artifact]["sha256"] = _sha(path)
    _write_json(authority_path, authority)
    assert load_downstream_after_authority(
        authority_path.name, execution_proof=proof, repository_root=tmp_path
    ) is None


@pytest.mark.parametrize(
    "payload_mutator",
    [
        lambda payloads: payloads["data06_per_ticker"]["tickers"].pop(),
        lambda payloads: payloads["data06_per_ticker"]["tickers"][1].update(
            ticker=payloads["data06_per_ticker"]["tickers"][0]["ticker"]
        ),
        lambda payloads: payloads["run31_per_ticker_index"]["instruments"][1].update(
            instrument_id=payloads["run31_per_ticker_index"]["instruments"][0]["instrument_id"]
        ),
        lambda payloads: payloads["run31_per_ticker_index"]["instruments"][0].update(
            canonical_advice_input_ready=True
        ),
    ],
)
def test_semantically_invalid_but_stage_bound_after_state_fails_closed(tmp_path: Path, payload_mutator) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    authority_path, proof, _, _ = _bound_after_state(tmp_path, decision, payload_mutator)
    assert load_downstream_after_authority(
        authority_path.name, execution_proof=proof, repository_root=tmp_path
    ) is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda authority: authority["trusted_artifacts"].pop("data07_receipt"),
        lambda authority: authority["lineage"].update(data06_run_id="wrong"),
        lambda authority: authority.update(canonical_universe_version="wrong"),
        lambda authority: authority["trusted_artifacts"]["data06_summary"].update(sha256="0" * 64),
    ],
)
def test_invalid_after_authority_fails_closed(tmp_path: Path, mutation) -> None:
    decision = _approve(_approval_bundle(tmp_path))
    authority_path, proof, authority, _ = _bound_after_state(tmp_path, decision)
    mutation(authority)
    _write_json(authority_path, authority)
    assert load_downstream_after_authority(
        authority_path.name, execution_proof=proof, repository_root=tmp_path
    ) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generated_at": "2026-08-13 00:00:00", "acquired_at": "2026-08-13T00:00:00Z", "source_publication_date": "2026-08-01", "trusted_now": "2026-08-13T00:00:00Z"},
        {"generated_at": "2026-08-13T00:00:00+00:00", "acquired_at": "2026-08-13T00:00:00Z", "source_publication_date": "2026-08-01", "trusted_now": "2026-08-13T00:00:00Z"},
        {"generated_at": "2026-08-13T00:00:00Z", "acquired_at": "2026-08-14T00:00:00Z", "source_publication_date": "2026-08-01", "trusted_now": "2026-08-13T00:00:00Z"},
        {"generated_at": "2026-08-13T00:00:00Z", "acquired_at": "2026-08-13T00:00:00Z", "source_publication_date": "2026-08-14", "trusted_now": "2026-08-13T00:00:00Z"},
    ],
)
def test_temporal_authority_fails_closed(kwargs: dict) -> None:
    with pytest.raises(Data11GovernanceError):
        validate_temporal_boundary(**kwargs)


def test_effective_freshness_is_recomputed_against_later_trusted_time() -> None:
    first = effective_freshness(source_publication_date="2026-05-01", acquired_at="2026-08-01T00:00:00Z", trusted_now="2026-08-01T00:00:00Z")
    later = effective_freshness(source_publication_date="2026-05-01", acquired_at="2026-08-01T00:00:00Z", trusted_now="2026-10-01T00:00:00Z")
    assert first["artifact_freshness"] == "current"
    assert later["artifact_freshness"] == "current"
    assert later["effective_freshness"] == "stale"


def test_every_new_governance_artifact_variant_passes_json_schema(tmp_path: Path) -> None:
    schema = json.loads(Path("config/market_engine/data11_governance_artifacts_v1.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    samples = [
        json.loads(DEFAULT_RUN30_AUTHORITY.read_text(encoding="utf-8")),
        json.loads(DEFAULT_DOWNSTREAM_AUTHORITY.read_text(encoding="utf-8")),
    ]
    bundle = _approval_bundle(tmp_path)
    samples.extend(
        json.loads((bundle / name).read_text(encoding="utf-8"))
        for name in ("source_evidence.json", "mapping_review.json", "approval_candidate.json")
    )
    authority_path, proof, authority, _ = _bound_after_state(tmp_path, _approve(bundle))
    samples.append(authority)
    for stage, binding in proof.stage_bindings.items():
        samples.append({
            "schema_version": "market-engine-data11-stage-result-v1", "stage": stage,
            "run_id": binding["run_id"], "status": "completed",
            "output_path": binding["output_path"], "output_sha256": binding["output_sha256"],
            "receipt_path": binding["receipt_path"], "receipt_sha256": binding["receipt_sha256"],
        })
        samples.append(json.loads(Path(binding["receipt_path"]).read_text(encoding="utf-8")))
    assert authority_path.is_file()
    state = load_downstream_prestate()
    samples.append(build_downstream_measurement(
        {"selected_tickers": ["ASB"]},
        {"instruments": [{"ticker": "ASB", "status": "blocked", "reason_codes": ["APPROVAL_REQUIRED"]}]},
        state,
        downstream_executed=False,
    ))
    for sample in samples:
        validator.validate(sample)
