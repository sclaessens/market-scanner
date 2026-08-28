from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import UTC, date, datetime, timedelta
from inspect import signature
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from market_engine.data.local_market_data_universe import UNIVERSE_SNAPSHOT_SCHEMA_VERSION
from market_engine.data.scheduled_canonical_price_refresh import expected_completed_session
from market_engine.run import current_technical_screening as screening
from market_engine.source_refresh import advisory_ohlc_history as history
from market_engine.source_refresh import advisory_ohlc_history_runtime as runtime


SHA = "a" * 40
NOW = datetime(2026, 8, 13, 6, tzinfo=UTC)
ACTION_DIGEST = "0fde654d4c6e659b45783a725dc92f1bfb0baa6c2de64b34e814dc206ff4aaaf"


def _instrument(index: int) -> dict[str, object]:
    ticker = f"T{index:03d}"
    return {
        "instrument_id": f"equity:{ticker.lower()}",
        "symbol": ticker,
        "source_symbol": ticker,
        "source_mapping_status": "mapped",
        "currency": "USD",
        "exchange": "US",
        "country": "US",
        "asset_type": "equity",
    }


def _weekdays(end: date, count: int) -> list[str]:
    result: list[str] = []
    current = end
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current -= timedelta(days=1)
    return list(reversed(result))


def _raw_result(instrument: dict[str, object], policy: dict[str, object]) -> dict[str, object]:
    _profile, expected = expected_completed_session(instrument, NOW)
    bars = [
        {
            "session": session,
            "open": "100.00",
            "high": "102.00",
            "low": "99.00",
            "close": "101.00",
            "volume": None if index == 0 else "1000",
            "volume_status": "not_reported" if index == 0 else "provider_reported",
        }
        for index, session in enumerate(_weekdays(expected, 3))
    ]
    return {
        "instrument_id": instrument["instrument_id"],
        "canonical_ticker": instrument["symbol"],
        "source_symbol": instrument["source_symbol"],
        "currency": instrument["currency"],
        "price_basis": policy["price_basis"],
        "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"],
        "bars": bars,
    }


@pytest.fixture
def governed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repository = Path.cwd()
    required = [
        history.DEFAULT_UNIVERSE_SNAPSHOT,
        history.DEFAULT_POLICY_PATH,
        runtime.DEFAULT_RUNTIME_CONFIG,
        runtime.DEFAULT_RUNTIME_SCHEMA,
        runtime.DEFAULT_CHECKPOINT_SCHEMA,
        runtime.DEFAULT_RECEIPT_SCHEMA,
        screening.DEFAULT_SCREENING_POLICY,
    ]
    for relative in required:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
    instruments = tuple(_instrument(index) for index in range(952))
    universe = {
        "schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "universe_version": "runtime-fixture-v1",
        "universe_checksum": "c" * 64,
        "instruments": list(instruments),
    }
    policy = json.loads((tmp_path / history.DEFAULT_POLICY_PATH).read_text(encoding="utf-8"))
    config = json.loads((tmp_path / runtime.DEFAULT_RUNTIME_CONFIG).read_text(encoding="utf-8"))
    monkeypatch.setattr(runtime, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(history, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(screening, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "_authority", lambda: (universe, policy, config, instruments))
    monkeypatch.setattr(runtime, "_load_runtime_config", lambda: config)
    monkeypatch.setattr(history, "_load_canonical_universe", lambda: universe)
    monkeypatch.setattr(history, "_load_canonical_policy", lambda: policy)
    monkeypatch.setattr(history, "load_authoritative_universe", lambda _path: universe)
    monkeypatch.setattr(history, "_current_repository_head_sha", lambda: SHA)
    monkeypatch.setattr(runtime, "_utc_now", lambda: NOW)
    return {
        "root": tmp_path,
        "universe": universe,
        "policy": policy,
        "config": config,
        "instruments": instruments,
    }


def _persist_and_gate(run_id: str, stage_id: str, outcome: str = "success") -> None:
    runtime.record_persistence_receipt(
        run_id,
        stage_id,
        artifact_name=runtime._expected_artifact_name(run_id, stage_id),
        artifact_id=f"artifact-{stage_id}",
        artifact_digest=ACTION_DIGEST,
    )
    runtime.gate_stage(run_id, stage_id, execution_outcome=outcome)


def _stage_success(
    run_id: str, plan: dict[str, object], stage_id: str,
    results: dict[str, dict[str, object]], cumulative: int,
) -> None:
    spec = runtime._stage_spec(plan, stage_id)
    payload = {
        "schema_version": runtime.STAGE_RESULT_VERSION,
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "stage_id": stage_id,
        "chunk_index": spec.chunk_index,
        "chunk_identity_digest": spec.identity_digest,
        "instrument_ids": list(spec.instrument_ids),
        "acquisition_started_at": plan["acquisition_started_at"],
        "fallbacks_used": len(spec.instrument_ids) if stage_id == runtime.FALLBACK_STAGE_ID else 0,
        "results": results,
    }
    runtime._atomic_write_json(runtime._result_path(run_id, stage_id), payload)
    successful, empty, errors = runtime._result_counts(results)
    checkpoint = runtime._checkpoint(
        run_id=run_id,
        plan=plan,
        spec=spec,
        phase="fallback_acquisition" if stage_id == runtime.FALLBACK_STAGE_ID else "primary_acquisition",
        execution_status="success",
        reason_code=None,
        attempted=len(spec.instrument_ids),
        completed=len(spec.instrument_ids),
        cumulative_attempted=cumulative,
        cumulative_completed=cumulative,
        successful_frames=successful,
        empty_frames=empty,
        provider_errors=errors,
        fallbacks_used=len(spec.instrument_ids) if stage_id == runtime.FALLBACK_STAGE_ID else 0,
    )
    runtime._atomic_write_json(runtime._checkpoint_path(run_id, stage_id), checkpoint)
    _persist_and_gate(run_id, stage_id)


def _complete_zero_fallback_runtime(
    governed_runtime: dict[str, object], run_id: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]], Path]:
    instruments = governed_runtime["instruments"]
    policy = governed_runtime["policy"]
    raw = {str(row["instrument_id"]): _raw_result(row, policy) for row in instruments}
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)
    plan = runtime._load_plan(run_id)
    cumulative = 0
    for row in plan["primary_chunks"]:
        stage_id = row["stage_id"]
        results = {instrument_id: raw[instrument_id] for instrument_id in row["instrument_ids"]}
        cumulative += len(results)
        _stage_success(run_id, plan, stage_id, results, cumulative)
    fallback_plan = json.loads(runtime.plan_fallbacks(run_id).read_text())
    assert fallback_plan["instrument_ids"] == []
    _stage_success(run_id, plan, runtime.FALLBACK_STAGE_ID, {}, 952)
    final_root = governed_runtime["root"] / history.DEFAULT_OUTPUT_ROOT / run_id
    return plan, raw, final_root


def test_runtime_config_and_canonical_chunk_plan_are_exact(governed_runtime) -> None:
    config = runtime._load_runtime_config()
    specs = runtime._primary_stage_specs(governed_runtime["instruments"], config["primary_chunk_size"])
    flattened = [identity for spec in specs for identity in spec.instrument_ids]
    assert config == {
        "schema_version": runtime.RUNTIME_VERSION,
        "primary_chunk_size": 64,
        "provider_stage_timeout_seconds": 600,
        "provider_termination_grace_seconds": 5,
        "heartbeat_interval_seconds": 60,
    }
    assert len(specs) == 15
    assert [len(spec.instrument_ids) for spec in specs] == [64] * 14 + [56]
    assert flattened == [row["instrument_id"] for row in governed_runtime["instruments"]]
    assert len(flattened) == len(set(flattened)) == 952
    assert [spec.stage_id for spec in specs] == [f"primary-chunk-{index:03d}" for index in range(15)]


def test_preflight_contains_no_environment_values_and_blocks_collision(governed_runtime, monkeypatch) -> None:
    monkeypatch.setenv("ME_SR28_SECRET_SENTINEL", "must-not-be-captured")
    root = runtime.create_preflight("preflight-fixture")
    environment = json.loads((root / "runtime_environment.json").read_text())
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    serialized = json.dumps(environment)
    assert "must-not-be-captured" not in serialized
    assert set(environment["packages"]) == {"pandas", "numpy", "yfinance", "curl_cffi", "requests", "lxml"}
    assert environment["operational_runtime_config_sha256"] == runtime._sha256_file(
        governed_runtime["root"] / runtime.DEFAULT_RUNTIME_CONFIG
    )
    assert checkpoint["complete"] is False
    assert checkpoint["authority_boundary"] == runtime.DIAGNOSTIC_AUTHORITY
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="RUN_COLLISION"):
        runtime.create_preflight("preflight-fixture")


def test_non_cooperative_process_is_term_kill_bounded_and_reaped() -> None:
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    started = time.monotonic()
    result = runtime._supervise_command(
        [sys.executable, "-c", code],
        timeout_seconds=0.15,
        grace_seconds=0.05,
        heartbeat_interval_seconds=0.05,
        heartbeat=lambda _elapsed, _pid: None,
    )
    assert time.monotonic() - started < 2
    assert result.timed_out is True
    assert result.termination_sequence == ("SIGTERM", "SIGKILL")
    assert result.returncode == -signal.SIGKILL


def test_receipt_requires_supported_action_outputs_and_true_outcome(governed_runtime) -> None:
    run_id = "receipt-fixture"
    runtime.create_preflight(run_id)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="ARTIFACT_ID_MISSING"):
        runtime.record_persistence_receipt(
            run_id, runtime.PREFLIGHT_STAGE_ID,
            artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
            artifact_id="", artifact_digest=ACTION_DIGEST,
        )
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="ARTIFACT_DIGEST_INVALID"):
        runtime.record_persistence_receipt(
            run_id, runtime.PREFLIGHT_STAGE_ID,
            artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
            artifact_id="123", artifact_digest="invented",
        )
    runtime.record_persistence_receipt(
        run_id, runtime.PREFLIGHT_STAGE_ID,
        artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
        artifact_id="123", artifact_digest=ACTION_DIGEST,
    )
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="STAGE_GATE_BLOCKED"):
        runtime.gate_stage(run_id, runtime.PREFLIGHT_STAGE_ID, execution_outcome="failure")
    assert not runtime._gate_path(run_id, runtime.PREFLIGHT_STAGE_ID).exists()


def test_upload_artifact_v4_raw_sha256_output_contract(governed_runtime) -> None:
    """upload-artifact v4 emits a bare lowercase SHA-256 hexadecimal digest."""
    run_id = "official-action-digest-fixture"
    runtime.create_preflight(run_id)
    assert runtime.ARTIFACT_DIGEST.fullmatch(ACTION_DIGEST)
    assert runtime.ARTIFACT_DIGEST.fullmatch(f"sha256:{ACTION_DIGEST}") is None
    runtime.record_persistence_receipt(
        run_id,
        runtime.PREFLIGHT_STAGE_ID,
        artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
        artifact_id="123",
        artifact_digest=ACTION_DIGEST,
    )
    receipt = runtime._read_json(runtime._receipt_path(run_id, runtime.PREFLIGHT_STAGE_ID))
    assert receipt["artifact_digest"] == ACTION_DIGEST


def test_checkpoint_authority_mismatch_blocks_persistence_receipt(governed_runtime) -> None:
    run_id = "checkpoint-binding"
    runtime.create_preflight(run_id)
    checkpoint_path = runtime._checkpoint_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["source_main_sha"] = "d" * 40
    runtime._atomic_write_json(checkpoint_path, checkpoint)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="CHECKPOINT_BINDING_INVALID"):
        runtime.record_persistence_receipt(
            run_id, runtime.PREFLIGHT_STAGE_ID,
            artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
            artifact_id="123", artifact_digest=ACTION_DIGEST,
        )


def test_workflow_has_static_ordered_upload_receipt_gate_groups() -> None:
    workflow = Path(".github/workflows/advisory-ohlc-history.yml").read_text(encoding="utf-8")
    assert "contents: read" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "timeout-minutes: 90" in workflow
    assert workflow.count("name: Execute primary chunk ") == 15
    assert workflow.count("name: Upload primary chunk ") == 15
    assert workflow.count("name: Record primary chunk ") == 15
    assert workflow.count("name: Gate primary chunk ") == 15
    assert workflow.count("continue-on-error: true") == 17
    assert workflow.count("retention-days: 14") == 18
    assert workflow.count("if-no-files-found: error") == 18
    assert "outputs.artifact-id" in workflow and "outputs.artifact-digest" in workflow
    assert "steps.execute_primary_000.outcome" in workflow
    assert "steps.execute_primary_000.conclusion" not in workflow
    assert "market-data" not in workflow and "git push" not in workflow
    positions = []
    for index in range(15):
        token = f"{index:03d}"
        group = [
            workflow.index(f"name: Execute primary chunk {token}"),
            workflow.index(f"name: Upload primary chunk {token} diagnostic"),
            workflow.index(f"name: Record primary chunk {token} persistence receipt"),
            workflow.index(f"name: Gate primary chunk {token}"),
        ]
        assert group == sorted(group)
        positions.extend(group)
    assert positions == sorted(positions)
    assert workflow.index("name: Plan global singleton fallbacks") > positions[-1]
    assert workflow.index("name: Assemble final advisory history") > workflow.index("name: Gate global fallback stage")


def test_workflow_is_manual_validation_only_and_has_no_automatic_trigger() -> None:
    workflow = Path(".github/workflows/advisory-ohlc-history.yml").read_text(encoding="utf-8")
    trigger_block = workflow.split("\nconcurrency:", maxsplit=1)[0]
    assert re.findall(r"(?m)^  ([a-z_]+):", trigger_block) == ["workflow_dispatch"]
    assert "schedule:" not in trigger_block


def test_runtime_cli_exposes_no_operational_authority_overrides() -> None:
    assert {
        "_timeout_seconds", "_grace_seconds", "_heartbeat_interval_seconds", "_supervisor"
    }.isdisjoint(signature(runtime.execute_stage).parameters)
    with pytest.raises(SystemExit):
        runtime.run_command(["preflight", "--run-id", "x", "--provider-stage-timeout-seconds", "1"])
    with pytest.raises(SystemExit):
        runtime.run_command(["execute-stage", "--run-id", "x", "--stage-id", "primary-chunk-000", "--chunk-size", "1"])
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="STAGE_ID_INVALID"):
        runtime._validate_stage_id("primary-chunk-015")


def test_json_config_and_diagnostic_schemas_validate() -> None:
    pairs = [
        (runtime.DEFAULT_RUNTIME_SCHEMA, runtime.DEFAULT_RUNTIME_CONFIG),
    ]
    for schema_path, value_path in pairs:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        value = json.loads(value_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def test_global_zero_fallback_stage_and_chunked_final_are_semantically_identical(
    governed_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "semantic-equivalence"
    instruments = governed_runtime["instruments"]
    policy = governed_runtime["policy"]
    raw = {str(row["instrument_id"]): _raw_result(row, policy) for row in instruments}
    monolithic_manifest, monolithic_root = history._build_advisory_ohlc_history_impl(
        run_id=run_id,
        source_main_sha=SHA,
        provider=lambda rows, at, configured: raw,
        clock=runtime._AssemblyClock(NOW, NOW),
    )
    monolithic_files = {
        path.relative_to(monolithic_root).as_posix(): path.read_bytes()
        for path in monolithic_root.rglob("*") if path.is_file()
    }
    _monolithic_screening_manifest, monolithic_screening_root = screening._run_current_technical_screening_impl(
        run_id="semantic-equivalence-screening",
        history_artifact_root=monolithic_root,
        universe_path=history.DEFAULT_UNIVERSE_SNAPSHOT,
        history_policy_path=history.DEFAULT_POLICY_PATH,
        screening_policy_path=governed_runtime["root"] / screening.DEFAULT_SCREENING_POLICY,
        now=NOW,
    )
    monolithic_screening_files = {
        path.relative_to(monolithic_screening_root).as_posix(): path.read_bytes()
        for path in monolithic_screening_root.rglob("*") if path.is_file()
    }
    shutil.rmtree(monolithic_screening_root)
    shutil.rmtree(monolithic_root)

    _plan, chunked_raw, _final_root = _complete_zero_fallback_runtime(governed_runtime, run_id)
    assert chunked_raw == raw
    chunked_manifest, chunked_root = runtime.assemble_final_history(run_id)
    chunked_files = {
        path.relative_to(chunked_root).as_posix(): path.read_bytes()
        for path in chunked_root.rglob("*") if path.is_file()
    }
    assert chunked_manifest == monolithic_manifest
    assert chunked_manifest["observations_sha256"] == monolithic_manifest["observations_sha256"]
    assert chunked_files == monolithic_files
    assert json.loads((chunked_root / "screening_eligibility.json").read_text()) == json.loads(
        monolithic_files["screening_eligibility.json"]
    )
    _chunked_screening_manifest, chunked_screening_root = screening._run_current_technical_screening_impl(
        run_id="semantic-equivalence-screening",
        history_artifact_root=chunked_root,
        universe_path=history.DEFAULT_UNIVERSE_SNAPSHOT,
        history_policy_path=history.DEFAULT_POLICY_PATH,
        screening_policy_path=governed_runtime["root"] / screening.DEFAULT_SCREENING_POLICY,
        now=NOW,
    )
    chunked_screening_files = {
        path.relative_to(chunked_screening_root).as_posix(): path.read_bytes()
        for path in chunked_screening_root.rglob("*") if path.is_file()
    }
    assert chunked_screening_files == monolithic_screening_files


def test_final_assembly_blocks_missing_preflight_receipt(governed_runtime) -> None:
    run_id = "missing-preflight-receipt"
    _plan, _raw, final_root = _complete_zero_fallback_runtime(governed_runtime, run_id)
    runtime._receipt_path(run_id, runtime.PREFLIGHT_STAGE_ID).unlink()
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()


def test_final_assembly_blocks_corrupt_preflight_receipt_binding(governed_runtime) -> None:
    run_id = "corrupt-preflight-receipt"
    _plan, _raw, final_root = _complete_zero_fallback_runtime(governed_runtime, run_id)
    receipt_path = runtime._receipt_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    receipt = json.loads(receipt_path.read_text())
    receipt["source_main_sha"] = "d" * 40
    runtime._atomic_write_json(receipt_path, receipt)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="RECEIPT_BINDING_INVALID"):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_final_assembly_blocks_missing_or_corrupt_preflight_gate(governed_runtime, mutation: str) -> None:
    run_id = f"{mutation}-preflight-gate"
    _plan, _raw, final_root = _complete_zero_fallback_runtime(governed_runtime, run_id)
    gate_path = runtime._gate_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    if mutation == "missing":
        gate_path.unlink()
    else:
        gate = json.loads(gate_path.read_text())
        gate["status"] = "failed"
        runtime._atomic_write_json(gate_path, gate)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()


@pytest.mark.parametrize("mutation", ["authority", "execution_status"])
def test_final_assembly_blocks_corrupt_preflight_checkpoint(governed_runtime, mutation: str) -> None:
    run_id = f"corrupt-preflight-checkpoint-{mutation}"
    _plan, _raw, final_root = _complete_zero_fallback_runtime(governed_runtime, run_id)
    checkpoint_path = runtime._checkpoint_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    checkpoint = json.loads(checkpoint_path.read_text())
    receipt_path = runtime._receipt_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    receipt = json.loads(receipt_path.read_text())
    if mutation == "authority":
        checkpoint["source_main_sha"] = "d" * 40
    else:
        checkpoint["execution_status"] = "failure"
        receipt["execution_status"] = "failure"
    runtime._atomic_write_json(checkpoint_path, checkpoint)
    if mutation == "execution_status":
        receipt["checkpoint_sha256"] = runtime._sha256_file(checkpoint_path)
        runtime._atomic_write_json(receipt_path, receipt)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()


def test_fallback_selection_is_global_canonical_and_capped(governed_runtime) -> None:
    run_id = "fallback-selection"
    instruments = governed_runtime["instruments"]
    policy = governed_runtime["policy"]
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)
    plan = runtime._load_plan(run_id)
    missing = {str(row["instrument_id"]) for row in instruments[:30]}
    cumulative = 0
    for row in plan["primary_chunks"]:
        results = {
            instrument_id: (
                {"error_code": "PROVIDER_HISTORY_MISSING"}
                if instrument_id in missing
                else _raw_result(next(item for item in instruments if item["instrument_id"] == instrument_id), policy)
            )
            for instrument_id in row["instrument_ids"]
        }
        cumulative += len(results)
        _stage_success(run_id, plan, row["stage_id"], results, cumulative)
    fallback = json.loads(runtime.plan_fallbacks(run_id).read_text())
    assert fallback["instrument_ids"] == [row["instrument_id"] for row in instruments[:25]]
    assert len(fallback["instrument_ids"]) == 25


def test_final_assembly_rejects_missing_extra_cross_bound_and_over_budget_evidence(governed_runtime) -> None:
    run_id = "assembly-rejections"
    instruments = governed_runtime["instruments"]
    policy = governed_runtime["policy"]
    raw = {str(row["instrument_id"]): _raw_result(row, policy) for row in instruments}
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)
    plan = runtime._load_plan(run_id)
    cumulative = 0
    for row in plan["primary_chunks"]:
        results = {instrument_id: raw[instrument_id] for instrument_id in row["instrument_ids"]}
        cumulative += len(results)
        _stage_success(run_id, plan, row["stage_id"], results, cumulative)
    runtime.plan_fallbacks(run_id)
    _stage_success(run_id, plan, runtime.FALLBACK_STAGE_ID, {}, 952)
    final_root = governed_runtime["root"] / history.DEFAULT_OUTPUT_ROOT / run_id

    receipt_path = runtime._receipt_path(run_id, "primary-chunk-007")
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.unlink()
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()
    receipt_path.write_bytes(receipt_bytes)

    extra = runtime._staging_run_root(run_id) / "stages" / "unexpected-stage"
    extra.mkdir()
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="missing or unexpected stage"):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()
    extra.rmdir()

    result_path = runtime._result_path(run_id, "primary-chunk-009")
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    result["source_main_sha"] = "d" * 40
    runtime._atomic_write_json(result_path, result)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="authority binding"):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()
    result_path.write_bytes(result_bytes)

    fallback_path = runtime._fallback_plan_path(run_id)
    fallback_bytes = fallback_path.read_bytes()
    fallback = json.loads(fallback_bytes)
    fallback["instrument_ids"] = [row["instrument_id"] for row in instruments[:26]]
    fallback["chunk_identity_digest"] = runtime._identity_digest(fallback["instrument_ids"])
    runtime._atomic_write_json(fallback_path, fallback)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)
    assert not final_root.exists()
    fallback_path.write_bytes(fallback_bytes)


def test_missing_receipt_and_tampered_binding_block_later_stage_and_final(governed_runtime) -> None:
    run_id = "receipt-binding"
    runtime.create_preflight(run_id)
    plan = runtime._load_plan(run_id)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="RUNTIME_EVIDENCE_INVALID"):
        runtime.gate_stage(run_id, runtime.PREFLIGHT_STAGE_ID, execution_outcome="success")
    runtime.record_persistence_receipt(
        run_id, runtime.PREFLIGHT_STAGE_ID,
        artifact_name=runtime._expected_artifact_name(run_id, runtime.PREFLIGHT_STAGE_ID),
        artifact_id="preflight-id", artifact_digest=ACTION_DIGEST,
    )
    receipt_path = runtime._receipt_path(run_id, runtime.PREFLIGHT_STAGE_ID)
    receipt = json.loads(receipt_path.read_text())
    receipt["source_main_sha"] = "d" * 40
    runtime._atomic_write_json(receipt_path, receipt)
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="RECEIPT_BINDING_INVALID"):
        runtime.gate_stage(run_id, runtime.PREFLIGHT_STAGE_ID, execution_outcome="success")
    assert not history.DEFAULT_OUTPUT_ROOT.joinpath(run_id).exists()
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue):
        runtime.assemble_final_history(run_id)


def test_diagnostic_roots_are_rejected_by_all_analytic_consumers(governed_runtime) -> None:
    run_id = "loader-isolation"
    diagnostic = runtime.create_preflight(run_id)
    with pytest.raises(history.AdvisoryHistoryIssue):
        history._load_advisory_ohlc_history_impl(
            diagnostic,
            universe_path=governed_runtime["root"] / history.DEFAULT_UNIVERSE_SNAPSHOT,
            policy_path=governed_runtime["root"] / history.DEFAULT_POLICY_PATH,
            now=NOW,
        )
    with pytest.raises(history.AdvisoryHistoryIssue):
        screening._run_current_technical_screening_impl(
            run_id="screening-isolation",
            history_artifact_root=diagnostic,
            universe_path=governed_runtime["root"] / history.DEFAULT_UNIVERSE_SNAPSHOT,
            history_policy_path=governed_runtime["root"] / history.DEFAULT_POLICY_PATH,
            now=NOW,
        )
    with pytest.raises(history.AdvisoryHistoryIssue):
        screening._build_run33_grounded_handoff_impl(
            run_id="run33-isolation",
            screening_root=diagnostic,
            history_root=diagnostic,
            price_root=diagnostic,
            universe_path=governed_runtime["root"] / history.DEFAULT_UNIVERSE_SNAPSHOT,
            history_policy_path=governed_runtime["root"] / history.DEFAULT_POLICY_PATH,
            now=NOW,
        )


def test_timeout_checkpoint_is_fail_closed_and_temp_only(governed_runtime) -> None:
    run_id = "timeout-stage"
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)

    def timed_out_supervisor(command, **kwargs):
        return runtime.SupervisedResult(-signal.SIGKILL, True, ("SIGTERM", "SIGKILL"), 0.2)

    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="provider_chunk_timeout"):
        runtime._execute_stage_impl(
            run_id,
            "primary-chunk-000",
            _timeout_seconds=0.1,
            _grace_seconds=0.01,
            _heartbeat_interval_seconds=0.01,
            _supervisor=timed_out_supervisor,
        )
    checkpoint = json.loads(runtime._checkpoint_path(run_id, "primary-chunk-000").read_text())
    assert checkpoint["complete"] is False
    assert checkpoint["execution_status"] == "failure"
    assert checkpoint["reason_code"] == "provider_chunk_timeout"
    assert not runtime._result_path(run_id, "primary-chunk-000").exists()
    assert not runtime._stage_staging_root(run_id, "primary-chunk-001").exists()
    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="PRIOR_GATE_INVALID"):
        runtime.execute_stage(run_id, "primary-chunk-001")
    assert not (governed_runtime["root"] / history.DEFAULT_OUTPUT_ROOT / run_id).exists()


def test_resource_snapshot_failure_is_null_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_read_proc_value", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unsupported")))
    monkeypatch.setattr(runtime.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("unsupported")))
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _path: (_ for _ in ()).throw(OSError("unsupported")))
    snapshot = runtime._resource_snapshot()
    for key in ("process_rss_bytes", "system_memory_available_bytes", "disk_free_bytes", "load_average_1m"):
        assert snapshot[key] is None
        assert snapshot["resource_metric_errors"][key] == "OSError"


def test_primary_adapter_receives_only_expected_symbols_and_uses_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    instruments = [_instrument(index) for index in range(3)]
    policy = json.loads(history.DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    seen: list[tuple[str, ...]] = []
    fallbacks: list[str] = []
    monkeypatch.setattr(history, "download_yfinance_batch", lambda symbols, start, end: seen.append(tuple(symbols)) or {})
    monkeypatch.setattr(history, "_download_yfinance_history", lambda symbol, start, end: fallbacks.append(symbol))
    result = history._acquire_primary_with_existing_adapter(instruments, NOW, policy, canonical_instruments=instruments)
    assert seen == [("T000", "T001", "T002")]
    assert fallbacks == []
    assert set(result) == {row["instrument_id"] for row in instruments}


def test_supervised_child_uses_new_process_session(tmp_path: Path) -> None:
    marker = tmp_path / "session.txt"
    code = f"import os,pathlib; pathlib.Path({str(marker)!r}).write_text(str(os.getsid(0) == os.getpid()))"
    result = runtime._supervise_command(
        [sys.executable, "-c", code],
        timeout_seconds=2,
        grace_seconds=0.1,
        heartbeat_interval_seconds=0.1,
        heartbeat=lambda _elapsed, _pid: None,
    )
    assert result.returncode == 0 and result.timed_out is False
    assert marker.read_text() == "True"


def test_malformed_worker_staging_is_rejected_without_promotion(governed_runtime) -> None:
    run_id = "invalid-worker"
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)

    def malformed_supervisor(command, **kwargs):
        token = command[command.index("--worker-token") + 1]
        stage = command[command.index("--stage-id") + 1]
        temp = runtime._worker_temp_root(run_id, stage, token)
        runtime._atomic_write_json(temp / "result.json", {"schema_version": "wrong"})
        (temp / "unexpected.txt").write_text("forbidden", encoding="utf-8")
        return runtime.SupervisedResult(0, False, (), 0.01)

    with pytest.raises(runtime.AdvisoryHistoryRuntimeIssue, match="invalid_worker_result"):
        runtime._execute_stage_impl(run_id, "primary-chunk-000", _supervisor=malformed_supervisor)
    checkpoint = json.loads(runtime._checkpoint_path(run_id, "primary-chunk-000").read_text())
    assert checkpoint["reason_code"] == "invalid_worker_result"
    assert checkpoint["execution_status"] == "failure"
    assert not runtime._stage_staging_root(run_id, "primary-chunk-000").exists()


def test_valid_worker_result_is_bound_validated_and_atomically_promoted(governed_runtime) -> None:
    run_id = "valid-worker"
    instruments = governed_runtime["instruments"]
    policy = governed_runtime["policy"]
    runtime.create_preflight(run_id)
    _persist_and_gate(run_id, runtime.PREFLIGHT_STAGE_ID)
    plan = runtime._load_plan(run_id)

    def successful_supervisor(command, **kwargs):
        token = command[command.index("--worker-token") + 1]
        stage_id = command[command.index("--stage-id") + 1]
        spec = runtime._stage_spec(plan, stage_id)
        by_id = {row["instrument_id"]: row for row in instruments}
        results = {identity: _raw_result(by_id[identity], policy) for identity in spec.instrument_ids}
        payload = {
            "schema_version": runtime.STAGE_RESULT_VERSION,
            "run_id": run_id,
            "source_main_sha": plan["source_main_sha"],
            "canonical_universe_sha256": plan["canonical_universe_sha256"],
            "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
            "history_policy_sha256": plan["history_policy_sha256"],
            "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
            "stage_id": stage_id,
            "chunk_index": spec.chunk_index,
            "chunk_identity_digest": spec.identity_digest,
            "instrument_ids": list(spec.instrument_ids),
            "acquisition_started_at": plan["acquisition_started_at"],
            "fallbacks_used": 0,
            "results": results,
        }
        runtime._atomic_write_json(runtime._worker_temp_root(run_id, stage_id, token) / "result.json", payload)
        return runtime.SupervisedResult(0, False, (), 0.01)

    checkpoint_path = runtime._execute_stage_impl(run_id, "primary-chunk-000", _supervisor=successful_supervisor)
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["execution_status"] == "success"
    assert checkpoint["chunk_completed_identity_count"] == 64
    assert runtime._result_path(run_id, "primary-chunk-000").is_file()


def test_heartbeat_is_structured_bounded_operational_telemetry() -> None:
    from io import StringIO

    output = StringIO()
    runtime._emit_event(
        event="provider_stage_heartbeat",
        run_id="heartbeat",
        phase="primary_acquisition",
        source_main_sha=SHA,
        stage_id="primary-chunk-000",
        chunk_index=0,
        completed_identities=0,
        elapsed_seconds=60,
        seconds_since_progress=60,
        resource_reader=lambda _pid: {
            "process_rss_bytes": 1,
            "system_memory_available_bytes": 2,
            "disk_free_bytes": 3,
            "load_average_1m": 0.5,
            "resource_metric_errors": {},
        },
        stdout=output,
    )
    payload = json.loads(output.getvalue())
    assert payload["event"] == "provider_stage_heartbeat"
    assert payload["elapsed_seconds"] == 60
    assert payload["seconds_since_progress"] == 60
    assert payload["total_identities"] == 952
    assert "cookie" not in output.getvalue().lower()
