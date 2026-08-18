from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


STAGE_RESULT_SCHEMA = "market-engine-data11-stage-result-v1"
STAGE_RECEIPT_SCHEMA = "market-engine-data11-stage-receipt-v1"
AFTER_AUTHORITY_SCHEMA = "market-engine-data11-downstream-after-authority-v1"
DATA07_OUTPUT_SCHEMA = "market-engine-data11-data07-import-output-v1"
DATA06_OUTPUT_SCHEMA = "market-engine-data11-data06-refresh-output-v1"
RUN31_OUTPUT_SCHEMA = "market-engine-data11-run31-refresh-output-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROOF_TOKEN = object()

_OPERATIONAL_ALLOWLISTS = {
    "data07": frozenset({"run_id", "output_root", "log_level"}),
    "data06": frozenset({"run_id", "output_root", "log_level"}),
    "run31": frozenset({"run_id", "output_root", "log_level"}),
}


class Data11ExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class _BundleArtifact:
    name: str
    filename: str
    original_path: str
    sha256: str
    content: bytes


@dataclass(frozen=True)
class _ExecutionContext:
    decision_id: str
    decision_path: str
    decision_sha256: str
    decision_content: bytes
    ticker: str
    instrument_id: str
    governed_package_path: str
    governed_package_sha256: str
    governed_package_id: str
    approved_metrics: tuple[str, ...]
    calculation_checksums: tuple[str, ...]
    artifacts: tuple[_BundleArtifact, ...]
    approval_validation: Mapping[str, Any]
    runtime_settings: Mapping[str, Mapping[str, Any]]
    repository_root: str

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "market-engine-data11-approved-execution-binding-v2",
            "decision_id": self.decision_id,
            "decision_path": self.decision_path,
            "decision_sha256": self.decision_sha256,
            "ticker": self.ticker,
            "instrument_id": self.instrument_id,
            "bundle_root": Path(self.decision_path).parent.as_posix(),
            "governed_package_path": self.governed_package_path,
            "governed_package_sha256": self.governed_package_sha256,
            "governed_package_id": self.governed_package_id,
            "approved_metrics": list(self.approved_metrics),
            "calculation_checksums": list(self.calculation_checksums),
            "approval_artifact_bindings": {
                artifact.name: {
                    "path": artifact.original_path,
                    "sha256": artifact.sha256,
                }
                for artifact in self.artifacts
            },
        }


@dataclass(frozen=True)
class ValidatedExecutionProof:
    _token: object
    approval_binding: Mapping[str, Any]
    stage_bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ValidatedDownstreamAuthorityState:
    _token: object
    payload: Mapping[str, Any]


def _execute_bound_stage_chain(
    *,
    context: _ExecutionContext | None,
    approval_validation: Mapping[str, Any],
    data07_runner: Callable[..., Any],
    data07_operational_kwargs: Mapping[str, Any] | None = None,
    data06_runner: Callable[..., Any] | None = None,
    data06_operational_kwargs: Mapping[str, Any] | None = None,
    run31_runner: Callable[..., Any] | None = None,
    run31_operational_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stages = {name: {"status": "not_started", "invocation_count": 0, "input_checksums": {}, "output_checksums": {}} for name in ("data07", "data06", "run31")}
    response: dict[str, Any] = {
        "status": "blocked",
        "approval_validation": dict(approval_validation),
        "execution_binding": approval_validation.get("execution_binding"),
        "stages": stages,
        "calls": {"data07": 0, "data06": 0, "run31": 0},
        "stop_reason": None,
        "downstream_authority_status": "not_established",
    }
    if approval_validation.get("validation_status") != "approved" or context is None:
        response["stop_reason"] = "APPROVAL_OR_EXECUTION_BINDING_INVALID"
        return response
    binding = context.public_payload()
    try:
        operationals = {
            "data07": _operational_kwargs("data07", data07_operational_kwargs, context=context),
            "data06": _operational_kwargs("data06", data06_operational_kwargs, context=context),
            "run31": _operational_kwargs("run31", run31_operational_kwargs, context=context),
        }
    except (Data11ExecutionError, OSError) as exc:
        if isinstance(exc, OSError):
            response["stop_reason"] = "APPROVAL_BUNDLE_REBIND_FAILED"
            return response
        response["stop_reason"] = str(exc)
        return response

    try:
        with tempfile.TemporaryDirectory(prefix="me-data11-approved-bundle-") as snapshot_root:
            snapshot = _materialize_execution_snapshot(context, Path(snapshot_root))
            return _execute_snapshot_stage_chain(
                context=context,
                binding=binding,
                snapshot=snapshot,
                operationals=operationals,
                data07_runner=data07_runner,
                data06_runner=data06_runner,
                run31_runner=run31_runner,
                stages=stages,
                response=response,
            )
    except (Data11ExecutionError, OSError) as exc:
        if isinstance(exc, OSError):
            response["stop_reason"] = "APPROVAL_BUNDLE_REBIND_FAILED"
            return response
        response["stop_reason"] = str(exc)
        return response


def _execute_snapshot_stage_chain(
    *,
    context: _ExecutionContext,
    binding: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    operationals: Mapping[str, Mapping[str, Any]],
    data07_runner: Callable[..., Any],
    data06_runner: Callable[..., Any] | None,
    run31_runner: Callable[..., Any] | None,
    stages: dict[str, dict[str, Any]],
    response: dict[str, Any],
) -> dict[str, Any]:
    snapshot_bindings = snapshot["artifact_bindings"]

    data07_input = {
        "decision_sha256": binding["decision_sha256"],
        "governed_package_path": snapshot["governed_package_path"],
        "governed_package_sha256": binding["governed_package_sha256"],
        "governed_package_id": binding["governed_package_id"],
        "decision_id": binding["decision_id"],
        "ticker": binding["ticker"],
        "instrument_id": binding["instrument_id"],
        "approved_metrics": binding["approved_metrics"],
        "calculation_checksums": binding["calculation_checksums"],
        "approval_artifact_bindings": snapshot_bindings,
    }
    data07_call = {
        **dict(context.runtime_settings["data07"]),
        **operationals["data07"],
        "source_mode": "operator_import",
        "operator_import_path": snapshot["governed_package_path"],
        "source_approval_decision_path": snapshot["decision_path"],
        "execute_downstream": False,
        "allow_overwrite": False,
    }
    first = _invoke_stage("data07", data07_runner, data07_call, data07_input, stages, response)
    if first is None:
        return response
    if data06_runner is None:
        response["status"] = "completed"
        response["stop_reason"] = "DATA07_COMPLETED_DOWNSTREAM_NOT_REQUESTED"
        response["downstream_authority_status"] = "data07_only"
        return response

    data06_input = {
        "data07_output_path": first["output_path"],
        "data07_output_sha256": first["output_sha256"],
        "data07_receipt_path": first["receipt_path"],
        "data07_receipt_sha256": first["receipt_sha256"],
    }
    data06_call = {**dict(context.runtime_settings["data06"]), **operationals["data06"], **data06_input}
    second = _invoke_stage("data06", data06_runner, data06_call, data06_input, stages, response)
    if second is None:
        return response
    if run31_runner is None:
        response["status"] = "completed"
        response["stop_reason"] = "DATA06_COMPLETED_RUN31_NOT_REQUESTED"
        response["downstream_authority_status"] = "data06_only"
        return response

    run31_input = {
        "data06_output_path": second["output_path"],
        "data06_output_sha256": second["output_sha256"],
        "data06_receipt_path": second["receipt_path"],
        "data06_receipt_sha256": second["receipt_sha256"],
    }
    run31_call = {**dict(context.runtime_settings["run31"]), **operationals["run31"], **run31_input}
    third = _invoke_stage("run31", run31_runner, run31_call, run31_input, stages, response)
    if third is None:
        return response
    proof = ValidatedExecutionProof(_PROOF_TOKEN, dict(binding), {"data07": first, "data06": second, "run31": third})
    response.update(
        status="completed",
        stop_reason=None,
        downstream_authority_status="stage_chain_validated_after_authority_required",
        validated_execution_proof=proof,
    )
    return response


def _operational_kwargs(
    stage: str,
    values: Mapping[str, Any] | None,
    *,
    context: _ExecutionContext,
) -> dict[str, Any]:
    supplied = dict(values or {})
    forbidden = sorted(set(supplied) - _OPERATIONAL_ALLOWLISTS[stage])
    if forbidden:
        raise Data11ExecutionError(f"{stage.upper()}_AUTHORITY_OR_UNKNOWN_KWARGS_FORBIDDEN:{','.join(forbidden)}")
    run_id = supplied.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None):
        raise Data11ExecutionError(f"{stage.upper()}_RUN_ID_INVALID")
    if "output_root" in supplied:
        supplied["output_root"] = _safe_output_root(
            supplied["output_root"], stage=stage, repository_root=Path(context.repository_root)
        )
    if "log_level" in supplied and supplied["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise Data11ExecutionError(f"{stage.upper()}_LOG_LEVEL_INVALID")
    return supplied


def _safe_output_root(value: Any, *, stage: str, repository_root: Path) -> str:
    candidate = Path(str(value or ""))
    if not candidate.parts or candidate.is_absolute() or ".." in candidate.parts:
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_ROOT_UNTRUSTED")
    if candidate.parts[:2] != ("artifacts", "market_engine"):
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_ROOT_OUTSIDE_APPROVED_ROOT")
    approved = (repository_root / "artifacts" / "market_engine").resolve()
    resolved = (repository_root / candidate).resolve()
    try:
        resolved.relative_to(approved)
    except ValueError as exc:
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_ROOT_SYMLINK_ESCAPE") from exc
    if resolved.exists():
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_ROOT_ALREADY_EXISTS")
    return resolved.as_posix()


def _materialize_execution_snapshot(context: _ExecutionContext, root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256(Path(context.decision_path)) != context.decision_sha256:
        raise Data11ExecutionError("APPROVAL_DECISION_CHECKSUM_MISMATCH")
    for artifact in context.artifacts:
        if _sha256(Path(artifact.original_path)) != artifact.sha256:
            code = "GOVERNED_PACKAGE_CHECKSUM_MISMATCH" if artifact.name == "governed_package_candidate" else "APPROVAL_BUNDLE_CHECKSUM_MISMATCH"
            raise Data11ExecutionError(code)
    decision_path = (root / Path(context.decision_path).name).resolve()
    decision_path.write_bytes(context.decision_content)
    if _sha256(decision_path) != context.decision_sha256:
        raise Data11ExecutionError("APPROVAL_DECISION_SNAPSHOT_CHECKSUM_MISMATCH")
    bindings: dict[str, dict[str, str]] = {}
    governed_path: Path | None = None
    for artifact in context.artifacts:
        target = (root / artifact.filename).resolve()
        target.write_bytes(artifact.content)
        if _sha256(target) != artifact.sha256:
            raise Data11ExecutionError("APPROVAL_BUNDLE_SNAPSHOT_CHECKSUM_MISMATCH")
        bindings[artifact.name] = {"path": target.as_posix(), "sha256": artifact.sha256}
        if artifact.name == "governed_package_candidate":
            governed_path = target
    if governed_path is None:
        raise Data11ExecutionError("GOVERNED_PACKAGE_SNAPSHOT_MISSING")
    return {
        "decision_path": decision_path.as_posix(),
        "governed_package_path": governed_path.as_posix(),
        "artifact_bindings": bindings,
    }


def _invoke_stage(
    stage: str,
    runner: Callable[..., Any],
    call_kwargs: Mapping[str, Any],
    expected_input: Mapping[str, Any],
    stages: dict[str, dict[str, Any]],
    response: dict[str, Any],
) -> dict[str, Any] | None:
    stages[stage]["invocation_count"] = 1
    response["calls"][stage] = 1
    stages[stage]["input_checksums"] = {key: value for key, value in expected_input.items() if key.endswith("sha256")}
    try:
        raw = runner(**dict(call_kwargs))
    except Exception as exc:
        stages[stage]["status"] = "failed"
        stages[stage]["reason_codes"] = [f"{stage.upper()}_EXCEPTION:{type(exc).__name__}"]
        response["status"] = "failed"
        response["stop_reason"] = stages[stage]["reason_codes"][0]
        return None
    if isinstance(raw, Mapping) and raw.get("status") in {"blocked", "failed"}:
        stages[stage]["status"] = str(raw["status"])
        stages[stage]["reason_codes"] = list(raw.get("reason_codes") or [f"{stage.upper()}_{str(raw['status']).upper()}"])
        response["status"] = str(raw["status"])
        response["stop_reason"] = stages[stage]["reason_codes"][0]
        return None
    try:
        validated = validate_stage_result(raw, stage=stage, expected_input=expected_input)
    except Data11ExecutionError as exc:
        stages[stage]["status"] = "failed"
        stages[stage]["reason_codes"] = [str(exc)]
        response["status"] = "failed"
        response["stop_reason"] = str(exc)
        return None
    stages[stage].update(status="completed", reason_codes=[], output_checksums={"output_sha256": validated["output_sha256"], "receipt_sha256": validated["receipt_sha256"]})
    return validated


def validate_stage_result(result: Any, *, stage: str, expected_input: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping) or result.get("schema_version") != STAGE_RESULT_SCHEMA:
        raise Data11ExecutionError(f"{stage.upper()}_RESULT_MALFORMED")
    if result.get("stage") != stage or result.get("status") != "completed" or not isinstance(result.get("run_id"), str):
        raise Data11ExecutionError(f"{stage.upper()}_RESULT_STATUS_OR_IDENTITY_INVALID")
    output_path = _strict_file(result.get("output_path"), f"{stage.upper()}_OUTPUT")
    receipt_path = _strict_file(result.get("receipt_path"), f"{stage.upper()}_RECEIPT")
    if result.get("output_sha256") != _sha256(output_path) or result.get("receipt_sha256") != _sha256(receipt_path):
        raise Data11ExecutionError(f"{stage.upper()}_RESULT_CHECKSUM_MISMATCH")
    output = _strict_json(output_path)
    receipt = _strict_json(receipt_path)
    expected_schema = {"data07": DATA07_OUTPUT_SCHEMA, "data06": DATA06_OUTPUT_SCHEMA, "run31": RUN31_OUTPUT_SCHEMA}[stage]
    if output.get("schema_version") != expected_schema or output.get("stage") != stage or output.get("run_id") != result.get("run_id") or output.get("status") != "completed":
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_CONTRACT_INVALID")
    if receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA or receipt.get("stage") != stage or receipt.get("run_id") != result.get("run_id") or receipt.get("status") != "completed":
        raise Data11ExecutionError(f"{stage.upper()}_RECEIPT_CONTRACT_INVALID")
    if stage == "data07":
        for label, payload in (("OUTPUT", output), ("RECEIPT", receipt)):
            inputs = payload.get("input_bindings")
            if not isinstance(inputs, Mapping) or inputs.get("decision_sha256") != expected_input.get("decision_sha256"):
                raise Data11ExecutionError(f"DATA07_{label}_DECISION_CHECKSUM_MISMATCH")
            if inputs.get("governed_package_sha256") != expected_input.get("governed_package_sha256"):
                raise Data11ExecutionError(f"DATA07_{label}_GOVERNED_PACKAGE_CHECKSUM_MISMATCH")
    if receipt.get("input_bindings") != dict(expected_input):
        raise Data11ExecutionError(f"{stage.upper()}_RECEIPT_INPUT_BINDING_MISMATCH")
    if receipt.get("output_path") != output_path.as_posix() or receipt.get("output_sha256") != result.get("output_sha256"):
        raise Data11ExecutionError(f"{stage.upper()}_RECEIPT_OUTPUT_BINDING_MISMATCH")
    if output.get("input_bindings") != dict(expected_input):
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_INPUT_BINDING_MISMATCH")
    if not isinstance(output.get("artifact_bindings"), Mapping):
        raise Data11ExecutionError(f"{stage.upper()}_OUTPUT_ARTIFACT_BINDINGS_INVALID")
    return {
        "run_id": result["run_id"],
        "output_path": output_path.as_posix(),
        "output_sha256": result["output_sha256"],
        "receipt_path": receipt_path.as_posix(),
        "receipt_sha256": result["receipt_sha256"],
    }


def load_downstream_after_authority(
    authority_path: str | Path,
    *,
    execution_proof: ValidatedExecutionProof | None,
    repository_root: str | Path = ".",
) -> ValidatedDownstreamAuthorityState | None:
    if not isinstance(execution_proof, ValidatedExecutionProof) or execution_proof._token is not _PROOF_TOKEN:
        return None
    root = Path(repository_root).resolve()
    try:
        authority_file = _repo_file(root, authority_path)
        authority = _strict_json(authority_file)
        if authority.get("schema_version") != AFTER_AUTHORITY_SCHEMA:
            return None
        artifacts = _load_bound(root, authority.get("trusted_artifacts"))
        _validate_after_lineage(authority, artifacts, execution_proof)
        payload = _reconcile_after_state(authority, artifacts)
        payload["authority_path"] = authority_file.relative_to(root).as_posix()
        payload["authority_sha256"] = _sha256(authority_file)
        payload["approval_binding"] = dict(execution_proof.approval_binding)
    except (Data11ExecutionError, KeyError, TypeError, ValueError):
        return None
    return ValidatedDownstreamAuthorityState(_PROOF_TOKEN, payload)


def validated_after_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, ValidatedDownstreamAuthorityState) and value._token is _PROOF_TOKEN:
        return value.payload
    return None


def _validate_after_lineage(authority: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], proof: ValidatedExecutionProof) -> None:
    stages = proof.stage_bindings
    for stage in ("data07", "data06", "run31"):
        bound = stages[stage]
        trusted = authority.get("trusted_artifacts") or {}
        if (
            trusted.get(f"{stage}_output", {}).get("sha256") != bound["output_sha256"]
            or trusted.get(f"{stage}_receipt", {}).get("sha256") != bound["receipt_sha256"]
        ):
            raise Data11ExecutionError("AFTER_STAGE_PROOF_CHECKSUM_MISMATCH")
        if authority.get("lineage", {}).get(f"{stage}_run_id") != bound["run_id"]:
            raise Data11ExecutionError("AFTER_STAGE_RUN_ID_MISMATCH")
    if artifacts["data07_receipt"].get("input_bindings", {}).get("governed_package_sha256") != proof.approval_binding.get("governed_package_sha256"):
        raise Data11ExecutionError("AFTER_DATA07_APPROVAL_LINEAGE_MISMATCH")
    if artifacts["data06_receipt"].get("input_bindings", {}).get("data07_receipt_sha256") != stages["data07"]["receipt_sha256"]:
        raise Data11ExecutionError("AFTER_DATA06_LINEAGE_MISMATCH")
    if artifacts["run31_receipt"].get("input_bindings", {}).get("data06_receipt_sha256") != stages["data06"]["receipt_sha256"]:
        raise Data11ExecutionError("AFTER_RUN31_LINEAGE_MISMATCH")
    for stage, names in {
        "data06": ("data06_manifest", "data06_summary", "data06_per_ticker"),
        "run31": ("run31_compact_index", "run31_evidence_summary", "run31_per_ticker_index", "run31_top_level_checksums"),
    }.items():
        declared = artifacts[f"{stage}_output"].get("artifact_bindings")
        expected = {name: authority["trusted_artifacts"].get(name) for name in names}
        if declared != expected or any(value is None for value in expected.values()):
            raise Data11ExecutionError(f"AFTER_{stage.upper()}_ARTIFACT_LINEAGE_MISMATCH")


def _reconcile_after_state(authority: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    universe_version = authority.get("canonical_universe_version")
    universe_size = authority.get("canonical_universe_size")
    if universe_size != 952:
        raise Data11ExecutionError("AFTER_UNIVERSE_SIZE_INVALID")
    data06_manifest = artifacts["data06_manifest"]
    data06_summary = artifacts["data06_summary"]
    data06_rows = artifacts["data06_per_ticker"].get("tickers")
    run31_index = artifacts["run31_compact_index"]
    run31_summary = artifacts["run31_evidence_summary"]
    run31_rows = artifacts["run31_per_ticker_index"].get("instruments")
    top = artifacts["run31_top_level_checksums"]
    if data06_manifest.get("run_id") != authority.get("lineage", {}).get("data06_run_id") or run31_index.get("run_id") != authority.get("lineage", {}).get("run31_run_id"):
        raise Data11ExecutionError("AFTER_PRODUCER_RUN_ID_MISMATCH")
    canonical = data06_manifest.get("canonical_universe") or {}
    if canonical.get("universe_version") != universe_version or canonical.get("total_instruments") != universe_size or run31_index.get("canonical_universe_version") != universe_version:
        raise Data11ExecutionError("AFTER_UNIVERSE_BINDING_MISMATCH")
    if not isinstance(data06_rows, list) or not isinstance(run31_rows, list) or len(data06_rows) != universe_size or len(run31_rows) != universe_size:
        raise Data11ExecutionError("AFTER_CANONICAL_ROW_SET_INCOMPLETE")
    data06_by = _unique_rows(data06_rows, ticker_key="ticker")
    run31_by = _unique_rows(run31_rows, ticker_key="symbol")
    if set(data06_by) != set(run31_by):
        raise Data11ExecutionError("AFTER_TICKER_SET_MISMATCH")
    by_ticker: dict[str, dict[str, Any]] = {}
    for ticker, fundamental in data06_by.items():
        readiness = run31_by[ticker]
        if fundamental.get("instrument_id") != readiness.get("instrument_id"):
            raise Data11ExecutionError("AFTER_INSTRUMENT_ID_MISMATCH")
        status = str(fundamental.get("overall_fundamental_status") or "missing")
        context = readiness.get("fundamental_context")
        if not isinstance(context, Mapping) or context.get("status") != status:
            raise Data11ExecutionError("AFTER_FUNDAMENTAL_CONSUMER_MISMATCH")
        advice_ready = readiness.get("canonical_advice_input_ready")
        full_ready = readiness.get("full_advice_ready")
        unable = readiness.get("canonical_advice_label") == "unable_to_advise"
        if not isinstance(advice_ready, bool) or not isinstance(full_ready, bool):
            raise Data11ExecutionError("AFTER_READINESS_FLAGS_INVALID")
        by_ticker[ticker] = {**dict(fundamental), "canonical_advice_input_ready": advice_ready, "full_advice_ready": full_ready, "unable_to_advise": unable}
    counts = _state_counts(by_ticker)
    declared = data06_summary.get("after") or {}
    run_summary = run31_summary.get("summary") or {}
    expected = {
        "fundamental_complete": declared.get("fundamental_complete"),
        "fundamental_partial": declared.get("fundamental_partial"),
        "fundamental_missing": declared.get("fundamental_missing"),
        "invalid_stale_conflicting": declared.get("invalid_stale_conflicting"),
        "advice_input_ready": run_summary.get("canonical_advice_input_ready"),
        "full_advice_ready": run_summary.get("full_advice_ready"),
        "unable_to_advise": run_summary.get("unable_to_advise"),
    }
    if counts != expected:
        raise Data11ExecutionError("AFTER_TOTAL_RECONCILIATION_FAILED")
    trusted = authority["trusted_artifacts"]
    for name in ("run31_evidence_summary", "run31_per_ticker_index"):
        filename = Path(trusted[name]["path"]).name
        if top.get(filename) != trusted[name]["sha256"]:
            raise Data11ExecutionError("AFTER_RUN31_CHECKSUM_INDEX_MISMATCH")
    return {
        "measurement_status": "measured",
        "reason_codes": [],
        "authority_path": None,
        "authority_sha256": None,
        "artifact_bindings": authority["trusted_artifacts"],
        "before": counts,
        "by_ticker": by_ticker,
        "data06_run_id": data06_manifest["run_id"],
        "run31_run_id": run31_index["run_id"],
    }


def _state_counts(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for row in rows.values():
        status = str(row.get("overall_fundamental_status") or "missing")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "fundamental_complete": statuses.get("complete", 0),
        "fundamental_partial": statuses.get("partial", 0),
        "fundamental_missing": statuses.get("missing", 0),
        "invalid_stale_conflicting": sum(statuses.get(name, 0) for name in ("invalid", "stale", "conflicting")),
        "advice_input_ready": sum(row.get("canonical_advice_input_ready") is True for row in rows.values()),
        "full_advice_ready": sum(row.get("full_advice_ready") is True for row in rows.values()),
        "unable_to_advise": sum(row.get("unable_to_advise") is True for row in rows.values()),
    }


def _unique_rows(rows: list[Any], *, ticker_key: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    instruments: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Data11ExecutionError("AFTER_ROW_INVALID")
        ticker = str(row.get(ticker_key) or "").upper()
        instrument = str(row.get("instrument_id") or "")
        if not ticker or ticker in result or not instrument or instrument in instruments:
            raise Data11ExecutionError("AFTER_DUPLICATE_OR_MISSING_IDENTITY")
        result[ticker] = row
        instruments.add(instrument)
    return result


def _load_bound(root: Path, bindings: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(bindings, Mapping):
        raise Data11ExecutionError("AFTER_BINDINGS_MISSING")
    result = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise Data11ExecutionError("AFTER_BINDING_INVALID")
        path = _repo_file(root, binding.get("path"))
        if binding.get("sha256") != _sha256(path):
            raise Data11ExecutionError("AFTER_BINDING_CHECKSUM_MISMATCH")
        result[str(name)] = _strict_json(path)
    return result


def _repo_file(root: Path, value: Any) -> Path:
    candidate = Path(str(value or ""))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise Data11ExecutionError("AFTER_PATH_UNTRUSTED")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Data11ExecutionError("AFTER_PATH_ESCAPES_ROOT") from exc
    if not resolved.is_file():
        raise Data11ExecutionError("AFTER_FILE_MISSING")
    return resolved


def _strict_file(value: Any, label: str) -> Path:
    path = Path(str(value or "")).resolve()
    if not path.is_file():
        raise Data11ExecutionError(f"{label}_MISSING")
    return path


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise Data11ExecutionError("STRICT_JSON_INVALID") from exc
    if not isinstance(value, Mapping):
        raise Data11ExecutionError("JSON_OBJECT_REQUIRED")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
