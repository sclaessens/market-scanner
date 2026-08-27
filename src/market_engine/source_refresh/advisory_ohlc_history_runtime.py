from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

from jsonschema import Draft202012Validator, FormatChecker

from market_engine.source_refresh import advisory_ohlc_history as history


RUNTIME_VERSION = "market-engine-advisory-ohlc-history-runtime-v1"
PLAN_VERSION = "market-engine-advisory-ohlc-history-runtime-plan-v1"
STAGE_RESULT_VERSION = "market-engine-advisory-ohlc-history-stage-result-v1"
FALLBACK_PLAN_VERSION = "market-engine-advisory-ohlc-history-fallback-plan-v1"
CHECKPOINT_VERSION = "market-engine-advisory-ohlc-history-diagnostic-checkpoint-v1"
RECEIPT_VERSION = "market-engine-advisory-ohlc-history-persistence-receipt-v1"
GATE_VERSION = "market-engine-advisory-ohlc-history-stage-gate-v1"
DIAGNOSTIC_AUTHORITY = "diagnostic_only_partial_never_analytic"
RECEIPT_AUTHORITY = "diagnostic_persistence_only_never_analytic"

DEFAULT_RUNTIME_CONFIG = Path("config/market_engine/advisory_ohlc_history_runtime.json")
DEFAULT_RUNTIME_SCHEMA = Path("config/market_engine/advisory_ohlc_history_runtime_v1.schema.json")
DEFAULT_CHECKPOINT_SCHEMA = Path("config/market_engine/advisory_ohlc_history_diagnostic_checkpoint_v1.schema.json")
DEFAULT_RECEIPT_SCHEMA = Path("config/market_engine/advisory_ohlc_history_persistence_receipt_v1.schema.json")
DIAGNOSTIC_ROOT = Path("artifacts/market_engine/advisory_ohlc_history_diagnostics")
STAGING_ROOT = Path("artifacts/market_engine/advisory_ohlc_history_staging")
PRIMARY_CHUNK_COUNT = 15
FALLBACK_STAGE_ID = "fallback-chunk-000"
PREFLIGHT_STAGE_ID = "preflight"
ARTIFACT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
STAGE_ID = re.compile(r"^(?:preflight|primary-chunk-0(?:0[0-9]|1[0-4])|fallback-chunk-000)$")
WORKER_TOKEN = re.compile(r"^[0-9a-f]{32}$")


class AdvisoryHistoryRuntimeIssue(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    chunk_index: int | None
    instrument_ids: tuple[str, ...]
    identity_digest: str


@dataclass(frozen=True)
class SupervisedResult:
    returncode: int
    timed_out: bool
    termination_sequence: tuple[str, ...]
    elapsed_seconds: float


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "symlink evidence files are forbidden")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_EVIDENCE_INVALID", f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_EVIDENCE_INVALID", f"{path.name} must contain an object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    repository = _repository_root().resolve()
    try:
        relative_parent = path.parent.absolute().relative_to(repository)
    except ValueError as exc:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "evidence path escapes repository authority") from exc
    cursor = repository
    for part in relative_parent.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "symlink evidence paths are forbidden")
    if path.parent.is_symlink() or path.is_symlink():
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "symlink evidence paths are forbidden")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_run_id(run_id: str) -> None:
    if not history.IDENTIFIER.fullmatch(run_id):
        raise AdvisoryHistoryRuntimeIssue("RUN_ID_INVALID", "run_id is invalid")


def _approved_run_root(base: Path, run_id: str) -> Path:
    _validate_run_id(run_id)
    repository = _repository_root().resolve()
    approved = (repository / base).resolve()
    destination = (approved / run_id).resolve()
    if approved not in destination.parents:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "run path escapes its repository-owned root")
    cursor = repository
    for part in base.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "symlink runtime roots are forbidden")
    if (approved / run_id).is_symlink():
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "symlink run roots are forbidden")
    return destination


def _diagnostic_run_root(run_id: str) -> Path:
    return _approved_run_root(DIAGNOSTIC_ROOT, run_id)


def _staging_run_root(run_id: str) -> Path:
    return _approved_run_root(STAGING_ROOT, run_id)


def _stage_diagnostic_root(run_id: str, stage_id: str) -> Path:
    _validate_stage_id(stage_id)
    return _diagnostic_run_root(run_id) / "stages" / stage_id


def _stage_staging_root(run_id: str, stage_id: str) -> Path:
    _validate_stage_id(stage_id)
    return _staging_run_root(run_id) / "stages" / stage_id


def _validate_stage_id(stage_id: str) -> None:
    if not STAGE_ID.fullmatch(stage_id):
        raise AdvisoryHistoryRuntimeIssue("STAGE_ID_INVALID", "stage ID is not repository-owned")


def _load_schema(path: Path) -> dict[str, Any]:
    schema = _read_json(_repository_root() / path)
    Draft202012Validator.check_schema(schema)
    return schema


def _validate_schema(payload: Mapping[str, Any], schema_path: Path, code: str) -> None:
    errors = sorted(
        Draft202012Validator(_load_schema(schema_path), format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.path),
    )
    if errors:
        raise AdvisoryHistoryRuntimeIssue(code, errors[0].message)


def _load_runtime_config() -> dict[str, Any]:
    path = _repository_root() / DEFAULT_RUNTIME_CONFIG
    value = _read_json(path)
    _validate_schema(value, DEFAULT_RUNTIME_SCHEMA, "RUNTIME_CONFIG_INVALID")
    governed = {
        "schema_version": RUNTIME_VERSION,
        "primary_chunk_size": 64,
        "provider_stage_timeout_seconds": 600,
        "provider_termination_grace_seconds": 5,
        "heartbeat_interval_seconds": 60,
    }
    if value != governed:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_CONFIG_INVALID", "governed operational values changed")
    return value


def _authority() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[Mapping[str, Any], ...]]:
    universe = history._load_canonical_universe()
    policy = history._load_canonical_policy()
    runtime = _load_runtime_config()
    instruments = tuple(sorted(universe["instruments"], key=lambda row: str(row["instrument_id"])))
    return universe, policy, runtime, instruments


def _identity_digest(instrument_ids: Sequence[str]) -> str:
    return _sha256(_canonical_json(list(instrument_ids)))


def _primary_stage_specs(instruments: Sequence[Mapping[str, Any]], chunk_size: int) -> tuple[StageSpec, ...]:
    identities = tuple(str(row["instrument_id"]) for row in instruments)
    if len(identities) != 952 or len(identities) != len(set(identities)):
        raise AdvisoryHistoryRuntimeIssue("CHUNK_PLAN_INVALID", "canonical identities do not reconcile exactly")
    chunks = tuple(identities[offset:offset + chunk_size] for offset in range(0, len(identities), chunk_size))
    if len(chunks) != PRIMARY_CHUNK_COUNT or any(len(chunk) > 64 for chunk in chunks):
        raise AdvisoryHistoryRuntimeIssue("CHUNK_PLAN_INVALID", "canonical plan must contain 15 bounded chunks")
    return tuple(
        StageSpec(f"primary-chunk-{index:03d}", index, chunk, _identity_digest(chunk))
        for index, chunk in enumerate(chunks)
    )


def _bindings(
    *, universe: Mapping[str, Any], runtime: Mapping[str, Any], source_main_sha: str
) -> dict[str, Any]:
    return {
        "source_main_sha": source_main_sha,
        "canonical_universe_sha256": _sha256_file(_repository_root() / history.DEFAULT_UNIVERSE_SNAPSHOT),
        "canonical_universe_identity_digest": universe["universe_checksum"],
        "history_policy_sha256": _sha256_file(_repository_root() / history.DEFAULT_POLICY_PATH),
        "operational_runtime_config_sha256": _sha256_file(_repository_root() / DEFAULT_RUNTIME_CONFIG),
    }


def _plan_path(run_id: str) -> Path:
    return _stage_diagnostic_root(run_id, PREFLIGHT_STAGE_ID) / "runtime_plan.json"


def _checkpoint_path(run_id: str, stage_id: str) -> Path:
    return _stage_diagnostic_root(run_id, stage_id) / "checkpoint.json"


def _receipt_path(run_id: str, stage_id: str) -> Path:
    return _diagnostic_run_root(run_id) / "receipts" / f"{stage_id}.json"


def _gate_path(run_id: str, stage_id: str) -> Path:
    return _diagnostic_run_root(run_id) / "gates" / f"{stage_id}.json"


def _result_path(run_id: str, stage_id: str) -> Path:
    return _stage_staging_root(run_id, stage_id) / "result.json"


def _fallback_plan_path(run_id: str) -> Path:
    return _staging_run_root(run_id) / "fallback_plan.json"


def _expected_artifact_name(run_id: str, stage_id: str) -> str:
    return f"advisory-ohlc-history-diagnostic-{run_id}-{stage_id}"


def _load_plan(run_id: str) -> dict[str, Any]:
    plan = _read_json(_plan_path(run_id))
    required = {
        "schema_version", "run_id", "acquisition_started_at", "source_main_sha",
        "canonical_universe_sha256", "canonical_universe_identity_digest",
        "history_policy_sha256", "operational_runtime_config_sha256", "primary_chunks",
        "fallback_stage_id",
    }
    if set(plan) != required or plan.get("schema_version") != PLAN_VERSION or plan.get("run_id") != run_id:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PLAN_INVALID", "runtime plan contract is invalid")
    universe, _policy, runtime, instruments = _authority()
    source = history._current_repository_head_sha()
    expected_bindings = _bindings(universe=universe, runtime=runtime, source_main_sha=source)
    if any(plan.get(key) != value for key, value in expected_bindings.items()):
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PLAN_INVALID", "runtime plan authority binding changed")
    specs = _primary_stage_specs(instruments, int(runtime["primary_chunk_size"]))
    expected_chunks = [
        {
            "stage_id": spec.stage_id,
            "chunk_index": spec.chunk_index,
            "instrument_ids": list(spec.instrument_ids),
            "chunk_identity_digest": spec.identity_digest,
        }
        for spec in specs
    ]
    if plan["primary_chunks"] != expected_chunks or plan["fallback_stage_id"] != FALLBACK_STAGE_ID:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PLAN_INVALID", "runtime plan differs from canonical chunk authority")
    history._timestamp(str(plan["acquisition_started_at"]), "acquisition_started_at")
    return plan


def _stage_spec(plan: Mapping[str, Any], stage_id: str) -> StageSpec:
    _validate_stage_id(stage_id)
    if stage_id == PREFLIGHT_STAGE_ID:
        return StageSpec(stage_id, None, (), _identity_digest(()))
    if stage_id == FALLBACK_STAGE_ID:
        fallback = _read_json(_fallback_plan_path(str(plan["run_id"])))
        identities = tuple(str(value) for value in fallback.get("instrument_ids", ()))
        return StageSpec(stage_id, 0, identities, _identity_digest(identities))
    for row in plan["primary_chunks"]:
        if row["stage_id"] == stage_id:
            return StageSpec(stage_id, int(row["chunk_index"]), tuple(row["instrument_ids"]), row["chunk_identity_digest"])
    raise AdvisoryHistoryRuntimeIssue("STAGE_ID_INVALID", "stage is absent from canonical plan")


def _read_proc_value(path: Path, key: str, multiplier: int = 1) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return int(line.split()[1]) * multiplier
    raise ValueError(f"{key} unavailable")


def _resource_snapshot(pid: int | None = None) -> dict[str, Any]:
    errors: dict[str, str] = {}
    try:
        rss = _read_proc_value(Path(f"/proc/{pid or os.getpid()}/status"), "VmRSS", 1024)
    except (OSError, UnicodeError, ValueError) as exc:
        rss = None
        errors["process_rss_bytes"] = type(exc).__name__
    try:
        available = _read_proc_value(Path("/proc/meminfo"), "MemAvailable", 1024)
    except (OSError, UnicodeError, ValueError) as exc:
        available = None
        errors["system_memory_available_bytes"] = type(exc).__name__
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError) as exc:
        load = None
        errors["load_average_1m"] = type(exc).__name__
    try:
        disk = shutil.disk_usage(_repository_root()).free
    except OSError as exc:
        disk = None
        errors["disk_free_bytes"] = type(exc).__name__
    return {
        "process_rss_bytes": rss,
        "system_memory_available_bytes": available,
        "disk_free_bytes": disk,
        "load_average_1m": load,
        "resource_metric_errors": errors,
    }


def _emit_event(
    *, event: str, run_id: str, phase: str, source_main_sha: str,
    stage_id: str | None = None, chunk_index: int | None = None,
    completed_identities: int = 0, fallbacks_used: int = 0,
    elapsed_seconds: float = 0, seconds_since_progress: float = 0,
    resource_reader: Callable[[int | None], Mapping[str, Any]] = _resource_snapshot,
    pid: int | None = None, stdout: TextIO = sys.stdout,
) -> None:
    payload = {
        "event": event,
        "run_id": run_id,
        "source_main_sha": source_main_sha,
        "phase": phase,
        "stage_id": stage_id,
        "chunk_index": chunk_index,
        "chunk_count": PRIMARY_CHUNK_COUNT,
        "completed_identities": completed_identities,
        "total_identities": 952,
        "fallbacks_used": fallbacks_used,
        "fallbacks_max": 25,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_since_progress": round(seconds_since_progress, 3),
        **dict(resource_reader(pid)),
        "timestamp_utc": _utc_text(_utc_now()),
    }
    print(json.dumps(payload, sort_keys=True), file=stdout, flush=True)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_packages() -> list[dict[str, str]]:
    packages = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def _checkpoint(
    *, run_id: str, plan: Mapping[str, Any], spec: StageSpec, phase: str,
    execution_status: str, reason_code: str | None, attempted: int, completed: int,
    cumulative_attempted: int, cumulative_completed: int,
    successful_frames: int, empty_frames: int, provider_errors: int,
    fallbacks_used: int,
) -> dict[str, Any]:
    value = {
        "schema_version": CHECKPOINT_VERSION,
        "complete": False,
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "phase": phase,
        "execution_status": execution_status,
        "reason_code": reason_code,
        "chunk_index": spec.chunk_index,
        "chunk_count": PRIMARY_CHUNK_COUNT if spec.stage_id.startswith("primary-") else (1 if spec.stage_id == FALLBACK_STAGE_ID else 0),
        "chunk_identity_digest": spec.identity_digest,
        "chunk_attempted_identity_count": attempted,
        "chunk_completed_identity_count": completed,
        "cumulative_attempted_identity_count": cumulative_attempted,
        "cumulative_completed_identity_count": cumulative_completed,
        "successful_frame_count": successful_frames,
        "empty_frame_count": empty_frames,
        "provider_error_count": provider_errors,
        "fallbacks_used": fallbacks_used,
        "generated_at": _utc_text(_utc_now()),
        "authority_boundary": DIAGNOSTIC_AUTHORITY,
    }
    _validate_schema(value, DEFAULT_CHECKPOINT_SCHEMA, "CHECKPOINT_INVALID")
    return value


def create_preflight(run_id: str, *, stdout: TextIO = sys.stdout) -> Path:
    _validate_run_id(run_id)
    diagnostic_root = _diagnostic_run_root(run_id)
    staging_root = _staging_run_root(run_id)
    if diagnostic_root.exists() or staging_root.exists():
        raise AdvisoryHistoryRuntimeIssue("RUN_COLLISION", "run-specific runtime roots already exist")
    universe, _policy, runtime, instruments = _authority()
    source = history._current_repository_head_sha()
    acquired_at = _utc_now()
    specs = _primary_stage_specs(instruments, int(runtime["primary_chunk_size"]))
    bindings = _bindings(universe=universe, runtime=runtime, source_main_sha=source)
    _emit_event(event="preflight_start", run_id=run_id, phase="preflight", source_main_sha=source, stdout=stdout)
    plan = {
        "schema_version": PLAN_VERSION,
        "run_id": run_id,
        "acquisition_started_at": _utc_text(acquired_at),
        **bindings,
        "primary_chunks": [
            {
                "stage_id": spec.stage_id,
                "chunk_index": spec.chunk_index,
                "instrument_ids": list(spec.instrument_ids),
                "chunk_identity_digest": spec.identity_digest,
            }
            for spec in specs
        ],
        "fallback_stage_id": FALLBACK_STAGE_ID,
    }
    environment = {
        "schema_version": "market-engine-advisory-ohlc-history-runtime-environment-v1",
        "run_id": run_id,
        **bindings,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("pandas", "numpy", "yfinance", "curl_cffi", "requests", "lxml")
        },
        "installed_packages": _installed_packages(),
        "generated_at": _utc_text(_utc_now()),
        "authority_boundary": DIAGNOSTIC_AUTHORITY,
    }
    spec = StageSpec(PREFLIGHT_STAGE_ID, None, (), _identity_digest(()))
    checkpoint = _checkpoint(
        run_id=run_id, plan=plan, spec=spec, phase="preflight", execution_status="success",
        reason_code=None, attempted=0, completed=0, cumulative_attempted=0,
        cumulative_completed=0, successful_frames=0, empty_frames=0,
        provider_errors=0, fallbacks_used=0,
    )
    stage_root = _stage_diagnostic_root(run_id, PREFLIGHT_STAGE_ID)
    _atomic_write_json(stage_root / "runtime_plan.json", plan)
    _atomic_write_json(stage_root / "runtime_environment.json", environment)
    _atomic_write_json(stage_root / "checkpoint.json", checkpoint)
    staging_root.mkdir(parents=True, exist_ok=False)
    _emit_event(event="preflight_end", run_id=run_id, phase="preflight", source_main_sha=source, stdout=stdout)
    return stage_root


def _validate_prior_gate(run_id: str, stage_id: str) -> None:
    try:
        gate = _read_json(_gate_path(run_id, stage_id))
    except AdvisoryHistoryRuntimeIssue as exc:
        raise AdvisoryHistoryRuntimeIssue("PRIOR_GATE_INVALID", f"{stage_id} gate has not passed") from exc
    if gate != {
        "schema_version": GATE_VERSION,
        "run_id": run_id,
        "stage_id": stage_id,
        "status": "passed",
    }:
        raise AdvisoryHistoryRuntimeIssue("PRIOR_GATE_INVALID", f"{stage_id} gate has not passed")


def _required_prior_stage(stage_id: str) -> str:
    if stage_id == "primary-chunk-000":
        return PREFLIGHT_STAGE_ID
    if stage_id.startswith("primary-chunk-"):
        return f"primary-chunk-{int(stage_id.rsplit('-', 1)[1]) - 1:03d}"
    if stage_id == FALLBACK_STAGE_ID:
        return "primary-chunk-014"
    raise AdvisoryHistoryRuntimeIssue("STAGE_ID_INVALID", "stage has no executable predecessor")


def _worker_temp_root(run_id: str, stage_id: str, token: str) -> Path:
    if not WORKER_TOKEN.fullmatch(token):
        raise AdvisoryHistoryRuntimeIssue("WORKER_TOKEN_INVALID", "worker token is invalid")
    root = _staging_run_root(run_id) / "worker-temp" / f"{stage_id}-{token}"
    approved = (_staging_run_root(run_id) / "worker-temp").resolve()
    if approved not in root.resolve().parents:
        raise AdvisoryHistoryRuntimeIssue("RUNTIME_PATH_INVALID", "worker root escapes staging authority")
    return root


def _validate_fallback_plan(run_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    fallback = _read_json(_fallback_plan_path(run_id))
    required = {
        "schema_version", "run_id", "source_main_sha", "canonical_universe_sha256",
        "canonical_universe_identity_digest", "history_policy_sha256",
        "operational_runtime_config_sha256", "stage_id", "instrument_ids",
        "chunk_identity_digest", "fallbacks_max",
    }
    if set(fallback) != required or fallback.get("schema_version") != FALLBACK_PLAN_VERSION:
        raise AdvisoryHistoryRuntimeIssue("FALLBACK_PLAN_INVALID", "fallback plan contract is invalid")
    for key in (
        "run_id", "source_main_sha", "canonical_universe_sha256",
        "canonical_universe_identity_digest", "history_policy_sha256",
        "operational_runtime_config_sha256",
    ):
        expected = run_id if key == "run_id" else plan[key]
        if fallback.get(key) != expected:
            raise AdvisoryHistoryRuntimeIssue("FALLBACK_PLAN_INVALID", f"fallback {key} binding is invalid")
    identities = fallback.get("instrument_ids")
    if (
        fallback.get("stage_id") != FALLBACK_STAGE_ID
        or not isinstance(identities, list)
        or len(identities) > 25
        or len(identities) != len(set(identities))
        or fallback.get("chunk_identity_digest") != _identity_digest(tuple(identities))
        or fallback.get("fallbacks_max") != 25
    ):
        raise AdvisoryHistoryRuntimeIssue("FALLBACK_PLAN_INVALID", "fallback selection is invalid")
    return fallback


def _worker_execute(run_id: str, stage_id: str, token: str) -> None:
    plan = _load_plan(run_id)
    _validate_prior_gate(run_id, _required_prior_stage(stage_id))
    universe, policy, _runtime, instruments = _authority()
    if stage_id == FALLBACK_STAGE_ID:
        _validate_fallback_plan(run_id, plan)
    spec = _stage_spec(plan, stage_id)
    by_id = {str(row["instrument_id"]): row for row in instruments}
    selected = [by_id[instrument_id] for instrument_id in spec.instrument_ids]
    acquired_at = history._timestamp(plan["acquisition_started_at"], "acquisition_started_at")
    if stage_id.startswith("primary-"):
        results = history._acquire_primary_with_existing_adapter(
            selected, acquired_at, policy, canonical_instruments=instruments
        )
        fallbacks_used = 0
    else:
        results = history._acquire_fallback_with_existing_adapter(selected, acquired_at, policy)
        fallbacks_used = len(selected)
    temp_root = _worker_temp_root(run_id, stage_id, token)
    if not temp_root.is_dir() or temp_root.is_symlink() or any(path.is_symlink() for path in temp_root.rglob("*")):
        raise AdvisoryHistoryRuntimeIssue("WORKER_STAGING_INVALID", "worker temporary root is invalid")
    payload = {
        "schema_version": STAGE_RESULT_VERSION,
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
        "fallbacks_used": fallbacks_used,
        "results": results,
    }
    _atomic_write_json(temp_root / "result.json", payload)


def _supervise_command(
    command: Sequence[str], *, timeout_seconds: float, grace_seconds: float,
    heartbeat_interval_seconds: float, heartbeat: Callable[[float, int], None],
    monotonic: Callable[[], float] = time.monotonic,
) -> SupervisedResult:
    started = monotonic()
    process = subprocess.Popen(list(command), start_new_session=True)
    termination: list[str] = []
    while True:
        elapsed = monotonic() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            break
        try:
            returncode = process.wait(timeout=min(heartbeat_interval_seconds, remaining))
            return SupervisedResult(returncode, False, (), monotonic() - started)
        except subprocess.TimeoutExpired:
            heartbeat(monotonic() - started, process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        termination.append("SIGTERM")
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            termination.append("SIGKILL")
        except ProcessLookupError:
            pass
        process.wait()
    return SupervisedResult(process.returncode, True, tuple(termination), monotonic() - started)


def _validate_stage_result(
    payload: Mapping[str, Any], *, run_id: str, plan: Mapping[str, Any], spec: StageSpec
) -> dict[str, Mapping[str, Any]]:
    required = {
        "schema_version", "run_id", "source_main_sha", "canonical_universe_sha256",
        "canonical_universe_identity_digest", "history_policy_sha256",
        "operational_runtime_config_sha256", "stage_id", "chunk_index",
        "chunk_identity_digest", "instrument_ids", "acquisition_started_at",
        "fallbacks_used", "results",
    }
    if set(payload) != required or payload.get("schema_version") != STAGE_RESULT_VERSION:
        raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker result schema is invalid")
    expected = {
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "stage_id": spec.stage_id,
        "chunk_index": spec.chunk_index,
        "chunk_identity_digest": spec.identity_digest,
        "instrument_ids": list(spec.instrument_ids),
        "acquisition_started_at": plan["acquisition_started_at"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker result authority binding is invalid")
    results = payload.get("results")
    if not isinstance(results, Mapping) or set(results) != set(spec.instrument_ids):
        raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker result identities do not reconcile")
    expected_fallbacks = len(spec.instrument_ids) if spec.stage_id == FALLBACK_STAGE_ID else 0
    if payload.get("fallbacks_used") != expected_fallbacks or expected_fallbacks > 25:
        raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker fallback accounting is invalid")
    if not all(isinstance(value, Mapping) for value in results.values()):
        raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker instrument results must be objects")
    return {str(key): dict(value) for key, value in results.items()}


def _primary_cumulative(run_id: str, chunk_index: int) -> tuple[int, int]:
    attempted = completed = 0
    for index in range(chunk_index):
        checkpoint = _read_json(_checkpoint_path(run_id, f"primary-chunk-{index:03d}"))
        if checkpoint.get("execution_status") != "success":
            raise AdvisoryHistoryRuntimeIssue("PRIOR_STAGE_INVALID", "prior primary stage did not succeed")
        attempted += int(checkpoint["chunk_attempted_identity_count"])
        completed += int(checkpoint["chunk_completed_identity_count"])
    return attempted, completed


def _result_counts(results: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int]:
    successful = sum(not value.get("error_code") for value in results.values())
    empty = sum(value.get("error_code") == "PROVIDER_HISTORY_MISSING" for value in results.values())
    errors = sum(str(value.get("error_code", "")).startswith("PROVIDER_") for value in results.values())
    return successful, empty, errors


def execute_stage(
    run_id: str, stage_id: str, *, stdout: TextIO = sys.stdout,
) -> Path:
    """Execute one canonical production stage with repository-owned timing."""
    return _execute_stage_impl(run_id, stage_id, stdout=stdout)


def _execute_stage_impl(
    run_id: str, stage_id: str, *, stdout: TextIO = sys.stdout,
    _timeout_seconds: float | None = None, _grace_seconds: float | None = None,
    _heartbeat_interval_seconds: float | None = None,
    _supervisor: Callable[..., SupervisedResult] = _supervise_command,
) -> Path:
    """Private deterministic stage seam; timing overrides are test-only."""
    plan = _load_plan(run_id)
    if stage_id in {PREFLIGHT_STAGE_ID}:
        raise AdvisoryHistoryRuntimeIssue("STAGE_ID_INVALID", "preflight is not a provider worker stage")
    _validate_prior_gate(run_id, _required_prior_stage(stage_id))
    runtime = _load_runtime_config()
    if stage_id == FALLBACK_STAGE_ID:
        _validate_fallback_plan(run_id, plan)
    spec = _stage_spec(plan, stage_id)
    final_root = _stage_staging_root(run_id, stage_id)
    if final_root.exists() or _checkpoint_path(run_id, stage_id).exists():
        raise AdvisoryHistoryRuntimeIssue("STAGE_COLLISION", "stage output already exists")
    token = uuid.uuid4().hex
    temp_root = _worker_temp_root(run_id, stage_id, token)
    temp_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cumulative_attempted, cumulative_completed = (
        _primary_cumulative(run_id, int(spec.chunk_index)) if stage_id.startswith("primary-") else (952, 952)
    )
    _emit_event(
        event="provider_stage_start", run_id=run_id, phase="primary_acquisition" if stage_id.startswith("primary-") else "fallback_acquisition",
        source_main_sha=plan["source_main_sha"], stage_id=stage_id, chunk_index=spec.chunk_index,
        completed_identities=cumulative_completed, fallbacks_used=len(spec.instrument_ids) if stage_id == FALLBACK_STAGE_ID else 0,
        stdout=stdout,
    )

    def heartbeat(elapsed: float, pid: int) -> None:
        _emit_event(
            event="provider_stage_heartbeat", run_id=run_id,
            phase="primary_acquisition" if stage_id.startswith("primary-") else "fallback_acquisition",
            source_main_sha=plan["source_main_sha"], stage_id=stage_id, chunk_index=spec.chunk_index,
            completed_identities=cumulative_completed,
            fallbacks_used=len(spec.instrument_ids) if stage_id == FALLBACK_STAGE_ID else 0,
            elapsed_seconds=elapsed, seconds_since_progress=elapsed, pid=pid, stdout=stdout,
        )

    command = [
        sys.executable, "-m", "market_engine.source_refresh.advisory_ohlc_history_runtime",
        "_worker", "--run-id", run_id, "--stage-id", stage_id, "--worker-token", token,
    ]
    timeout = float(runtime["provider_stage_timeout_seconds"]) if _timeout_seconds is None else _timeout_seconds
    grace = float(runtime["provider_termination_grace_seconds"]) if _grace_seconds is None else _grace_seconds
    heartbeat_interval = float(runtime["heartbeat_interval_seconds"]) if _heartbeat_interval_seconds is None else _heartbeat_interval_seconds
    supervised = _supervisor(
        command, timeout_seconds=timeout, grace_seconds=grace,
        heartbeat_interval_seconds=heartbeat_interval, heartbeat=heartbeat,
    )
    reason: str | None = None
    results: dict[str, Mapping[str, Any]] = {}
    status = "failure"
    try:
        if supervised.timed_out:
            reason = "provider_chunk_timeout"
        elif supervised.returncode != 0:
            reason = "provider_chunk_failure"
        else:
            entries = sorted(path.relative_to(temp_root).as_posix() for path in temp_root.rglob("*"))
            if temp_root.is_symlink() or any(path.is_symlink() for path in temp_root.rglob("*")) or entries != ["result.json"]:
                raise AdvisoryHistoryRuntimeIssue("INVALID_WORKER_RESULT", "worker staging contains unexpected files")
            payload = _read_json(temp_root / "result.json")
            results = _validate_stage_result(payload, run_id=run_id, plan=plan, spec=spec)
            final_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_root, final_root)
            status = "success"
    except AdvisoryHistoryRuntimeIssue:
        reason = "invalid_worker_result"
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    attempted = len(spec.instrument_ids)
    completed = len(results) if status == "success" else 0
    successful, empty, provider_errors = _result_counts(results)
    checkpoint = _checkpoint(
        run_id=run_id, plan=plan, spec=spec,
        phase="primary_acquisition" if stage_id.startswith("primary-") else "fallback_acquisition",
        execution_status=status, reason_code=reason, attempted=attempted, completed=completed,
        cumulative_attempted=cumulative_attempted + attempted,
        cumulative_completed=cumulative_completed + completed,
        successful_frames=successful, empty_frames=empty, provider_errors=provider_errors,
        fallbacks_used=attempted if stage_id == FALLBACK_STAGE_ID else 0,
    )
    checkpoint_path = _checkpoint_path(run_id, stage_id)
    _atomic_write_json(checkpoint_path, checkpoint)
    _emit_event(
        event="provider_stage_exit", run_id=run_id,
        phase="primary_acquisition" if stage_id.startswith("primary-") else "fallback_acquisition",
        source_main_sha=plan["source_main_sha"], stage_id=stage_id, chunk_index=spec.chunk_index,
        completed_identities=cumulative_completed + completed,
        fallbacks_used=attempted if stage_id == FALLBACK_STAGE_ID else 0,
        elapsed_seconds=time.monotonic() - started,
        seconds_since_progress=0 if status == "success" else time.monotonic() - started,
        stdout=stdout,
    )
    if status != "success":
        raise AdvisoryHistoryRuntimeIssue(reason or "provider_chunk_failure", "provider stage failed closed")
    return checkpoint_path


def record_persistence_receipt(
    run_id: str, stage_id: str, *, artifact_name: str, artifact_id: str,
    artifact_digest: str, stdout: TextIO = sys.stdout,
) -> Path:
    plan = _load_plan(run_id)
    spec = _stage_spec(plan, stage_id)
    expected_name = _expected_artifact_name(run_id, stage_id)
    if artifact_name != expected_name:
        raise AdvisoryHistoryRuntimeIssue("ARTIFACT_NAME_INVALID", "artifact name differs from canonical stage name")
    if not artifact_id.strip():
        raise AdvisoryHistoryRuntimeIssue("ARTIFACT_ID_MISSING", "upload action artifact ID is required")
    if not ARTIFACT_DIGEST.fullmatch(artifact_digest):
        raise AdvisoryHistoryRuntimeIssue("ARTIFACT_DIGEST_INVALID", "upload action artifact digest is required")
    checkpoint_path = _checkpoint_path(run_id, stage_id)
    checkpoint = _read_json(checkpoint_path)
    _validate_schema(checkpoint, DEFAULT_CHECKPOINT_SCHEMA, "CHECKPOINT_INVALID")
    _validate_checkpoint_binding(checkpoint, run_id=run_id, stage_id=stage_id, plan=plan, spec=spec)
    receipt_path = _receipt_path(run_id, stage_id)
    if receipt_path.exists():
        raise AdvisoryHistoryRuntimeIssue("RECEIPT_COLLISION", "persistence receipt already exists")
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "stage_id": stage_id,
        "chunk_identity_digest": spec.identity_digest,
        "expected_artifact_name": expected_name,
        "artifact_id": artifact_id.strip(),
        "artifact_digest": artifact_digest,
        "execution_status": checkpoint["execution_status"],
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "generated_at": _utc_text(_utc_now()),
        "authority_boundary": RECEIPT_AUTHORITY,
    }
    _validate_schema(receipt, DEFAULT_RECEIPT_SCHEMA, "RECEIPT_INVALID")
    _atomic_write_json(receipt_path, receipt)
    _emit_event(
        event="persistence_receipt_validated", run_id=run_id, phase="diagnostic_persistence",
        source_main_sha=plan["source_main_sha"], stage_id=stage_id, chunk_index=spec.chunk_index,
        stdout=stdout,
    )
    return receipt_path


def _validate_checkpoint_binding(
    checkpoint: Mapping[str, Any], *, run_id: str, stage_id: str,
    plan: Mapping[str, Any], spec: StageSpec,
) -> None:
    expected = {
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "chunk_index": spec.chunk_index,
        "chunk_identity_digest": spec.identity_digest,
        "chunk_attempted_identity_count": len(spec.instrument_ids),
        "fallbacks_used": len(spec.instrument_ids) if stage_id == FALLBACK_STAGE_ID else 0,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise AdvisoryHistoryRuntimeIssue("CHECKPOINT_BINDING_INVALID", "checkpoint authority binding is invalid")
    if checkpoint.get("execution_status") == "success" and checkpoint.get("chunk_completed_identity_count") != len(spec.instrument_ids):
        raise AdvisoryHistoryRuntimeIssue("CHECKPOINT_BINDING_INVALID", "successful checkpoint count is invalid")


def _validate_receipt(run_id: str, stage_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    spec = _stage_spec(plan, stage_id)
    checkpoint_path = _checkpoint_path(run_id, stage_id)
    checkpoint = _read_json(checkpoint_path)
    _validate_schema(checkpoint, DEFAULT_CHECKPOINT_SCHEMA, "CHECKPOINT_INVALID")
    _validate_checkpoint_binding(checkpoint, run_id=run_id, stage_id=stage_id, plan=plan, spec=spec)
    receipt = _read_json(_receipt_path(run_id, stage_id))
    _validate_schema(receipt, DEFAULT_RECEIPT_SCHEMA, "RECEIPT_INVALID")
    expected = {
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": plan["canonical_universe_identity_digest"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "stage_id": stage_id,
        "chunk_identity_digest": spec.identity_digest,
        "expected_artifact_name": _expected_artifact_name(run_id, stage_id),
        "execution_status": checkpoint["execution_status"],
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise AdvisoryHistoryRuntimeIssue("RECEIPT_BINDING_INVALID", "persistence receipt binding is invalid")
    return receipt


def gate_stage(run_id: str, stage_id: str, *, execution_outcome: str) -> Path:
    plan = _load_plan(run_id)
    checkpoint = _read_json(_checkpoint_path(run_id, stage_id))
    _validate_schema(checkpoint, DEFAULT_CHECKPOINT_SCHEMA, "CHECKPOINT_INVALID")
    _validate_receipt(run_id, stage_id, plan)
    if execution_outcome != "success" or checkpoint["execution_status"] != "success":
        raise AdvisoryHistoryRuntimeIssue("STAGE_GATE_BLOCKED", "true execution outcome did not succeed")
    gate = {
        "schema_version": GATE_VERSION,
        "run_id": run_id,
        "stage_id": stage_id,
        "status": "passed",
    }
    path = _gate_path(run_id, stage_id)
    if path.exists():
        raise AdvisoryHistoryRuntimeIssue("GATE_COLLISION", "stage gate already exists")
    _atomic_write_json(path, gate)
    return path


def plan_fallbacks(run_id: str, *, stdout: TextIO = sys.stdout) -> Path:
    plan = _load_plan(run_id)
    for index in range(PRIMARY_CHUNK_COUNT):
        _validate_prior_gate(run_id, f"primary-chunk-{index:03d}")
    universe, policy, _runtime, instruments = _authority()
    if int(policy["max_individual_fallbacks"]) != 25:
        raise AdvisoryHistoryRuntimeIssue("FALLBACK_BUDGET_INVALID", "canonical fallback budget changed")
    primary_results: dict[str, Mapping[str, Any]] = {}
    for row in plan["primary_chunks"]:
        stage_id = str(row["stage_id"])
        spec = _stage_spec(plan, stage_id)
        payload = _read_json(_result_path(run_id, stage_id))
        results = _validate_stage_result(payload, run_id=run_id, plan=plan, spec=spec)
        overlap = set(primary_results) & set(results)
        if overlap:
            raise AdvisoryHistoryRuntimeIssue("PRIMARY_RECONCILIATION_INVALID", "primary stage identities overlap")
        primary_results.update(results)
    canonical_ids = [str(row["instrument_id"]) for row in instruments]
    if set(primary_results) != set(canonical_ids) or len(primary_results) != 952:
        raise AdvisoryHistoryRuntimeIssue("PRIMARY_RECONCILIATION_INVALID", "primary results do not reconcile 952 identities")
    selected = [
        instrument_id for instrument_id in canonical_ids
        if primary_results[instrument_id].get("error_code") == "PROVIDER_HISTORY_MISSING"
    ][:25]
    fallback = {
        "schema_version": FALLBACK_PLAN_VERSION,
        "run_id": run_id,
        "source_main_sha": plan["source_main_sha"],
        "canonical_universe_sha256": plan["canonical_universe_sha256"],
        "canonical_universe_identity_digest": universe["universe_checksum"],
        "history_policy_sha256": plan["history_policy_sha256"],
        "operational_runtime_config_sha256": plan["operational_runtime_config_sha256"],
        "stage_id": FALLBACK_STAGE_ID,
        "instrument_ids": selected,
        "chunk_identity_digest": _identity_digest(selected),
        "fallbacks_max": 25,
    }
    path = _fallback_plan_path(run_id)
    if path.exists():
        raise AdvisoryHistoryRuntimeIssue("FALLBACK_PLAN_COLLISION", "fallback plan already exists")
    _atomic_write_json(path, fallback)
    _emit_event(
        event="fallback_planned", run_id=run_id, phase="fallback_planning",
        source_main_sha=plan["source_main_sha"], stage_id=FALLBACK_STAGE_ID,
        chunk_index=0, completed_identities=952, fallbacks_used=len(selected), stdout=stdout,
    )
    return path


class _AssemblyClock:
    def __init__(self, started: datetime, completed: datetime) -> None:
        self._values = iter((started, completed))

    def __call__(self) -> datetime:
        return next(self._values)


def _validate_exact_runtime_envelope(run_id: str, provider_stages: Sequence[str]) -> None:
    staging = _staging_run_root(run_id)
    diagnostic = _diagnostic_run_root(run_id)
    if staging.is_symlink() or diagnostic.is_symlink():
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "runtime roots cannot be symlinks")
    stage_root = staging / "stages"
    actual_provider_stages = sorted(path.name for path in stage_root.iterdir() if path.is_dir()) if stage_root.is_dir() else []
    if actual_provider_stages != sorted(provider_stages):
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "staging contains a missing or unexpected stage")
    for stage_id in provider_stages:
        if sorted(path.name for path in (stage_root / stage_id).iterdir()) != ["result.json"]:
            raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "stage result files are not exact")
    worker_temp = staging / "worker-temp"
    if worker_temp.exists() and any(worker_temp.iterdir()):
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "worker temporary staging is not empty")
    allowed_staging_entries = {"stages", "fallback_plan.json", "worker-temp"}
    if {path.name for path in staging.iterdir()} - allowed_staging_entries:
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "staging contains unexpected entries")
    expected_all = sorted([PREFLIGHT_STAGE_ID, *provider_stages])
    diagnostic_stages = diagnostic / "stages"
    actual_diagnostic_stages = sorted(path.name for path in diagnostic_stages.iterdir() if path.is_dir())
    if actual_diagnostic_stages != expected_all:
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "diagnostics contain a missing or unexpected stage")
    if sorted(path.name for path in (diagnostic_stages / PREFLIGHT_STAGE_ID).iterdir()) != [
        "checkpoint.json", "runtime_environment.json", "runtime_plan.json"
    ]:
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "preflight evidence files are not exact")
    for stage_id in provider_stages:
        if sorted(path.name for path in (diagnostic_stages / stage_id).iterdir()) != ["checkpoint.json"]:
            raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "checkpoint files are not exact")
    for directory in (diagnostic / "receipts", diagnostic / "gates"):
        actual_files = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
        expected_files = sorted(f"{stage_id}.json" for stage_id in expected_all)
        if actual_files != expected_files or any(path.is_symlink() for path in directory.rglob("*")):
            raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "persistence envelope is incomplete or unexpected")
    if sorted(path.name for path in diagnostic.iterdir()) != ["gates", "receipts", "stages"]:
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "diagnostic run root contains unexpected entries")


def assemble_final_history(run_id: str, *, stdout: TextIO = sys.stdout) -> tuple[dict[str, Any], Path]:
    plan = _load_plan(run_id)
    expected_stages = [f"primary-chunk-{index:03d}" for index in range(PRIMARY_CHUNK_COUNT)] + [FALLBACK_STAGE_ID]
    _validate_exact_runtime_envelope(run_id, expected_stages)
    for stage_id in expected_stages:
        _validate_prior_gate(run_id, stage_id)
        checkpoint = _read_json(_checkpoint_path(run_id, stage_id))
        if checkpoint.get("execution_status") != "success":
            raise AdvisoryHistoryRuntimeIssue("FINAL_ASSEMBLY_BLOCKED", "a provider stage did not succeed")
        _validate_receipt(run_id, stage_id, plan)
    fallback_plan = _validate_fallback_plan(run_id, plan)
    primary_results: dict[str, Mapping[str, Any]] = {}
    primary_identities: list[str] = []
    for row in plan["primary_chunks"]:
        stage_id = str(row["stage_id"])
        spec = _stage_spec(plan, stage_id)
        payload = _read_json(_result_path(run_id, stage_id))
        results = _validate_stage_result(payload, run_id=run_id, plan=plan, spec=spec)
        primary_identities.extend(spec.instrument_ids)
        if set(primary_results) & set(results):
            raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "duplicate primary identity")
        primary_results.update(results)
    _universe, _policy, _runtime, instruments = _authority()
    canonical_ids = [str(row["instrument_id"]) for row in instruments]
    if primary_identities != canonical_ids or len(primary_results) != 952 or set(primary_results) != set(canonical_ids):
        raise AdvisoryHistoryRuntimeIssue("FINAL_RECONCILIATION_INVALID", "primary envelope does not exactly reconcile 952 identities")
    fallback_spec = _stage_spec(plan, FALLBACK_STAGE_ID)
    fallback_payload = _read_json(_result_path(run_id, FALLBACK_STAGE_ID))
    fallback_results = _validate_stage_result(fallback_payload, run_id=run_id, plan=plan, spec=fallback_spec)
    if list(fallback_spec.instrument_ids) != fallback_plan["instrument_ids"] or len(fallback_results) > 25:
        raise AdvisoryHistoryRuntimeIssue("FALLBACK_BUDGET_INVALID", "fallback result differs from global fallback plan")
    combined = dict(primary_results)
    combined.update(fallback_results)
    started = history._timestamp(plan["acquisition_started_at"], "acquisition_started_at")
    completed = _utc_now()
    if completed < started:
        raise AdvisoryHistoryRuntimeIssue("CLOCK_INVALID", "assembly completion precedes acquisition start")
    _emit_event(
        event="final_assembly_start", run_id=run_id, phase="final_assembly",
        source_main_sha=plan["source_main_sha"], completed_identities=952,
        fallbacks_used=len(fallback_results), stdout=stdout,
    )
    try:
        manifest, path = history._build_advisory_ohlc_history_impl(
            run_id=run_id,
            source_main_sha=plan["source_main_sha"],
            provider=lambda rows, at, policy: combined,
            clock=_AssemblyClock(started, completed),
        )
    except history.AdvisoryHistoryIssue as exc:
        raise AdvisoryHistoryRuntimeIssue(exc.code, str(exc)) from exc
    _emit_event(
        event="final_assembly_end", run_id=run_id, phase="final_assembly",
        source_main_sha=plan["source_main_sha"], completed_identities=952,
        fallbacks_used=len(fallback_results), stdout=stdout,
    )
    return manifest, path


def observed_quality_gate(run_id: str, *, stdout: TextIO = sys.stdout) -> int:
    plan = _load_plan(run_id)
    _emit_event(
        event="quality_gate_start", run_id=run_id, phase="quality_gate",
        source_main_sha=plan["source_main_sha"], completed_identities=952, stdout=stdout,
    )
    context = history.load_advisory_ohlc_history(history.DEFAULT_OUTPUT_ROOT / run_id)
    status = history._effective_analytic_authority_status(context)
    _emit_event(
        event="quality_gate_end", run_id=run_id, phase="quality_gate",
        source_main_sha=plan["source_main_sha"], completed_identities=952, stdout=stdout,
    )
    print(json.dumps({"status": status, "run_status": context.manifest["run_status"]}, sort_keys=True), file=stdout)
    return 0 if status == "usable" else 3


def emit_named_event(run_id: str, event: str, *, stdout: TextIO = sys.stdout) -> None:
    allowed = {
        "technical_screening_start": "technical_screening",
        "technical_screening_end": "technical_screening",
    }
    if event not in allowed:
        raise AdvisoryHistoryRuntimeIssue("EVENT_INVALID", "event is not an approved workflow boundary")
    plan = _load_plan(run_id)
    _emit_event(
        event=event, run_id=run_id, phase=allowed[event], source_main_sha=plan["source_main_sha"],
        completed_identities=952, stdout=stdout,
    )


def run_command(
    argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr
) -> int:
    parser = argparse.ArgumentParser(description="Run bounded advisory OHLC history acquisition")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "plan-fallbacks", "assemble", "quality-gate"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
    execute = commands.add_parser("execute-stage")
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--stage-id", required=True)
    receipt = commands.add_parser("record-receipt")
    receipt.add_argument("--run-id", required=True)
    receipt.add_argument("--stage-id", required=True)
    receipt.add_argument("--artifact-name", required=True)
    receipt.add_argument("--artifact-id", required=True)
    receipt.add_argument("--artifact-digest", required=True)
    gate = commands.add_parser("gate-stage")
    gate.add_argument("--run-id", required=True)
    gate.add_argument("--stage-id", required=True)
    gate.add_argument("--execution-outcome", required=True)
    event = commands.add_parser("emit-event")
    event.add_argument("--run-id", required=True)
    event.add_argument("--event", required=True)
    worker = commands.add_parser("_worker")
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--stage-id", required=True)
    worker.add_argument("--worker-token", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            create_preflight(args.run_id, stdout=stdout)
        elif args.command == "execute-stage":
            execute_stage(args.run_id, args.stage_id, stdout=stdout)
        elif args.command == "record-receipt":
            record_persistence_receipt(
                args.run_id, args.stage_id, artifact_name=args.artifact_name,
                artifact_id=args.artifact_id, artifact_digest=args.artifact_digest, stdout=stdout,
            )
        elif args.command == "gate-stage":
            gate_stage(args.run_id, args.stage_id, execution_outcome=args.execution_outcome)
        elif args.command == "plan-fallbacks":
            plan_fallbacks(args.run_id, stdout=stdout)
        elif args.command == "assemble":
            manifest, path = assemble_final_history(args.run_id, stdout=stdout)
            print(json.dumps({"status": manifest["run_status"], "artifact_path": path.as_posix()}, sort_keys=True), file=stdout)
        elif args.command == "quality-gate":
            return observed_quality_gate(args.run_id, stdout=stdout)
        elif args.command == "emit-event":
            emit_named_event(args.run_id, args.event, stdout=stdout)
        else:
            _worker_execute(args.run_id, args.stage_id, args.worker_token)
    except (AdvisoryHistoryRuntimeIssue, history.AdvisoryHistoryIssue) as exc:
        code = exc.code
        print(json.dumps({"status": "blocked", "code": code}, sort_keys=True), file=stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_command())
