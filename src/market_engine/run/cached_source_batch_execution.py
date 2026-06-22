from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.run.cached_source_execution import (
    CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
    CachedSourceLocalExecutionError,
    build_cached_source_local_execution_stage_payloads,
)
from market_engine.run.end_to_end_dry_run import (
    MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
    build_market_engine_end_to_end_dry_run,
)
from market_engine.run.local_dry_run_artifacts import (
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION,
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION,
    SEC_COMPANYFACTS_SOURCE_NAME,
    SecCompanyFactsRawSnapshot,
    SecCompanyFactsSnapshotError,
    load_sec_companyfacts_raw_snapshot,
)


MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION = (
    "market-engine-cached-source-batch-dry-run-v1"
)
CACHED_SOURCE_BATCH_INPUT_MODE = "cached_source_batch"

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class CachedSourceBatchDryRunError(ValueError):
    pass


def build_cached_source_batch_dry_run(
    *,
    source_snapshot_root: str | Path,
    batch_id: str,
    generated_at: str | None,
    requested_tickers: Sequence[str] | None = None,
    discover_cached_tickers: bool = False,
    portfolio_contexts_by_ticker: Mapping[str, Mapping[str, Any]] | None = None,
    ticker_limit: int | None = None,
    write_local_artifacts: bool = False,
    artifact_output_root: str | Path = MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    artifact_created_at: str | None = None,
) -> dict[str, Any]:
    safe_batch_id = _safe_path_segment(batch_id, field_name="batch_id")
    root = _validated_existing_root(Path(source_snapshot_root))
    portfolio_contexts = {
        _normalize_ticker(ticker): payload
        for ticker, payload in (portfolio_contexts_by_ticker or {}).items()
    }
    discovered = _discover_cached_snapshots(root)
    requested = _requested_ticker_universe(
        requested_tickers=requested_tickers,
        discover_cached_tickers=discover_cached_tickers,
        discovered=discovered,
        ticker_limit=ticker_limit,
    )

    batch_artifact_root: Path | None = None
    if write_local_artifacts:
        if not artifact_created_at:
            raise CachedSourceBatchDryRunError(
                "artifact_created_at is required when local artifact writing is enabled."
            )
        batch_artifact_root = _prepare_batch_artifact_root(
            artifact_output_root=Path(artifact_output_root),
            batch_id=safe_batch_id,
        )

    per_ticker_results = []
    for ticker in requested:
        per_ticker_results.append(
            _build_ticker_result(
                ticker=ticker,
                batch_id=safe_batch_id,
                generated_at=generated_at,
                source_snapshot_root=root,
                discovered=discovered,
                portfolio_context_payload=portfolio_contexts.get(ticker),
                batch_artifact_root=batch_artifact_root,
                artifact_created_at=artifact_created_at,
            )
        )

    batch_counts = _batch_counts(
        requested_tickers=requested,
        discovered=discovered,
        per_ticker_results=per_ticker_results,
    )
    batch_payload = {
        "contract_version": MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
        "batch_id": safe_batch_id,
        "generated_at": generated_at,
        "input_mode": CACHED_SOURCE_BATCH_INPUT_MODE,
        "source_mode": "cached_source_local_only",
        "source_snapshot_root": root.as_posix(),
        "operator_ticker_input_reference": (
            "explicit_requested_tickers"
            if requested_tickers is not None
            else "discovered_cached_snapshots"
        ),
        "requested_tickers": requested,
        "ticker_universe_metadata": {
            "requested_count": len(requested),
            "discovered_cached_source_tickers": sorted(discovered.valid_by_ticker),
            "discovered_but_unrequested_tickers": tuple(
                ticker
                for ticker in sorted(discovered.valid_by_ticker)
                if ticker not in set(requested)
            ),
            "ticker_limit": ticker_limit,
            "discovery_policy": "scan_local_sec_companyfacts_raw_snapshot_layouts",
            "ambiguity_policy": "block_ambiguous_ticker",
        },
        "batch_execution_state": _batch_execution_state(per_ticker_results),
        "batch_counts": batch_counts,
        "per_ticker_results": per_ticker_results,
        "batch_blocked_reasons": (),
        "batch_warnings": _batch_warnings(discovered),
        "artifact_manifest_reference": None,
        "forbidden_side_effect_confirmation": (
            "No provider, live market data, broker, message-delivery, scheduler, UI, "
            "portfolio, watchlist, production-report, or execution side effects are performed."
        ),
        "authority_boundary_confirmation": (
            "Decision Engine remains the only future action/allocation authority; "
            "the batch dry-run summarizes cached-source local execution state only."
        ),
        "provenance": {
            "batch_contract_version": (
                MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION
            ),
            "batch_id": safe_batch_id,
            "command_input_mode": CACHED_SOURCE_BATCH_INPUT_MODE,
            "per_ticker_output_contract": MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
            "source_snapshot_root": root.as_posix(),
            "generated_at": generated_at,
            "implementation_module": "market_engine.run.cached_source_batch_execution",
            "artifact_policy": (
                "write_local_artifacts"
                if write_local_artifacts
                else "stdout_or_in_memory_only"
            ),
        },
        "live_provider_call_made": False,
        "non_production_batch": True,
    }

    if batch_artifact_root is not None:
        manifest_path = _write_batch_manifest(
            batch_payload=batch_payload,
            batch_artifact_root=batch_artifact_root,
            artifact_created_at=str(artifact_created_at),
        )
        batch_payload["artifact_manifest_reference"] = _relative_posix(
            manifest_path,
            Path(artifact_output_root).resolve(),
        )
        _write_json(manifest_path, batch_payload, allow_overwrite=True)

    return batch_payload


def persist_cached_source_batch_dry_run_artifacts(
    batch_payload: Mapping[str, Any],
    *,
    output_root: str | Path = MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    artifact_created_at: str,
) -> dict[str, Any]:
    if batch_payload.get("contract_version") != (
        MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION
    ):
        raise CachedSourceBatchDryRunError(
            "Batch artifact payload uses an unsupported contract version."
        )
    safe_batch_id = _safe_path_segment(
        batch_payload.get("batch_id"),
        field_name="batch_id",
    )
    root = _validated_output_root(Path(output_root)).resolve()
    batch_artifact_root = _resolved_child(root, safe_batch_id)
    if batch_artifact_root.exists():
        raise CachedSourceBatchDryRunError(
            f"Cached-source batch artifact directory already exists: {batch_artifact_root}"
        )
    batch_artifact_root.mkdir(parents=True)
    for result in batch_payload.get("per_ticker_results", ()):
        if isinstance(result, Mapping) and isinstance(result.get("dry_run_payload"), Mapping):
            _write_ticker_artifact(
                batch_artifact_root=batch_artifact_root,
                ticker=str(result["ticker"]),
                dry_run_payload=result["dry_run_payload"],
                artifact_created_at=artifact_created_at,
            )
    manifest_path = _write_batch_manifest(
        batch_payload=dict(batch_payload),
        batch_artifact_root=batch_artifact_root,
        artifact_created_at=artifact_created_at,
    )
    return {
        "batch_artifact_root": batch_artifact_root.as_posix(),
        "batch_manifest_path": manifest_path.as_posix(),
    }


class _DiscoveredSnapshots:
    def __init__(self) -> None:
        self.valid_by_ticker: dict[str, list[tuple[Path, SecCompanyFactsRawSnapshot]]] = {}
        self.invalid_by_ticker: dict[str, list[tuple[Path, str]]] = {}
        self.unsupported_by_ticker: dict[str, list[tuple[Path, str]]] = {}


def _discover_cached_snapshots(root: Path) -> _DiscoveredSnapshots:
    discovered = _DiscoveredSnapshots()
    for path in _candidate_snapshot_paths(root):
        try:
            snapshot = load_sec_companyfacts_raw_snapshot(path)
        except SecCompanyFactsSnapshotError as exc:
            ticker = _infer_ticker_from_snapshot_path(path)
            message = str(exc)
            if "unsupported" in message.lower():
                discovered.unsupported_by_ticker.setdefault(ticker, []).append(
                    (path, message)
                )
            else:
                discovered.invalid_by_ticker.setdefault(ticker, []).append((path, message))
            continue
        discovered.valid_by_ticker.setdefault(snapshot.ticker, []).append((path, snapshot))
    return discovered


def _candidate_snapshot_paths(root: Path) -> tuple[Path, ...]:
    candidates = {
        path
        for pattern in (
            "*/raw/*.json",
            f"{SEC_COMPANYFACTS_SOURCE_NAME}/*/raw/*.json",
        )
        for path in root.glob(pattern)
    }
    return tuple(sorted(candidates, key=lambda candidate: candidate.as_posix()))


def _requested_ticker_universe(
    *,
    requested_tickers: Sequence[str] | None,
    discover_cached_tickers: bool,
    discovered: _DiscoveredSnapshots,
    ticker_limit: int | None,
) -> tuple[str, ...]:
    if requested_tickers is None:
        if not discover_cached_tickers:
            raise CachedSourceBatchDryRunError(
                "Cached-source batch dry-run requires requested_tickers or "
                "discover_cached_tickers=true."
            )
        tickers = sorted(discovered.valid_by_ticker)
    else:
        tickers = [_normalize_ticker(ticker) for ticker in requested_tickers]
        if not tickers:
            raise CachedSourceBatchDryRunError(
                "Cached-source batch dry-run requires at least one ticker."
            )
    if len(set(tickers)) != len(tickers):
        raise CachedSourceBatchDryRunError(
            "Cached-source batch dry-run requested tickers must be unique."
        )
    if ticker_limit is not None:
        if ticker_limit < 1:
            raise CachedSourceBatchDryRunError("ticker_limit must be a positive integer.")
        tickers = tickers[:ticker_limit]
    return tuple(tickers)


def _build_ticker_result(
    *,
    ticker: str,
    batch_id: str,
    generated_at: str | None,
    source_snapshot_root: Path,
    discovered: _DiscoveredSnapshots,
    portfolio_context_payload: Mapping[str, Any] | None,
    batch_artifact_root: Path | None,
    artifact_created_at: str | None,
) -> dict[str, Any]:
    valid_candidates = discovered.valid_by_ticker.get(ticker, [])
    if len(valid_candidates) > 1:
        return _blocked_ticker_result(
            ticker=ticker,
            execution_state="blocked_ambiguous_cached_source",
            blocked_reasons=("Multiple cached source snapshots matched requested ticker.",),
            candidate_paths=tuple(path for path, _snapshot in valid_candidates),
        )
    if not valid_candidates:
        invalid_candidates = discovered.invalid_by_ticker.get(ticker, [])
        unsupported_candidates = discovered.unsupported_by_ticker.get(ticker, [])
        if unsupported_candidates:
            return _blocked_ticker_result(
                ticker=ticker,
                execution_state="blocked_unsupported_cached_source",
                blocked_reasons=tuple(message for _path, message in unsupported_candidates),
                candidate_paths=tuple(path for path, _message in unsupported_candidates),
            )
        if invalid_candidates:
            return _blocked_ticker_result(
                ticker=ticker,
                execution_state="blocked_invalid_cached_source",
                blocked_reasons=tuple(message for _path, message in invalid_candidates),
                candidate_paths=tuple(path for path, _message in invalid_candidates),
            )
        return _blocked_ticker_result(
            ticker=ticker,
            execution_state="blocked_missing_cached_source",
            blocked_reasons=("No matching cached source snapshot was found.",),
            candidate_paths=(),
        )

    snapshot_path, snapshot = valid_candidates[0]
    dry_run_id = f"{batch_id}-{ticker.lower()}"
    try:
        stage_payloads = build_cached_source_local_execution_stage_payloads(
            source_snapshot_path=snapshot_path,
            source_snapshot_root=source_snapshot_root,
            dry_run_id=dry_run_id,
            generated_at=generated_at,
            portfolio_context_payload=portfolio_context_payload,
        )
        dry_run = build_market_engine_end_to_end_dry_run(
            stage_payloads,
            dry_run_id=dry_run_id,
            input_mode=CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            generated_at=generated_at,
        )
        dry_run_payload = dry_run.to_payload()
    except CachedSourceLocalExecutionError as exc:
        return _blocked_ticker_result(
            ticker=ticker,
            cik=snapshot.cik,
            source_snapshot_path=snapshot_path,
            snapshot=snapshot,
            execution_state="blocked_downstream_contract_failure",
            blocked_reasons=(str(exc),),
            candidate_paths=(snapshot_path,),
        )
    except Exception as exc:
        return _blocked_ticker_result(
            ticker=ticker,
            cik=snapshot.cik,
            source_snapshot_path=snapshot_path,
            snapshot=snapshot,
            execution_state="failed_unexpected_local_error",
            blocked_reasons=(f"{type(exc).__name__}: {exc}",),
            candidate_paths=(snapshot_path,),
        )

    artifact_reference = None
    if batch_artifact_root is not None:
        artifact_reference = _write_ticker_artifact(
            batch_artifact_root=batch_artifact_root,
            ticker=ticker,
            dry_run_payload=dry_run_payload,
            artifact_created_at=str(artifact_created_at),
        )

    return {
        "ticker": ticker,
        "cik": snapshot.cik,
        "source_snapshot_reference": _relative_posix(snapshot_path, source_snapshot_root),
        "source_snapshot_format_version": snapshot.payload_format_version,
        "source_snapshot_created_at": snapshot.fetched_at,
        "source_snapshot_age_days": _age_days(snapshot.fetched_at, generated_at),
        "source_snapshot_stale_for_review": False,
        "execution_state": _ticker_execution_state(dry_run_payload),
        "blocked_reasons": tuple(dry_run_payload.get("blocked_reasons") or ()),
        "warnings": (),
        "end_to_end_dry_run_reference": {
            "dry_run_id": dry_run_payload["dry_run_id"],
            "dry_run_format_version": dry_run_payload["dry_run_format_version"],
            "run_state": dry_run_payload["run_state"],
        },
        "artifact_reference": artifact_reference,
        "missing_data_markers": tuple(dry_run_payload.get("missing_data_summary") or ()),
        "stale_data_markers": tuple(dry_run_payload.get("stale_data_summary") or ()),
        "numeric_zero_evidence_present": bool(
            dry_run_payload.get("numeric_zero_evidence_summary")
        ),
        "provenance": {
            "source_snapshot_path": snapshot_path.as_posix(),
            "source_name": snapshot.source_name,
            "retrieval_timestamp": snapshot.fetched_at,
            "selected_snapshot_rule": "single_valid_snapshot_for_ticker",
            "downstream_dry_run_id": dry_run_payload["dry_run_id"],
        },
        "dry_run_payload": dry_run_payload,
    }


def _blocked_ticker_result(
    *,
    ticker: str,
    execution_state: str,
    blocked_reasons: Sequence[str],
    candidate_paths: Sequence[Path],
    cik: str = "",
    source_snapshot_path: Path | None = None,
    snapshot: SecCompanyFactsRawSnapshot | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "cik": cik,
        "source_snapshot_reference": (
            source_snapshot_path.as_posix() if source_snapshot_path is not None else None
        ),
        "source_snapshot_format_version": (
            snapshot.payload_format_version if snapshot is not None else None
        ),
        "source_snapshot_created_at": snapshot.fetched_at if snapshot is not None else None,
        "source_snapshot_age_days": None,
        "source_snapshot_stale_for_review": False,
        "execution_state": execution_state,
        "blocked_reasons": tuple(str(reason) for reason in blocked_reasons),
        "warnings": (),
        "end_to_end_dry_run_reference": None,
        "artifact_reference": None,
        "missing_data_markers": (),
        "stale_data_markers": (),
        "numeric_zero_evidence_present": False,
        "provenance": {
            "candidate_snapshot_paths": tuple(path.as_posix() for path in candidate_paths),
            "omitted_stage_reason": execution_state,
        },
    }


def _ticker_execution_state(dry_run_payload: Mapping[str, Any]) -> str:
    run_state = dry_run_payload.get("run_state")
    if run_state == "dry_run_completed":
        return "completed"
    if run_state == "dry_run_completed_with_limitations":
        return "completed_with_limitations"
    if run_state == "dry_run_blocked":
        return "blocked_downstream_contract_failure"
    if run_state == "dry_run_unsupported_input":
        return "blocked_unsupported_cached_source"
    if run_state == "dry_run_contract_violation":
        return "blocked_downstream_contract_failure"
    return "failed_unexpected_local_error"


def _batch_execution_state(per_ticker_results: Sequence[Mapping[str, Any]]) -> str:
    if not per_ticker_results:
        return "blocked_no_tickers"
    states = {str(result["execution_state"]) for result in per_ticker_results}
    if states <= {"completed"}:
        return "completed"
    if states <= {"completed", "completed_with_limitations"}:
        return "completed_with_limitations"
    if any(state.startswith("failed") or state.startswith("blocked") for state in states):
        return "completed_with_ticker_failures"
    return "completed_with_limitations"


def _batch_counts(
    *,
    requested_tickers: Sequence[str],
    discovered: _DiscoveredSnapshots,
    per_ticker_results: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "requested_count": len(requested_tickers),
        "discovered_cached_source_count": len(discovered.valid_by_ticker),
        "eligible_count": sum(
            1 for result in per_ticker_results if result["execution_state"] in {
                "completed",
                "completed_with_limitations",
            }
        ),
        "executed_count": sum(
            1 for result in per_ticker_results if result.get("end_to_end_dry_run_reference")
        ),
        "completed_count": _state_count(per_ticker_results, "completed"),
        "completed_with_limitations_count": _state_count(
            per_ticker_results,
            "completed_with_limitations",
        ),
        "blocked_count": sum(
            1 for result in per_ticker_results if str(result["execution_state"]).startswith("blocked")
        ),
        "failed_count": sum(
            1 for result in per_ticker_results if str(result["execution_state"]).startswith("failed")
        ),
        "skipped_count": sum(
            1 for result in per_ticker_results if str(result["execution_state"]).startswith("skipped")
        ),
        "missing_cached_source_count": _state_count(
            per_ticker_results,
            "blocked_missing_cached_source",
        ),
        "ambiguous_cached_source_count": _state_count(
            per_ticker_results,
            "blocked_ambiguous_cached_source",
        ),
        "unsupported_cached_source_count": _state_count(
            per_ticker_results,
            "blocked_unsupported_cached_source",
        ),
        "stale_source_count": _state_count(
            per_ticker_results,
            "blocked_stale_source_without_safe_downstream_contract",
        ),
    }


def _state_count(
    per_ticker_results: Sequence[Mapping[str, Any]],
    state: str,
) -> int:
    return sum(1 for result in per_ticker_results if result["execution_state"] == state)


def _batch_warnings(discovered: _DiscoveredSnapshots) -> tuple[str, ...]:
    warnings: list[str] = []
    if discovered.invalid_by_ticker:
        warnings.append("Invalid cached source snapshots were discovered.")
    if discovered.unsupported_by_ticker:
        warnings.append("Unsupported cached source snapshots were discovered.")
    return tuple(warnings)


def _write_ticker_artifact(
    *,
    batch_artifact_root: Path,
    ticker: str,
    dry_run_payload: Mapping[str, Any],
    artifact_created_at: str,
) -> dict[str, str]:
    ticker_dir = _resolved_child(batch_artifact_root, ticker)
    ticker_dir.mkdir(parents=True)
    dry_run_path = _resolved_child(ticker_dir, "dry_run.json")
    manifest_path = _resolved_child(ticker_dir, "manifest.json")
    dry_run_artifact = {
        "artifact_format_version": MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION,
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": artifact_created_at,
        "non_production_artifact": True,
        "source_dry_run_format_version": dry_run_payload["dry_run_format_version"],
        "source_dry_run_id": dry_run_payload["dry_run_id"],
        "source_input_mode": dry_run_payload["input_mode"],
        "source_run_state": dry_run_payload["run_state"],
        "payload": dry_run_payload,
    }
    manifest = {
        "manifest_format_version": MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION,
        "artifact_count": 1,
        "artifact_created_at": artifact_created_at,
        "non_production_artifact": True,
        "source_dry_run_format_version": dry_run_payload["dry_run_format_version"],
        "source_dry_run_id": dry_run_payload["dry_run_id"],
        "source_input_mode": dry_run_payload["input_mode"],
        "source_run_state": dry_run_payload["run_state"],
        "artifacts": (
            {
                "artifact_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION
                ),
                "artifact_type": "market_engine_end_to_end_dry_run",
                "artifact_relative_path": dry_run_path.name,
            },
        ),
    }
    _write_json(dry_run_path, dry_run_artifact)
    _write_json(manifest_path, manifest)
    return {
        "dry_run_artifact_path": dry_run_path.as_posix(),
        "ticker_manifest_path": manifest_path.as_posix(),
    }


def _write_batch_manifest(
    *,
    batch_payload: Mapping[str, Any],
    batch_artifact_root: Path,
    artifact_created_at: str,
) -> Path:
    manifest_path = _resolved_child(batch_artifact_root, "batch_manifest.json")
    manifest_payload = {
        "contract_version": MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
        "batch_id": batch_payload["batch_id"],
        "artifact_created_at": artifact_created_at,
        "non_production_artifact": True,
        "batch_execution_state": batch_payload["batch_execution_state"],
        "batch_counts": batch_payload["batch_counts"],
        "per_ticker_artifacts": tuple(
            {
                "ticker": result["ticker"],
                "execution_state": result["execution_state"],
                "artifact_reference": result.get("artifact_reference"),
            }
            for result in batch_payload.get("per_ticker_results", ())
        ),
        "forbidden_side_effect_confirmation": batch_payload[
            "forbidden_side_effect_confirmation"
        ],
        "authority_boundary_confirmation": batch_payload[
            "authority_boundary_confirmation"
        ],
    }
    _write_json(manifest_path, manifest_payload)
    return manifest_path


def _write_json(path: Path, payload: Mapping[str, Any], *, allow_overwrite: bool = False) -> None:
    if path.exists() and not allow_overwrite:
        raise CachedSourceBatchDryRunError(
            f"Cached-source batch artifact already exists: {path}"
        )
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested_value) for nested_value in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validated_existing_root(root: Path) -> Path:
    if any(part == ".." for part in root.parts):
        raise CachedSourceBatchDryRunError(
            "Cached-source batch snapshot root may not contain parent traversal."
        )
    resolved = root.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise CachedSourceBatchDryRunError(
            f"Cached-source batch snapshot root does not exist: {root}"
        )
    return resolved


def _validated_output_root(root: Path) -> Path:
    if any(part == ".." for part in root.parts):
        raise CachedSourceBatchDryRunError(
            "Cached-source batch artifact output root may not contain parent traversal."
        )
    return root


def _prepare_batch_artifact_root(*, artifact_output_root: Path, batch_id: str) -> Path:
    root = _validated_output_root(artifact_output_root).resolve()
    batch_root = _resolved_child(root, batch_id)
    if batch_root.exists():
        raise CachedSourceBatchDryRunError(
            f"Cached-source batch artifact directory already exists: {batch_root}"
        )
    batch_root.mkdir(parents=True)
    return batch_root


def _resolved_child(parent: Path, child: str) -> Path:
    safe_child = _safe_path_segment(child, field_name="path_component")
    candidate = (parent / safe_child).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as exc:
        raise CachedSourceBatchDryRunError(
            "Cached-source batch path escaped the approved root."
        ) from exc
    return candidate


def _safe_path_segment(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CachedSourceBatchDryRunError(f"{field_name} must be a non-empty string.")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise CachedSourceBatchDryRunError(f"{field_name} is not a safe path segment.")
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(value):
        raise CachedSourceBatchDryRunError(f"{field_name} contains unsafe characters.")
    return value


def _normalize_ticker(ticker: Any) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise CachedSourceBatchDryRunError("Ticker must be a non-empty string.")
    normalized = ticker.strip().upper()
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(normalized):
        raise CachedSourceBatchDryRunError(f"Ticker contains unsafe characters: {ticker}")
    return normalized


def _infer_ticker_from_snapshot_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_companyfacts"):
        stem = stem[: -len("_companyfacts")]
    try:
        return _normalize_ticker(stem)
    except CachedSourceBatchDryRunError:
        return "UNKNOWN"


def _age_days(fetched_at: str, generated_at: str | None) -> int | None:
    if not generated_at:
        return None
    try:
        fetched = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((generated - fetched).days, 0)


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
