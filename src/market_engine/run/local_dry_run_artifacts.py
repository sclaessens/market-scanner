from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION = (
    "market-engine-local-dry-run-artifact-v1"
)

MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION = (
    "market-engine-local-dry-run-artifact-manifest-v1"
)

MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY = (
    "artifacts/market_engine/dry_runs"
)

APPROVED_DRY_RUN_PAYLOAD_FORMAT_VERSION = "market-engine-end-to-end-dry-run-v1"

LOCAL_DRY_RUN_PERSISTENCE_MODE = "local_dry_run_only"

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class LocalDryRunArtifactError(ValueError):
    """Raised when a local dry-run artifact cannot be persisted safely."""


@dataclass(frozen=True)
class LocalDryRunArtifactPersistenceResult:
    run_directory: Path
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def persist_market_engine_local_dry_run_artifact(
    dry_run_payload: Mapping[str, Any],
    *,
    output_root: Path | str = MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    artifact_created_at: str,
    allow_overwrite: bool = False,
) -> LocalDryRunArtifactPersistenceResult:
    """Persist an already-built Market Engine end-to-end dry-run payload locally."""

    payload = _validated_dry_run_payload(dry_run_payload)
    safe_dry_run_id = _safe_path_segment(
        payload["dry_run_id"],
        field_name="dry_run_id",
    )
    artifact_date = _artifact_date(
        payload.get("generated_at") or artifact_created_at,
    )
    root = _validated_output_root(output_root)
    root_resolved = root.resolve()
    run_directory = _resolved_child(root_resolved, safe_dry_run_id)
    artifact_directory = _resolved_child(run_directory, "artifacts")
    artifact_filename = (
        f"market_engine_dry_run_{safe_dry_run_id}_{artifact_date}.json"
    )
    artifact_path = _resolved_child(artifact_directory, artifact_filename)
    manifest_path = _resolved_child(run_directory, "manifest.json")

    if run_directory.exists() and not allow_overwrite:
        raise LocalDryRunArtifactError(
            f"Local dry-run artifact directory already exists: {run_directory}"
        )

    artifact_payload = _artifact_payload(
        payload,
        artifact_created_at=artifact_created_at,
        artifact_path=artifact_path,
        root_resolved=root_resolved,
    )
    manifest = _manifest_payload(
        payload,
        artifact_created_at=artifact_created_at,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        root_resolved=root_resolved,
    )

    artifact_directory.mkdir(parents=True, exist_ok=allow_overwrite)
    _write_json(artifact_path, artifact_payload, allow_overwrite=allow_overwrite)
    _write_json(manifest_path, manifest, allow_overwrite=allow_overwrite)

    return LocalDryRunArtifactPersistenceResult(
        run_directory=run_directory,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _validated_dry_run_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LocalDryRunArtifactError("Dry-run artifact payload must be a mapping.")

    dry_run_format_version = payload.get("dry_run_format_version")
    if dry_run_format_version != APPROVED_DRY_RUN_PAYLOAD_FORMAT_VERSION:
        raise LocalDryRunArtifactError(
            "Dry-run artifact payload uses an unsupported format version."
        )

    required_fields = ("dry_run_id", "input_mode", "run_state")
    missing_fields = tuple(
        field_name
        for field_name in required_fields
        if not isinstance(payload.get(field_name), str) or not payload.get(field_name)
    )
    if missing_fields:
        raise LocalDryRunArtifactError(
            "Dry-run artifact payload is missing required identity fields: "
            + ", ".join(missing_fields)
        )

    return _json_ready(payload, path="dry_run_payload")


def _artifact_payload(
    dry_run_payload: Mapping[str, Any],
    *,
    artifact_created_at: str,
    artifact_path: Path,
    root_resolved: Path,
) -> dict[str, Any]:
    return {
        "artifact_format_version": MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION,
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": artifact_created_at,
        "artifact_persistence_mode": LOCAL_DRY_RUN_PERSISTENCE_MODE,
        "artifact_path_category": MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
        "artifact_relative_path": _relative_posix(artifact_path, root_resolved),
        "non_production_artifact": True,
        "source_dry_run_format_version": dry_run_payload["dry_run_format_version"],
        "source_dry_run_id": dry_run_payload["dry_run_id"],
        "source_dry_run_generated_at": dry_run_payload.get("generated_at"),
        "source_input_mode": dry_run_payload["input_mode"],
        "source_run_state": dry_run_payload["run_state"],
        "payload": dry_run_payload,
    }


def _manifest_payload(
    dry_run_payload: Mapping[str, Any],
    *,
    artifact_created_at: str,
    artifact_path: Path,
    manifest_path: Path,
    root_resolved: Path,
) -> dict[str, Any]:
    return {
        "manifest_format_version": MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION,
        "artifact_count": 1,
        "artifact_created_at": artifact_created_at,
        "artifact_persistence_mode": LOCAL_DRY_RUN_PERSISTENCE_MODE,
        "artifact_path_category": MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
        "manifest_relative_path": _relative_posix(manifest_path, root_resolved),
        "non_production_artifact": True,
        "source_dry_run_format_version": dry_run_payload["dry_run_format_version"],
        "source_dry_run_id": dry_run_payload["dry_run_id"],
        "source_dry_run_generated_at": dry_run_payload.get("generated_at"),
        "source_input_mode": dry_run_payload["input_mode"],
        "source_run_state": dry_run_payload["run_state"],
        "artifacts": (
            {
                "artifact_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION
                ),
                "artifact_type": "market_engine_end_to_end_dry_run",
                "artifact_relative_path": _relative_posix(
                    artifact_path,
                    root_resolved,
                ),
                "source_dry_run_format_version": dry_run_payload[
                    "dry_run_format_version"
                ],
                "source_dry_run_id": dry_run_payload["dry_run_id"],
                "source_run_state": dry_run_payload["run_state"],
            },
        ),
    }


def _json_ready(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, nested_value in value.items():
            if isinstance(key, Enum):
                key = str(key.value)
            else:
                key = str(key)
            ready[key] = _json_ready(nested_value, path=f"{path}.{key}")
        return ready
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(nested_value, path=f"{path}[{index}]")
            for index, nested_value in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise LocalDryRunArtifactError(
        f"Dry-run artifact value is not JSON serializable at {path}."
    )


def _write_json(path: Path, payload: Mapping[str, Any], *, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise LocalDryRunArtifactError(f"Local dry-run artifact already exists: {path}")
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except TypeError as exc:
        raise LocalDryRunArtifactError(
            f"Local dry-run artifact could not be serialized: {path}"
        ) from exc
    except OSError as exc:
        raise LocalDryRunArtifactError(
            f"Local dry-run artifact could not be written: {path}"
        ) from exc


def _safe_path_segment(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalDryRunArtifactError(f"{field_name} must be a non-empty string.")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise LocalDryRunArtifactError(f"{field_name} is not a safe path segment.")
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(value):
        raise LocalDryRunArtifactError(f"{field_name} contains unsafe characters.")
    return value


def _validated_output_root(output_root: Path | str) -> Path:
    root = Path(output_root)
    if not root.parts:
        raise LocalDryRunArtifactError("Local dry-run artifact output root is required.")
    if any(part == ".." for part in root.parts):
        raise LocalDryRunArtifactError(
            "Local dry-run artifact output root may not contain parent traversal."
        )
    return root


def _artifact_date(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise LocalDryRunArtifactError("Artifact creation timestamp is required.")
    return _safe_path_segment(value[:10], field_name="artifact_date")


def _resolved_child(parent: Path, child: str) -> Path:
    safe_child = _safe_path_segment(child, field_name="path_component")
    candidate = (parent / safe_child).resolve()
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise LocalDryRunArtifactError(
            f"Resolved local dry-run artifact path escapes output root: {candidate}"
        ) from exc
    return candidate


def _relative_posix(path: Path, root_resolved: Path) -> str:
    try:
        return path.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise LocalDryRunArtifactError(
            f"Resolved local dry-run artifact path escapes output root: {path}"
        ) from exc
