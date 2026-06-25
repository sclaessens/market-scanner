from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    build_cached_source_snapshot_staging_validation,
)


DEFAULT_CACHED_SOURCE_SNAPSHOT_IMPORT_ROOT = Path(
    "data/market_engine/cached_source_snapshots"
)
IMPORT_COMMAND_IDENTIFIER = (
    "market_engine.source_refresh.cached_source_snapshot_import_command"
)


class CachedSourceSnapshotImportError(Exception):
    def __init__(
        self,
        reason: str,
        *,
        source_path: str | Path | None = None,
        expected_manifest: str | Path | None = None,
        issues: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.source_path = Path(source_path) if source_path is not None else None
        self.expected_manifest = (
            Path(expected_manifest) if expected_manifest is not None else None
        )
        self.issues = issues


def import_cached_source_snapshot(
    *,
    source_path: str | Path | None,
    destination_root: str | Path = DEFAULT_CACHED_SOURCE_SNAPSHOT_IMPORT_ROOT,
    validated_at: str | None = None,
) -> dict[str, Any]:
    if source_path is None or not str(source_path).strip():
        raise CachedSourceSnapshotImportError(
            "source path is required",
            source_path=source_path,
        )

    resolved_source = Path(source_path)
    manifest_path = _resolve_single_manifest_path(resolved_source)
    source_snapshot_dir = manifest_path.parent
    manifest_payload = _read_manifest(manifest_path)
    validation_entry = _accepted_validation_entry(
        source_snapshot_dir=source_snapshot_dir,
        validated_at=validated_at,
    )

    ticker = _required_text(manifest_payload, "ticker")
    snapshot_id = _required_text(manifest_payload, "snapshot_id")
    batch_id = _required_text(manifest_payload, "batch_id")
    source_family = _required_text(manifest_payload, "source_family")
    destination = Path(destination_root) / batch_id / ticker / snapshot_id
    if destination.exists():
        raise CachedSourceSnapshotImportError(
            "destination already exists",
            source_path=resolved_source,
            expected_manifest=destination / "manifest.json",
            issues=("destination_already_exists",),
        )

    _copy_snapshot_directory(
        source_snapshot_dir=source_snapshot_dir,
        destination=destination,
    )
    destination_validation_entry = _accepted_validation_entry(
        source_snapshot_dir=destination,
        validated_at=validated_at,
    )

    return {
        "import_status": "completed",
        "import_command": IMPORT_COMMAND_IDENTIFIER,
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
        "ticker": ticker,
        "source_family": source_family,
        "source_path": resolved_source.as_posix(),
        "source_snapshot_directory": source_snapshot_dir.as_posix(),
        "destination_path": destination.as_posix(),
        "manifest_path": (destination / "manifest.json").as_posix(),
        "imported_entities": (ticker,),
        "validation_status": destination_validation_entry["staging_validation_status"],
        "validation_issues": tuple(destination_validation_entry["issues"]),
        "warnings": tuple(validation_entry["issues"]),
        "forbidden_side_effect_confirmation": (
            "Cached-source snapshot import copied local files only. No provider, "
            "network, SEC/EDGAR, yfinance, broker, Telegram, portfolio, watchlist, "
            "production write outside the configured destination root, Decision "
            "Engine, Recommendation Review, ranking, scoring, allocation, order, "
            "execution, or tradeability behavior was invoked."
        ),
    }


def _resolve_single_manifest_path(source_path: Path) -> Path:
    if not source_path.exists():
        raise CachedSourceSnapshotImportError(
            "source path does not exist",
            source_path=source_path,
        )
    if not _is_readable(source_path):
        raise CachedSourceSnapshotImportError(
            "source path is not readable",
            source_path=source_path,
        )

    if source_path.is_file():
        if source_path.name != "manifest.json":
            raise CachedSourceSnapshotImportError(
                "source file must be manifest.json",
                source_path=source_path,
                expected_manifest=source_path.parent / "manifest.json",
                issues=("source_file_not_manifest",),
            )
        return source_path

    if not source_path.is_dir():
        raise CachedSourceSnapshotImportError(
            "source path must be a snapshot directory or manifest.json file",
            source_path=source_path,
            issues=("source_path_not_file_or_directory",),
        )

    manifest_paths = sorted(
        source_path.rglob("manifest.json"),
        key=lambda path: path.as_posix(),
    )
    if not manifest_paths:
        raise CachedSourceSnapshotImportError(
            "manifest file not found",
            source_path=source_path,
            expected_manifest=source_path / "manifest.json",
            issues=("manifest_missing",),
        )
    if len(manifest_paths) > 1:
        raise CachedSourceSnapshotImportError(
            "multiple manifest files found; supply one snapshot directory or manifest",
            source_path=source_path,
            issues=("multiple_manifests_found",),
        )
    return manifest_paths[0]


def _read_manifest(manifest_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CachedSourceSnapshotImportError(
            "manifest JSON is malformed",
            source_path=manifest_path,
            expected_manifest=manifest_path,
            issues=("manifest_json_malformed",),
        ) from exc
    except OSError as exc:
        raise CachedSourceSnapshotImportError(
            "manifest is not readable",
            source_path=manifest_path,
            expected_manifest=manifest_path,
            issues=("manifest_unreadable",),
        ) from exc
    if not isinstance(payload, Mapping):
        raise CachedSourceSnapshotImportError(
            "manifest must be a JSON object",
            source_path=manifest_path,
            expected_manifest=manifest_path,
            issues=("manifest_must_be_json_object",),
        )
    return payload


def _accepted_validation_entry(
    *,
    source_snapshot_dir: Path,
    validated_at: str | None,
) -> Mapping[str, Any]:
    report = build_cached_source_snapshot_staging_validation(
        staging_root=source_snapshot_dir,
        validated_at=validated_at,
    )
    entries = report["entries"]
    if len(entries) != 1:
        raise CachedSourceSnapshotImportError(
            "source must contain exactly one cached-source snapshot manifest",
            source_path=source_snapshot_dir,
            expected_manifest=source_snapshot_dir / "manifest.json",
            issues=("snapshot_identity_ambiguous",),
        )
    entry = entries[0]
    if entry["staging_validation_status"] != "accepted":
        raise CachedSourceSnapshotImportError(
            "cached-source snapshot validation failed",
            source_path=source_snapshot_dir,
            expected_manifest=source_snapshot_dir / "manifest.json",
            issues=tuple(entry["issues"]),
        )
    return entry


def _required_text(manifest_payload: Mapping[str, Any], field_name: str) -> str:
    value = manifest_payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CachedSourceSnapshotImportError(
            f"manifest field {field_name} is required",
            issues=(f"{field_name}_missing",),
        )
    return value.strip()


def _copy_snapshot_directory(
    *,
    source_snapshot_dir: Path,
    destination: Path,
) -> None:
    temporary_destination = destination.with_name(f"{destination.name}.importing")
    if temporary_destination.exists():
        raise CachedSourceSnapshotImportError(
            "temporary import destination already exists",
            expected_manifest=temporary_destination / "manifest.json",
            issues=("temporary_destination_already_exists",),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source_snapshot_dir, temporary_destination)
        temporary_destination.rename(destination)
    except OSError as exc:
        if temporary_destination.exists():
            shutil.rmtree(temporary_destination)
        raise CachedSourceSnapshotImportError(
            f"unable to copy cached-source snapshot: {exc}",
            source_path=source_snapshot_dir,
            expected_manifest=destination / "manifest.json",
            issues=("snapshot_copy_failed",),
        ) from exc


def _is_readable(path: Path) -> bool:
    try:
        if path.is_dir():
            tuple(path.iterdir())
        else:
            with path.open("rb"):
                pass
    except OSError:
        return False
    return True
