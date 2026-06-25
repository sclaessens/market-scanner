from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CACHED_SOURCE_SNAPSHOT_INVENTORY_FORMAT_VERSION = (
    "market-engine-cached-source-snapshot-inventory-v1"
)
CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION = (
    "market-engine-cached-source-snapshot-acquisition-manifest-v1"
)


def build_cached_source_snapshot_inventory(
    *,
    input_root: str | Path,
    inspected_at: str | None = None,
    tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(input_root)
    selected_tickers = _normalize_tickers(tickers or ())
    entries = _inventory_entries(root=root, selected_tickers=set(selected_tickers))
    counts = _counts(entries)
    return {
        "report_format_version": CACHED_SOURCE_SNAPSHOT_INVENTORY_FORMAT_VERSION,
        "inspected_at": inspected_at or _utc_now_text(),
        "input_root": root.as_posix(),
        "ticker_filter": selected_tickers,
        "counts": counts,
        "entries": entries,
        "forbidden_side_effect_confirmation": (
            "Inventory inspected local files only. No provider, network, broker, "
            "Telegram, portfolio, watchlist, production write, Decision Engine, "
            "recommendation, allocation, order, or execution behavior was invoked."
        ),
    }


def _inventory_entries(
    *,
    root: Path,
    selected_tickers: set[str],
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    manifest_paths = sorted(root.rglob("manifest.json"), key=lambda path: path.as_posix())
    manifest_parent_paths = {path.parent.resolve() for path in manifest_paths}

    entries = [
        _entry_from_manifest(root=root, manifest_path=manifest_path)
        for manifest_path in manifest_paths
    ]
    entries.extend(
        _missing_manifest_entries(
            root=root,
            manifest_parent_paths=manifest_parent_paths,
        )
    )

    filtered = [
        entry
        for entry in entries
        if not selected_tickers or str(entry.get("ticker") or "").upper() in selected_tickers
    ]
    return sorted(
        filtered,
        key=lambda entry: (
            str(entry.get("ticker") or ""),
            str(entry.get("snapshot_id") or ""),
            str(entry.get("manifest_path") or ""),
            str(entry.get("directory_path") or ""),
        ),
    )


def _entry_from_manifest(*, root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_reference = _relative_path(manifest_path, root)
    base_entry = {
        "ticker": None,
        "snapshot_id": None,
        "source_family": None,
        "source_name": None,
        "manifest_path": manifest_reference,
        "snapshot_path": None,
        "manifest_format_version": None,
        "inventory_status": "unusable",
        "usable_for_cached_source_dry_run": False,
        "validation_status": None,
        "staleness_status": None,
        "issues": (),
    }

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            **base_entry,
            "inventory_status": "malformed_manifest",
            "issues": ("manifest_json_malformed",),
        }
    except OSError:
        return {
            **base_entry,
            "inventory_status": "malformed_manifest",
            "issues": ("manifest_unreadable",),
        }
    if not isinstance(payload, Mapping):
        return {
            **base_entry,
            "inventory_status": "malformed_manifest",
            "issues": ("manifest_must_be_json_object",),
        }

    manifest_format_version = payload.get("manifest_format_version")
    referenced_path = _referenced_snapshot_path(
        root=root,
        manifest_path=manifest_path,
        manifest_payload=payload,
    )
    issues = _manifest_issues(
        root=root,
        manifest_path=manifest_path,
        manifest_payload=payload,
        referenced_path=referenced_path,
    )
    inventory_status = _inventory_status(
        manifest_format_version=manifest_format_version,
        issues=issues,
        usable=payload.get("usable_for_cached_source_dry_run"),
    )
    return {
        "ticker": _string_or_none(payload.get("ticker")),
        "snapshot_id": _string_or_none(payload.get("snapshot_id")),
        "source_family": _string_or_none(payload.get("source_family")),
        "source_name": _string_or_none(payload.get("source_name")),
        "manifest_path": manifest_reference,
        "snapshot_path": _relative_path(referenced_path, root) if referenced_path else None,
        "manifest_format_version": _string_or_none(manifest_format_version),
        "inventory_status": inventory_status,
        "usable_for_cached_source_dry_run": bool(
            payload.get("usable_for_cached_source_dry_run")
        )
        and inventory_status == "usable",
        "validation_status": _string_or_none(payload.get("validation_status")),
        "staleness_status": _string_or_none(payload.get("staleness_status")),
        "issues": tuple(issues),
    }


def _missing_manifest_entries(
    *,
    root: Path,
    manifest_parent_paths: set[Path],
) -> Iterable[dict[str, Any]]:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: path.as_posix(),
    ):
        if directory.resolve() in manifest_parent_paths:
            continue
        try:
            children = tuple(directory.iterdir())
        except OSError:
            children = ()
        if any(child.is_dir() for child in children):
            continue
        if directory == root:
            continue
        yield {
            "ticker": _infer_ticker_from_directory(root=root, directory=directory),
            "snapshot_id": directory.name,
            "source_family": None,
            "source_name": None,
            "manifest_path": None,
            "directory_path": _relative_path(directory, root),
            "snapshot_path": None,
            "manifest_format_version": None,
            "inventory_status": "missing_manifest",
            "usable_for_cached_source_dry_run": False,
            "validation_status": None,
            "staleness_status": None,
            "issues": ("manifest_missing",),
        }


def _manifest_issues(
    *,
    root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    referenced_path: Path | None,
) -> list[str]:
    issues: list[str] = []
    if (
        manifest_payload.get("manifest_format_version")
        != CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION
    ):
        issues.append("manifest_format_unknown")
    for field_name in (
        "snapshot_id",
        "ticker",
        "source_family",
        "validation_status",
        "staleness_status",
        "usable_for_cached_source_dry_run",
    ):
        if field_name not in manifest_payload:
            issues.append(f"{field_name}_missing")
    if not referenced_path:
        issues.append("local_snapshot_path_missing")
    elif not referenced_path.exists():
        issues.append("referenced_snapshot_file_missing")
    else:
        expected_hash = manifest_payload.get("local_payload_sha256")
        if isinstance(expected_hash, str) and expected_hash:
            observed_hash = _sha256(referenced_path)
            if observed_hash != expected_hash:
                issues.append("referenced_snapshot_hash_mismatch")
        expected_size = manifest_payload.get("local_payload_size_bytes")
        if isinstance(expected_size, int) and expected_size > 0:
            try:
                if referenced_path.stat().st_size != expected_size:
                    issues.append("referenced_snapshot_size_mismatch")
            except OSError:
                issues.append("referenced_snapshot_unreadable")

    validation_status = manifest_payload.get("validation_status")
    if validation_status in {"failed", "not_validated"}:
        issues.append(f"validation_status_{validation_status}")
    if manifest_payload.get("staleness_status") == "stale":
        issues.append("snapshot_stale")
    if manifest_payload.get("source_material_type") == "synthetic_fixture":
        issues.append("synthetic_fixture_not_real_coverage")
    if manifest_payload.get("acquisition_mode") == "test_fixture":
        issues.append("test_fixture_not_real_coverage")
    if manifest_payload.get("local_manifest_path"):
        resolved_manifest_reference = _resolve_local_path(
            root=root,
            manifest_path=manifest_path,
            value=manifest_payload.get("local_manifest_path"),
        )
        if resolved_manifest_reference and resolved_manifest_reference.resolve() != manifest_path.resolve():
            issues.append("local_manifest_path_mismatch")
    if manifest_payload.get("usable_for_cached_source_dry_run") is True and issues:
        issues.append("usable_flag_conflicts_with_inventory_issues")
    return sorted(set(issues))


def _inventory_status(
    *,
    manifest_format_version: object,
    issues: Sequence[str],
    usable: object,
) -> str:
    if manifest_format_version != CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION:
        return "unknown_format"
    if usable is True and not issues:
        return "usable"
    return "unusable"


def _counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total_inspected_entries": len(entries),
        "usable_entries": sum(1 for entry in entries if entry["inventory_status"] == "usable"),
        "unusable_entries": sum(1 for entry in entries if entry["inventory_status"] != "usable"),
        "missing_manifest_count": sum(
            1 for entry in entries if entry["inventory_status"] == "missing_manifest"
        ),
        "malformed_manifest_count": sum(
            1 for entry in entries if entry["inventory_status"] == "malformed_manifest"
        ),
        "unknown_format_count": sum(
            1 for entry in entries if entry["inventory_status"] == "unknown_format"
        ),
        "missing_referenced_file_count": sum(
            1 for entry in entries if "referenced_snapshot_file_missing" in entry["issues"]
        ),
        "stale_count": sum(
            1
            for entry in entries
            if entry.get("staleness_status") == "stale"
            or "snapshot_stale" in entry["issues"]
        ),
    }


def _referenced_snapshot_path(
    *,
    root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
) -> Path | None:
    value = manifest_payload.get("local_snapshot_path")
    return _resolve_local_path(root=root, manifest_path=manifest_path, value=value)


def _resolve_local_path(
    *,
    root: Path,
    manifest_path: Path,
    value: object,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = root / path
    if root_candidate.exists():
        return root_candidate
    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate
    return root_candidate


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _infer_ticker_from_directory(*, root: Path, directory: Path) -> str | None:
    try:
        parts = directory.relative_to(root).parts
    except ValueError:
        parts = directory.parts
    if len(parts) >= 2:
        return parts[-2].upper()
    if parts:
        return parts[-1].upper()
    return None


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in tickers:
        for part in str(value).split(","):
            ticker = part.strip().upper()
            if ticker:
                normalized.add(ticker)
    return tuple(sorted(normalized))


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
