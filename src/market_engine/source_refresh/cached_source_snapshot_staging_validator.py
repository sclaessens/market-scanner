from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)


CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION = (
    "market-engine-cached-source-snapshot-staging-validation-v1"
)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "manifest_format_version",
    "snapshot_id",
    "batch_id",
    "created_at_utc",
    "acquired_at_utc",
    "acquisition_mode",
    "source_family",
    "source_name",
    "source_license_note",
    "redistribution_allowed",
    "local_use_allowed",
    "commit_allowed",
    "source_material_type",
    "ticker",
    "entity_name",
    "entity_country",
    "entity_exchange",
    "requested_document_type",
    "resolved_document_type",
    "source_retrieved_at_utc",
    "local_snapshot_path",
    "local_manifest_path",
    "local_payload_sha256",
    "local_payload_size_bytes",
    "payload_mime_type",
    "payload_encoding",
    "normalization_status",
    "validation_status",
    "validation_errors",
    "validation_warnings",
    "staleness_status",
    "staleness_reason",
    "usable_for_cached_source_dry_run",
    "blocked_reason",
    "notes",
)

_REQUIRED_TEXT_FIELDS: tuple[str, ...] = (
    "snapshot_id",
    "batch_id",
    "created_at_utc",
    "acquired_at_utc",
    "acquisition_mode",
    "source_family",
    "source_name",
    "source_license_note",
    "source_material_type",
    "ticker",
    "entity_name",
    "entity_country",
    "entity_exchange",
    "requested_document_type",
    "resolved_document_type",
    "source_retrieved_at_utc",
    "local_snapshot_path",
    "local_manifest_path",
    "local_payload_sha256",
    "payload_mime_type",
    "payload_encoding",
    "normalization_status",
    "validation_status",
    "staleness_status",
    "staleness_reason",
    "notes",
)


def build_cached_source_snapshot_staging_validation(
    *,
    staging_root: str | Path,
    validated_at: str | None = None,
    tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(staging_root)
    ticker_filter = _normalize_tickers(tickers or ())
    entries = _validation_entries(root=root, selected_tickers=set(ticker_filter))
    counts = _counts(entries)
    return {
        "report_format_version": CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION,
        "validated_at": validated_at or _utc_now_text(),
        "staging_root": root.as_posix(),
        "ticker_filter": ticker_filter,
        "counts": counts,
        "acceptance_summary": {
            "accepted_entries": counts["accepted_entries"],
            "rejected_entries": counts["rejected_entries"],
            "all_entries_accepted": counts["total_inspected_entries"] > 0
            and counts["rejected_entries"] == 0,
        },
        "entries": entries,
        "forbidden_side_effect_confirmation": (
            "Staging validation inspected local files only. No provider, network, "
            "broker, Telegram, portfolio, watchlist, production write, Decision "
            "Engine, Recommendation Review, ranking, scoring, allocation, order, "
            "execution, or tradeability behavior was invoked."
        ),
    }


def _validation_entries(
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
        _missing_manifest_entries(root=root, manifest_parent_paths=manifest_parent_paths)
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
        "payload_path": None,
        "manifest_format_version": None,
        "staging_validation_status": "rejected",
        "accepted_for_cached_source_staging": False,
        "validation_status": None,
        "staleness_status": None,
        "issues": (),
    }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            **base_entry,
            "staging_validation_status": "malformed_manifest",
            "issues": ("manifest_json_malformed",),
        }
    except OSError:
        return {
            **base_entry,
            "staging_validation_status": "malformed_manifest",
            "issues": ("manifest_unreadable",),
        }
    if not isinstance(payload, Mapping):
        return {
            **base_entry,
            "staging_validation_status": "malformed_manifest",
            "issues": ("manifest_must_be_json_object",),
        }

    payload_path = _referenced_payload_path(
        root=root,
        manifest_path=manifest_path,
        manifest_payload=payload,
    )
    issues = _manifest_issues(
        root=root,
        manifest_path=manifest_path,
        manifest_payload=payload,
        payload_path=payload_path,
    )
    manifest_format_version = payload.get("manifest_format_version")
    staging_status = _staging_status(
        manifest_format_version=manifest_format_version,
        issues=issues,
    )
    accepted = staging_status == "accepted"
    return {
        "ticker": _string_or_none(payload.get("ticker")),
        "snapshot_id": _string_or_none(payload.get("snapshot_id")),
        "source_family": _string_or_none(payload.get("source_family")),
        "source_name": _string_or_none(payload.get("source_name")),
        "manifest_path": manifest_reference,
        "payload_path": _relative_path(payload_path, root) if payload_path else None,
        "manifest_format_version": _string_or_none(manifest_format_version),
        "staging_validation_status": staging_status,
        "accepted_for_cached_source_staging": accepted,
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
        if directory == root or directory.resolve() in manifest_parent_paths:
            continue
        try:
            children = tuple(directory.iterdir())
        except OSError:
            children = ()
        if any(child.is_dir() for child in children):
            continue
        yield {
            "ticker": _infer_ticker_from_directory(root=root, directory=directory),
            "snapshot_id": directory.name,
            "source_family": None,
            "source_name": None,
            "manifest_path": None,
            "directory_path": _relative_path(directory, root),
            "payload_path": None,
            "manifest_format_version": None,
            "staging_validation_status": "missing_manifest",
            "accepted_for_cached_source_staging": False,
            "validation_status": None,
            "staleness_status": None,
            "issues": ("manifest_missing",),
        }


def _manifest_issues(
    *,
    root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    payload_path: Path | None,
) -> list[str]:
    issues: list[str] = []
    if (
        manifest_payload.get("manifest_format_version")
        != CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION
    ):
        issues.append("manifest_format_unknown")

    for field_name in _REQUIRED_FIELDS:
        if field_name not in manifest_payload:
            issues.append(f"{field_name}_missing")
    for field_name in _REQUIRED_TEXT_FIELDS:
        if field_name in manifest_payload and not _is_non_empty_text(
            manifest_payload.get(field_name)
        ):
            issues.append(f"{field_name}_invalid")

    _append_boolean_type_issue(issues, manifest_payload, "redistribution_allowed")
    _append_boolean_type_issue(issues, manifest_payload, "local_use_allowed")
    _append_boolean_type_issue(issues, manifest_payload, "commit_allowed")
    _append_boolean_type_issue(
        issues,
        manifest_payload,
        "usable_for_cached_source_dry_run",
    )
    if "local_payload_size_bytes" in manifest_payload and not isinstance(
        manifest_payload.get("local_payload_size_bytes"),
        int,
    ):
        issues.append("local_payload_size_bytes_invalid")
    if "validation_errors" in manifest_payload and not isinstance(
        manifest_payload.get("validation_errors"),
        list,
    ):
        issues.append("validation_errors_invalid")
    if "validation_warnings" in manifest_payload and not isinstance(
        manifest_payload.get("validation_warnings"),
        list,
    ):
        issues.append("validation_warnings_invalid")

    if payload_path is None:
        issues.append("local_snapshot_path_missing")
    elif not _path_is_under_root_or_manifest_dir(
        path=payload_path,
        root=root,
        manifest_path=manifest_path,
    ):
        issues.append("referenced_snapshot_path_outside_staging_root")
    elif not payload_path.exists():
        issues.append("referenced_snapshot_file_missing")
    elif not payload_path.is_file():
        issues.append("referenced_snapshot_not_file")
    else:
        _append_payload_integrity_issues(
            issues,
            manifest_payload=manifest_payload,
            payload_path=payload_path,
        )

    manifest_reference = _resolve_local_path(
        root=root,
        manifest_path=manifest_path,
        value=manifest_payload.get("local_manifest_path"),
    )
    if manifest_reference is None:
        issues.append("local_manifest_path_missing")
    elif not _path_is_under_root_or_manifest_dir(
        path=manifest_reference,
        root=root,
        manifest_path=manifest_path,
    ):
        issues.append("local_manifest_path_outside_staging_root")
    elif manifest_reference.resolve() != manifest_path.resolve():
        issues.append("local_manifest_path_mismatch")

    if manifest_payload.get("staleness_status") == "stale":
        issues.append("snapshot_stale")
    if manifest_payload.get("validation_status") in {"failed", "not_validated"}:
        issues.append(f"validation_status_{manifest_payload['validation_status']}")
    if manifest_payload.get("acquisition_mode") == "test_fixture":
        issues.append("test_fixture_not_real_coverage")
    if manifest_payload.get("source_material_type") == "synthetic_fixture":
        issues.append("synthetic_fixture_not_real_coverage")
    if manifest_payload.get("usable_for_cached_source_dry_run") is False:
        issues.append("usable_for_cached_source_dry_run_false")
    if manifest_payload.get("usable_for_cached_source_dry_run") is True and issues:
        issues.append("usable_flag_conflicts_with_staging_issues")
    return tuple(sorted(set(issues)))


def _append_payload_integrity_issues(
    issues: list[str],
    *,
    manifest_payload: Mapping[str, Any],
    payload_path: Path,
) -> None:
    expected_hash = manifest_payload.get("local_payload_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        issues.append("local_payload_sha256_invalid")
    else:
        try:
            if _sha256(payload_path) != expected_hash:
                issues.append("referenced_snapshot_hash_mismatch")
        except OSError:
            issues.append("referenced_snapshot_unreadable")
    expected_size = manifest_payload.get("local_payload_size_bytes")
    if not isinstance(expected_size, int) or expected_size <= 0:
        issues.append("local_payload_size_bytes_invalid")
    else:
        try:
            if payload_path.stat().st_size != expected_size:
                issues.append("referenced_snapshot_size_mismatch")
        except OSError:
            issues.append("referenced_snapshot_unreadable")


def _append_boolean_type_issue(
    issues: list[str],
    manifest_payload: Mapping[str, Any],
    field_name: str,
) -> None:
    if field_name in manifest_payload and not isinstance(
        manifest_payload.get(field_name),
        bool,
    ):
        issues.append(f"{field_name}_invalid")


def _staging_status(
    *,
    manifest_format_version: object,
    issues: Sequence[str],
) -> str:
    if manifest_format_version != CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION:
        return "unknown_format"
    if issues:
        return "rejected"
    return "accepted"


def _counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total_inspected_entries": len(entries),
        "accepted_entries": sum(
            1 for entry in entries if entry["staging_validation_status"] == "accepted"
        ),
        "rejected_entries": sum(
            1 for entry in entries if entry["staging_validation_status"] != "accepted"
        ),
        "missing_manifest_count": sum(
            1 for entry in entries if entry["staging_validation_status"] == "missing_manifest"
        ),
        "malformed_manifest_count": sum(
            1
            for entry in entries
            if entry["staging_validation_status"] == "malformed_manifest"
        ),
        "unknown_format_count": sum(
            1 for entry in entries if entry["staging_validation_status"] == "unknown_format"
        ),
        "missing_referenced_file_count": _issue_count(
            entries,
            "referenced_snapshot_file_missing",
        ),
        "hash_mismatch_count": _issue_count(entries, "referenced_snapshot_hash_mismatch"),
        "size_mismatch_count": _issue_count(entries, "referenced_snapshot_size_mismatch"),
        "stale_count": _issue_count(entries, "snapshot_stale"),
        "fixture_or_test_material_count": sum(
            1
            for entry in entries
            if "test_fixture_not_real_coverage" in entry["issues"]
            or "synthetic_fixture_not_real_coverage" in entry["issues"]
        ),
        "validation_status_blocked_count": sum(
            1
            for entry in entries
            if "validation_status_failed" in entry["issues"]
            or "validation_status_not_validated" in entry["issues"]
        ),
        "usable_flag_conflict_count": _issue_count(
            entries,
            "usable_flag_conflicts_with_staging_issues",
        ),
    }


def _issue_count(entries: Sequence[Mapping[str, Any]], issue_code: str) -> int:
    return sum(1 for entry in entries if issue_code in entry["issues"])


def _referenced_payload_path(
    *,
    root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
) -> Path | None:
    return _resolve_local_path(
        root=root,
        manifest_path=manifest_path,
        value=manifest_payload.get("local_snapshot_path"),
    )


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


def _path_is_under_root_or_manifest_dir(
    *,
    path: Path,
    root: Path,
    manifest_path: Path,
) -> bool:
    resolved_path = path.resolve()
    allowed_roots = (root.resolve(), manifest_path.parent.resolve())
    for allowed_root in allowed_roots:
        try:
            resolved_path.relative_to(allowed_root)
            return True
        except ValueError:
            continue
    return False


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


def _is_non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
