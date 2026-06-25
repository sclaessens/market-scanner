from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

from market_engine.source_refresh import cached_source_snapshot_import_command as command
from market_engine.source_refresh.cached_source_snapshot_importer import (
    CachedSourceSnapshotImportError,
    import_cached_source_snapshot,
)
from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)


VALIDATED_AT = "2026-06-25T13:00:00Z"


def test_import_valid_operator_snapshot_copies_into_destination_root(
    tmp_path: Path,
) -> None:
    source_snapshot_dir = _write_valid_operator_snapshot(tmp_path / "operator")
    destination_root = tmp_path / "workspace"

    result = import_cached_source_snapshot(
        source_path=source_snapshot_dir,
        destination_root=destination_root,
        validated_at=VALIDATED_AT,
    )

    destination = destination_root / "batch-001" / "NVDA" / "NVDA-snapshot"
    assert result["import_status"] == "completed"
    assert result["snapshot_id"] == "NVDA-snapshot"
    assert result["ticker"] == "NVDA"
    assert result["source_family"] == "sec_companyfacts"
    assert result["destination_path"] == destination.as_posix()
    assert result["manifest_path"] == (destination / "manifest.json").as_posix()
    assert result["imported_entities"] == ("NVDA",)
    assert result["validation_status"] == "accepted"
    assert result["warnings"] == ()
    assert (destination / "manifest.json").exists()
    assert (destination / "payload.json").read_text(encoding="utf-8") == (
        '{"fixture": true}\n'
    )
    assert (source_snapshot_dir / "payload.json").exists()


def test_import_accepts_direct_manifest_path(tmp_path: Path) -> None:
    source_snapshot_dir = _write_valid_operator_snapshot(tmp_path / "operator")
    destination_root = tmp_path / "workspace"

    result = import_cached_source_snapshot(
        source_path=source_snapshot_dir / "manifest.json",
        destination_root=destination_root,
        validated_at=VALIDATED_AT,
    )

    assert result["snapshot_id"] == "NVDA-snapshot"
    assert (destination_root / "batch-001" / "NVDA" / "NVDA-snapshot").exists()


def test_import_missing_source_path_fails_closed() -> None:
    try:
        import_cached_source_snapshot(source_path=None)
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "source path is required"
    else:
        raise AssertionError("expected missing source path to fail")


def test_import_nonexistent_source_path_fails_closed(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    try:
        import_cached_source_snapshot(source_path=missing_path)
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "source path does not exist"
        assert exc.source_path == missing_path
    else:
        raise AssertionError("expected nonexistent source path to fail")


def test_import_missing_manifest_fails_closed(tmp_path: Path) -> None:
    source_dir = tmp_path / "operator"
    source_dir.mkdir()
    (source_dir / "payload.json").write_text("{}", encoding="utf-8")

    try:
        import_cached_source_snapshot(source_path=source_dir)
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "manifest file not found"
        assert exc.issues == ("manifest_missing",)
    else:
        raise AssertionError("expected missing manifest to fail")


def test_import_invalid_manifest_fails_closed(tmp_path: Path) -> None:
    source_dir = tmp_path / "operator"
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    try:
        import_cached_source_snapshot(source_path=source_dir)
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "manifest JSON is malformed"
        assert exc.issues == ("manifest_json_malformed",)
    else:
        raise AssertionError("expected malformed manifest to fail")


def test_import_validation_failure_does_not_copy(tmp_path: Path) -> None:
    source_snapshot_dir = _write_valid_operator_snapshot(
        tmp_path / "operator",
        validation_status="not_validated",
        usable_for_cached_source_dry_run=True,
    )
    destination_root = tmp_path / "workspace"

    try:
        import_cached_source_snapshot(
            source_path=source_snapshot_dir,
            destination_root=destination_root,
            validated_at=VALIDATED_AT,
        )
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "cached-source snapshot validation failed"
        assert "validation_status_not_validated" in exc.issues
    else:
        raise AssertionError("expected validation failure to fail")
    assert not destination_root.exists()


def test_import_destination_already_exists_fails_without_overwrite(
    tmp_path: Path,
) -> None:
    source_snapshot_dir = _write_valid_operator_snapshot(tmp_path / "operator")
    destination_root = tmp_path / "workspace"
    existing_destination = destination_root / "batch-001" / "NVDA" / "NVDA-snapshot"
    existing_destination.mkdir(parents=True)
    (existing_destination / "sentinel.txt").write_text("keep me", encoding="utf-8")

    try:
        import_cached_source_snapshot(
            source_path=source_snapshot_dir,
            destination_root=destination_root,
            validated_at=VALIDATED_AT,
        )
    except CachedSourceSnapshotImportError as exc:
        assert exc.reason == "destination already exists"
        assert exc.issues == ("destination_already_exists",)
    else:
        raise AssertionError("expected existing destination to fail")
    assert (existing_destination / "sentinel.txt").read_text(encoding="utf-8") == "keep me"


def test_import_command_prints_stable_success_summary(tmp_path: Path) -> None:
    source_snapshot_dir = _write_valid_operator_snapshot(tmp_path / "operator")
    destination_root = tmp_path / "workspace"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--source-path",
            str(source_snapshot_dir),
            "--destination-root",
            str(destination_root),
            "--validated-at",
            VALIDATED_AT,
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    output = stdout.getvalue()
    assert "Cached-source snapshot import completed" in output
    assert "Snapshot ID: NVDA-snapshot" in output
    assert f"Source path: {source_snapshot_dir}" in output
    assert f"Destination path: {destination_root / 'batch-001' / 'NVDA' / 'NVDA-snapshot'}" in output
    assert "Imported entities: NVDA" in output
    assert "Validation: accepted" in output
    assert "Warnings: none" in output
    assert "No provider, network" in output


def test_import_command_prints_stable_failure_summary(tmp_path: Path) -> None:
    source_dir = tmp_path / "operator"
    source_dir.mkdir()
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--source-path",
            str(source_dir),
            "--destination-root",
            str(tmp_path / "workspace"),
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    output = stderr.getvalue()
    assert "Cached-source snapshot import failed" in output
    assert "Reason: manifest file not found" in output
    assert f"Source path: {source_dir}" in output
    assert f"Expected manifest: {source_dir / 'manifest.json'}" in output
    assert "Issues: manifest_missing" in output


def test_import_command_path_does_not_import_provider_modules() -> None:
    assert "SecCompanyFactsProvider" not in command.__dict__
    assert "requests" not in command.__dict__
    assert "yfinance" not in command.__dict__
    assert "telegram" not in command.__dict__


def _write_valid_operator_snapshot(
    root: Path,
    *,
    ticker: str = "NVDA",
    validation_status: str = "passed",
    usable_for_cached_source_dry_run: object = True,
) -> Path:
    snapshot_dir = root / "snapshot"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"fixture": true}\n', encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker=ticker,
        snapshot_id=f"{ticker}-snapshot",
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status=validation_status,
        usable_for_cached_source_dry_run=usable_for_cached_source_dry_run,
    )
    return snapshot_dir


def _write_manifest(
    path: Path,
    *,
    ticker: str,
    snapshot_id: str,
    local_payload_sha256: str,
    local_payload_size_bytes: int,
    validation_status: str,
    usable_for_cached_source_dry_run: object,
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_format_version": (
                    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION
                ),
                "snapshot_id": snapshot_id,
                "batch_id": "batch-001",
                "created_at_utc": VALIDATED_AT,
                "acquired_at_utc": VALIDATED_AT,
                "acquisition_mode": "operator_supplied",
                "source_family": "sec_companyfacts",
                "source_name": "SEC CompanyFacts",
                "source_url": "https://www.example.invalid/sec/companyfacts.json",
                "source_license_note": "Synthetic import test fixture.",
                "redistribution_allowed": False,
                "local_use_allowed": True,
                "commit_allowed": False,
                "source_material_type": "official_regulatory_source",
                "ticker": ticker,
                "entity_name": f"{ticker} Fixture",
                "entity_country": "US",
                "entity_exchange": "NASDAQ",
                "source_entity_identifier": "0000000000",
                "cik": "0000000000",
                "requested_document_type": "companyfacts",
                "resolved_document_type": "companyfacts",
                "requested_period": "latest",
                "resolved_period": "latest",
                "source_publication_date": None,
                "source_retrieved_at_utc": VALIDATED_AT,
                "local_snapshot_path": "payload.json",
                "local_manifest_path": "manifest.json",
                "local_payload_sha256": local_payload_sha256,
                "local_payload_size_bytes": local_payload_size_bytes,
                "payload_mime_type": "application/json",
                "payload_encoding": "utf-8",
                "normalization_status": "raw_only",
                "validation_status": validation_status,
                "validation_errors": [],
                "validation_warnings": [],
                "staleness_status": "fresh",
                "staleness_reason": "Synthetic deterministic import test fixture.",
                "usable_for_cached_source_dry_run": usable_for_cached_source_dry_run,
                "blocked_reason": None,
                "notes": "Synthetic import test fixture only; no source data was acquired.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
