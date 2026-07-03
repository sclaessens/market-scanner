from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

from market_engine.source_refresh import (
    cached_source_snapshot_staging_validator_command as command,
)
from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)
from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION,
    build_cached_source_snapshot_staging_validation,
)


VALIDATED_AT = "2026-06-25T11:00:00Z"


def test_staging_validation_empty_root_is_deterministic(tmp_path: Path) -> None:
    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["report_format_version"] == (
        CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION
    )
    assert report["validated_at"] == VALIDATED_AT
    assert report["entries"] == []
    assert report["acceptance_summary"] == {
        "accepted_entries": 0,
        "rejected_entries": 0,
        "all_entries_accepted": False,
    }
    assert report["counts"] == {
        "total_inspected_entries": 0,
        "accepted_entries": 0,
        "rejected_entries": 0,
        "missing_manifest_count": 0,
        "malformed_manifest_count": 0,
        "unknown_format_count": 0,
        "missing_referenced_file_count": 0,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "stale_count": 0,
        "fixture_or_test_material_count": 0,
        "validation_status_blocked_count": 0,
        "usable_flag_conflict_count": 0,
    }


def test_staging_validation_accepts_valid_manual_manifest_and_payload(
    tmp_path: Path,
) -> None:
    _write_simple_valid_staged_snapshot(tmp_path, ticker="NVDA")

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["accepted_entries"] == 1
    assert report["counts"]["rejected_entries"] == 0
    assert report["acceptance_summary"]["all_entries_accepted"] is True
    entry = report["entries"][0]
    assert entry["ticker"] == "NVDA"
    assert entry["market"] == "NASDAQ"
    assert entry["source_family"] == "sec_companyfacts"
    assert entry["source_retrieved_at_utc"] == VALIDATED_AT
    assert entry["staging_validation_status"] == "accepted"
    assert entry["accepted_for_cached_source_staging"] is True
    assert entry["usable_for_cached_source_dry_run"] is True
    assert entry["validation_errors"] == ()
    assert entry["validation_warnings"] == ()
    assert entry["issues"] == ()


def test_staging_validation_rejects_missing_manifest(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "AMD" / "snapshot-001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "payload.json").write_text("{}", encoding="utf-8")

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["missing_manifest_count"] == 1
    entry = report["entries"][0]
    assert entry["ticker"] == "AMD"
    assert entry["staging_validation_status"] == "missing_manifest"
    assert entry["accepted_for_cached_source_staging"] is False
    assert entry["issues"] == ("manifest_missing",)


def test_staging_validation_rejects_malformed_manifest_without_crashing(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "batch-001" / "MSFT" / "snapshot-001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["malformed_manifest_count"] == 1
    entry = report["entries"][0]
    assert entry["staging_validation_status"] == "malformed_manifest"
    assert entry["issues"] == ("manifest_json_malformed",)


def test_staging_validation_rejects_unknown_manifest_format(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="AVGO",
        manifest_format_version="unsupported-format",
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["unknown_format_count"] == 1
    entry = report["entries"][0]
    assert entry["staging_validation_status"] == "unknown_format"
    assert entry["accepted_for_cached_source_staging"] is False
    assert "manifest_format_unknown" in entry["issues"]


def test_staging_validation_rejects_missing_referenced_payload(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "META" / "snapshot-001"
    snapshot_dir.mkdir(parents=True)
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="META",
        snapshot_id="snapshot-001",
        local_payload_path="missing-payload.json",
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["missing_referenced_file_count"] == 1
    entry = report["entries"][0]
    assert entry["staging_validation_status"] == "rejected"
    assert "referenced_snapshot_file_missing" in entry["issues"]
    assert "usable_flag_conflicts_with_staging_issues" in entry["issues"]


def test_staging_validation_rejects_hash_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "COST" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="COST",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        local_payload_sha256="f" * 64,
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["hash_mismatch_count"] == 1
    assert "referenced_snapshot_hash_mismatch" in report["entries"][0]["issues"]


def test_staging_validation_rejects_size_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "CRDO" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="CRDO",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=999,
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["size_mismatch_count"] == 1
    assert "referenced_snapshot_size_mismatch" in report["entries"][0]["issues"]


def test_staging_validation_rejects_stale_snapshot(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="VRT",
        staleness_status="stale",
        usable_for_cached_source_dry_run=True,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["stale_count"] == 1
    entry = report["entries"][0]
    assert entry["staging_validation_status"] == "rejected"
    assert "snapshot_stale" in entry["issues"]


def test_staging_validation_rejects_failed_validation_status(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="CLS",
        validation_status="failed",
        usable_for_cached_source_dry_run=True,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["validation_status_blocked_count"] == 1
    assert "validation_status_failed" in report["entries"][0]["issues"]


def test_staging_validation_rejects_not_validated_status(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="IREN",
        validation_status="not_validated",
        usable_for_cached_source_dry_run=True,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["validation_status_blocked_count"] == 1
    assert "validation_status_not_validated" in report["entries"][0]["issues"]


def test_staging_validation_rejects_usable_false(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="TSM",
        usable_for_cached_source_dry_run=False,
        blocked_reason="operator_blocked",
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    entry = report["entries"][0]
    assert entry["accepted_for_cached_source_staging"] is False
    assert "usable_for_cached_source_dry_run_false" in entry["issues"]


def test_staging_validation_rejects_true_usable_with_blocking_issues(
    tmp_path: Path,
) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="ASML",
        validation_status="failed",
        usable_for_cached_source_dry_run=True,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["usable_flag_conflict_count"] == 1
    assert "usable_flag_conflicts_with_staging_issues" in report["entries"][0]["issues"]


def test_staging_validation_rejects_fixture_and_test_material(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(
        tmp_path,
        ticker="NVDA",
        acquisition_mode="test_fixture",
        source_material_type="synthetic_fixture",
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    assert report["counts"]["fixture_or_test_material_count"] == 1
    entry = report["entries"][0]
    assert entry["staging_validation_status"] == "rejected"
    assert "test_fixture_not_real_coverage" in entry["issues"]
    assert "synthetic_fixture_not_real_coverage" in entry["issues"]


def test_staging_validation_rejects_required_field_type_errors(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "batch-001" / "AMD" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="AMD",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run="true",
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
    )

    entry = report["entries"][0]
    assert entry["accepted_for_cached_source_staging"] is False
    assert "usable_for_cached_source_dry_run_invalid" in entry["issues"]


def test_staging_validation_ticker_filter_is_deterministic(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(tmp_path, ticker="NVDA")
    _write_simple_valid_staged_snapshot(tmp_path, ticker="AMD")

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path,
        validated_at=VALIDATED_AT,
        tickers=("amd",),
    )

    assert [entry["ticker"] for entry in report["entries"]] == ["AMD"]
    assert report["ticker_filter"] == ("AMD",)


def test_staging_validator_command_writes_json_report(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(tmp_path, ticker="NVDA")
    output_path = tmp_path / "reports" / "staging-validation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--staging-root",
            str(tmp_path),
            "--validated-at",
            VALIDATED_AT,
            "--output-json",
            str(output_path),
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    emitted = json.loads(stdout.getvalue())
    assert written == emitted
    assert written["counts"]["accepted_entries"] == 1


def test_staging_validator_command_prints_human_output(tmp_path: Path) -> None:
    _write_simple_valid_staged_snapshot(tmp_path, ticker="NVDA")
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--staging-root",
            str(tmp_path),
            "--validated-at",
            VALIDATED_AT,
            "--human",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "CACHED-SOURCE SNAPSHOT STAGING VALIDATION" in output
    assert "accepted=1" in output
    assert "NVDA | NVDA-snapshot | sec_companyfacts | accepted" in output
    assert "No provider, network, broker" in output


def test_staging_validator_path_does_not_import_provider_modules() -> None:
    assert "SecCompanyFactsProvider" not in command.__dict__
    assert "requests" not in command.__dict__
    assert "yfinance" not in command.__dict__
    assert "telegram" not in command.__dict__


def _write_simple_valid_staged_snapshot(
    root: Path,
    *,
    ticker: str,
    manifest_format_version: str = CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
    validation_status: str = "passed",
    staleness_status: str = "fresh",
    usable_for_cached_source_dry_run: object = True,
    blocked_reason: str | None = None,
    acquisition_mode: str = "operator_supplied",
    source_material_type: str = "official_regulatory_source",
) -> None:
    snapshot_dir = root / "batch-001" / ticker / f"{ticker}-snapshot"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"fixture": true}\n', encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker=ticker,
        snapshot_id=f"{ticker}-snapshot",
        local_payload_path="payload.json",
        manifest_format_version=manifest_format_version,
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status=validation_status,
        staleness_status=staleness_status,
        usable_for_cached_source_dry_run=usable_for_cached_source_dry_run,
        blocked_reason=blocked_reason,
        acquisition_mode=acquisition_mode,
        source_material_type=source_material_type,
    )


def _write_manifest(
    path: Path,
    *,
    ticker: str,
    snapshot_id: str,
    local_payload_path: str,
    manifest_format_version: str = CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
    local_payload_sha256: str = "0" * 64,
    local_payload_size_bytes: int = 2,
    validation_status: str,
    staleness_status: str,
    usable_for_cached_source_dry_run: object,
    blocked_reason: str | None,
    acquisition_mode: str = "operator_supplied",
    source_material_type: str = "official_regulatory_source",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest_format_version": manifest_format_version,
                "snapshot_id": snapshot_id,
                "batch_id": "batch-001",
                "created_at_utc": VALIDATED_AT,
                "acquired_at_utc": VALIDATED_AT,
                "acquisition_mode": acquisition_mode,
                "source_family": "sec_companyfacts",
                "source_name": "SEC CompanyFacts",
                "source_url": "https://www.example.invalid/sec/companyfacts.json",
                "source_license_note": "Synthetic test fixture representing a valid manifest shape.",
                "redistribution_allowed": False,
                "local_use_allowed": True,
                "commit_allowed": False,
                "source_material_type": source_material_type,
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
                "local_snapshot_path": local_payload_path,
                "local_manifest_path": "manifest.json",
                "local_payload_sha256": local_payload_sha256,
                "local_payload_size_bytes": local_payload_size_bytes,
                "payload_mime_type": "application/json",
                "payload_encoding": "utf-8",
                "normalization_status": "raw_only",
                "validation_status": validation_status,
                "validation_errors": [],
                "validation_warnings": [],
                "staleness_status": staleness_status,
                "staleness_reason": "Synthetic deterministic test fixture.",
                "usable_for_cached_source_dry_run": usable_for_cached_source_dry_run,
                "blocked_reason": blocked_reason,
                "notes": "Synthetic staging validator test fixture only; no source data was acquired.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
