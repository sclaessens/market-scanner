from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

from market_engine.source_refresh import cached_source_snapshot_inventory_command as command
from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
    CACHED_SOURCE_SNAPSHOT_INVENTORY_FORMAT_VERSION,
    build_cached_source_snapshot_inventory,
)


INSPECTED_AT = "2026-06-25T10:00:00Z"


def test_inventory_empty_input_root_is_deterministic(tmp_path: Path) -> None:
    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["report_format_version"] == CACHED_SOURCE_SNAPSHOT_INVENTORY_FORMAT_VERSION
    assert report["inspected_at"] == INSPECTED_AT
    assert report["entries"] == []
    assert report["counts"] == {
        "total_inspected_entries": 0,
        "usable_entries": 0,
        "unusable_entries": 0,
        "missing_manifest_count": 0,
        "malformed_manifest_count": 0,
        "unknown_format_count": 0,
        "missing_referenced_file_count": 0,
        "stale_count": 0,
    }


def test_inventory_valid_manifest_and_snapshot_fixture_is_usable(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "NVDA" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"ok": true}\n', encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="NVDA",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["total_inspected_entries"] == 1
    assert report["counts"]["usable_entries"] == 1
    assert report["counts"]["unusable_entries"] == 0
    entry = report["entries"][0]
    assert entry["ticker"] == "NVDA"
    assert entry["source_family"] == "sec_companyfacts"
    assert entry["inventory_status"] == "usable"
    assert entry["usable_for_cached_source_dry_run"] is True
    assert entry["issues"] == ()


def test_inventory_reports_missing_manifest_fail_closed(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "AMD" / "snapshot-001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "payload.json").write_text("{}", encoding="utf-8")

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["missing_manifest_count"] == 1
    entry = report["entries"][0]
    assert entry["ticker"] == "AMD"
    assert entry["inventory_status"] == "missing_manifest"
    assert entry["usable_for_cached_source_dry_run"] is False
    assert entry["issues"] == ("manifest_missing",)


def test_inventory_reports_malformed_manifest_without_crashing(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "MSFT" / "snapshot-001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["malformed_manifest_count"] == 1
    entry = report["entries"][0]
    assert entry["inventory_status"] == "malformed_manifest"
    assert entry["issues"] == ("manifest_json_malformed",)


def test_inventory_reports_missing_referenced_snapshot_file(tmp_path: Path) -> None:
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

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["missing_referenced_file_count"] == 1
    entry = report["entries"][0]
    assert entry["inventory_status"] == "unusable"
    assert "referenced_snapshot_file_missing" in entry["issues"]
    assert "usable_flag_conflicts_with_inventory_issues" in entry["issues"]


def test_inventory_reports_unknown_manifest_format(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "AVGO" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="AVGO",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        manifest_format_version="unsupported-format",
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
    )

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["unknown_format_count"] == 1
    entry = report["entries"][0]
    assert entry["inventory_status"] == "unknown_format"
    assert "manifest_format_unknown" in entry["issues"]


def test_inventory_counts_stale_manifest(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "COST" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="COST",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        validation_status="warning",
        staleness_status="stale",
        usable_for_cached_source_dry_run=False,
        blocked_reason="stale_source",
    )

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    assert report["counts"]["stale_count"] == 1
    assert report["entries"][0]["inventory_status"] == "unusable"
    assert "snapshot_stale" in report["entries"][0]["issues"]


def test_inventory_does_not_treat_test_fixture_as_real_coverage(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "batch-001" / "TSM" / "snapshot-001"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker="TSM",
        snapshot_id="snapshot-001",
        local_payload_path="payload.json",
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
        acquisition_mode="test_fixture",
        source_material_type="synthetic_fixture",
    )

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
    )

    entry = report["entries"][0]
    assert entry["inventory_status"] == "unusable"
    assert entry["usable_for_cached_source_dry_run"] is False
    assert "test_fixture_not_real_coverage" in entry["issues"]
    assert "synthetic_fixture_not_real_coverage" in entry["issues"]


def test_inventory_ticker_filter_is_deterministic(tmp_path: Path) -> None:
    _write_simple_valid_snapshot(tmp_path, ticker="NVDA")
    _write_simple_valid_snapshot(tmp_path, ticker="AMD")

    report = build_cached_source_snapshot_inventory(
        input_root=tmp_path,
        inspected_at=INSPECTED_AT,
        tickers=("amd",),
    )

    assert [entry["ticker"] for entry in report["entries"]] == ["AMD"]
    assert report["ticker_filter"] == ("AMD",)


def test_command_writes_json_report_when_output_path_is_provided(tmp_path: Path) -> None:
    _write_simple_valid_snapshot(tmp_path, ticker="NVDA")
    output_path = tmp_path / "reports" / "inventory.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--input-root",
            str(tmp_path),
            "--inspected-at",
            INSPECTED_AT,
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
    assert written["counts"]["usable_entries"] == 1


def test_command_prints_human_readable_output(tmp_path: Path) -> None:
    _write_simple_valid_snapshot(tmp_path, ticker="NVDA")
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--input-root",
            str(tmp_path),
            "--inspected-at",
            INSPECTED_AT,
            "--human",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "CACHED-SOURCE SNAPSHOT INVENTORY" in output
    assert "usable=1" in output
    assert "NVDA | NVDA-snapshot | sec_companyfacts | usable" in output
    assert "No provider, network, broker" in output


def test_inventory_path_does_not_import_provider_modules() -> None:
    assert "SecCompanyFactsProvider" not in command.__dict__
    assert "requests" not in command.__dict__
    assert "yfinance" not in command.__dict__


def _write_simple_valid_snapshot(root: Path, *, ticker: str) -> None:
    snapshot_dir = root / "batch-001" / ticker / f"{ticker}-snapshot"
    payload_path = snapshot_dir / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"fixture": true}\n', encoding="utf-8")
    _write_manifest(
        snapshot_dir / "manifest.json",
        ticker=ticker,
        snapshot_id=f"{ticker}-snapshot",
        local_payload_path="payload.json",
        local_payload_sha256=_sha256(payload_path),
        local_payload_size_bytes=payload_path.stat().st_size,
        validation_status="passed",
        staleness_status="fresh",
        usable_for_cached_source_dry_run=True,
        blocked_reason=None,
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
    usable_for_cached_source_dry_run: bool,
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
                "created_at_utc": INSPECTED_AT,
                "acquired_at_utc": INSPECTED_AT,
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
                "source_retrieved_at_utc": INSPECTED_AT,
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
                "notes": "Synthetic inventory test fixture only; no source data was acquired.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
