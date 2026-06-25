from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from market_engine.source_refresh import (
    cached_source_snapshot_acquisition_dry_run_command as command,
)
from market_engine.source_refresh.cached_source_snapshot_acquisition_dry_run import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_DRY_RUN_FORMAT_VERSION,
    REQUIRED_ACQUISITION_MANIFEST_FIELDS,
    build_cached_source_snapshot_acquisition_dry_run,
)
from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)


DRY_RUN_AT = "2026-06-25T12:00:00Z"


def test_acquisition_dry_run_plans_one_valid_ticker_and_source_family(
    tmp_path: Path,
) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("NVDA",),
        source_families=("sec_companyfacts",),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
        batch_id="batch-001",
    )

    assert report["report_format_version"] == (
        CACHED_SOURCE_SNAPSHOT_ACQUISITION_DRY_RUN_FORMAT_VERSION
    )
    assert report["dry_run_at"] == DRY_RUN_AT
    assert report["counts"]["planned_entries"] == 1
    assert report["counts"]["blocked_entries"] == 0
    entry = report["entries"][0]
    assert entry["ticker"] == "NVDA"
    assert entry["source_family"] == "sec_companyfacts"
    assert entry["acquisition_mode"] == "dry_run_only"
    assert entry["acquisition_dry_run_status"] == "planned"
    assert entry["would_acquire"] is True
    assert entry["would_acquire_external_data"] is False
    assert entry["would_write_payload"] is False
    assert entry["would_write_manifest"] is False
    assert entry["expected_manifest_format_version"] == (
        CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION
    )
    assert entry["required_manifest_fields"] == REQUIRED_ACQUISITION_MANIFEST_FIELDS
    assert "local_payload_sha256" in entry["required_payload_metadata_fields"]
    assert "cached_source_snapshot_staging_validator_command" in entry["validation_handoff"]
    assert entry["issues"] == ()


def test_acquisition_dry_run_sorts_multiple_tickers_and_source_families(
    tmp_path: Path,
) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("nvda,AMD",),
        source_families=("unknown_family,sec_companyfacts",),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
        batch_id="batch-001",
    )

    assert [
        (entry["ticker"], entry["source_family"])
        for entry in report["entries"]
    ] == [
        ("AMD", "sec_companyfacts"),
        ("AMD", "unknown_family"),
        ("NVDA", "sec_companyfacts"),
        ("NVDA", "unknown_family"),
    ]
    assert report["counts"]["total_requested_entries"] == 4
    assert report["counts"]["planned_entries"] == 2
    assert report["counts"]["blocked_entries"] == 2
    assert report["counts"]["unsupported_source_family_count"] == 1


def test_acquisition_dry_run_rejects_invalid_ticker(tmp_path: Path) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("BAD TICKER",),
        source_families=("sec_companyfacts",),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
    )

    assert report["counts"]["invalid_ticker_count"] == 1
    entry = report["entries"][0]
    assert entry["acquisition_dry_run_status"] == "blocked"
    assert entry["issues"] == ("ticker_invalid",)


def test_acquisition_dry_run_no_ticker_request_is_no_op_report(
    tmp_path: Path,
) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=(),
        source_families=("sec_companyfacts",),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
    )

    assert report["entries"] == []
    assert report["counts"]["missing_ticker_count"] == 1
    assert report["counts"]["planned_entries"] == 0
    assert report["counts"]["blocked_entries"] == 0


def test_acquisition_dry_run_no_source_family_request_is_no_op_report(
    tmp_path: Path,
) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("NVDA",),
        source_families=(),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
    )

    assert report["entries"] == []
    assert report["counts"]["missing_source_family_count"] == 1
    assert report["counts"]["planned_entries"] == 0
    assert report["counts"]["blocked_entries"] == 0


def test_acquisition_dry_run_rejects_unsupported_source_family(
    tmp_path: Path,
) -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("NVDA",),
        source_families=("broker_feed",),
        output_root=tmp_path,
        dry_run_at=DRY_RUN_AT,
    )

    assert report["counts"]["unsupported_source_family_count"] == 1
    entry = report["entries"][0]
    assert entry["acquisition_dry_run_status"] == "blocked"
    assert entry["issues"] == ("source_family_unsupported",)


def test_acquisition_dry_run_missing_output_root_blocks_entry() -> None:
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=("NVDA",),
        source_families=("sec_companyfacts",),
        output_root=None,
        dry_run_at=DRY_RUN_AT,
    )

    entry = report["entries"][0]
    assert report["output_root"] is None
    assert entry["acquisition_dry_run_status"] == "blocked"
    assert entry["proposed_staging_path"] is None
    assert entry["proposed_manifest_path"] is None
    assert entry["issues"] == ("output_root_missing",)


def test_acquisition_dry_run_command_writes_only_json_report(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "staging"
    output_path = tmp_path / "reports" / "acquisition-dry-run.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--ticker",
            "NVDA",
            "--source-family",
            "sec_companyfacts",
            "--output-root",
            str(output_root),
            "--dry-run-at",
            DRY_RUN_AT,
            "--batch-id",
            "batch-001",
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
    assert written["counts"]["planned_entries"] == 1
    assert output_path.exists()
    assert not output_root.exists()


def test_acquisition_dry_run_command_prints_human_output(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        (
            "--ticker",
            "NVDA",
            "--source-family",
            "sec_companyfacts",
            "--output-root",
            str(tmp_path),
            "--dry-run-at",
            DRY_RUN_AT,
            "--human",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "CACHED-SOURCE SNAPSHOT ACQUISITION DRY-RUN" in output
    assert "planned=1" in output
    assert "NVDA | sec_companyfacts | planned" in output
    assert "No provider, network" in output


def test_acquisition_dry_run_path_does_not_import_provider_modules() -> None:
    assert "SecCompanyFactsProvider" not in command.__dict__
    assert "requests" not in command.__dict__
    assert "yfinance" not in command.__dict__
    assert "telegram" not in command.__dict__
