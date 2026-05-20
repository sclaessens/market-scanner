from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_sources import common


def test_json_provider_export_reads_row_payload(tmp_path: Path):
    input_path = tmp_path / "provider.json"
    input_path.write_text(json.dumps({"rows": [{"ticker": " aaa "}]}))

    df = common.read_provider_export(input_path)

    assert df.loc[0, "ticker"] == " aaa "


def test_reject_blank_tickers_is_deterministic():
    df = pd.DataFrame([{"ticker": " "}])

    with pytest.raises(ValueError, match="blank ticker"):
        common.reject_blank_tickers(df, "provider export")


def test_atomic_write_does_not_replace_existing_artifact_without_overwrite(tmp_path: Path):
    output_path = tmp_path / "artifact.csv"
    output_path.write_text("ticker\nAAA\n")

    with pytest.raises(FileExistsError, match="allow-overwrite"):
        common.ensure_output_path(output_path, allow_overwrite=False)

    assert output_path.read_text() == "ticker\nAAA\n"


def test_governed_output_path_rejects_ungoverned_targets(tmp_path: Path):
    with pytest.raises(ValueError, match="governed source artifact"):
        common.require_governed_output_path(tmp_path / "reports" / "fundamentals.csv", ("data", "raw", "fundamentals.csv"))


def test_audit_output_is_credential_safe(capsys):
    audit = common.PrefillAudit(
        run_timestamp="2026-05-20 12:00:00",
        provider_source_label="local_export",
        requested_ticker_count=1,
        matched_ticker_count=1,
        missing_ticker_count=0,
        written_row_count=0,
        stale_row_count=0,
        invalid_row_count=0,
        partial_row_count=0,
        duplicate_detection_result="PASSED",
        artifact_write_path="data/raw/fundamentals.csv",
        validation_status="VALIDATED",
        failure_reason="",
        refresh_mode="provider_assisted_prefill",
        source_artifact_target="data/raw/fundamentals.csv",
        dry_run=True,
    )

    common.print_audit(audit)

    output = capsys.readouterr().out.lower()
    assert "secret" not in output
    assert "credential" in output
