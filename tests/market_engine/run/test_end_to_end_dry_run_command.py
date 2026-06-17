from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from market_engine.run.end_to_end_dry_run import (
    MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
)
from market_engine.run.end_to_end_dry_run_command import (
    build_synthetic_dry_run_stage_payloads,
    run_market_engine_end_to_end_dry_run_command,
)
from market_engine.run.local_dry_run_inputs import (
    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION,
)


def test_local_dry_run_command_emits_synthetic_payload(capsys) -> None:
    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--dry-run-id",
            "local-run-001",
            "--generated-at",
            "2026-06-17T13:30:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["dry_run_format_version"] == (
        MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION
    )
    assert payload["dry_run_id"] == "local-run-001"
    assert payload["generated_at"] == "2026-06-17T13:30:00Z"
    assert payload["input_mode"] == "synthetic_contract_fixture"
    assert payload["run_state"] == "dry_run_completed"
    assert payload["ticker"] == "NVDA"
    assert payload["stage_results"][-1]["stage_name"] == "dry_run_summary"
    assert "No provider" in payload["forbidden_side_effect_confirmation"]
    assert captured.err == ""


def test_local_dry_run_command_accepts_explicit_json_payload_file(tmp_path: Path, capsys) -> None:
    stage_payloads: dict[str, dict[str, Any]] = build_synthetic_dry_run_stage_payloads()
    stage_payloads["analysis_review"] = {
        **stage_payloads["analysis_review"],
        "missing_data_markers": ["analysis_review.free_cash_flow_component"],
    }
    payload_path = tmp_path / "stage_payloads.json"
    payload_path.write_text(json.dumps(stage_payloads), encoding="utf-8")

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "explicit_in_memory_payload",
            "--stage-payloads-json",
            str(payload_path),
            "--dry-run-id",
            "explicit-run-001",
            "--generated-at",
            "2026-06-17T13:31:00Z",
            "--compact",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["input_mode"] == "explicit_in_memory_payload"
    assert payload["run_state"] == "dry_run_completed_with_limitations"
    assert payload["missing_data_summary"] == [
        "analysis_review.free_cash_flow_component"
    ]
    assert "\n" not in captured.out.rstrip("\n")
    assert captured.err == ""


def test_local_dry_run_command_accepts_local_snapshot_fixture_file(
    tmp_path: Path,
    capsys,
) -> None:
    stage_payloads: dict[str, dict[str, Any]] = build_synthetic_dry_run_stage_payloads()
    stage_payloads["source_context"] = {
        **stage_payloads["source_context"],
        "ticker": "MSFT",
        "cik": "0000789019",
    }
    payload_path = tmp_path / "local_snapshot_fixture.json"
    payload_path.write_text(
        json.dumps(
            {
                "dry_run_input_fixture_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
                ),
                "fixture_id": "local-snapshot-fixture-001",
                "input_mode": "local_snapshot_fixture",
                "non_production_fixture": True,
                "stage_payloads": stage_payloads,
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(payload_path),
            "--dry-run-id",
            "local-snapshot-run-001",
            "--generated-at",
            "2026-06-17T13:32:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["input_mode"] == "local_snapshot_fixture"
    assert payload["dry_run_id"] == "local-snapshot-run-001"
    assert payload["ticker"] == "MSFT"
    assert payload["cik"] == "0000789019"
    assert payload["run_state"] == "dry_run_completed"
    assert captured.err == ""


def test_local_dry_run_command_requires_json_file_for_non_synthetic_mode(capsys) -> None:
    exit_code = run_market_engine_end_to_end_dry_run_command(
        ["--input-mode", "local_snapshot_fixture"]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--stage-payloads-json is required" in captured.err
    assert captured.out == ""


def test_local_dry_run_command_rejects_malformed_json_file(tmp_path: Path, capsys) -> None:
    payload_path = tmp_path / "stage_payloads.json"
    payload_path.write_text("not-json", encoding="utf-8")

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "explicit_in_memory_payload",
            "--stage-payloads-json",
            str(payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Stage payload JSON is invalid" in captured.err
    assert captured.out == ""


def test_local_dry_run_command_rejects_raw_payload_for_local_snapshot_mode(
    tmp_path: Path,
    capsys,
) -> None:
    payload_path = tmp_path / "raw_stage_payloads.json"
    payload_path.write_text(json.dumps(build_synthetic_dry_run_stage_payloads()), encoding="utf-8")

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "local-dry-run-input-fixture-v1" in captured.err
    assert captured.out == ""


def test_local_dry_run_command_module_does_not_import_side_effect_dependencies() -> None:
    module_source = Path("src/market_engine/run/end_to_end_dry_run_command.py").read_text(
        encoding="utf-8"
    )

    forbidden_terms = (
        "from scripts",
        "import scripts",
        "from market_scanner",
        "import market_scanner",
        "telegram",
        "smtplib",
        "yfinance",
        "urllib",
        "requests",
        "socket",
        "subprocess",
    )

    assert not any(term in module_source for term in forbidden_terms)
