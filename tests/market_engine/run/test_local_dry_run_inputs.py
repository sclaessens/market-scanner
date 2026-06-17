from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.run.end_to_end_dry_run_command import build_synthetic_dry_run_stage_payloads
from market_engine.run.local_dry_run_inputs import (
    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION,
    LocalDryRunInputError,
    load_market_engine_local_dry_run_input,
)


def test_loads_approved_local_snapshot_fixture(tmp_path: Path) -> None:
    stage_payloads = build_synthetic_dry_run_stage_payloads()
    stage_payloads["portfolio_review"] = {
        **stage_payloads["portfolio_review"],
        "portfolio_context_reference": {
            **stage_payloads["portfolio_review"]["portfolio_context_reference"],
            "current_quantity": 0,
            "current_market_value": 0.0,
        },
    }
    fixture_path = tmp_path / "local_snapshot_fixture.json"
    fixture_path.write_text(
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

    loaded = load_market_engine_local_dry_run_input(
        fixture_path,
        input_mode="local_snapshot_fixture",
    )

    assert loaded["source_context"]["ticker"] == "NVDA"
    assert loaded["portfolio_review"]["portfolio_context_reference"]["current_quantity"] == 0
    assert loaded["portfolio_review"]["portfolio_context_reference"]["current_market_value"] == 0.0


def test_rejects_local_snapshot_fixture_without_fixture_contract(tmp_path: Path) -> None:
    fixture_path = tmp_path / "raw_stage_payloads.json"
    fixture_path.write_text(json.dumps(build_synthetic_dry_run_stage_payloads()), encoding="utf-8")

    with pytest.raises(LocalDryRunInputError, match="local-dry-run-input-fixture-v1"):
        load_market_engine_local_dry_run_input(
            fixture_path,
            input_mode="local_snapshot_fixture",
        )


def test_rejects_local_snapshot_fixture_without_non_production_marker(tmp_path: Path) -> None:
    fixture_path = tmp_path / "missing_non_production_marker.json"
    fixture_path.write_text(
        json.dumps(
            {
                "dry_run_input_fixture_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
                ),
                "stage_payloads": build_synthetic_dry_run_stage_payloads(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalDryRunInputError, match="non_production_fixture=true"):
        load_market_engine_local_dry_run_input(
            fixture_path,
            input_mode="local_snapshot_fixture",
        )


def test_rejects_local_snapshot_fixture_without_stage_payloads_object(tmp_path: Path) -> None:
    fixture_path = tmp_path / "invalid_stage_payloads.json"
    fixture_path.write_text(
        json.dumps(
            {
                "dry_run_input_fixture_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
                ),
                "non_production_fixture": True,
                "stage_payloads": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalDryRunInputError, match="stage_payloads object"):
        load_market_engine_local_dry_run_input(
            fixture_path,
            input_mode="local_snapshot_fixture",
        )


def test_explicit_in_memory_payload_still_accepts_raw_stage_payload_mapping(tmp_path: Path) -> None:
    payload_path = tmp_path / "stage_payloads.json"
    payload_path.write_text(json.dumps(build_synthetic_dry_run_stage_payloads()), encoding="utf-8")

    loaded = load_market_engine_local_dry_run_input(
        payload_path,
        input_mode="explicit_in_memory_payload",
    )

    assert loaded["delivery_reporting"]["report_format_version"] == "market-engine-delivery-report-v1"


def test_explicit_in_memory_payload_can_accept_fixture_wrapper(tmp_path: Path) -> None:
    payload_path = tmp_path / "wrapped_stage_payloads.json"
    payload_path.write_text(
        json.dumps(
            {
                "dry_run_input_fixture_format_version": (
                    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
                ),
                "non_production_fixture": True,
                "stage_payloads": build_synthetic_dry_run_stage_payloads(),
            }
        ),
        encoding="utf-8",
    )

    loaded = load_market_engine_local_dry_run_input(
        payload_path,
        input_mode="explicit_in_memory_payload",
    )

    assert loaded["decision_engine_handoff"]["handoff_run_id"] == "handoff-run-001"


def test_rejects_non_object_json_top_level(tmp_path: Path) -> None:
    payload_path = tmp_path / "list_payload.json"
    payload_path.write_text("[]", encoding="utf-8")

    with pytest.raises(LocalDryRunInputError, match="object at the top level"):
        load_market_engine_local_dry_run_input(
            payload_path,
            input_mode="explicit_in_memory_payload",
        )


def test_local_dry_run_input_loader_does_not_import_side_effect_dependencies() -> None:
    module_source = Path("src/market_engine/run/local_dry_run_inputs.py").read_text(
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
