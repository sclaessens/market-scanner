import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

from market_scanner.app import (
    CANONICAL_ENTRYPOINT,
    LEGACY_RUNTIME_AUTHORITIES,
    build_canonical_runtime_plan,
    run_canonical_app,
)


APPROVED_STAGE_ORDER = (
    "application_entrypoint",
    "scanner_universe_selection",
    "provider_source_access",
    "fundamentals_normalization_evidence",
    "analysis",
    "decision_review_boundary",
    "message_composition",
    "report_generation_where_approved",
    "delivery_telegram_where_approved",
)

FORBIDDEN_OUTPUT_TERMS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "urgency",
    "scoring",
    "target-price",
    "tradeability",
    "recommendation",
}


def _flatten_values(value):
    if is_dataclass(value):
        yield from _flatten_values(asdict(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield str(value)


def test_canonical_runtime_plan_exposes_approved_stage_order():
    plan = build_canonical_runtime_plan()

    assert plan.entrypoint == CANONICAL_ENTRYPOINT
    assert tuple(stage.name for stage in plan.stages) == APPROVED_STAGE_ORDER
    assert plan.legacy_runtime_authorities == LEGACY_RUNTIME_AUTHORITIES


def test_canonical_runtime_plan_marks_all_stages_side_effect_free_by_default():
    plan = build_canonical_runtime_plan()

    assert all(stage.side_effects_allowed is False for stage in plan.stages)
    assert {
        stage.name: stage.status for stage in plan.stages
    } == {
        "application_entrypoint": "canonical_boundary_established",
        "scanner_universe_selection": "planned_for_migration",
        "provider_source_access": "canonical_boundary_available",
        "fundamentals_normalization_evidence": "canonical_boundary_available",
        "analysis": "planned_for_migration",
        "decision_review_boundary": "planned_for_migration",
        "message_composition": "planned_for_migration",
        "report_generation_where_approved": "approval_required",
        "delivery_telegram_where_approved": "approval_required",
    }


def test_canonical_app_dry_run_returns_side_effect_guarantees():
    result = run_canonical_app()

    assert result.mode == "dry_run"
    assert result.runtime_plan == build_canonical_runtime_plan()
    assert result.side_effect_guarantees.provider_calls_made is False
    assert result.side_effect_guarantees.production_data_writes is False
    assert result.side_effect_guarantees.reports_generated is False
    assert result.side_effect_guarantees.telegram_artifacts_created is False
    assert result.side_effect_guarantees.portfolio_or_watchlist_updates is False
    assert result.side_effect_guarantees.legacy_runners_invoked is False


def test_canonical_app_rejects_non_dry_run_execution():
    with pytest.raises(ValueError, match="Only dry-run canonical app planning"):
        run_canonical_app(dry_run=False)


def test_canonical_app_import_and_dry_run_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.app")
    run_canonical_app()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_canonical_app_boundary_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.app")
    run_canonical_app()

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before


def test_canonical_app_output_contains_no_investment_behavior():
    output_text = " ".join(_flatten_values(run_canonical_app()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_canonical_app_is_deterministic():
    assert build_canonical_runtime_plan() == build_canonical_runtime_plan()
    assert run_canonical_app() == run_canonical_app()
