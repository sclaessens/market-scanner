import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.scanner.scanner_boundary import (
    LEGACY_SCANNER_AUTHORITIES,
    SCANNER_CANONICAL_OWNER,
    build_scanner_plan,
    build_universe_selection_plan,
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


def test_scanner_plan_is_deterministic():
    assert build_scanner_plan() == build_scanner_plan()
    assert build_universe_selection_plan() == build_universe_selection_plan()


def test_scanner_plan_exposes_canonical_owner_and_stage_order():
    plan = build_scanner_plan()

    assert plan.canonical_owner == SCANNER_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "universe_selection",
        "candidate_construction",
    )
    assert plan.legacy_scanner_authorities == LEGACY_SCANNER_AUTHORITIES
    assert plan.migration_status == "canonical_scanner_boundary_established"


def test_scanner_plan_forbids_side_effects_by_default():
    for stage in build_scanner_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.data_writes_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.reports_allowed is False
        assert stage.telegram_delivery_allowed is False


def test_scanner_plan_describes_input_and_universe_sources():
    universe_stage, candidate_stage = build_scanner_plan().stages

    assert universe_stage.input_source_category == "configured_universe_reference"
    assert universe_stage.candidate_universe_source == (
        "canonical_scanner_boundary_plan"
    )
    assert candidate_stage.input_source_category == "scanner_candidate_inputs"
    assert candidate_stage.candidate_universe_source == (
        "canonical_scanner_boundary_plan"
    )


def test_scanner_boundary_import_and_plan_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.scanner.scanner_boundary")
    build_scanner_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_scanner_boundary_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.scanner.scanner_boundary")
    build_scanner_plan()

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before


def test_scanner_plan_contains_no_investment_behavior():
    output_text = " ".join(_flatten_values(build_scanner_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_runners_are_not_expanded_to_import_canonical_scanner():
    legacy_sources = (
        Path("scripts/run_scan.py").read_text(encoding="utf-8"),
        Path("scripts/run_full_pipeline.py").read_text(encoding="utf-8"),
    )

    for source in legacy_sources:
        assert "market_scanner.scanner" not in source
        assert "market_scanner.app" not in source
