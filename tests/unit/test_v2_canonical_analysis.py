import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.analysis.analysis_boundary import (
    ANALYSIS_CANONICAL_OWNER,
    LEGACY_ANALYSIS_AUTHORITIES,
    build_analysis_input_policy,
    build_analysis_plan,
    build_fundamental_analysis_plan,
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


def test_analysis_plan_is_deterministic():
    assert build_analysis_plan() == build_analysis_plan()
    assert build_fundamental_analysis_plan() == build_fundamental_analysis_plan()
    assert build_analysis_input_policy() == build_analysis_input_policy()


def test_analysis_plan_exposes_canonical_owner_and_stage_order():
    plan = build_analysis_plan()

    assert plan.canonical_owner == ANALYSIS_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "fundamental_evidence_review",
        "profile_evidence_review",
        "limitation_review",
    )
    assert plan.legacy_analysis_authorities == LEGACY_ANALYSIS_AUTHORITIES
    assert plan.migration_status == "canonical_analysis_boundary_established"


def test_analysis_plan_forbids_side_effects_and_final_outputs_by_default():
    for stage in build_analysis_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.data_writes_allowed is False
        assert stage.reports_allowed is False
        assert stage.telegram_delivery_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False
        assert stage.execution_quality_outputs_allowed is False


def test_analysis_plan_describes_governed_evidence_inputs():
    plan = build_analysis_plan()
    policy = plan.input_policy

    assert policy.governed_evidence_required is True
    assert policy.accepts_source_derived_free_cash_flow is True
    assert policy.accepts_growth_available_evidence is True
    assert policy.preserves_review_limitations is True
    assert policy.missing_values_must_remain_explicit is True
    assert policy.final_outcomes_allowed is False
    assert tuple(stage.input_evidence_category for stage in plan.stages) == (
        "scanner_candidates_and_governed_fundamentals",
        "cash_flow_growth_quality_evidence",
        "review_limitation_evidence",
    )


def test_analysis_boundary_import_and_plan_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.analysis.analysis_boundary")
    build_analysis_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_analysis_boundary_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.analysis.analysis_boundary")
    build_analysis_plan()

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before


def test_analysis_plan_contains_no_investment_behavior():
    output_text = " ".join(_flatten_values(build_analysis_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_runners_and_analysis_files_are_not_expanded_to_import_canonical_analysis():
    legacy_sources = (
        Path("scripts/run_scan.py").read_text(encoding="utf-8"),
        Path("scripts/run_full_pipeline.py").read_text(encoding="utf-8"),
        Path("scripts/fundamentals/build_analysis.py").read_text(encoding="utf-8"),
        Path("scripts/fundamentals/build_metrics.py").read_text(encoding="utf-8"),
        Path("scripts/fundamentals/build_quality.py").read_text(encoding="utf-8"),
    )

    for source in legacy_sources:
        assert "market_scanner.analysis" not in source
        assert "market_scanner.app" not in source

