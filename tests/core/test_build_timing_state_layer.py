from __future__ import annotations

from pathlib import Path

LEGACY_TIMING_LAYER_MODULE_PATH = Path("scripts/core/build_timing_state_layer.py")

UPSTREAM_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "profitability_profile",
    "balance_sheet_profile",
    "earnings_quality_profile",
    "capital_efficiency_profile",
    "cashflow_profile",
    "stability_profile",
    "quality_metadata_status",
    "source_data_status",
    "source_timestamp",
    "generated_at",
]

TIMING_COLUMNS = [
    "timing_state",
    "timing_reason",
    "breakout_state",
    "pullback_state",
    "compression_state",
    "extension_state",
    "participation_state",
    "timing_environment",
    "timing_pattern_state",
    "trend_participation_state",
    "timing_structure_state",
    "timing_metadata_status",
    "timing_source_data_status",
    "timing_source_timestamp",
    "timing_generated_at",
]

EXPECTED_OUTPUT_COLUMNS = UPSTREAM_COLUMNS + TIMING_COLUMNS

EXPECTED_TIMING_STATES = {
    "UNCLASSIFIED",
}

EXPECTED_METADATA_STATUSES = {
    "SOURCE_MISSING",
    "SOURCE_AVAILABLE",
}

FORBIDDEN_TIMING_FIELDS = {
    "tradeable",
    "approved",
    "rejected",
    "high_conviction",
    "conviction",
    "conviction_score",
    "priority",
    "rank",
    "ranking",
    "score",
    "scoring",
    "actionable",
    "buy_candidate",
    "sell_candidate",
    "execution_ready",
    "readiness",
    "best_opportunity",
    "allocation",
    "allocation_weight",
    "expected_return",
    "alpha_score",
    "opportunity_rank",
    "preferred_setup",
    "timing_grade",
    "timing_signal",
    "final_action",
    "final_score",
}


def test_timing_layer_script_remains_legacy_reference_only():
    assert LEGACY_TIMING_LAYER_MODULE_PATH == Path("scripts/core/build_timing_state_layer.py")


def test_timing_contract_preserves_upstream_and_appends_timing_metadata():
    assert EXPECTED_OUTPUT_COLUMNS[: len(UPSTREAM_COLUMNS)] == UPSTREAM_COLUMNS
    assert EXPECTED_OUTPUT_COLUMNS[len(UPSTREAM_COLUMNS) :] == TIMING_COLUMNS
    assert set(EXPECTED_OUTPUT_COLUMNS).isdisjoint(FORBIDDEN_TIMING_FIELDS)


def test_timing_contract_is_non_filtering_metadata_enrichment():
    assert "ticker" in UPSTREAM_COLUMNS
    assert "date" in UPSTREAM_COLUMNS
    assert "timing_metadata_status" in TIMING_COLUMNS
    assert "timing_source_data_status" in TIMING_COLUMNS
    assert "timing_generated_at" in TIMING_COLUMNS


def test_timing_contract_keeps_descriptive_states_only():
    assert EXPECTED_TIMING_STATES == {"UNCLASSIFIED"}
    assert {"SOURCE_MISSING", "SOURCE_AVAILABLE"}.issubset(EXPECTED_METADATA_STATUSES)


def test_timing_contract_has_no_trade_or_decision_authority():
    serialized = " ".join(
        EXPECTED_OUTPUT_COLUMNS + sorted(EXPECTED_TIMING_STATES) + sorted(EXPECTED_METADATA_STATUSES)
    )
    for field in FORBIDDEN_TIMING_FIELDS:
        assert field not in EXPECTED_OUTPUT_COLUMNS
        assert field not in serialized


def test_active_code_no_longer_imports_timing_layer_script():
    for path in [Path("src"), Path("tests"), Path(".github")]:
        if not path.exists():
            continue
        for source_path in path.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            if source_path == Path("tests/core/test_build_timing_state_layer.py"):
                continue
            source = source_path.read_text(encoding="utf-8")
            assert "from scripts.core import build_timing_state_layer" not in source
            assert "import scripts.core.build_timing_state_layer" not in source
