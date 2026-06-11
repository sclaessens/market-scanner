from __future__ import annotations

from pathlib import Path

LEGACY_CONTEXT_LAYER_MODULE_PATH = Path("scripts/core/build_context_layer.py")

EXPECTED_CONTEXT_COLUMNS = [
    "ticker",
    "date",
    "rs_score",
    "rs_percentile",
    "rs_rank",
    "rs_vs_market",
    "rs_vs_sector",
    "context_strength",
    "context_reason",
    "leadership_state",
]

EXPECTED_CONTEXT_STATES = {
    "LEADING",
    "STRONG",
    "NEUTRAL",
    "WEAK",
    "UNKNOWN",
}

EXPECTED_CONTEXT_REASONS = {
    "top_decile_leadership",
    "upper_quartile_leadership",
    "middle_distribution",
    "lower_distribution",
    "missing_percentile",
}

FORBIDDEN_CONTEXT_FIELDS = {
    "context_tradeable",
    "tradeability",
    "conviction",
    "allocation_priority",
    "final_action",
    "urgency",
    "actionable",
    "BUY",
    "SELL",
    "HOLD",
    "TRIM",
    "REMOVE",
}


def test_context_layer_script_remains_legacy_reference_only():
    assert LEGACY_CONTEXT_LAYER_MODULE_PATH == Path("scripts/core/build_context_layer.py")


def test_context_contract_schema_is_classification_only():
    assert EXPECTED_CONTEXT_COLUMNS == [
        "ticker",
        "date",
        "rs_score",
        "rs_percentile",
        "rs_rank",
        "rs_vs_market",
        "rs_vs_sector",
        "context_strength",
        "context_reason",
        "leadership_state",
    ]
    assert set(EXPECTED_CONTEXT_COLUMNS).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_context_contract_keeps_expected_states_and_reasons():
    assert EXPECTED_CONTEXT_STATES == {
        "LEADING",
        "STRONG",
        "NEUTRAL",
        "WEAK",
        "UNKNOWN",
    }
    assert "missing_percentile" in EXPECTED_CONTEXT_REASONS
    assert "top_decile_leadership" in EXPECTED_CONTEXT_REASONS


def test_context_contract_has_no_trade_or_decision_authority():
    serialized = " ".join(EXPECTED_CONTEXT_COLUMNS + sorted(EXPECTED_CONTEXT_STATES) + sorted(EXPECTED_CONTEXT_REASONS))
    for field in FORBIDDEN_CONTEXT_FIELDS:
        assert field not in EXPECTED_CONTEXT_COLUMNS
        assert field not in serialized


def test_active_code_no_longer_imports_context_layer_script():
    for path in [Path("src"), Path("tests"), Path(".github")]:
        if not path.exists():
            continue
        for source_path in path.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            if source_path == Path("tests/core/test_build_context_layer.py"):
                continue
            source = source_path.read_text(encoding="utf-8")
            assert "from scripts.core import build_context_layer" not in source
            assert "import scripts.core.build_context_layer" not in source
