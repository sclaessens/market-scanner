from __future__ import annotations

from pathlib import Path

LEGACY_VALIDATION_LAYER_MODULE_PATH = Path("scripts/core/build_validation_layer.py")

REQUIRED_SCANNER_COLUMNS = [
    "ticker",
    "date",
    "primary_setup",
    "rr",
    "close",
    "ma20",
    "ma50",
    "ma200",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
]

VALIDATION_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "structure_state",
    "structure_reason",
    "setup_type",
    "valid_setup",
    "validation_reason",
]

EXPECTED_STRUCTURE_STATES = {
    "COHERENT",
    "BROKEN",
    "INCOMPLETE",
}

EXPECTED_VALIDATION_REASONS = {
    "coherent_breakout",
    "coherent_pullback",
    "coherent_vcp",
    "structure_broken",
    "missing_data",
    "no_setup",
}

FORBIDDEN_VALIDATION_FIELDS = {
    "tradeable_setup",
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


def test_validation_layer_script_remains_legacy_reference_only():
    assert LEGACY_VALIDATION_LAYER_MODULE_PATH == Path("scripts/core/build_validation_layer.py")


def test_validation_contract_schema_is_exact_and_governance_clean():
    assert VALIDATION_OUTPUT_COLUMNS == [
        "ticker",
        "date",
        "structure_state",
        "structure_reason",
        "setup_type",
        "valid_setup",
        "validation_reason",
    ]
    assert set(VALIDATION_OUTPUT_COLUMNS).isdisjoint(FORBIDDEN_VALIDATION_FIELDS)


def test_validation_contract_requires_scanner_input_fields():
    assert REQUIRED_SCANNER_COLUMNS == [
        "ticker",
        "date",
        "primary_setup",
        "rr",
        "close",
        "ma20",
        "ma50",
        "ma200",
        "high_20d",
        "low_20d",
        "atr14",
        "volume_ratio",
        "extension_atr",
    ]


def test_validation_contract_keeps_descriptive_structure_states_only():
    assert EXPECTED_STRUCTURE_STATES == {"COHERENT", "BROKEN", "INCOMPLETE"}
    assert {"coherent_breakout", "structure_broken", "missing_data", "no_setup"}.issubset(
        EXPECTED_VALIDATION_REASONS
    )


def test_validation_contract_has_no_trade_or_decision_authority():
    serialized = " ".join(
        VALIDATION_OUTPUT_COLUMNS
        + REQUIRED_SCANNER_COLUMNS
        + sorted(EXPECTED_STRUCTURE_STATES)
        + sorted(EXPECTED_VALIDATION_REASONS)
    )
    for field in FORBIDDEN_VALIDATION_FIELDS:
        assert field not in VALIDATION_OUTPUT_COLUMNS
        assert field not in serialized


def test_active_code_no_longer_imports_validation_layer_script():
    for path in [Path("src"), Path("tests"), Path(".github")]:
        if not path.exists():
            continue
        for source_path in path.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            if source_path in {
                Path("tests/core/test_build_validation_layer.py"),
                Path("tests/core/test_entry_quality.py"),
            }:
                continue
            source = source_path.read_text(encoding="utf-8")
            assert "from scripts.core import build_validation_layer" not in source
            assert "import scripts.core.build_validation_layer" not in source
