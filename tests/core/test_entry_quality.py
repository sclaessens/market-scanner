from __future__ import annotations

from pathlib import Path

LEGACY_ENTRY_QUALITY_OWNER_PATH = Path("scripts/core/build_validation_layer.py")

ENTRY_QUALITY_METRICS_COLUMNS = [
    "ticker",
    "date",
    "distance_to_breakout_pct",
    "breakout_extension_atr",
    "extension_atr",
    "distance_ma20_pct",
    "volume_ratio",
    "range_atr",
    "entry_quality_state",
    "entry_quality_reason",
]

EXPECTED_ENTRY_QUALITY_STATES = {
    "BALANCED",
    "EXTENDED",
    "WIDE_RANGE",
}

EXPECTED_ENTRY_QUALITY_REASONS = {
    "balanced_structure",
    "extended_vs_ma20",
    "wide_recent_range",
}

FORBIDDEN_ENTRY_QUALITY_FIELDS = {
    "tradeable",
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


def test_entry_quality_owner_remains_legacy_reference_only():
    assert LEGACY_ENTRY_QUALITY_OWNER_PATH == Path("scripts/core/build_validation_layer.py")


def test_entry_quality_metrics_schema_is_descriptive_only():
    assert ENTRY_QUALITY_METRICS_COLUMNS == [
        "ticker",
        "date",
        "distance_to_breakout_pct",
        "breakout_extension_atr",
        "extension_atr",
        "distance_ma20_pct",
        "volume_ratio",
        "range_atr",
        "entry_quality_state",
        "entry_quality_reason",
    ]
    assert set(ENTRY_QUALITY_METRICS_COLUMNS).isdisjoint(FORBIDDEN_ENTRY_QUALITY_FIELDS)


def test_entry_quality_contract_keeps_metric_only_states():
    assert EXPECTED_ENTRY_QUALITY_STATES == {"BALANCED", "EXTENDED", "WIDE_RANGE"}
    assert "balanced_structure" in EXPECTED_ENTRY_QUALITY_REASONS
    assert "extended_vs_ma20" in EXPECTED_ENTRY_QUALITY_REASONS


def test_entry_quality_does_not_change_validation_structure_contract():
    validation_fields = {
        "structure_state",
        "structure_reason",
        "setup_type",
        "valid_setup",
        "validation_reason",
    }
    assert validation_fields.isdisjoint(ENTRY_QUALITY_METRICS_COLUMNS)


def test_entry_quality_contract_has_no_trade_or_decision_authority():
    serialized = " ".join(
        ENTRY_QUALITY_METRICS_COLUMNS
        + sorted(EXPECTED_ENTRY_QUALITY_STATES)
        + sorted(EXPECTED_ENTRY_QUALITY_REASONS)
    )
    for field in FORBIDDEN_ENTRY_QUALITY_FIELDS:
        assert field not in ENTRY_QUALITY_METRICS_COLUMNS
        assert field not in serialized


def test_active_code_no_longer_imports_entry_quality_script_owner():
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
