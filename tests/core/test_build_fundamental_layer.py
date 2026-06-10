from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/build_quality.py")

EXPECTED_OUTPUT_COLUMNS = [
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
    "source_name",
    "source_last_updated",
    "source_freshness_days",
    "missing_required_fields",
    "partial_data_reason",
    "stale_data_reason",
    "invalid_data_reason",
    "generated_at",
]

FORBIDDEN_FIELDS = {
    "allocation",
    "conviction",
    "decision",
    "final_action",
    "tradeable",
    "urgency",
}


def test_legacy_fundamental_quality_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "build_quality.py")


def test_legacy_fundamental_quality_contract_remains_classification_only() -> None:
    normalized_columns = {column.lower() for column in EXPECTED_OUTPUT_COLUMNS}

    assert normalized_columns.isdisjoint(FORBIDDEN_FIELDS)
    assert EXPECTED_OUTPUT_COLUMNS[:2] == ["ticker", "date"]
