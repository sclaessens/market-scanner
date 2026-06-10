from __future__ import annotations

from market_scanner.fundamentals.fundamental_contracts import (
    FundamentalDatasetRole,
    SourceDataReadinessState,
    generated_dataset_roles,
    required_fundamental_history_fields,
    source_dataset_roles,
)


def test_fundamental_quality_layer_contract_is_now_canonical_evidence_only() -> None:
    assert FundamentalDatasetRole.NORMALIZED_FUNDAMENTAL_HISTORY in source_dataset_roles()
    assert FundamentalDatasetRole.GENERATED_FUNDAMENTAL_CLASSIFICATION in (
        generated_dataset_roles()
    )


def test_fundamental_quality_layer_preserves_source_data_readiness_states() -> None:
    assert SourceDataReadinessState.AVAILABLE.value == "available"
    assert SourceDataReadinessState.SOURCE_MISSING.value == "source_missing"
    assert SourceDataReadinessState.REVIEW_REQUIRED.value == "review_required"


def test_fundamental_quality_layer_does_not_define_final_action_fields() -> None:
    fields = required_fundamental_history_fields()

    assert "source_reference" in fields
    assert "final_action" not in fields
    assert "allocation" not in fields
    assert "position_size" not in fields