from market_scanner.fundamentals.source_data_records import (
    SourceDataReadinessRecord,
    SourceDataReadinessResult,
    SourceDataStatus,
)


FORBIDDEN_SOURCE_DATA_FIELD_NAMES = {
    "final_action",
    "decision_state",
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "quality_score",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
}


def test_source_data_statuses_are_readiness_only():
    assert tuple(SourceDataStatus) == (
        SourceDataStatus.AVAILABLE,
        SourceDataStatus.MISSING,
        SourceDataStatus.PARTIAL,
        SourceDataStatus.STALE,
        SourceDataStatus.REVIEW_REQUIRED,
    )


def test_source_data_readiness_record_has_no_decision_or_quality_fields():
    assert set(SourceDataReadinessRecord.__dataclass_fields__).isdisjoint(
        FORBIDDEN_SOURCE_DATA_FIELD_NAMES
    )


def test_source_data_readiness_result_has_no_decision_or_quality_fields():
    assert set(SourceDataReadinessResult.__dataclass_fields__).isdisjoint(
        FORBIDDEN_SOURCE_DATA_FIELD_NAMES
    )
