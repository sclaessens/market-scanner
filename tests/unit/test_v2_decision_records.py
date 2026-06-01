from market_scanner.decisions.decision_records import (
    DecisionEngineResult,
    DecisionRecord,
    DecisionState,
)


FORBIDDEN_DECISION_FIELD_NAMES = {
    "allocation_amount",
    "allocation_weight",
    "position_size",
    "rank",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
}


def test_reset_6_decision_state_contains_only_review():
    assert tuple(DecisionState) == (DecisionState.REVIEW,)


def test_decision_record_contains_only_review_scaffold_fields():
    assert set(DecisionRecord.__dataclass_fields__) == {
        "row_id",
        "source_name",
        "final_action",
        "rationale",
    }
    assert set(DecisionRecord.__dataclass_fields__).isdisjoint(
        FORBIDDEN_DECISION_FIELD_NAMES
    )


def test_decision_engine_result_contains_no_sizing_or_execution_fields():
    assert set(DecisionEngineResult.__dataclass_fields__).isdisjoint(
        FORBIDDEN_DECISION_FIELD_NAMES
    )
