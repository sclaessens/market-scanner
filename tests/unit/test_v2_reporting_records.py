from market_scanner.reporting.report_records import ReportRecord, ReportResult


FORBIDDEN_REPORTING_FIELD_NAMES = {
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
    "priority",
}


def test_report_record_contains_only_communication_fields():
    assert set(ReportRecord.__dataclass_fields__) == {
        "row_id",
        "source_name",
        "communicated_action",
        "communicated_rationale",
        "line",
    }
    assert set(ReportRecord.__dataclass_fields__).isdisjoint(
        FORBIDDEN_REPORTING_FIELD_NAMES
    )


def test_report_result_contains_no_priority_or_allocation_fields():
    assert set(ReportResult.__dataclass_fields__).isdisjoint(
        FORBIDDEN_REPORTING_FIELD_NAMES
    )
