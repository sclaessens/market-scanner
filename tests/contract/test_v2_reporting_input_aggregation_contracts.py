from importlib import reload
from pathlib import Path

from market_scanner.portfolio.portfolio_source_contracts import (
    PortfolioSourceDatasetRole,
    is_portfolio_source_of_truth,
)
from market_scanner.reporting import reporting_input_contracts
from market_scanner.reporting.reporting_input_contracts import (
    AGGREGATION_CONTRACT_VERSION,
    ReportingInputBoundaryRole,
    ReportingInputContractIssue,
    ReportingInputContractIssueCode,
    ReportingInputRole,
    forbidden_reporting_aggregation_fields,
    reporting_input_boundary_roles,
    reporting_input_roles,
    required_aggregation_trace_fields,
    required_candidate_display_input_fields,
    required_data_warning_input_fields,
    required_decision_status_input_fields,
    required_portfolio_display_input_fields,
    required_source_data_status_input_fields,
    telegram_renderer_input_roles,
    upstream_input_roles,
    validate_candidate_display_input_shape,
    validate_data_warning_input_shape,
    validate_decision_status_input_shape,
    validate_portfolio_display_input_shape,
    validate_reporting_input_trace_shape,
    validate_source_data_status_input_shape,
)
from market_scanner.reporting.telegram_contracts import (
    required_candidate_display_fields as telegram_candidate_display_fields,
)
from market_scanner.reporting.telegram_contracts import (
    required_portfolio_display_fields as telegram_portfolio_display_fields,
)


def _trace(**overrides):
    record = {
        "source_role": "portfolio_display_input",
        "source_reference": "synthetic-source-row-001",
        "aggregation_contract_version": AGGREGATION_CONTRACT_VERSION,
    }
    record.update(overrides)
    return record


def _portfolio_input(**overrides):
    record = {
        "ticker": "ASML",
        "profit_loss_percent_display": "+8.4%",
        "current_price_display": "€XXX",
        "target_price_display": "€XXX",
        "action_status": "REVIEW",
        "currency": "EUR",
        "source_reference": "synthetic-portfolio-row-001",
    }
    record.update(overrides)
    return record


def _candidate_input(**overrides):
    record = {
        "ticker": "AMD",
        "candidate_group": "buy_on_pullback",
        "threshold_price_display": "$XXX",
        "threshold_direction": "below",
        "action_status": "REVIEW",
        "currency": "USD",
        "source_reference": "synthetic-candidate-row-001",
    }
    record.update(overrides)
    return record


def _decision_status_input(**overrides):
    record = {
        "row_id": "row-001",
        "action_status": "REVIEW",
        "decision_rationale": "review_only_scaffold",
        "source_reference": "synthetic-decision-row-001",
    }
    record.update(overrides)
    return record


def _source_data_status_input(**overrides):
    record = {
        "data_status": "Fundamental data incomplete",
        "review_reason": "many REVIEW.",
        "source_reference": "synthetic-source-data-row-001",
    }
    record.update(overrides)
    return record


def _data_warning_input(**overrides):
    record = {
        "warning_type": "missing_fundamentals",
        "warning_text": "Fundamental data incomplete.",
        "source_reference": "synthetic-warning-row-001",
    }
    record.update(overrides)
    return record


def test_reporting_input_aggregation_roles_exist():
    assert reporting_input_roles() == (
        ReportingInputRole.PORTFOLIO_DISPLAY_INPUT,
        ReportingInputRole.CANDIDATE_DISPLAY_INPUT,
        ReportingInputRole.DECISION_STATUS_INPUT,
        ReportingInputRole.SOURCE_DATA_STATUS_INPUT,
        ReportingInputRole.DATA_WARNING_INPUT,
        ReportingInputRole.TELEGRAM_RENDERER_INPUT,
    )
    assert reporting_input_boundary_roles() == (
        ReportingInputBoundaryRole.UPSTREAM_SOURCE_INPUT,
        ReportingInputBoundaryRole.DERIVED_DISPLAY_INPUT,
        ReportingInputBoundaryRole.GENERATED_REPORT_INPUT,
        ReportingInputBoundaryRole.RENDERER_INPUT,
    )


def test_portfolio_display_input_contains_telegram_ux_fields():
    fields = set(required_portfolio_display_input_fields())

    assert fields >= {
        "ticker",
        "profit_loss_percent_display",
        "current_price_display",
        "target_price_display",
        "action_status",
        "currency",
        "source_reference",
    }
    assert {
        field.replace("_display", "")
        for field in fields
        if field.endswith("_display")
    } >= set(telegram_portfolio_display_fields()) - {
        "ticker",
        "action_status",
        "currency",
    }
    assert validate_portfolio_display_input_shape(_portfolio_input()) == ()


def test_candidate_display_input_contains_telegram_ux_fields():
    fields = set(required_candidate_display_input_fields())

    assert fields >= {
        "ticker",
        "candidate_group",
        "threshold_price_display",
        "threshold_direction",
        "action_status",
        "currency",
        "source_reference",
    }
    assert {
        "ticker",
        "candidate_group",
        "threshold_direction",
        "action_status",
        "currency",
    } <= set(telegram_candidate_display_fields())
    assert validate_candidate_display_input_shape(_candidate_input()) == ()


def test_source_data_status_input_contains_status_and_review_reason():
    assert required_source_data_status_input_fields() == (
        "data_status",
        "review_reason",
        "source_reference",
    )
    assert validate_source_data_status_input_shape(_source_data_status_input()) == ()


def test_data_warning_input_is_traceable_without_report_generation():
    assert required_data_warning_input_fields() == (
        "warning_type",
        "warning_text",
        "source_reference",
    )
    assert validate_data_warning_input_shape(_data_warning_input()) == ()


def test_decision_status_input_preserves_decision_engine_status_only():
    assert required_decision_status_input_fields() == (
        "row_id",
        "action_status",
        "decision_rationale",
        "source_reference",
    )
    assert validate_decision_status_input_shape(_decision_status_input()) == ()


def test_each_reporting_input_requires_traceability():
    assert required_aggregation_trace_fields() == (
        "source_role",
        "source_reference",
        "aggregation_contract_version",
    )
    assert validate_reporting_input_trace_shape(_trace()) == ()


def test_invalid_trace_source_role_is_reported():
    issues = validate_reporting_input_trace_shape(_trace(source_role="source_truth"))

    assert issues == (
        ReportingInputContractIssue(
            field_name="source_role",
            issue_code=ReportingInputContractIssueCode.INVALID_SOURCE_ROLE,
            observed_value="source_truth",
        ),
    )


def test_reporting_input_aggregation_does_not_define_source_of_truth_roles():
    assert set(upstream_input_roles()).isdisjoint(telegram_renderer_input_roles())
    assert not is_portfolio_source_of_truth(
        PortfolioSourceDatasetRole.REPORTING_DISPLAY_INPUT
    )
    assert "source_of_truth" in forbidden_reporting_aggregation_fields()
    assert "source_of_truth_overwrite" in forbidden_reporting_aggregation_fields()
    assert "portfolio_source_overwrite" in forbidden_reporting_aggregation_fields()


def test_telegram_renderer_input_is_downstream_of_reporting_aggregation():
    assert upstream_input_roles() == (
        ReportingInputRole.PORTFOLIO_DISPLAY_INPUT,
        ReportingInputRole.CANDIDATE_DISPLAY_INPUT,
        ReportingInputRole.DECISION_STATUS_INPUT,
        ReportingInputRole.SOURCE_DATA_STATUS_INPUT,
        ReportingInputRole.DATA_WARNING_INPUT,
    )
    assert telegram_renderer_input_roles() == (
        ReportingInputRole.TELEGRAM_RENDERER_INPUT,
    )


def test_reporting_input_aggregation_does_not_calculate_display_values():
    fields = set(required_portfolio_display_input_fields()) | set(
        required_candidate_display_input_fields()
    )

    for forbidden in (
        "target_price_calculation",
        "buy_threshold_calculation",
        "breakout_threshold_calculation",
        "threshold_calculation_authority",
        "profit_loss_calculation",
        "current_price_fetch",
    ):
        assert forbidden not in fields
        assert forbidden in forbidden_reporting_aggregation_fields()


def test_reporting_input_aggregation_does_not_create_decisions():
    fields = set(required_decision_status_input_fields())

    assert "decision_override" not in fields
    assert "decision_override" in forbidden_reporting_aggregation_fields()


def test_reporting_input_aggregation_has_no_allocation_or_ranking_authority():
    all_fields = (
        set(required_portfolio_display_input_fields())
        | set(required_candidate_display_input_fields())
        | set(required_decision_status_input_fields())
        | set(required_source_data_status_input_fields())
        | set(required_data_warning_input_fields())
    )

    for forbidden in (
        "allocation",
        "execution_instruction",
        "urgency",
        "conviction",
        "tradeability",
        "ranking",
        "score",
        "recommendation",
    ):
        assert forbidden not in all_fields
        assert forbidden in forbidden_reporting_aggregation_fields()


def test_missing_display_values_remain_explicit_not_zero():
    issues = validate_portfolio_display_input_shape(
        _portfolio_input(profit_loss_percent_display="")
    )

    assert issues == (
        ReportingInputContractIssue(
            field_name="profit_loss_percent_display",
            issue_code=ReportingInputContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_forbidden_fields_are_reported_explicitly():
    issues = validate_candidate_display_input_shape(
        _candidate_input(ranking="top")
    )

    assert issues == (
        ReportingInputContractIssue(
            field_name="ranking",
            issue_code=ReportingInputContractIssueCode.FORBIDDEN_FIELD,
            observed_value="top",
        ),
    )


def test_reporting_input_contract_module_does_not_import_legacy_scripts():
    source = Path(reporting_input_contracts.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source


def test_importing_reporting_input_contracts_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(reporting_input_contracts)

    assert list(tmp_path.iterdir()) == []


def test_reporting_input_contract_helpers_create_no_files_or_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    validate_portfolio_display_input_shape(_portfolio_input(target_price_display=None))
    validate_candidate_display_input_shape(_candidate_input(threshold_price_display=""))
    validate_source_data_status_input_shape(_source_data_status_input())
    validate_reporting_input_trace_shape(_trace())

    assert list(tmp_path.iterdir()) == []
    assert not Path("reports/daily/telegram_message.txt").exists()
