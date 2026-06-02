from pathlib import Path

from market_scanner.reporting import telegram_contracts
from market_scanner.reporting.telegram_contracts import (
    CandidateDisplayGroup,
    TelegramContractIssue,
    TelegramContractIssueCode,
    TelegramSection,
    ThresholdDirection,
    empty_section_text,
    expected_threshold_direction,
    forbidden_telegram_authority_fields,
    required_candidate_display_fields,
    required_data_status_fields,
    required_portfolio_display_fields,
    telegram_section_order,
    validate_candidate_display_shape,
    validate_data_status_shape,
    validate_portfolio_display_shape,
)


def _portfolio_row(**overrides):
    row = {
        "ticker": "ASML",
        "profit_loss_percent": "+8.4",
        "current_price": "650.00",
        "target_price": "700.00",
        "action_status": "REVIEW",
        "currency": "EUR",
    }
    row.update(overrides)
    return row


def _candidate_row(**overrides):
    row = {
        "ticker": "AMD",
        "candidate_group": "buy_on_pullback",
        "threshold_price": "150.00",
        "threshold_direction": "below",
        "action_status": "REVIEW",
        "currency": "USD",
    }
    row.update(overrides)
    return row


def _data_status_row(**overrides):
    row = {
        "data_status": "fundamental_data_incomplete",
        "review_reason": "many review states from partial fundamentals",
    }
    row.update(overrides)
    return row


def test_telegram_section_order_is_portfolio_first_after_header():
    assert telegram_section_order() == (
        TelegramSection.HEADER,
        TelegramSection.PORTFOLIO,
        TelegramSection.BUY_NOW,
        TelegramSection.BUY_ON_PULLBACK,
        TelegramSection.BUY_ON_BREAKOUT,
        TelegramSection.DATA_STATUS,
    )


def test_portfolio_display_fields_include_required_ux_concepts():
    assert required_portfolio_display_fields() == (
        "ticker",
        "profit_loss_percent",
        "current_price",
        "target_price",
        "action_status",
        "currency",
    )
    assert validate_portfolio_display_shape(_portfolio_row()) == ()


def test_candidate_display_groups_include_approved_baseline_sections():
    assert {group for group in CandidateDisplayGroup} == {
        CandidateDisplayGroup.BUY_NOW,
        CandidateDisplayGroup.BUY_ON_PULLBACK,
        CandidateDisplayGroup.BUY_ON_BREAKOUT,
    }
    assert required_candidate_display_fields() == (
        "ticker",
        "candidate_group",
        "threshold_price",
        "threshold_direction",
        "action_status",
        "currency",
    )


def test_pullback_candidates_require_below_threshold_semantics():
    assert expected_threshold_direction(
        CandidateDisplayGroup.BUY_ON_PULLBACK
    ) == ThresholdDirection.BELOW
    assert validate_candidate_display_shape(_candidate_row()) == ()


def test_breakout_candidates_require_above_threshold_semantics():
    row = _candidate_row(
        candidate_group="buy_on_breakout",
        threshold_direction="above",
    )

    assert expected_threshold_direction(
        CandidateDisplayGroup.BUY_ON_BREAKOUT
    ) == ThresholdDirection.ABOVE
    assert validate_candidate_display_shape(row) == ()


def test_buy_now_candidates_use_no_threshold_direction():
    row = _candidate_row(
        candidate_group="buy_now",
        threshold_price="not_applicable",
        threshold_direction="not_applicable",
    )

    assert expected_threshold_direction(
        CandidateDisplayGroup.BUY_NOW
    ) == ThresholdDirection.NOT_APPLICABLE
    assert validate_candidate_display_shape(row) == ()


def test_candidate_threshold_direction_mismatch_is_reported():
    issues = validate_candidate_display_shape(
        _candidate_row(candidate_group="buy_on_pullback", threshold_direction="above")
    )

    assert issues == (
        TelegramContractIssue(
            field_name="threshold_direction",
            issue_code=TelegramContractIssueCode.INVALID_THRESHOLD_DIRECTION,
            observed_value="above",
        ),
    )


def test_empty_state_support_exists_for_candidate_sections():
    assert empty_section_text() == {
        CandidateDisplayGroup.BUY_NOW: "No candidates today.",
        CandidateDisplayGroup.BUY_ON_PULLBACK: "No pullback candidates today.",
        CandidateDisplayGroup.BUY_ON_BREAKOUT: "No breakout candidates today.",
    }


def test_data_status_section_remains_explicit():
    assert required_data_status_fields() == ("data_status", "review_reason")
    assert validate_data_status_shape(_data_status_row()) == ()


def test_reporting_telegram_contracts_do_not_include_authority_fields():
    all_display_fields = (
        set(required_portfolio_display_fields())
        | set(required_candidate_display_fields())
        | set(required_data_status_fields())
    )

    for field_name in forbidden_telegram_authority_fields():
        assert field_name not in all_display_fields


def test_forbidden_reporting_authority_fields_are_reported_as_issues():
    issues = validate_portfolio_display_shape(_portfolio_row(allocation="10%"))

    assert issues == (
        TelegramContractIssue(
            field_name="allocation",
            issue_code=TelegramContractIssueCode.FORBIDDEN_FIELD,
            observed_value="10%",
        ),
    )


def test_missing_target_price_remains_explicit_and_not_zero():
    issues = validate_portfolio_display_shape(_portfolio_row(target_price=""))

    assert issues == (
        TelegramContractIssue(
            field_name="target_price",
            issue_code=TelegramContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_missing_profit_loss_remains_explicit_and_not_zero():
    issues = validate_portfolio_display_shape(_portfolio_row(profit_loss_percent=None))

    assert issues == (
        TelegramContractIssue(
            field_name="profit_loss_percent",
            issue_code=TelegramContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value=None,
        ),
    )


def test_invalid_candidate_group_is_reported_without_reclassification():
    issues = validate_candidate_display_shape(
        _candidate_row(candidate_group="priority_watch")
    )

    assert issues == (
        TelegramContractIssue(
            field_name="candidate_group",
            issue_code=TelegramContractIssueCode.INVALID_CANDIDATE_GROUP,
            observed_value="priority_watch",
        ),
    )


def test_telegram_contract_module_does_not_import_legacy_or_delivery_modules():
    source = Path(telegram_contracts.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "TELEGRAM_API",
        "send_telegram",
    ):
        assert forbidden not in source


def test_telegram_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validate_portfolio_display_shape(_portfolio_row(target_price=None))
    validate_candidate_display_shape(_candidate_row(threshold_price=None))
    validate_data_status_shape(_data_status_row(review_reason=None))

    assert list(tmp_path.iterdir()) == []
