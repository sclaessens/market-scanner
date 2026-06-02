"""Contract metadata for portfolio-first Telegram reporting UX."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class TelegramSection(StrEnum):
    """Sections required by the approved v2 Telegram UX baseline."""

    HEADER = "header"
    PORTFOLIO = "portfolio"
    BUY_NOW = "buy_now"
    BUY_ON_PULLBACK = "buy_on_pullback"
    BUY_ON_BREAKOUT = "buy_on_breakout"
    DATA_STATUS = "data_status"


class CandidateDisplayGroup(StrEnum):
    """Candidate display groups approved for Telegram communication."""

    BUY_NOW = "buy_now"
    BUY_ON_PULLBACK = "buy_on_pullback"
    BUY_ON_BREAKOUT = "buy_on_breakout"


class ThresholdDirection(StrEnum):
    """Display-only threshold direction supplied by upstream records."""

    NOT_APPLICABLE = "not_applicable"
    BELOW = "below"
    ABOVE = "above"


class TelegramContractIssueCode(StrEnum):
    """Metadata-only issue codes for Telegram display contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_CANDIDATE_GROUP = "invalid_candidate_group"
    INVALID_THRESHOLD_DIRECTION = "invalid_threshold_direction"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class TelegramContractIssue:
    """Explicit Telegram contract issue without reporting authority."""

    field_name: str
    issue_code: TelegramContractIssueCode
    observed_value: object


REQUIRED_PORTFOLIO_DISPLAY_FIELDS: tuple[str, ...] = (
    "ticker",
    "profit_loss_percent",
    "current_price",
    "target_price",
    "action_status",
    "currency",
)

OPTIONAL_PORTFOLIO_DISPLAY_FIELDS: tuple[str, ...] = ("instrument_group",)

REQUIRED_CANDIDATE_DISPLAY_FIELDS: tuple[str, ...] = (
    "ticker",
    "candidate_group",
    "threshold_price",
    "threshold_direction",
    "action_status",
    "currency",
)

REQUIRED_DATA_STATUS_FIELDS: tuple[str, ...] = (
    "data_status",
    "review_reason",
)

OPTIONAL_DATA_STATUS_FIELDS: tuple[str, ...] = (
    "missing_count",
    "partial_count",
    "stale_count",
)

EXPECTED_THRESHOLD_DIRECTION_BY_GROUP: dict[CandidateDisplayGroup, ThresholdDirection] = {
    CandidateDisplayGroup.BUY_NOW: ThresholdDirection.NOT_APPLICABLE,
    CandidateDisplayGroup.BUY_ON_PULLBACK: ThresholdDirection.BELOW,
    CandidateDisplayGroup.BUY_ON_BREAKOUT: ThresholdDirection.ABOVE,
}

EMPTY_SECTION_TEXT: dict[CandidateDisplayGroup, str] = {
    CandidateDisplayGroup.BUY_NOW: "No candidates today.",
    CandidateDisplayGroup.BUY_ON_PULLBACK: "No pullback candidates today.",
    CandidateDisplayGroup.BUY_ON_BREAKOUT: "No breakout candidates today.",
}

FORBIDDEN_TELEGRAM_AUTHORITY_FIELDS: tuple[str, ...] = (
    "allocation",
    "allocation_amount",
    "allocation_priority",
    "execution_instruction",
    "execution_ready",
    "urgency",
    "conviction",
    "tradeability",
    "tradeable_setup",
    "rank",
    "ranking",
    "score",
    "recommendation",
    "target_price_calculation",
    "buy_threshold_calculation",
    "breakout_threshold_calculation",
)


def telegram_section_order() -> tuple[TelegramSection, ...]:
    """Return the approved portfolio-first Telegram section order."""

    return (
        TelegramSection.HEADER,
        TelegramSection.PORTFOLIO,
        TelegramSection.BUY_NOW,
        TelegramSection.BUY_ON_PULLBACK,
        TelegramSection.BUY_ON_BREAKOUT,
        TelegramSection.DATA_STATUS,
    )


def required_portfolio_display_fields() -> tuple[str, ...]:
    """Return required portfolio display fields for Telegram."""

    return REQUIRED_PORTFOLIO_DISPLAY_FIELDS


def required_candidate_display_fields() -> tuple[str, ...]:
    """Return required candidate display fields for Telegram."""

    return REQUIRED_CANDIDATE_DISPLAY_FIELDS


def required_data_status_fields() -> tuple[str, ...]:
    """Return required data-status display fields for Telegram."""

    return REQUIRED_DATA_STATUS_FIELDS


def forbidden_telegram_authority_fields() -> tuple[str, ...]:
    """Return fields Telegram contracts must not accept as authority."""

    return FORBIDDEN_TELEGRAM_AUTHORITY_FIELDS


def empty_section_text() -> Mapping[CandidateDisplayGroup, str]:
    """Return compact empty-state text for candidate sections."""

    return EMPTY_SECTION_TEXT


def expected_threshold_direction(
    group: CandidateDisplayGroup,
) -> ThresholdDirection:
    """Return the upstream-supplied threshold direction expected for a group."""

    return EXPECTED_THRESHOLD_DIRECTION_BY_GROUP[group]


def validate_portfolio_display_shape(
    record: Mapping[str, object],
) -> tuple[TelegramContractIssue, ...]:
    """Check portfolio display shape without formatting or delivery."""

    return _validate_required_shape(record, REQUIRED_PORTFOLIO_DISPLAY_FIELDS)


def validate_candidate_display_shape(
    record: Mapping[str, object],
) -> tuple[TelegramContractIssue, ...]:
    """Check candidate display shape without ranking or threshold calculation."""

    issues = _validate_required_shape(record, REQUIRED_CANDIDATE_DISPLAY_FIELDS)

    raw_group = record.get("candidate_group")
    try:
        group = CandidateDisplayGroup(raw_group)
    except ValueError:
        if raw_group is not None and raw_group != "":
            issues += (
                TelegramContractIssue(
                    field_name="candidate_group",
                    issue_code=TelegramContractIssueCode.INVALID_CANDIDATE_GROUP,
                    observed_value=raw_group,
                ),
            )
        return issues

    raw_direction = record.get("threshold_direction")
    expected_direction = EXPECTED_THRESHOLD_DIRECTION_BY_GROUP[group]
    if raw_direction != expected_direction.value:
        issues += (
            TelegramContractIssue(
                field_name="threshold_direction",
                issue_code=TelegramContractIssueCode.INVALID_THRESHOLD_DIRECTION,
                observed_value=raw_direction,
            ),
        )

    return issues


def validate_data_status_shape(
    record: Mapping[str, object],
) -> tuple[TelegramContractIssue, ...]:
    """Check data-status display shape without quality inference."""

    return _validate_required_shape(record, REQUIRED_DATA_STATUS_FIELDS)


def _validate_required_shape(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
) -> tuple[TelegramContractIssue, ...]:
    issues: list[TelegramContractIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                TelegramContractIssue(
                    field_name=field_name,
                    issue_code=TelegramContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                TelegramContractIssue(
                    field_name=field_name,
                    issue_code=TelegramContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    for field_name in FORBIDDEN_TELEGRAM_AUTHORITY_FIELDS:
        if field_name in record:
            issues.append(
                TelegramContractIssue(
                    field_name=field_name,
                    issue_code=TelegramContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)
