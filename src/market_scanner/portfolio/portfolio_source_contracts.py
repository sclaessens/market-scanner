"""Portfolio source-of-truth contract metadata for v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class PortfolioSourceDatasetRole(StrEnum):
    """Portfolio dataset roles for source ownership and downstream display."""

    MANUAL_SOURCE_TRANSACTIONS = "manual_source_transactions"
    MANUAL_SOURCE_POSITIONS = "manual_source_positions"
    NORMALIZED_POSITIONS = "normalized_positions"
    GENERATED_PORTFOLIO_REVIEW = "generated_portfolio_review"
    GENERATED_PORTFOLIO_INTELLIGENCE = "generated_portfolio_intelligence"
    REPORTING_DISPLAY_INPUT = "reporting_display_input"


class PortfolioSourceContractIssueCode(StrEnum):
    """Metadata-only issue codes for portfolio source contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class PortfolioSourceContractIssue:
    """Explicit source contract issue without portfolio runtime behavior."""

    field_name: str
    issue_code: PortfolioSourceContractIssueCode
    observed_value: object


MANUAL_SOURCE_DATASET_ROLES: tuple[PortfolioSourceDatasetRole, ...] = (
    PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS,
    PortfolioSourceDatasetRole.MANUAL_SOURCE_POSITIONS,
)

NORMALIZED_DATASET_ROLES: tuple[PortfolioSourceDatasetRole, ...] = (
    PortfolioSourceDatasetRole.NORMALIZED_POSITIONS,
)

GENERATED_DATASET_ROLES: tuple[PortfolioSourceDatasetRole, ...] = (
    PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_REVIEW,
    PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_INTELLIGENCE,
)

REPORTING_DISPLAY_DATASET_ROLES: tuple[PortfolioSourceDatasetRole, ...] = (
    PortfolioSourceDatasetRole.REPORTING_DISPLAY_INPUT,
)

PORTFOLIO_SOURCE_OF_TRUTH_ROLES: tuple[PortfolioSourceDatasetRole, ...] = (
    PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS,
    PortfolioSourceDatasetRole.MANUAL_SOURCE_POSITIONS,
)

REQUIRED_MANUAL_POSITION_SOURCE_FIELDS: tuple[str, ...] = (
    "portfolio_id",
    "ticker",
    "quantity",
    "currency",
    "source_type",
    "as_of_date",
)

REQUIRED_MANUAL_TRANSACTION_SOURCE_FIELDS: tuple[str, ...] = (
    "portfolio_id",
    "transaction_id",
    "ticker",
    "transaction_type",
    "quantity",
    "price",
    "currency",
    "transaction_date",
    "source_type",
)

REQUIRED_PORTFOLIO_DISPLAY_INPUT_FIELDS: tuple[str, ...] = (
    "ticker",
    "profit_loss_percent_display",
    "current_price_display",
    "target_price_display",
    "action_status",
    "currency",
    "source_reference",
)

FORBIDDEN_PORTFOLIO_SOURCE_FIELDS: tuple[str, ...] = (
    "final_action",
    "action_status",
    "allocation",
    "allocation_amount",
    "allocation_instruction",
    "execution_instruction",
    "urgency",
    "conviction",
    "tradeability",
    "tradeable_setup",
    "rank",
    "ranking",
    "score",
    "recommendation",
    "profit_loss_percent_display",
    "current_price_display",
    "target_price_display",
    "telegram_message",
    "reporting_line",
    "generated_review_overwrite",
)

FORBIDDEN_PORTFOLIO_DISPLAY_AUTHORITY_FIELDS: tuple[str, ...] = (
    "allocation",
    "allocation_amount",
    "allocation_instruction",
    "execution_instruction",
    "urgency",
    "conviction",
    "tradeability",
    "tradeable_setup",
    "rank",
    "ranking",
    "score",
    "recommendation",
    "telegram_message",
    "target_price_calculation",
    "profit_loss_calculation",
    "current_price_fetch",
)


def portfolio_dataset_roles() -> tuple[PortfolioSourceDatasetRole, ...]:
    """Return all portfolio source-of-truth contract dataset roles."""

    return tuple(PortfolioSourceDatasetRole)


def manual_source_dataset_roles() -> tuple[PortfolioSourceDatasetRole, ...]:
    """Return roles that may be authoritative manual portfolio sources."""

    return MANUAL_SOURCE_DATASET_ROLES


def generated_dataset_roles() -> tuple[PortfolioSourceDatasetRole, ...]:
    """Return portfolio roles that are generated outputs only."""

    return GENERATED_DATASET_ROLES


def reporting_display_dataset_roles() -> tuple[PortfolioSourceDatasetRole, ...]:
    """Return portfolio roles used only for downstream communication display."""

    return REPORTING_DISPLAY_DATASET_ROLES


def is_portfolio_source_of_truth(role: PortfolioSourceDatasetRole) -> bool:
    """Return whether a dataset role may own portfolio source truth."""

    return role in PORTFOLIO_SOURCE_OF_TRUTH_ROLES


def required_manual_position_fields() -> tuple[str, ...]:
    """Return required manual source position fields."""

    return REQUIRED_MANUAL_POSITION_SOURCE_FIELDS


def required_manual_transaction_fields() -> tuple[str, ...]:
    """Return required manual source transaction fields."""

    return REQUIRED_MANUAL_TRANSACTION_SOURCE_FIELDS


def required_portfolio_display_input_fields() -> tuple[str, ...]:
    """Return required portfolio fields supplied to reporting display."""

    return REQUIRED_PORTFOLIO_DISPLAY_INPUT_FIELDS


def forbidden_portfolio_source_fields() -> tuple[str, ...]:
    """Return fields manual source records must not contain."""

    return FORBIDDEN_PORTFOLIO_SOURCE_FIELDS


def forbidden_portfolio_display_authority_fields() -> tuple[str, ...]:
    """Return fields reporting display input must not treat as authority."""

    return FORBIDDEN_PORTFOLIO_DISPLAY_AUTHORITY_FIELDS


def validate_manual_position_source_shape(
    record: Mapping[str, object],
) -> tuple[PortfolioSourceContractIssue, ...]:
    """Check manual position source shape without portfolio calculations."""

    return _validate_shape(
        record=record,
        required_fields=REQUIRED_MANUAL_POSITION_SOURCE_FIELDS,
        forbidden_fields=FORBIDDEN_PORTFOLIO_SOURCE_FIELDS,
    )


def validate_manual_transaction_source_shape(
    record: Mapping[str, object],
) -> tuple[PortfolioSourceContractIssue, ...]:
    """Check manual transaction source shape without ingestion behavior."""

    return _validate_shape(
        record=record,
        required_fields=REQUIRED_MANUAL_TRANSACTION_SOURCE_FIELDS,
        forbidden_fields=FORBIDDEN_PORTFOLIO_SOURCE_FIELDS,
    )


def validate_portfolio_display_input_shape(
    record: Mapping[str, object],
) -> tuple[PortfolioSourceContractIssue, ...]:
    """Check reporting display input shape without rendering or authority."""

    return _validate_shape(
        record=record,
        required_fields=REQUIRED_PORTFOLIO_DISPLAY_INPUT_FIELDS,
        forbidden_fields=FORBIDDEN_PORTFOLIO_DISPLAY_AUTHORITY_FIELDS,
    )


def _validate_shape(
    *,
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
    forbidden_fields: tuple[str, ...],
) -> tuple[PortfolioSourceContractIssue, ...]:
    issues: list[PortfolioSourceContractIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                PortfolioSourceContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioSourceContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                PortfolioSourceContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioSourceContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    for field_name in forbidden_fields:
        if field_name in record:
            issues.append(
                PortfolioSourceContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioSourceContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)
