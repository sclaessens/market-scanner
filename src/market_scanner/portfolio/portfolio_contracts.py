"""Contract metadata for v2 portfolio source and generated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class PortfolioDatasetType(str, Enum):
    """Portfolio dataset roles in the v2 data lifecycle."""

    MANUAL_TRANSACTION_INPUT = "manual_transaction_input"
    NORMALIZED_POSITION_INPUT = "normalized_position_input"
    GENERATED_PORTFOLIO_REVIEW = "generated_portfolio_review"
    GENERATED_PORTFOLIO_CLASSIFICATION = "generated_portfolio_classification"


class PortfolioContractIssueCode(str, Enum):
    """Metadata-only issue codes for portfolio contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class PortfolioContractIssue:
    """Explicit portfolio contract issue without decisions or file effects."""

    field_name: str
    issue_code: PortfolioContractIssueCode
    observed_value: object


PORTFOLIO_TRANSACTION_IDENTITY_FIELDS: tuple[str, ...] = ("transaction_id",)
PORTFOLIO_POSITION_IDENTITY_FIELDS: tuple[str, ...] = (
    "portfolio_account",
    "symbol",
)

REQUIRED_PORTFOLIO_TRANSACTION_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "portfolio_account",
    "symbol",
    "transaction_kind",
    "quantity_delta",
    "cash_amount",
    "currency",
    "occurred_at",
    "source_reference",
)

REQUIRED_PORTFOLIO_POSITION_FIELDS: tuple[str, ...] = (
    "portfolio_account",
    "symbol",
    "quantity",
    "average_cost",
    "currency",
    "source_reference",
)

PORTFOLIO_NUMERIC_FIELDS: tuple[str, ...] = (
    "quantity_delta",
    "cash_amount",
    "quantity",
    "average_cost",
)

PORTFOLIO_SOURCE_DATASET_TYPES: tuple[PortfolioDatasetType, ...] = (
    PortfolioDatasetType.MANUAL_TRANSACTION_INPUT,
    PortfolioDatasetType.NORMALIZED_POSITION_INPUT,
)

PORTFOLIO_GENERATED_DATASET_TYPES: tuple[PortfolioDatasetType, ...] = (
    PortfolioDatasetType.GENERATED_PORTFOLIO_REVIEW,
    PortfolioDatasetType.GENERATED_PORTFOLIO_CLASSIFICATION,
)

FORBIDDEN_PORTFOLIO_UPSTREAM_FIELDS: tuple[str, ...] = (
    "final_action",
    "portfolio_action",
    "allocation",
    "allocation_decision",
    "allocation_amount",
    "position_size",
    "target_weight",
    "recommended_weight",
    "rank",
    "score",
    "urgency",
    "conviction",
    "execution_instruction",
    "tradeability",
    "tradeable_setup",
)


def required_portfolio_transaction_fields() -> tuple[str, ...]:
    """Return required fields for manual portfolio transaction input."""

    return REQUIRED_PORTFOLIO_TRANSACTION_FIELDS


def required_portfolio_position_fields() -> tuple[str, ...]:
    """Return required fields for normalized portfolio position input."""

    return REQUIRED_PORTFOLIO_POSITION_FIELDS


def portfolio_transaction_identity_fields() -> tuple[str, ...]:
    """Return identity fields for manual transaction records."""

    return PORTFOLIO_TRANSACTION_IDENTITY_FIELDS


def portfolio_position_identity_fields() -> tuple[str, ...]:
    """Return identity fields for normalized position records."""

    return PORTFOLIO_POSITION_IDENTITY_FIELDS


def portfolio_source_dataset_types() -> tuple[PortfolioDatasetType, ...]:
    """Return portfolio dataset types that may represent source/input data."""

    return PORTFOLIO_SOURCE_DATASET_TYPES


def portfolio_generated_dataset_types() -> tuple[PortfolioDatasetType, ...]:
    """Return portfolio dataset types that are generated outputs only."""

    return PORTFOLIO_GENERATED_DATASET_TYPES


def forbidden_portfolio_upstream_fields() -> tuple[str, ...]:
    """Return fields portfolio contracts must not accept as authority."""

    return FORBIDDEN_PORTFOLIO_UPSTREAM_FIELDS


def validate_portfolio_transaction_shape(
    record: Mapping[str, object],
) -> tuple[PortfolioContractIssue, ...]:
    """Check manual transaction shape without ingesting or generating positions."""

    return _validate_shape(record, REQUIRED_PORTFOLIO_TRANSACTION_FIELDS)


def validate_portfolio_position_shape(
    record: Mapping[str, object],
) -> tuple[PortfolioContractIssue, ...]:
    """Check normalized position shape without portfolio review behavior."""

    return _validate_shape(record, REQUIRED_PORTFOLIO_POSITION_FIELDS)


def _validate_shape(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
) -> tuple[PortfolioContractIssue, ...]:
    issues: list[PortfolioContractIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                PortfolioContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                PortfolioContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )
            continue

        if field_name in PORTFOLIO_NUMERIC_FIELDS:
            try:
                float(value)
            except (TypeError, ValueError):
                issues.append(
                    PortfolioContractIssue(
                        field_name=field_name,
                        issue_code=PortfolioContractIssueCode.INVALID_NUMERIC_VALUE,
                        observed_value=value,
                    )
                )

    for field_name in FORBIDDEN_PORTFOLIO_UPSTREAM_FIELDS:
        if field_name in record:
            issues.append(
                PortfolioContractIssue(
                    field_name=field_name,
                    issue_code=PortfolioContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)
