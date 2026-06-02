"""Contract metadata for the v2 validation layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ValidationState(str, Enum):
    """Allowed structure-classification states for v2 validation contracts."""

    COHERENT = "coherent"
    BROKEN = "broken"
    INCOMPLETE = "incomplete"
    REVIEW_REQUIRED = "review_required"


class ValidationIssueCode(str, Enum):
    """Metadata-only issue codes emitted by validation contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    INVALID_NON_POSITIVE_VALUE = "invalid_non_positive_value"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class ValidationIssue:
    """Explicit validation contract issue without filtering or decisions."""

    field_name: str
    issue_code: ValidationIssueCode
    observed_value: object


CANDIDATE_IDENTITY_FIELDS: tuple[str, ...] = ("ticker", "date")

REQUIRED_CANDIDATE_FIELDS: tuple[str, ...] = (
    "ticker",
    "date",
    "primary_setup",
    "rr",
    "close",
    "ma20",
    "ma50",
    "ma200",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
)

NUMERIC_CANDIDATE_FIELDS: tuple[str, ...] = (
    "rr",
    "close",
    "ma20",
    "ma50",
    "ma200",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
)

POSITIVE_NUMERIC_FIELDS: tuple[str, ...] = ("atr14",)

VALIDATION_CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "row_id",
    "ticker",
    "date",
    "validation_state",
    "validation_reason",
    "missing_fields",
    "source_reference",
)

FORBIDDEN_UPSTREAM_DECISION_FIELDS: tuple[str, ...] = (
    "final_action",
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "urgency",
    "conviction",
    "execution_instruction",
    "tradeability",
    "tradeable_setup",
)


def required_candidate_fields() -> tuple[str, ...]:
    """Return required candidate-input fields for v2 validation."""

    return REQUIRED_CANDIDATE_FIELDS


def candidate_identity_fields() -> tuple[str, ...]:
    """Return fields that establish validation row identity."""

    return CANDIDATE_IDENTITY_FIELDS


def validation_classification_fields() -> tuple[str, ...]:
    """Return metadata-only v2 validation output fields."""

    return VALIDATION_CLASSIFICATION_FIELDS


def forbidden_upstream_decision_fields() -> tuple[str, ...]:
    """Return fields that validation contracts must not emit or accept."""

    return FORBIDDEN_UPSTREAM_DECISION_FIELDS


def validate_candidate_record_shape(
    record: Mapping[str, object],
) -> tuple[ValidationIssue, ...]:
    """Check candidate shape without mutating, filtering, or interpreting rows."""

    issues: list[ValidationIssue] = []

    for field_name in REQUIRED_CANDIDATE_FIELDS:
        if field_name not in record:
            issues.append(
                ValidationIssue(
                    field_name=field_name,
                    issue_code=ValidationIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                ValidationIssue(
                    field_name=field_name,
                    issue_code=ValidationIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )
            continue

        if field_name in NUMERIC_CANDIDATE_FIELDS:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_NUMERIC_VALUE,
                        observed_value=value,
                    )
                )
                continue

            if field_name in POSITIVE_NUMERIC_FIELDS and numeric_value <= 0:
                issues.append(
                    ValidationIssue(
                        field_name=field_name,
                        issue_code=ValidationIssueCode.INVALID_NON_POSITIVE_VALUE,
                        observed_value=value,
                    )
                )

    for field_name in FORBIDDEN_UPSTREAM_DECISION_FIELDS:
        if field_name in record:
            issues.append(
                ValidationIssue(
                    field_name=field_name,
                    issue_code=ValidationIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)
