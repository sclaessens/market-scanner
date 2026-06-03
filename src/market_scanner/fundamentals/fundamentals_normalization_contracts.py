"""Contract metadata for raw-to-normalized fundamentals boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from market_scanner.fundamentals.fundamental_contracts import (
    SourceDataReadinessState,
)


class FundamentalsNormalizationDatasetRole(StrEnum):
    """Dataset roles for fundamentals raw-to-normalized lifecycle stages."""

    RAW_SOURCE_CAPTURE = "raw_source_capture"
    NORMALIZED_FUNDAMENTALS_INPUT = "normalized_fundamentals_input"
    SOURCE_DATA_READINESS = "source_data_readiness"
    GENERATED_FUNDAMENTAL_QUALITY = "generated_fundamental_quality"
    GENERATED_FUNDAMENTAL_ANALYSIS = "generated_fundamental_analysis"
    REPORTING_DISPLAY_INPUT = "reporting_display_input"


class FundamentalsNormalizationIssueCode(StrEnum):
    """Issue codes for metadata-only contract validation."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_READINESS_STATE = "invalid_readiness_state"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class FundamentalsNormalizationIssue:
    """Explicit contract issue without transformation, scoring, or decisions."""

    field_name: str
    issue_code: FundamentalsNormalizationIssueCode
    observed_value: object


RAW_SOURCE_DATASET_ROLES: tuple[FundamentalsNormalizationDatasetRole, ...] = (
    FundamentalsNormalizationDatasetRole.RAW_SOURCE_CAPTURE,
)

NORMALIZED_FUNDAMENTALS_DATASET_ROLES: tuple[
    FundamentalsNormalizationDatasetRole, ...
] = (
    FundamentalsNormalizationDatasetRole.NORMALIZED_FUNDAMENTALS_INPUT,
    FundamentalsNormalizationDatasetRole.SOURCE_DATA_READINESS,
)

GENERATED_FUNDAMENTALS_DATASET_ROLES: tuple[
    FundamentalsNormalizationDatasetRole, ...
] = (
    FundamentalsNormalizationDatasetRole.GENERATED_FUNDAMENTAL_QUALITY,
    FundamentalsNormalizationDatasetRole.GENERATED_FUNDAMENTAL_ANALYSIS,
)

REPORTING_DISPLAY_DATASET_ROLES: tuple[
    FundamentalsNormalizationDatasetRole, ...
] = (
    FundamentalsNormalizationDatasetRole.REPORTING_DISPLAY_INPUT,
)

REQUIRED_RAW_SOURCE_FIELDS: tuple[str, ...] = (
    "source_provider",
    "source_record_id",
    "ticker",
    "fiscal_period",
    "fiscal_year",
    "captured_at",
    "source_reference",
    "raw_payload_hash",
)

REQUIRED_NORMALIZED_FUNDAMENTALS_FIELDS: tuple[str, ...] = (
    "ticker",
    "fiscal_period",
    "fiscal_year",
    "metric_name",
    "metric_value",
    "metric_unit",
    "currency",
    "normalized_at",
    "source_provider",
    "source_reference",
    "source_record_identity",
)

REQUIRED_SOURCE_READINESS_FIELDS: tuple[str, ...] = (
    "ticker",
    "fiscal_period",
    "readiness_state",
    "source_data_status",
    "missing_fundamentals_count",
    "partial_data_count",
    "stale_data_count",
    "source_reference",
)

FORBIDDEN_NORMALIZED_FUNDAMENTALS_FIELDS: tuple[str, ...] = (
    "investment_quality",
    "investment_quality_score",
    "quality_score",
    "final_action",
    "allocation",
    "allocation_amount",
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
    "target_price",
    "threshold_price",
    "telegram_message",
    "reporting_line",
    "report_message",
)


def fundamentals_dataset_roles() -> tuple[FundamentalsNormalizationDatasetRole, ...]:
    """Return all dataset roles in the fundamentals normalization boundary."""

    return tuple(FundamentalsNormalizationDatasetRole)


def raw_source_dataset_roles() -> tuple[FundamentalsNormalizationDatasetRole, ...]:
    """Return roles that represent immutable raw source evidence."""

    return RAW_SOURCE_DATASET_ROLES


def normalized_fundamentals_dataset_roles() -> tuple[
    FundamentalsNormalizationDatasetRole, ...
]:
    """Return roles that represent program-ready normalized inputs."""

    return NORMALIZED_FUNDAMENTALS_DATASET_ROLES


def generated_fundamentals_dataset_roles() -> tuple[
    FundamentalsNormalizationDatasetRole, ...
]:
    """Return roles that represent generated fundamentals outputs."""

    return GENERATED_FUNDAMENTALS_DATASET_ROLES


def reporting_display_dataset_roles() -> tuple[
    FundamentalsNormalizationDatasetRole, ...
]:
    """Return roles that represent downstream communication inputs only."""

    return REPORTING_DISPLAY_DATASET_ROLES


def required_raw_source_fields() -> tuple[str, ...]:
    """Return required provenance fields for raw source capture records."""

    return REQUIRED_RAW_SOURCE_FIELDS


def required_normalized_fundamentals_fields() -> tuple[str, ...]:
    """Return required fields for normalized fundamentals metric records."""

    return REQUIRED_NORMALIZED_FUNDAMENTALS_FIELDS


def required_source_readiness_fields() -> tuple[str, ...]:
    """Return required fields for source-data readiness records."""

    return REQUIRED_SOURCE_READINESS_FIELDS


def forbidden_normalized_fundamentals_fields() -> tuple[str, ...]:
    """Return fields forbidden in normalized fundamentals input contracts."""

    return FORBIDDEN_NORMALIZED_FUNDAMENTALS_FIELDS


def validate_raw_source_shape(
    record: Mapping[str, object],
) -> tuple[FundamentalsNormalizationIssue, ...]:
    """Check raw source capture shape without reading files or providers."""

    return _validate_required_shape(record, REQUIRED_RAW_SOURCE_FIELDS)


def validate_normalized_fundamentals_shape(
    record: Mapping[str, object],
) -> tuple[FundamentalsNormalizationIssue, ...]:
    """Check normalized fundamentals shape without transformation logic."""

    return _validate_required_shape(
        record,
        REQUIRED_NORMALIZED_FUNDAMENTALS_FIELDS,
        include_forbidden_fields=True,
    )


def validate_source_readiness_shape(
    record: Mapping[str, object],
) -> tuple[FundamentalsNormalizationIssue, ...]:
    """Check source-readiness shape without quality or decision inference."""

    issues = _validate_required_shape(
        record,
        REQUIRED_SOURCE_READINESS_FIELDS,
        include_forbidden_fields=True,
    )

    if "readiness_state" in record and record.get("readiness_state") not in {
        state.value for state in SourceDataReadinessState
    }:
        issues += (
            FundamentalsNormalizationIssue(
                field_name="readiness_state",
                issue_code=FundamentalsNormalizationIssueCode.INVALID_READINESS_STATE,
                observed_value=record.get("readiness_state"),
            ),
        )

    return issues


def _validate_required_shape(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
    *,
    include_forbidden_fields: bool = False,
) -> tuple[FundamentalsNormalizationIssue, ...]:
    issues: list[FundamentalsNormalizationIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                FundamentalsNormalizationIssue(
                    field_name=field_name,
                    issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                FundamentalsNormalizationIssue(
                    field_name=field_name,
                    issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    if include_forbidden_fields:
        for field_name in FORBIDDEN_NORMALIZED_FUNDAMENTALS_FIELDS:
            if field_name in record:
                issues.append(
                    FundamentalsNormalizationIssue(
                        field_name=field_name,
                        issue_code=FundamentalsNormalizationIssueCode.FORBIDDEN_FIELD,
                        observed_value=record[field_name],
                    )
                )

    return tuple(issues)
