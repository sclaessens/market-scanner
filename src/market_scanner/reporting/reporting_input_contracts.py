"""Reporting input aggregation contract metadata for v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


AGGREGATION_CONTRACT_VERSION = "REPORTING_INPUT_AGGREGATION_V1"


class ReportingInputRole(StrEnum):
    """Reporting input roles allowed at the aggregation boundary."""

    PORTFOLIO_DISPLAY_INPUT = "portfolio_display_input"
    CANDIDATE_DISPLAY_INPUT = "candidate_display_input"
    DECISION_STATUS_INPUT = "decision_status_input"
    SOURCE_DATA_STATUS_INPUT = "source_data_status_input"
    DATA_WARNING_INPUT = "data_warning_input"
    TELEGRAM_RENDERER_INPUT = "telegram_renderer_input"


class ReportingInputBoundaryRole(StrEnum):
    """Lifecycle boundary roles for reporting input aggregation."""

    UPSTREAM_SOURCE_INPUT = "upstream_source_input"
    DERIVED_DISPLAY_INPUT = "derived_display_input"
    GENERATED_REPORT_INPUT = "generated_report_input"
    RENDERER_INPUT = "renderer_input"


class ReportingInputContractIssueCode(StrEnum):
    """Metadata-only issue codes for reporting input contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    FORBIDDEN_FIELD = "forbidden_field"
    INVALID_SOURCE_ROLE = "invalid_source_role"


@dataclass(frozen=True)
class ReportingInputContractIssue:
    """Explicit reporting input issue without aggregation runtime behavior."""

    field_name: str
    issue_code: ReportingInputContractIssueCode
    observed_value: object


UPSTREAM_INPUT_ROLES: tuple[ReportingInputRole, ...] = (
    ReportingInputRole.PORTFOLIO_DISPLAY_INPUT,
    ReportingInputRole.CANDIDATE_DISPLAY_INPUT,
    ReportingInputRole.DECISION_STATUS_INPUT,
    ReportingInputRole.SOURCE_DATA_STATUS_INPUT,
    ReportingInputRole.DATA_WARNING_INPUT,
)

RENDERER_INPUT_ROLES: tuple[ReportingInputRole, ...] = (
    ReportingInputRole.TELEGRAM_RENDERER_INPUT,
)

REQUIRED_AGGREGATION_TRACE_FIELDS: tuple[str, ...] = (
    "source_role",
    "source_reference",
    "aggregation_contract_version",
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

REQUIRED_CANDIDATE_DISPLAY_INPUT_FIELDS: tuple[str, ...] = (
    "ticker",
    "candidate_group",
    "threshold_price_display",
    "threshold_direction",
    "action_status",
    "currency",
    "source_reference",
)

REQUIRED_DECISION_STATUS_INPUT_FIELDS: tuple[str, ...] = (
    "row_id",
    "action_status",
    "decision_rationale",
    "source_reference",
)

REQUIRED_SOURCE_DATA_STATUS_INPUT_FIELDS: tuple[str, ...] = (
    "data_status",
    "review_reason",
    "source_reference",
)

OPTIONAL_SOURCE_DATA_STATUS_INPUT_FIELDS: tuple[str, ...] = (
    "missing_count",
    "partial_count",
    "stale_count",
)

REQUIRED_DATA_WARNING_INPUT_FIELDS: tuple[str, ...] = (
    "warning_type",
    "warning_text",
    "source_reference",
)

FORBIDDEN_REPORTING_AGGREGATION_FIELDS: tuple[str, ...] = (
    "source_of_truth",
    "source_of_truth_overwrite",
    "portfolio_source_overwrite",
    "telegram_message",
    "report_artifact_path",
    "allocation",
    "allocation_amount",
    "allocation_instruction",
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
    "target_price_calculation_authority",
    "buy_threshold_calculation",
    "breakout_threshold_calculation",
    "threshold_calculation_authority",
    "profit_loss_calculation",
    "current_price_fetch",
    "decision_override",
)


def reporting_input_roles() -> tuple[ReportingInputRole, ...]:
    """Return reporting input roles allowed at the aggregation boundary."""

    return tuple(ReportingInputRole)


def upstream_input_roles() -> tuple[ReportingInputRole, ...]:
    """Return upstream roles that may supply reporting display inputs."""

    return UPSTREAM_INPUT_ROLES


def telegram_renderer_input_roles() -> tuple[ReportingInputRole, ...]:
    """Return downstream renderer input roles."""

    return RENDERER_INPUT_ROLES


def reporting_input_boundary_roles() -> tuple[ReportingInputBoundaryRole, ...]:
    """Return lifecycle boundary roles for reporting input aggregation."""

    return tuple(ReportingInputBoundaryRole)


def required_portfolio_display_input_fields() -> tuple[str, ...]:
    """Return required portfolio display input fields."""

    return REQUIRED_PORTFOLIO_DISPLAY_INPUT_FIELDS


def required_candidate_display_input_fields() -> tuple[str, ...]:
    """Return required candidate display input fields."""

    return REQUIRED_CANDIDATE_DISPLAY_INPUT_FIELDS


def required_decision_status_input_fields() -> tuple[str, ...]:
    """Return required decision status input fields."""

    return REQUIRED_DECISION_STATUS_INPUT_FIELDS


def required_source_data_status_input_fields() -> tuple[str, ...]:
    """Return required source-data status input fields."""

    return REQUIRED_SOURCE_DATA_STATUS_INPUT_FIELDS


def required_data_warning_input_fields() -> tuple[str, ...]:
    """Return required data warning input fields."""

    return REQUIRED_DATA_WARNING_INPUT_FIELDS


def required_aggregation_trace_fields() -> tuple[str, ...]:
    """Return required traceability fields for every reporting input."""

    return REQUIRED_AGGREGATION_TRACE_FIELDS


def forbidden_reporting_aggregation_fields() -> tuple[str, ...]:
    """Return fields aggregation contracts must not accept as authority."""

    return FORBIDDEN_REPORTING_AGGREGATION_FIELDS


def validate_portfolio_display_input_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check portfolio display input shape without aggregation behavior."""

    return _validate_shape(record, REQUIRED_PORTFOLIO_DISPLAY_INPUT_FIELDS)


def validate_candidate_display_input_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check candidate display input shape without threshold calculation."""

    return _validate_shape(record, REQUIRED_CANDIDATE_DISPLAY_INPUT_FIELDS)


def validate_decision_status_input_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check Decision Engine status input shape without creating decisions."""

    return _validate_shape(record, REQUIRED_DECISION_STATUS_INPUT_FIELDS)


def validate_source_data_status_input_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check source-data status input shape without quality inference."""

    return _validate_shape(record, REQUIRED_SOURCE_DATA_STATUS_INPUT_FIELDS)


def validate_data_warning_input_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check data warning input shape without reporting output behavior."""

    return _validate_shape(record, REQUIRED_DATA_WARNING_INPUT_FIELDS)


def validate_reporting_input_trace_shape(
    record: Mapping[str, object],
) -> tuple[ReportingInputContractIssue, ...]:
    """Check reporting input traceability fields."""

    issues = _validate_required_fields(record, REQUIRED_AGGREGATION_TRACE_FIELDS)

    source_role = record.get("source_role")
    if source_role not in {role.value for role in ReportingInputRole}:
        issues.append(
            ReportingInputContractIssue(
                field_name="source_role",
                issue_code=ReportingInputContractIssueCode.INVALID_SOURCE_ROLE,
                observed_value=source_role,
            )
        )

    return tuple(issues)


def _validate_shape(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
) -> tuple[ReportingInputContractIssue, ...]:
    issues = _validate_required_fields(record, required_fields)
    issues.extend(_validate_forbidden_fields(record))
    return tuple(issues)


def _validate_required_fields(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
) -> list[ReportingInputContractIssue]:
    issues: list[ReportingInputContractIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                ReportingInputContractIssue(
                    field_name=field_name,
                    issue_code=ReportingInputContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "":
            issues.append(
                ReportingInputContractIssue(
                    field_name=field_name,
                    issue_code=ReportingInputContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    return issues


def _validate_forbidden_fields(
    record: Mapping[str, object],
) -> list[ReportingInputContractIssue]:
    issues: list[ReportingInputContractIssue] = []

    for field_name in FORBIDDEN_REPORTING_AGGREGATION_FIELDS:
        if field_name in record:
            issues.append(
                ReportingInputContractIssue(
                    field_name=field_name,
                    issue_code=ReportingInputContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return issues
