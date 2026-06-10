"""Contract metadata for v2 fundamentals and source-data datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Mapping, Sequence


class FundamentalDatasetRole(StrEnum):
    """Fundamentals dataset roles in the v2 data lifecycle."""

    RAW_SOURCE_CAPTURE = "raw_source_capture"
    NORMALIZED_SOURCE_READINESS = "normalized_source_readiness"
    NORMALIZED_FUNDAMENTAL_HISTORY = "normalized_fundamental_history"
    GENERATED_FUNDAMENTAL_CLASSIFICATION = "generated_fundamental_classification"
    GENERATED_FUNDAMENTAL_ANALYSIS = "generated_fundamental_analysis"
    LOCAL_ONLY_REVIEW = "local_only_review"


class SourceDataReadinessState(StrEnum):
    """Source-data readiness states, not investment-quality states."""

    AVAILABLE = "available"
    MISSING = "missing"
    SOURCE_MISSING = "source_missing"
    ROW_MISSING = "row_missing"
    PARTIAL = "partial"
    STALE = "stale"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    REVIEW_REQUIRED = "review_required"


class FundamentalContractIssueCode(StrEnum):
    """Metadata-only issue codes for fundamentals contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    INVALID_READINESS_STATE = "invalid_readiness_state"
    INVALID_FISCAL_YEAR = "invalid_fiscal_year"
    INVALID_FISCAL_PERIOD = "invalid_fiscal_period"
    INVALID_DATE_VALUE = "invalid_date_value"
    DUPLICATE_HISTORY_KEY = "duplicate_history_key"
    FORBIDDEN_FIELD = "forbidden_field"


@dataclass(frozen=True)
class FundamentalContractIssue:
    """Explicit fundamentals/source-data issue without scoring or decisions."""

    field_name: str
    issue_code: FundamentalContractIssueCode
    observed_value: object


SOURCE_READINESS_IDENTITY_FIELDS: tuple[str, ...] = ("source_record_id",)

FUNDAMENTAL_HISTORY_IDENTITY_FIELDS: tuple[str, ...] = (
    "ticker",
    "fiscal_year",
    "fiscal_period",
)

REQUIRED_SOURCE_READINESS_FIELDS: tuple[str, ...] = (
    "source_record_id",
    "symbol",
    "source_name",
    "metric_name",
    "metric_value",
    "metric_unit",
    "as_of_date",
    "readiness_state",
    "missing_value_policy",
    "review_required_reason",
)

REQUIRED_FUNDAMENTAL_HISTORY_FIELDS: tuple[str, ...] = (
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "report_date",
    "currency",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
    "source_name",
    "source_reference",
    "source_freshness_date",
    "extraction_date",
    "notes",
)

FUNDAMENTAL_HISTORY_NUMERIC_FIELDS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
)

FUNDAMENTAL_HISTORY_DATE_FIELDS: tuple[str, ...] = (
    "period_end_date",
    "report_date",
    "source_freshness_date",
    "extraction_date",
)

SUPPORTED_FUNDAMENTAL_HISTORY_PERIODS: tuple[str, ...] = (
    "FY",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "TTM",
)

MIN_FUNDAMENTAL_HISTORY_FISCAL_YEAR = 1900
MAX_FUNDAMENTAL_HISTORY_FISCAL_YEAR = 2200

OPTIONAL_EMPTY_VALUE_FIELDS: tuple[str, ...] = (
    "notes",
    "review_required_reason",
)

SOURCE_DATASET_ROLES: tuple[FundamentalDatasetRole, ...] = (
    FundamentalDatasetRole.RAW_SOURCE_CAPTURE,
    FundamentalDatasetRole.NORMALIZED_SOURCE_READINESS,
    FundamentalDatasetRole.NORMALIZED_FUNDAMENTAL_HISTORY,
)

GENERATED_DATASET_ROLES: tuple[FundamentalDatasetRole, ...] = (
    FundamentalDatasetRole.GENERATED_FUNDAMENTAL_CLASSIFICATION,
    FundamentalDatasetRole.GENERATED_FUNDAMENTAL_ANALYSIS,
)

FORBIDDEN_FUNDAMENTAL_UPSTREAM_FIELDS: tuple[str, ...] = (
    "final_action",
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "quality_score",
    "urgency",
    "conviction",
    "execution_instruction",
    "tradeability",
    "tradeable_setup",
    "buy_candidate",
    "sell_candidate",
    "investment_quality",
)


def required_source_readiness_fields() -> tuple[str, ...]:
    """Return required fields for normalized source-readiness records."""

    return REQUIRED_SOURCE_READINESS_FIELDS


def required_fundamental_history_fields() -> tuple[str, ...]:
    """Return required fields for normalized fundamentals history records."""

    return REQUIRED_FUNDAMENTAL_HISTORY_FIELDS


def fundamental_history_numeric_fields() -> tuple[str, ...]:
    """Return numeric fields used by normalized fundamentals history records."""

    return FUNDAMENTAL_HISTORY_NUMERIC_FIELDS


def fundamental_history_date_fields() -> tuple[str, ...]:
    """Return date fields used by normalized fundamentals history records."""

    return FUNDAMENTAL_HISTORY_DATE_FIELDS


def supported_fundamental_history_periods() -> tuple[str, ...]:
    """Return supported fiscal periods for normalized fundamentals history."""

    return SUPPORTED_FUNDAMENTAL_HISTORY_PERIODS


def source_readiness_identity_fields() -> tuple[str, ...]:
    """Return identity fields for normalized source-readiness records."""

    return SOURCE_READINESS_IDENTITY_FIELDS


def fundamental_history_identity_fields() -> tuple[str, ...]:
    """Return identity fields for normalized fundamentals history records."""

    return FUNDAMENTAL_HISTORY_IDENTITY_FIELDS


def source_dataset_roles() -> tuple[FundamentalDatasetRole, ...]:
    """Return fundamentals/source-data roles that may represent inputs."""

    return SOURCE_DATASET_ROLES


def generated_dataset_roles() -> tuple[FundamentalDatasetRole, ...]:
    """Return fundamentals roles that are generated outputs only."""

    return GENERATED_DATASET_ROLES


def forbidden_fundamental_upstream_fields() -> tuple[str, ...]:
    """Return fields fundamentals/source-data contracts must not accept."""

    return FORBIDDEN_FUNDAMENTAL_UPSTREAM_FIELDS


def validate_source_readiness_shape(
    record: Mapping[str, object],
) -> tuple[FundamentalContractIssue, ...]:
    """Check source-readiness shape without file IO or source integration."""

    issues = _validate_required_shape(record, REQUIRED_SOURCE_READINESS_FIELDS)

    if "readiness_state" in record and record.get("readiness_state") not in {
        state.value for state in SourceDataReadinessState
    }:
        issues += (
            FundamentalContractIssue(
                field_name="readiness_state",
                issue_code=FundamentalContractIssueCode.INVALID_READINESS_STATE,
                observed_value=record.get("readiness_state"),
            ),
        )

    return issues


def validate_fundamental_history_shape(
    record: Mapping[str, object],
) -> tuple[FundamentalContractIssue, ...]:
    """Check normalized fundamentals history shape without metric calculation."""

    issues = _validate_required_shape(record, REQUIRED_FUNDAMENTAL_HISTORY_FIELDS)

    for field_name in FUNDAMENTAL_HISTORY_NUMERIC_FIELDS:
        if field_name not in record:
            continue
        value = record[field_name]
        if value is None or value == "":
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            issues += (
                FundamentalContractIssue(
                    field_name=field_name,
                    issue_code=FundamentalContractIssueCode.INVALID_NUMERIC_VALUE,
                    observed_value=value,
                ),
            )

    return issues


def validate_fundamental_history_contract_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[FundamentalContractIssue, ...]:
    """Validate normalized fundamentals history records as a contract batch.

    This function is pure and side-effect-free. It performs no file I/O, source
    calls, metric calculations, scoring, ranking, or investment recommendations.
    """

    issues: list[FundamentalContractIssue] = []
    duplicate_counts: dict[tuple[str, int, str], int] = {}

    for record in records:
        issues.extend(validate_fundamental_history_shape(record))
        issues.extend(_validate_fiscal_year(record))
        issues.extend(_validate_fiscal_period(record))
        issues.extend(_validate_history_dates(record))

        key = _history_key(record)
        if key is not None:
            duplicate_counts[key] = duplicate_counts.get(key, 0) + 1

    for key, count in duplicate_counts.items():
        if count > 1:
            ticker, fiscal_year, fiscal_period = key
            issues.append(
                FundamentalContractIssue(
                    field_name="ticker,fiscal_year,fiscal_period",
                    issue_code=FundamentalContractIssueCode.DUPLICATE_HISTORY_KEY,
                    observed_value=f"{ticker}|{fiscal_year}|{fiscal_period}",
                )
            )

    return tuple(issues)


def _validate_required_shape(
    record: Mapping[str, object],
    required_fields: tuple[str, ...],
) -> tuple[FundamentalContractIssue, ...]:
    issues: list[FundamentalContractIssue] = []

    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                FundamentalContractIssue(
                    field_name=field_name,
                    issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or (
            value == "" and field_name not in OPTIONAL_EMPTY_VALUE_FIELDS
        ):
            issues.append(
                FundamentalContractIssue(
                    field_name=field_name,
                    issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    for field_name in FORBIDDEN_FUNDAMENTAL_UPSTREAM_FIELDS:
        if field_name in record:
            issues.append(
                FundamentalContractIssue(
                    field_name=field_name,
                    issue_code=FundamentalContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)


def _validate_fiscal_year(
    record: Mapping[str, object],
) -> tuple[FundamentalContractIssue, ...]:
    value = record.get("fiscal_year")

    if value is None or value == "":
        return ()

    fiscal_year = _to_int(value)

    if (
        fiscal_year is None
        or fiscal_year < MIN_FUNDAMENTAL_HISTORY_FISCAL_YEAR
        or fiscal_year > MAX_FUNDAMENTAL_HISTORY_FISCAL_YEAR
    ):
        return (
            FundamentalContractIssue(
                field_name="fiscal_year",
                issue_code=FundamentalContractIssueCode.INVALID_FISCAL_YEAR,
                observed_value=value,
            ),
        )

    return ()


def _validate_fiscal_period(
    record: Mapping[str, object],
) -> tuple[FundamentalContractIssue, ...]:
    value = record.get("fiscal_period")

    if value is None or value == "":
        return ()

    normalized_value = _normalized_text(value)

    if normalized_value not in SUPPORTED_FUNDAMENTAL_HISTORY_PERIODS:
        return (
            FundamentalContractIssue(
                field_name="fiscal_period",
                issue_code=FundamentalContractIssueCode.INVALID_FISCAL_PERIOD,
                observed_value=value,
            ),
        )

    return ()


def _validate_history_dates(
    record: Mapping[str, object],
) -> tuple[FundamentalContractIssue, ...]:
    issues: list[FundamentalContractIssue] = []

    for field_name in FUNDAMENTAL_HISTORY_DATE_FIELDS:
        value = record.get(field_name)
        if value is None or value == "":
            continue

        if _to_iso_date(value) is None:
            issues.append(
                FundamentalContractIssue(
                    field_name=field_name,
                    issue_code=FundamentalContractIssueCode.INVALID_DATE_VALUE,
                    observed_value=value,
                )
            )

    return tuple(issues)


def _history_key(record: Mapping[str, object]) -> tuple[str, int, str] | None:
    ticker = _normalized_text(record.get("ticker"))
    fiscal_year = _to_int(record.get("fiscal_year"))
    fiscal_period = _normalized_text(record.get("fiscal_period"))

    if ticker == "" or fiscal_year is None or fiscal_period == "":
        return None

    return (ticker, fiscal_year, fiscal_period)


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_iso_date(value: object) -> date | None:
    if value is None or value == "":
        return None

    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _normalized_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()