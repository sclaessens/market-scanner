"""Pure canonical fundamentals metrics contracts.

This module contains deterministic, side-effect-free contract helpers for
fundamentals history-to-metrics calculations.

It does not read files, write files, call providers, or create investment
decisions.
"""

from __future__ import annotations

from typing import Mapping, Sequence


FUNDAMENTAL_METRICS_IDENTITY_FIELDS: tuple[str, ...] = (
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "report_date",
    "currency",
    "source_name",
    "source_reference",
    "source_freshness_date",
    "extraction_date",
)

FUNDAMENTAL_METRICS_INPUT_FIELDS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
)

FUNDAMENTAL_DERIVED_METRIC_FIELDS: tuple[str, ...] = (
    "gross_margin",
    "operating_margin",
    "net_margin",
    "free_cash_flow_margin",
    "debt_to_equity",
    "return_on_equity",
    "revenue_yoy_growth",
    "eps_yoy_growth",
    "free_cash_flow_yoy_growth",
)

FUNDAMENTAL_METRICS_HELPER_FIELDS: tuple[str, ...] = (
    "metric_status",
    "metric_missing_inputs",
    "metric_warnings",
)

FORBIDDEN_FUNDAMENTAL_METRICS_AUTHORITY_FIELDS: tuple[str, ...] = (
    "buy",
    "sell",
    "action",
    "final_action",
    "decision",
    "allocation",
    "allocation_amount",
    "position_size",
    "urgency",
    "conviction",
    "tradeability",
    "eligible",
    "eligibility",
    "ranking",
    "rank",
    "score",
    "priority",
    "entry",
    "stop",
    "target",
    "target_price",
    "threshold_price",
    "recommendation",
    "investment_quality",
    "investment_quality_score",
    "quality_score",
)


def fundamental_metrics_identity_fields() -> tuple[str, ...]:
    """Return identity and provenance fields preserved in metrics output."""

    return FUNDAMENTAL_METRICS_IDENTITY_FIELDS


def fundamental_metrics_input_fields() -> tuple[str, ...]:
    """Return numeric source fields used by canonical metrics calculations."""

    return FUNDAMENTAL_METRICS_INPUT_FIELDS


def fundamental_derived_metric_fields() -> tuple[str, ...]:
    """Return deterministic derived metric fields."""

    return FUNDAMENTAL_DERIVED_METRIC_FIELDS


def fundamental_metrics_helper_fields() -> tuple[str, ...]:
    """Return helper/status fields used for missing-data transparency."""

    return FUNDAMENTAL_METRICS_HELPER_FIELDS


def forbidden_fundamental_metrics_authority_fields() -> tuple[str, ...]:
    """Return fields that metrics contracts must never introduce."""

    return FORBIDDEN_FUNDAMENTAL_METRICS_AUTHORITY_FIELDS


def build_fundamental_metrics_contract_records(
    history_records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Build pure, deterministic fundamentals metrics records.

    The function is intentionally in-memory only. It separates calculations from
    persistence and does not perform file I/O, provider calls, scoring, ranking,
    or investment recommendations.
    """

    prior_year_lookup = {
        _history_key(record): record
        for record in history_records
        if _history_key(record) is not None
    }

    metric_records: list[dict[str, object]] = []

    for record in history_records:
        metric_missing_inputs: list[str] = []
        metric_warnings: list[str] = []

        output: dict[str, object] = {
            field_name: record.get(field_name, "")
            for field_name in FUNDAMENTAL_METRICS_IDENTITY_FIELDS
        }

        output["gross_margin"] = _ratio(
            "gross_margin",
            record,
            numerator_name="gross_profit",
            denominator_name="revenue",
            issues=metric_missing_inputs,
        )
        output["operating_margin"] = _ratio(
            "operating_margin",
            record,
            numerator_name="operating_income",
            denominator_name="revenue",
            issues=metric_missing_inputs,
        )
        output["net_margin"] = _ratio(
            "net_margin",
            record,
            numerator_name="net_income",
            denominator_name="revenue",
            issues=metric_missing_inputs,
        )
        output["free_cash_flow_margin"] = _ratio(
            "free_cash_flow_margin",
            record,
            numerator_name="free_cash_flow",
            denominator_name="revenue",
            issues=metric_missing_inputs,
        )
        output["debt_to_equity"] = _ratio(
            "debt_to_equity",
            record,
            numerator_name="total_debt",
            denominator_name="total_equity",
            issues=metric_missing_inputs,
        )
        output["return_on_equity"] = _ratio(
            "return_on_equity",
            record,
            numerator_name="net_income",
            denominator_name="total_equity",
            issues=metric_missing_inputs,
        )

        prior_record = _prior_year_record(record, prior_year_lookup)

        output["revenue_yoy_growth"] = _yoy_growth(
            "revenue_yoy_growth",
            record,
            prior_record,
            field_name="revenue",
            prior_field_label="prior_revenue",
            warnings=metric_warnings,
        )
        output["eps_yoy_growth"] = _yoy_growth(
            "eps_yoy_growth",
            record,
            prior_record,
            field_name="diluted_eps",
            prior_field_label="prior_diluted_eps",
            warnings=metric_warnings,
        )
        output["free_cash_flow_yoy_growth"] = _yoy_growth(
            "free_cash_flow_yoy_growth",
            record,
            prior_record,
            field_name="free_cash_flow",
            prior_field_label="prior_free_cash_flow",
            warnings=metric_warnings,
        )

        output["metric_status"] = (
            "complete"
            if all(output[field_name] is not None for field_name in FUNDAMENTAL_DERIVED_METRIC_FIELDS)
            else "partial"
        )
        output["metric_missing_inputs"] = ";".join(metric_missing_inputs)
        output["metric_warnings"] = ";".join(metric_warnings)

        metric_records.append(output)

    return tuple(metric_records)


def _ratio(
    metric_name: str,
    record: Mapping[str, object],
    *,
    numerator_name: str,
    denominator_name: str,
    issues: list[str],
) -> float | None:
    numerator = _to_float(record.get(numerator_name))
    denominator = _to_float(record.get(denominator_name))

    missing_inputs: list[str] = []

    if numerator is None:
        missing_inputs.append(numerator_name)
    if denominator is None:
        missing_inputs.append(denominator_name)

    if missing_inputs:
        issues.append(f"{metric_name}:missing:{'|'.join(missing_inputs)}")
        return None

    if denominator == 0:
        issues.append(f"{metric_name}:zero_denominator:{denominator_name}")
        return None

    return numerator / denominator


def _yoy_growth(
    metric_name: str,
    record: Mapping[str, object],
    prior_record: Mapping[str, object] | None,
    *,
    field_name: str,
    prior_field_label: str,
    warnings: list[str],
) -> float | None:
    if prior_record is None:
        _append_once(warnings, "yoy_growth:missing_prior_year")
        return None

    current_value = _to_float(record.get(field_name))
    prior_value = _to_float(prior_record.get(field_name))

    if current_value is None:
        warnings.append(f"{metric_name}:missing:{field_name}")
        return None

    if prior_value is None:
        warnings.append(f"{metric_name}:missing:{prior_field_label}")
        return None

    if prior_value == 0:
        warnings.append(f"{metric_name}:zero_denominator:{prior_field_label}")
        return None

    return (current_value - prior_value) / abs(prior_value)


def _prior_year_record(
    record: Mapping[str, object],
    prior_year_lookup: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> Mapping[str, object] | None:
    fiscal_year = _to_int(record.get("fiscal_year"))
    if fiscal_year is None:
        return None

    ticker = _normalized_text(record.get("ticker"))
    fiscal_period = _normalized_text(record.get("fiscal_period"))

    if ticker == "" or fiscal_period == "":
        return None

    return prior_year_lookup.get((ticker, fiscal_year - 1, fiscal_period))


def _history_key(record: Mapping[str, object]) -> tuple[str, int, str] | None:
    fiscal_year = _to_int(record.get("fiscal_year"))
    if fiscal_year is None:
        return None

    ticker = _normalized_text(record.get("ticker"))
    fiscal_period = _normalized_text(record.get("fiscal_period"))

    if ticker == "" or fiscal_period == "":
        return None

    return (ticker, fiscal_year, fiscal_period)


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalized_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)