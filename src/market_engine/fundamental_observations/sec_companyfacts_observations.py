from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.source_context.sec_companyfacts_context import (
    SecCompanyFactsContextField,
    SecCompanyFactsContextFieldState,
    SecCompanyFactsContextState,
    SecCompanyFactsSourceContext,
)


SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION = (
    "sec-companyfacts-fundamental-observations-v1"
)
SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_ROOT = Path(
    "data/market_engine/fundamental_observations"
)

NON_DECISION_BOUNDARY = (
    "Fundamental Observations are source-grounded, non-decision outputs only."
)


class SecCompanyFactsFundamentalObservationCategory(str, Enum):
    SOURCE_CONTEXT_AVAILABILITY = "SOURCE_CONTEXT_AVAILABILITY"
    REVENUE_SOURCE_PRESENCE = "REVENUE_SOURCE_PRESENCE"
    NET_INCOME_SOURCE_VALUE = "NET_INCOME_SOURCE_VALUE"
    OPERATING_CASH_FLOW_SOURCE_VALUE = "OPERATING_CASH_FLOW_SOURCE_VALUE"
    CAPEX_SOURCE_PRESENCE = "CAPEX_SOURCE_PRESENCE"
    CASH_GENERATION_SOURCE_COMPLETENESS = "CASH_GENERATION_SOURCE_COMPLETENESS"
    DATA_LIMITATION = "DATA_LIMITATION"


class SecCompanyFactsFundamentalObservationState(str, Enum):
    PRESENT = "PRESENT"
    MISSING_DATA = "MISSING_DATA"
    POSITIVE_SOURCE_VALUE = "POSITIVE_SOURCE_VALUE"
    NEGATIVE_SOURCE_VALUE = "NEGATIVE_SOURCE_VALUE"
    ZERO_SOURCE_VALUE = "ZERO_SOURCE_VALUE"
    NOT_ASSESSED = "NOT_ASSESSED"
    SOURCE_LIMITED = "SOURCE_LIMITED"


@dataclass(frozen=True)
class SecCompanyFactsFundamentalObservation:
    ticker: str
    cik: str
    provider_name: str
    category: SecCompanyFactsFundamentalObservationCategory
    state: SecCompanyFactsFundamentalObservationState
    message: str
    source_context_state: str
    canonical_fields: tuple[str, ...]
    source_values: dict[str, Any | None]
    source_references: dict[str, dict[str, Any | None]]
    missing_source_fields: tuple[str, ...]
    observation_format_version: str = SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION
    non_decision_boundary: str = NON_DECISION_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsFundamentalObservationSet:
    ticker: str
    cik: str
    provider_name: str
    observation_format_version: str
    source_context_format_version: str
    source_context_state: str
    source_context_reference: str | None
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    observations: tuple[SecCompanyFactsFundamentalObservation, ...]
    non_decision_boundary: str = NON_DECISION_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_fundamental_observations(
    source_context: SecCompanyFactsSourceContext,
) -> SecCompanyFactsFundamentalObservationSet:
    observations = [
        _source_context_availability(source_context),
        _field_presence_observation(
            source_context,
            canonical_field="revenue",
            category=SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE,
            present_message="Revenue source field is present for the selected SEC period.",
            missing_message="Revenue source field is missing for the selected SEC period.",
        ),
        _field_value_observation(
            source_context,
            canonical_field="net_income",
            category=SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE,
            positive_message="Net income source value is positive for the selected SEC period.",
            negative_message="Net income source value is negative for the selected SEC period.",
            zero_message="Net income source value is zero for the selected SEC period.",
            missing_message="Net income source field is missing for the selected SEC period.",
        ),
        _field_value_observation(
            source_context,
            canonical_field="operating_cash_flow",
            category=SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE,
            positive_message="Operating cash flow source value is positive for the selected SEC period.",
            negative_message="Operating cash flow source value is negative for the selected SEC period.",
            zero_message="Operating cash flow source value is zero for the selected SEC period.",
            missing_message="Operating cash flow source field is missing for the selected SEC period.",
        ),
        _field_presence_observation(
            source_context,
            canonical_field="capital_expenditures",
            category=SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE,
            present_message="Capital expenditures source field is present for the selected SEC period.",
            missing_message="Capital expenditures source field is missing for the selected SEC period.",
        ),
        _cash_generation_source_completeness(source_context),
    ]

    data_limitation = _data_limitation(source_context)
    if data_limitation is not None:
        observations.append(data_limitation)

    return SecCompanyFactsFundamentalObservationSet(
        ticker=source_context.ticker,
        cik=source_context.cik,
        provider_name=source_context.provider_name,
        observation_format_version=SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
        source_context_format_version=source_context.context_format_version,
        source_context_state=source_context.source_context_state.value,
        source_context_reference=source_context.source_refresh_snapshot_path,
        source_refresh_snapshot_id=source_context.source_refresh_snapshot_id,
        source_refresh_fetched_at=source_context.source_refresh_fetched_at,
        source_refresh_payload_format_version=source_context.source_refresh_payload_format_version,
        observations=tuple(observations),
    )


def persist_sec_companyfacts_fundamental_observations(
    observation_set: SecCompanyFactsFundamentalObservationSet,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_ROOT
    observation_dir = root / run_id / observation_set.ticker
    observation_dir.mkdir(parents=True, exist_ok=True)
    observation_path = observation_dir / "fundamental_observations.json"

    if observation_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Fundamental Observations: "
            f"{observation_path}"
        )

    observation_path.write_text(
        json.dumps(_to_jsonable(observation_set), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return observation_path


def _source_context_availability(
    source_context: SecCompanyFactsSourceContext,
) -> SecCompanyFactsFundamentalObservation:
    source_context_state = source_context.source_context_state

    if source_context_state == SecCompanyFactsContextState.AVAILABLE:
        state = SecCompanyFactsFundamentalObservationState.PRESENT
        message = "All required SEC CompanyFacts Source Context fields are available."
    elif source_context_state == SecCompanyFactsContextState.PARTIAL:
        state = SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED
        message = "One or more required SEC CompanyFacts Source Context fields are missing."
    elif source_context_state == SecCompanyFactsContextState.MISSING:
        state = SecCompanyFactsFundamentalObservationState.NOT_ASSESSED
        message = "SEC CompanyFacts Source Context is missing for this ticker."
    else:
        state = SecCompanyFactsFundamentalObservationState.NOT_ASSESSED
        message = f"SEC CompanyFacts Source Context is {source_context_state.value} for this ticker."

    return _observation(
        source_context,
        category=SecCompanyFactsFundamentalObservationCategory.SOURCE_CONTEXT_AVAILABILITY,
        state=state,
        message=message,
        canonical_fields=tuple(source_context.canonical_fields),
        missing_source_fields=source_context.missing_canonical_fields,
    )


def _field_presence_observation(
    source_context: SecCompanyFactsSourceContext,
    *,
    canonical_field: str,
    category: SecCompanyFactsFundamentalObservationCategory,
    present_message: str,
    missing_message: str,
) -> SecCompanyFactsFundamentalObservation:
    context_field = source_context.fields[canonical_field]
    if context_field.state == SecCompanyFactsContextFieldState.PRESENT:
        return _observation(
            source_context,
            category=category,
            state=SecCompanyFactsFundamentalObservationState.PRESENT,
            message=present_message,
            canonical_fields=(canonical_field,),
        )

    return _observation(
        source_context,
        category=category,
        state=SecCompanyFactsFundamentalObservationState.MISSING_DATA,
        message=missing_message,
        canonical_fields=(canonical_field,),
        missing_source_fields=(canonical_field,),
    )


def _field_value_observation(
    source_context: SecCompanyFactsSourceContext,
    *,
    canonical_field: str,
    category: SecCompanyFactsFundamentalObservationCategory,
    positive_message: str,
    negative_message: str,
    zero_message: str,
    missing_message: str,
) -> SecCompanyFactsFundamentalObservation:
    context_field = source_context.fields[canonical_field]
    if context_field.state != SecCompanyFactsContextFieldState.PRESENT:
        return _observation(
            source_context,
            category=category,
            state=SecCompanyFactsFundamentalObservationState.MISSING_DATA,
            message=missing_message,
            canonical_fields=(canonical_field,),
            missing_source_fields=(canonical_field,),
        )

    raw_value = context_field.raw_value
    if raw_value is None:
        return _observation(
            source_context,
            category=category,
            state=SecCompanyFactsFundamentalObservationState.MISSING_DATA,
            message=missing_message,
            canonical_fields=(canonical_field,),
            missing_source_fields=(canonical_field,),
        )

    if raw_value > 0:
        state = SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
        message = positive_message
    elif raw_value < 0:
        state = SecCompanyFactsFundamentalObservationState.NEGATIVE_SOURCE_VALUE
        message = negative_message
    else:
        state = SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE
        message = zero_message

    return _observation(
        source_context,
        category=category,
        state=state,
        message=message,
        canonical_fields=(canonical_field,),
    )


def _cash_generation_source_completeness(
    source_context: SecCompanyFactsSourceContext,
) -> SecCompanyFactsFundamentalObservation:
    cash_generation_fields = ("operating_cash_flow", "capital_expenditures")
    missing_fields = tuple(
        field_name
        for field_name in cash_generation_fields
        if source_context.field_states[field_name] != SecCompanyFactsContextFieldState.PRESENT
    )

    if not missing_fields:
        return _observation(
            source_context,
            category=SecCompanyFactsFundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
            state=SecCompanyFactsFundamentalObservationState.PRESENT,
            message="Operating cash flow and capital expenditures source fields are both present.",
            canonical_fields=cash_generation_fields,
        )

    return _observation(
        source_context,
        category=SecCompanyFactsFundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
        state=SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED,
        message="One or more cash-generation source fields are missing.",
        canonical_fields=cash_generation_fields,
        missing_source_fields=missing_fields,
    )


def _data_limitation(
    source_context: SecCompanyFactsSourceContext,
) -> SecCompanyFactsFundamentalObservation | None:
    if source_context.source_context_state == SecCompanyFactsContextState.AVAILABLE:
        return None

    if source_context.source_context_state in {
        SecCompanyFactsContextState.PARTIAL,
        SecCompanyFactsContextState.MISSING,
    }:
        message = "Fundamental observation is limited because required source fields are missing."
        state = SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED
    else:
        message = "Fundamental observation is not assessed because Source Context is unavailable."
        state = SecCompanyFactsFundamentalObservationState.NOT_ASSESSED

    return _observation(
        source_context,
        category=SecCompanyFactsFundamentalObservationCategory.DATA_LIMITATION,
        state=state,
        message=message,
        canonical_fields=tuple(source_context.canonical_fields),
        missing_source_fields=source_context.missing_canonical_fields,
    )


def _observation(
    source_context: SecCompanyFactsSourceContext,
    *,
    category: SecCompanyFactsFundamentalObservationCategory,
    state: SecCompanyFactsFundamentalObservationState,
    message: str,
    canonical_fields: tuple[str, ...],
    missing_source_fields: tuple[str, ...] = (),
) -> SecCompanyFactsFundamentalObservation:
    return SecCompanyFactsFundamentalObservation(
        ticker=source_context.ticker,
        cik=source_context.cik,
        provider_name=source_context.provider_name,
        category=category,
        state=state,
        message=message,
        source_context_state=source_context.source_context_state.value,
        canonical_fields=canonical_fields,
        source_values={
            field_name: source_context.canonical_fields.get(field_name)
            for field_name in canonical_fields
        },
        source_references={
            field_name: _source_reference(source_context.fields[field_name])
            for field_name in canonical_fields
            if field_name in source_context.fields
            and source_context.field_states[field_name] == SecCompanyFactsContextFieldState.PRESENT
        },
        missing_source_fields=missing_source_fields,
    )


def _source_reference(
    context_field: SecCompanyFactsContextField,
) -> dict[str, Any | None]:
    return {
        "sec_tag_selected": context_field.sec_tag_selected,
        "provider_name": context_field.provider_name,
        "taxonomy_namespace": context_field.taxonomy_namespace,
        "unit": context_field.unit,
        "fiscal_year": context_field.fiscal_year,
        "fiscal_period": context_field.fiscal_period,
        "filing_form": context_field.filing_form,
        "filing_date": context_field.filing_date,
        "period_start_date": context_field.period_start_date,
        "period_end_date": context_field.period_end_date,
        "accession_number": context_field.accession_number,
        "frame": context_field.frame,
        "selection_reason": context_field.selection_reason,
        "fallback_alias_used": context_field.fallback_alias_used,
    }


def _to_jsonable(
    observation_set: SecCompanyFactsFundamentalObservationSet,
) -> dict[str, Any]:
    payload = asdict(observation_set)
    payload["observations"] = [
        {
            **asdict(observation),
            "category": observation.category.value,
            "state": observation.state.value,
        }
        for observation in observation_set.observations
    ]
    return payload