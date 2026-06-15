from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from market_engine.fundamentals.source_context import FundamentalSourceContext
from market_engine.source_intake.readiness import SourceReadinessStatus


class FundamentalObservationState(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MISSING_DATA = "MISSING_DATA"
    NOT_ASSESSED = "NOT_ASSESSED"


class FundamentalObservationCategory(str, Enum):
    SOURCE_READINESS = "SOURCE_READINESS"
    REVENUE_PRESENCE = "REVENUE_PRESENCE"
    PROFITABILITY_PRESENCE = "PROFITABILITY_PRESENCE"
    OPERATING_CASH_FLOW_PRESENCE = "OPERATING_CASH_FLOW_PRESENCE"
    CAPEX_PRESENCE = "CAPEX_PRESENCE"
    CASH_GENERATION_SOURCE_COMPLETENESS = "CASH_GENERATION_SOURCE_COMPLETENESS"
    DATA_QUALITY = "DATA_QUALITY"


@dataclass(frozen=True)
class FundamentalObservation:
    ticker: str
    provider: str
    category: FundamentalObservationCategory
    state: FundamentalObservationState
    message: str
    source_readiness: SourceReadinessStatus
    canonical_fields: tuple[str, ...] = ()
    source_values: dict[str, Any | None] = field(default_factory=dict)
    source_references: dict[str, dict[str, Any | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class FundamentalAnalysisPass:
    ticker: str
    provider: str
    source_readiness: SourceReadinessStatus
    observations: tuple[FundamentalObservation, ...]


def build_fundamental_analysis_pass(
    source_context: FundamentalSourceContext,
) -> FundamentalAnalysisPass:
    observations = (
        _source_readiness_observation(source_context),
        _presence_observation(
            source_context=source_context,
            field_name="revenue",
            category=FundamentalObservationCategory.REVENUE_PRESENCE,
            present_message="Revenue source field is present for the selected period.",
            missing_message="Revenue source field is missing for the selected period.",
        ),
        _sign_observation(
            source_context=source_context,
            field_name="net_income",
            category=FundamentalObservationCategory.PROFITABILITY_PRESENCE,
            positive_message="Net income source value is positive for the selected period.",
            negative_message="Net income source value is negative for the selected period.",
            zero_message="Net income source value is zero for the selected period.",
            missing_message="Net income source field is missing for the selected period.",
        ),
        _sign_observation(
            source_context=source_context,
            field_name="operating_cash_flow",
            category=FundamentalObservationCategory.OPERATING_CASH_FLOW_PRESENCE,
            positive_message="Operating cash flow source value is positive for the selected period.",
            negative_message="Operating cash flow source value is negative for the selected period.",
            zero_message="Operating cash flow source value is zero for the selected period.",
            missing_message="Operating cash flow source field is missing for the selected period.",
        ),
        _presence_observation(
            source_context=source_context,
            field_name="capital_expenditures",
            category=FundamentalObservationCategory.CAPEX_PRESENCE,
            present_message="Capital expenditures source field is present for the selected period.",
            missing_message="Capital expenditures source field is missing for the selected period.",
        ),
        _cash_generation_source_completeness_observation(source_context),
    )
    return FundamentalAnalysisPass(
        ticker=source_context.ticker,
        provider=source_context.provider,
        source_readiness=source_context.source_status,
        observations=observations,
    )


def _source_readiness_observation(
    source_context: FundamentalSourceContext,
) -> FundamentalObservation:
    if source_context.source_status == SourceReadinessStatus.AVAILABLE:
        state = FundamentalObservationState.POSITIVE
        message = "All required source fields are available for the selected period."
    elif source_context.source_status == SourceReadinessStatus.PARTIAL:
        state = FundamentalObservationState.MISSING_DATA
        message = "One or more required source fields are missing for the selected period."
    elif source_context.source_status == SourceReadinessStatus.PROVIDER_ERROR:
        state = FundamentalObservationState.NOT_ASSESSED
        message = "Source context has a controlled provider error."
    elif source_context.source_status == SourceReadinessStatus.UNSUPPORTED:
        state = FundamentalObservationState.NOT_ASSESSED
        message = "Ticker is unsupported for the selected provider source context."
    elif source_context.source_status == SourceReadinessStatus.INVALID_TICKER:
        state = FundamentalObservationState.NOT_ASSESSED
        message = "Ticker input is invalid for the selected provider source context."
    else:
        state = FundamentalObservationState.MISSING_DATA
        message = "No approved source fields are available for the selected period."
    return FundamentalObservation(
        ticker=source_context.ticker,
        provider=source_context.provider,
        category=FundamentalObservationCategory.SOURCE_READINESS,
        state=state,
        message=message,
        source_readiness=source_context.source_status,
        canonical_fields=tuple(source_context.canonical_fields),
        source_values=dict(source_context.canonical_fields),
        source_references=_source_references(source_context, tuple(source_context.canonical_fields)),
    )


def _presence_observation(
    *,
    source_context: FundamentalSourceContext,
    field_name: str,
    category: FundamentalObservationCategory,
    present_message: str,
    missing_message: str,
) -> FundamentalObservation:
    value = source_context.canonical_fields.get(field_name)
    state = (
        FundamentalObservationState.POSITIVE
        if value is not None
        else FundamentalObservationState.MISSING_DATA
    )
    return FundamentalObservation(
        ticker=source_context.ticker,
        provider=source_context.provider,
        category=category,
        state=state,
        message=present_message if value is not None else missing_message,
        source_readiness=source_context.source_status,
        canonical_fields=(field_name,),
        source_values={field_name: value},
        source_references=_source_references(source_context, (field_name,)),
    )


def _sign_observation(
    *,
    source_context: FundamentalSourceContext,
    field_name: str,
    category: FundamentalObservationCategory,
    positive_message: str,
    negative_message: str,
    zero_message: str,
    missing_message: str,
) -> FundamentalObservation:
    value = source_context.canonical_fields.get(field_name)
    if value is None:
        state = FundamentalObservationState.MISSING_DATA
        message = missing_message
    elif value > 0:
        state = FundamentalObservationState.POSITIVE
        message = positive_message
    elif value < 0:
        state = FundamentalObservationState.NEGATIVE
        message = negative_message
    else:
        state = FundamentalObservationState.NEUTRAL
        message = zero_message
    return FundamentalObservation(
        ticker=source_context.ticker,
        provider=source_context.provider,
        category=category,
        state=state,
        message=message,
        source_readiness=source_context.source_status,
        canonical_fields=(field_name,),
        source_values={field_name: value},
        source_references=_source_references(source_context, (field_name,)),
    )


def _cash_generation_source_completeness_observation(
    source_context: FundamentalSourceContext,
) -> FundamentalObservation:
    fields = ("operating_cash_flow", "capital_expenditures")
    missing = tuple(
        field_name
        for field_name in fields
        if source_context.canonical_fields.get(field_name) is None
    )
    if missing:
        state = FundamentalObservationState.MISSING_DATA
        message = "One or more cash-generation source fields are missing for the selected period."
    else:
        state = FundamentalObservationState.POSITIVE
        message = "Operating cash flow and capital expenditures source fields are present."
    return FundamentalObservation(
        ticker=source_context.ticker,
        provider=source_context.provider,
        category=FundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
        state=state,
        message=message,
        source_readiness=source_context.source_status,
        canonical_fields=fields,
        source_values={
            field_name: source_context.canonical_fields.get(field_name)
            for field_name in fields
        },
        source_references=_source_references(source_context, fields),
    )


def _source_references(
    source_context: FundamentalSourceContext,
    field_names: tuple[str, ...],
) -> dict[str, dict[str, Any | None]]:
    references: dict[str, dict[str, Any | None]] = {}
    for field_name in field_names:
        provenance = source_context.provenance.get(field_name)
        if provenance is None:
            continue
        references[field_name] = {
            "sec_tag_selected": provenance.sec_tag_selected,
            "provider_name": provenance.provider_name,
            "unit": provenance.unit,
            "fiscal_year": provenance.fiscal_year,
            "fiscal_period": provenance.fiscal_period,
            "filing_form": provenance.filing_form,
            "filing_date": provenance.filing_date,
            "period_start_date": provenance.period_start_date,
            "period_end_date": provenance.period_end_date,
            "accession_number": provenance.accession_number,
            "frame": provenance.frame,
        }
    return references
