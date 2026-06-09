"""Canonical SEC CompanyFacts smoke boundary for injected source-shaped input."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

from market_scanner.fundamentals.fundamentals_provider_adapter import (
    ProviderFundamentalsIngestionResult,
    build_prior_year_growth_evidence,
    ingest_provider_fundamentals,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderPriorYearGrowthEvidenceRecord,
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
)


SEC_COMPANYFACTS_SOURCE_FAMILY = "SEC EDGAR / SEC CompanyFacts"
SEC_COMPANYFACTS_PROVIDER_NAME = "SEC CompanyFacts"
SEC_COMPANYFACTS_CAPTURE_VERSION = "v2-sec-companyfacts-smoke-boundary-v1"
SEC_COMPANYFACTS_RAW_PAYLOAD_HASH = "sha256:redacted-sec-companyfacts-smoke-input"

SEC_COMPANYFACTS_METRIC_MAPPINGS: Mapping[str, tuple[str, ...]] = {
    "revenue": ("Revenues",),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "free_cash_flow": ("FreeCashFlow",),
}

SEC_COMPANYFACTS_GROWTH_METRICS: tuple[str, ...] = (
    "revenue",
    "free_cash_flow",
    "net_income",
    "operating_income",
)


@dataclass(frozen=True)
class SecCompanyFactsFact:
    """Minimal redacted SEC CompanyFacts-shaped fact candidate."""

    concept: str
    value: object
    unit: str
    currency: str
    fiscal_year: str
    fiscal_period: str
    period_end_date: str
    accession: str
    source_reference: str
    source_timestamp: str
    ticker: str = ""
    cik: str = ""
    fiscal_quarter: str = ""


@dataclass(frozen=True)
class SecCompanyFactsSmokeInput:
    """Injected one-ticker SEC CompanyFacts-shaped smoke input."""

    ticker: str
    cik: str
    company_name: str
    fiscal_year: str
    fiscal_period: str
    period_end_date: str
    retrieval_timestamp: str
    facts: Mapping[str, Sequence[SecCompanyFactsFact]]
    source_family: str = SEC_COMPANYFACTS_SOURCE_FAMILY
    provider_name: str = SEC_COMPANYFACTS_PROVIDER_NAME
    source_timestamp: str = ""
    fiscal_quarter: str = ""
    ticker_candidates: tuple[str, ...] = ()
    cik_candidates: tuple[str, ...] = ()
    prior_fiscal_year: str = ""
    prior_facts: Mapping[str, Sequence[SecCompanyFactsFact]] | None = None
    attempted_live_mode: bool = False
    production_persistence_requested: bool = False


@dataclass(frozen=True)
class SecFactSelectionResult:
    """Deterministic SEC fact-selection outcome."""

    metric_name: str
    selected_fact: SecCompanyFactsFact | None
    issue: str


@dataclass(frozen=True)
class SecCompanyFactsSmokeResult:
    """In-memory SEC CompanyFacts smoke-boundary result."""

    ticker: str
    cik: str
    source_family: str
    provider_name: str
    company_name: str
    fiscal_year: str
    fiscal_period: str
    period_end_date: str
    smoke_status: str
    fact_selection: tuple[SecFactSelectionResult, ...]
    ingestion_result: ProviderFundamentalsIngestionResult | None
    prior_ingestion_result: ProviderFundamentalsIngestionResult | None
    growth_evidence: tuple[ProviderPriorYearGrowthEvidenceRecord, ...]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


def build_sec_companyfacts_smoke_result(
    smoke_input: SecCompanyFactsSmokeInput,
    *,
    expected_ticker: str = "NVDA",
) -> SecCompanyFactsSmokeResult:
    """Build an in-memory smoke result from injected SEC-shaped input only."""

    policy_issues = _input_policy_issues(smoke_input, expected_ticker)
    if policy_issues:
        return _blocked_result(smoke_input, issues=policy_issues)

    fact_selection = _select_current_facts(smoke_input)
    selection_issues = tuple(
        selection.issue for selection in fact_selection if selection.issue
    )
    if selection_issues:
        return _blocked_result(
            smoke_input,
            fact_selection=fact_selection,
            issues=selection_issues,
        )

    response = _provider_response_from_selection(smoke_input, fact_selection)
    ingestion_result = ingest_provider_fundamentals(
        response,
        metric_mappings=SEC_COMPANYFACTS_METRIC_MAPPINGS,
    )

    prior_ingestion_result = None
    growth_evidence: tuple[ProviderPriorYearGrowthEvidenceRecord, ...] = ()
    prior_warnings: tuple[str, ...] = ()
    if smoke_input.prior_facts:
        prior_result = _prior_ingestion_for(smoke_input)
        prior_ingestion_result = prior_result.ingestion_result
        prior_warnings = prior_result.warnings
        if prior_ingestion_result is not None:
            growth_evidence = build_prior_year_growth_evidence(
                ingestion_result.normalized_records,
                prior_ingestion_result.normalized_records,
                metric_names=SEC_COMPANYFACTS_GROWTH_METRICS,
            )

    warnings = (
        *_warnings_from_ingestion(ingestion_result),
        *prior_warnings,
        *_warnings_from_growth(growth_evidence),
    )
    smoke_status = (
        "review_required"
        if ingestion_result.issues
        or ingestion_result.readiness_record.readiness_state != "available"
        or (
            prior_ingestion_result is not None
            and (
                prior_ingestion_result.issues
                or prior_ingestion_result.readiness_record.readiness_state
                != "available"
            )
        )
        or any(
            record.growth_status != "growth_available"
            for record in growth_evidence
        )
        else "passed"
    )

    return SecCompanyFactsSmokeResult(
        ticker=smoke_input.ticker,
        cik=smoke_input.cik,
        source_family=smoke_input.source_family,
        provider_name=smoke_input.provider_name,
        company_name=smoke_input.company_name,
        fiscal_year=smoke_input.fiscal_year,
        fiscal_period=smoke_input.fiscal_period,
        period_end_date=smoke_input.period_end_date,
        smoke_status=smoke_status,
        fact_selection=fact_selection,
        ingestion_result=ingestion_result,
        prior_ingestion_result=prior_ingestion_result,
        growth_evidence=growth_evidence,
        issues=(),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _input_policy_issues(
    smoke_input: SecCompanyFactsSmokeInput,
    expected_ticker: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    if smoke_input.attempted_live_mode:
        issues.append("attempted_live_or_network_mode")
    if smoke_input.production_persistence_requested:
        issues.append("attempted_production_persistence")
    if smoke_input.source_family != SEC_COMPANYFACTS_SOURCE_FAMILY:
        issues.append("wrong_source_family")
    if smoke_input.provider_name != SEC_COMPANYFACTS_PROVIDER_NAME:
        issues.append("wrong_provider_name")
    if smoke_input.ticker != expected_ticker:
        issues.append("ticker_mismatch")
    if not smoke_input.cik:
        issues.append("missing_cik")
    if not smoke_input.company_name:
        issues.append("missing_company_identity")
    if not smoke_input.fiscal_year or not smoke_input.fiscal_period:
        issues.append("missing_required_fiscal_context")
    if not smoke_input.period_end_date:
        issues.append("missing_period_end_date")
    if not smoke_input.retrieval_timestamp:
        issues.append("missing_retrieval_timestamp")
    if _candidate_values(smoke_input.ticker, smoke_input.ticker_candidates) != (
        smoke_input.ticker,
    ):
        issues.append("multi_ticker_input")
    if _candidate_values(smoke_input.cik, smoke_input.cik_candidates) != (
        smoke_input.cik,
    ):
        issues.append("ambiguous_cik")

    observed_fact_tickers = _fact_values(smoke_input.facts, "ticker")
    if observed_fact_tickers and observed_fact_tickers != (smoke_input.ticker,):
        issues.append("multi_ticker_input")
    observed_fact_ciks = _fact_values(smoke_input.facts, "cik")
    if observed_fact_ciks and observed_fact_ciks != (smoke_input.cik,):
        issues.append("ambiguous_cik")
    return tuple(dict.fromkeys(issues))


def _select_current_facts(
    smoke_input: SecCompanyFactsSmokeInput,
) -> tuple[SecFactSelectionResult, ...]:
    return _select_facts_for_period(
        smoke_input,
        facts=smoke_input.facts,
        fiscal_year=smoke_input.fiscal_year,
    )


def _select_facts_for_period(
    smoke_input: SecCompanyFactsSmokeInput,
    *,
    facts: Mapping[str, Sequence[SecCompanyFactsFact]],
    fiscal_year: str,
) -> tuple[SecFactSelectionResult, ...]:
    selections = tuple(
        _select_one_fact(
            metric_name,
            concept_names,
            smoke_input=smoke_input,
            facts=facts,
            fiscal_year=fiscal_year,
        )
        for metric_name, concept_names in SEC_COMPANYFACTS_METRIC_MAPPINGS.items()
    )

    return selections


def _select_one_fact(
    metric_name: str,
    concept_names: Sequence[str],
    *,
    smoke_input: SecCompanyFactsSmokeInput,
    facts: Mapping[str, Sequence[SecCompanyFactsFact]],
    fiscal_year: str,
) -> SecFactSelectionResult:
    candidates = tuple(
        fact
        for concept_name in concept_names
        for fact in facts.get(concept_name, ())
    )
    matching_candidates = tuple(
        fact
        for fact in candidates
        if fact.fiscal_year == fiscal_year
        and fact.fiscal_period == smoke_input.fiscal_period
    )

    if not matching_candidates:
        if metric_name == "free_cash_flow":
            return SecFactSelectionResult(metric_name, None, "")
        return SecFactSelectionResult(
            metric_name,
            None,
            f"{metric_name}:missing_fact",
        )
    if len(matching_candidates) > 1:
        return SecFactSelectionResult(
            metric_name,
            None,
            f"{metric_name}:ambiguous_fact_candidates",
        )

    fact = matching_candidates[0]
    issue = _fact_issue_for(
        metric_name,
        fact,
        smoke_input=smoke_input,
        fiscal_year=fiscal_year,
    )
    return SecFactSelectionResult(metric_name, None if issue else fact, issue)


def _fact_issue_for(
    metric_name: str,
    fact: SecCompanyFactsFact,
    *,
    smoke_input: SecCompanyFactsSmokeInput,
    fiscal_year: str,
) -> str:
    if fact.ticker and fact.ticker != smoke_input.ticker:
        return f"{metric_name}:ticker_mismatch"
    if fact.cik and fact.cik != smoke_input.cik:
        return f"{metric_name}:cik_mismatch"
    if fact.fiscal_year != fiscal_year:
        return f"{metric_name}:fiscal_year_mismatch"
    if fact.fiscal_period != smoke_input.fiscal_period:
        return f"{metric_name}:period_mismatch"
    if not fact.period_end_date:
        return f"{metric_name}:missing_period_end_date"
    if not fact.concept or not fact.accession or not fact.source_reference:
        return f"{metric_name}:missing_provenance"
    if not fact.unit:
        return f"{metric_name}:unit_mismatch"
    if not fact.currency:
        return f"{metric_name}:currency_mismatch"
    if _parse_decimal(fact.value) is None:
        return f"{metric_name}:non_numeric_fact_value"
    if metric_name == "free_cash_flow":
        return ""

    context_issue = _shared_context_issue(metric_name, fact, smoke_input)
    if context_issue:
        return context_issue
    return ""


def _shared_context_issue(
    metric_name: str,
    fact: SecCompanyFactsFact,
    smoke_input: SecCompanyFactsSmokeInput,
) -> str:
    selected = [
        candidate
        for selection in _select_facts_without_shared_checks(smoke_input)
        for candidate in (selection.selected_fact,)
        if candidate is not None
    ]
    if not selected:
        return ""
    reference = selected[0]
    if fact.unit != reference.unit:
        return f"{metric_name}:unit_mismatch"
    if fact.currency != reference.currency:
        return f"{metric_name}:currency_mismatch"
    return ""


def _select_facts_without_shared_checks(
    smoke_input: SecCompanyFactsSmokeInput,
) -> tuple[SecFactSelectionResult, ...]:
    selections: list[SecFactSelectionResult] = []
    for metric_name, concept_names in SEC_COMPANYFACTS_METRIC_MAPPINGS.items():
        candidates = tuple(
            fact
            for concept_name in concept_names
            for fact in smoke_input.facts.get(concept_name, ())
            if fact.fiscal_year == smoke_input.fiscal_year
            and fact.fiscal_period == smoke_input.fiscal_period
        )
        if len(candidates) == 1:
            fact = candidates[0]
            if (
                fact.concept
                and fact.accession
                and fact.source_reference
                and fact.unit
                and fact.currency
                and _parse_decimal(fact.value) is not None
            ):
                selections.append(SecFactSelectionResult(metric_name, fact, ""))
    return tuple(selections)


def _provider_response_from_selection(
    smoke_input: SecCompanyFactsSmokeInput,
    fact_selection: Sequence[SecFactSelectionResult],
) -> ProviderSourceResponse:
    selected_facts = tuple(
        selection.selected_fact
        for selection in fact_selection
        if selection.selected_fact is not None
    )
    raw_fields = {
        fact.concept: ProviderRawFieldEvidence(
            original_field_name=fact.concept,
            original_field_value=fact.value,
            original_currency=fact.currency,
            original_unit=fact.unit,
            reported_period=fact.fiscal_period,
            fiscal_year=fact.fiscal_year,
            fiscal_quarter=fact.fiscal_quarter,
        )
        for fact in selected_facts
    }
    missing_fields = tuple(
        next(iter(SEC_COMPANYFACTS_METRIC_MAPPINGS[selection.metric_name]))
        for selection in fact_selection
        if selection.selected_fact is None
    )
    first_fact = selected_facts[0]
    accession = first_fact.accession
    source_timestamp = smoke_input.source_timestamp or first_fact.source_timestamp
    return ProviderSourceResponse(
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        provider_category=ProviderCategory.REGULATORY_FILING.value,
        provider_record_id=f"{smoke_input.cik}-{accession}-{smoke_input.fiscal_year}",
        original_source_reference=(
            f"sec-companyfacts-smoke:{smoke_input.cik}:"
            f"{accession}:{smoke_input.fiscal_year}:{smoke_input.fiscal_period}"
        ),
        ticker=smoke_input.ticker,
        symbol=smoke_input.ticker,
        entity_identifier=smoke_input.cik,
        source_timestamp=source_timestamp,
        retrieval_timestamp=smoke_input.retrieval_timestamp,
        reported_period=smoke_input.fiscal_period,
        fiscal_year=smoke_input.fiscal_year,
        fiscal_quarter=smoke_input.fiscal_quarter,
        raw_fields=raw_fields,
        provider_status=ProviderSourceStatus.AVAILABLE.value,
        provider_error_status="",
        missing_field_evidence=missing_fields,
        provenance_metadata=(
            f"source_family={SEC_COMPANYFACTS_SOURCE_FAMILY};"
            f"company={smoke_input.company_name};"
            f"period_end_date={smoke_input.period_end_date};"
            f"accession={accession};payload=redacted"
        ),
        raw_payload_hash=SEC_COMPANYFACTS_RAW_PAYLOAD_HASH,
        capture_version=SEC_COMPANYFACTS_CAPTURE_VERSION,
    )


def _prior_ingestion_for(
    smoke_input: SecCompanyFactsSmokeInput,
) -> SecCompanyFactsSmokeResult:
    prior_year = smoke_input.prior_fiscal_year or _previous_year(
        smoke_input.fiscal_year
    )
    if not prior_year:
        return _blocked_result(smoke_input, issues=("prior_fiscal_year_missing",))
    assert smoke_input.prior_facts is not None
    prior_selection = _select_facts_for_period(
        smoke_input,
        facts=smoke_input.prior_facts,
        fiscal_year=prior_year,
    )
    prior_issues = tuple(
        selection.issue for selection in prior_selection if selection.issue
    )
    if prior_issues:
        return _blocked_result(
            smoke_input,
            fact_selection=prior_selection,
            issues=tuple(f"prior:{issue}" for issue in prior_issues),
        )
    prior_input = SecCompanyFactsSmokeInput(
        ticker=smoke_input.ticker,
        cik=smoke_input.cik,
        company_name=smoke_input.company_name,
        fiscal_year=prior_year,
        fiscal_period=smoke_input.fiscal_period,
        fiscal_quarter=smoke_input.fiscal_quarter,
        period_end_date=smoke_input.period_end_date,
        retrieval_timestamp=smoke_input.retrieval_timestamp,
        facts=smoke_input.prior_facts,
        source_family=smoke_input.source_family,
        provider_name=smoke_input.provider_name,
        source_timestamp=smoke_input.source_timestamp,
    )
    prior_response = _provider_response_from_selection(prior_input, prior_selection)
    prior_ingestion = ingest_provider_fundamentals(
        prior_response,
        metric_mappings=SEC_COMPANYFACTS_METRIC_MAPPINGS,
    )
    return SecCompanyFactsSmokeResult(
        ticker=smoke_input.ticker,
        cik=smoke_input.cik,
        source_family=smoke_input.source_family,
        provider_name=smoke_input.provider_name,
        company_name=smoke_input.company_name,
        fiscal_year=prior_year,
        fiscal_period=smoke_input.fiscal_period,
        period_end_date=smoke_input.period_end_date,
        smoke_status="passed",
        fact_selection=prior_selection,
        ingestion_result=prior_ingestion,
        prior_ingestion_result=None,
        growth_evidence=(),
        issues=(),
        warnings=_warnings_from_ingestion(prior_ingestion),
    )


def _blocked_result(
    smoke_input: SecCompanyFactsSmokeInput,
    *,
    fact_selection: Sequence[SecFactSelectionResult] = (),
    issues: Sequence[str],
) -> SecCompanyFactsSmokeResult:
    return SecCompanyFactsSmokeResult(
        ticker=smoke_input.ticker,
        cik=smoke_input.cik,
        source_family=smoke_input.source_family,
        provider_name=smoke_input.provider_name,
        company_name=smoke_input.company_name,
        fiscal_year=smoke_input.fiscal_year,
        fiscal_period=smoke_input.fiscal_period,
        period_end_date=smoke_input.period_end_date,
        smoke_status="review_required",
        fact_selection=tuple(fact_selection),
        ingestion_result=None,
        prior_ingestion_result=None,
        growth_evidence=(),
        issues=tuple(dict.fromkeys(issues)),
        warnings=tuple(dict.fromkeys(issues)),
    )


def _warnings_from_ingestion(
    ingestion_result: ProviderFundamentalsIngestionResult,
) -> tuple[str, ...]:
    warnings = [
        f"issue:{issue.field_name}:{issue.issue_code.value}"
        for issue in ingestion_result.issues
    ]
    warnings.extend(ingestion_result.readiness_record.readiness_warnings)
    if ingestion_result.readiness_record.missing_fundamentals_count > 0:
        warnings.append(
            "missing_fundamentals:"
            + str(ingestion_result.readiness_record.missing_fundamentals_count)
        )
    return tuple(dict.fromkeys(warnings))


def _warnings_from_growth(
    growth_evidence: Sequence[ProviderPriorYearGrowthEvidenceRecord],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for record in growth_evidence:
        warnings.extend(record.validation_warnings)
    return tuple(dict.fromkeys(warnings))


def _candidate_values(
    required_value: str,
    candidates: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(value for value in candidates if value))
    return values or ((required_value,) if required_value else ())


def _fact_values(
    facts: Mapping[str, Sequence[SecCompanyFactsFact]],
    attribute_name: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for fact_candidates in facts.values()
            for fact in fact_candidates
            for value in (str(getattr(fact, attribute_name)),)
            if value
        )
    )


def _parse_decimal(value: object) -> Decimal | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _previous_year(fiscal_year: str) -> str:
    try:
        return str(int(fiscal_year) - 1)
    except ValueError:
        return ""
