"""Controlled live SEC CompanyFacts one-ticker smoke boundary."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_scanner.fundamentals.sec_companyfacts_smoke_boundary import (
    SEC_COMPANYFACTS_PROVIDER_NAME,
    SEC_COMPANYFACTS_SOURCE_FAMILY,
    SecCompanyFactsFact,
    SecCompanyFactsSmokeInput,
    SecCompanyFactsSmokeResult,
    build_sec_companyfacts_smoke_result,
)


APPROVED_LIVE_SMOKE_TICKER = "NVDA"
APPROVED_LIVE_SMOKE_CIK = "0001045810"
APPROVED_LIVE_SMOKE_COMPANY = "NVIDIA Corporation"
SEC_COMPANYFACTS_ENDPOINT = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
)
SEC_USER_AGENT_ENV_VAR = "SEC_USER_AGENT"

SUPPORTED_SEC_CONCEPTS: tuple[str, ...] = (
    "Revenues",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "FreeCashFlow",
)

APPROVED_SEC_ANNUAL_FORMS: tuple[str, ...] = (
    "10-K",
    "10-K/A",
)


@dataclass(frozen=True)
class SecCompanyFactsHttpResponse:
    """In-memory SEC HTTP response container."""

    status_code: int
    body: str


@dataclass(frozen=True)
class SecCompanyFactsLiveSmokeResult:
    """Redacted result for the controlled one-ticker live smoke."""

    ticker: str
    cik: str
    company_name: str
    source_family: str
    provider_name: str
    endpoint: str
    status: str
    request_executed: bool
    request_count: int
    http_status_category: str
    failure_category: str
    retrieval_timestamp: str
    fiscal_context_summary: str
    canonical_fields_found: tuple[str, ...]
    canonical_fields_missing: tuple[str, ...]
    free_cash_flow_status: str
    growth_evidence_status: str
    readiness_state: str
    missingness_reasons: tuple[str, ...]
    provenance_summary: str
    issues: tuple[str, ...]
    boundary_result: SecCompanyFactsSmokeResult | None


NetworkFetcher = Callable[[str, str], SecCompanyFactsHttpResponse]


def sec_user_agent_from_env(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the local operator-supplied SEC User-Agent without logging it."""

    values = os.environ if environ is None else environ
    return values.get(SEC_USER_AGENT_ENV_VAR, "")


def run_controlled_live_sec_companyfacts_smoke(
    *,
    ticker: str,
    cik: str,
    user_agent: str,
    execute_live: bool = False,
    network_fetcher: NetworkFetcher | None = None,
    retrieval_timestamp: str = "",
) -> SecCompanyFactsLiveSmokeResult:
    """Run the controlled live smoke only when every pre-flight gate passes."""

    timestamp = retrieval_timestamp or _utc_timestamp()
    preflight_issues = _preflight_issues(
        ticker=ticker,
        cik=cik,
        user_agent=user_agent,
        execute_live=execute_live,
    )
    if preflight_issues:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category=_failure_category_for(preflight_issues),
            issues=preflight_issues,
            request_executed=False,
            request_count=0,
        )

    fetcher = _fetch_companyfacts_once if network_fetcher is None else network_fetcher
    try:
        response = fetcher(SEC_COMPANYFACTS_ENDPOINT, user_agent)
    except HTTPError as exc:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category="http_error",
            issues=(f"http_error:{exc.code}",),
            request_executed=True,
            request_count=1,
            http_status_category=_http_status_category(exc.code),
        )
    except (TimeoutError, URLError, OSError) as exc:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category="network_error",
            issues=(f"network_error:{exc.__class__.__name__}",),
            request_executed=True,
            request_count=1,
        )

    if response.status_code < 200 or response.status_code >= 300:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category="http_error",
            issues=(f"http_status:{response.status_code}",),
            request_executed=True,
            request_count=1,
            http_status_category=_http_status_category(response.status_code),
        )

    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category="invalid_json",
            issues=("invalid_json",),
            request_executed=True,
            request_count=1,
            http_status_category=_http_status_category(response.status_code),
        )

    smoke_input, extraction_issues = _smoke_input_from_companyfacts_payload(
        payload,
        retrieval_timestamp=timestamp,
    )
    if extraction_issues or smoke_input is None:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category=_failure_category_for(extraction_issues),
            issues=extraction_issues,
            request_executed=True,
            request_count=1,
            http_status_category=_http_status_category(response.status_code),
        )

    boundary_result = build_sec_companyfacts_smoke_result(smoke_input)
    if boundary_result.issues:
        return _failed_result(
            ticker=ticker,
            cik=cik,
            retrieval_timestamp=timestamp,
            failure_category="canonical_boundary_rejected_input",
            issues=boundary_result.issues,
            request_executed=True,
            request_count=1,
            http_status_category=_http_status_category(response.status_code),
            boundary_result=boundary_result,
        )

    return _result_from_boundary(
        boundary_result,
        retrieval_timestamp=timestamp,
        request_executed=True,
        request_count=1,
        http_status_category=_http_status_category(response.status_code),
    )


def _preflight_issues(
    *,
    ticker: str,
    cik: str,
    user_agent: str,
    execute_live: bool,
) -> tuple[str, ...]:
    issues: list[str] = []
    if not execute_live:
        issues.append("live_smoke_disabled_by_default")
    if ticker != APPROVED_LIVE_SMOKE_TICKER:
        issues.append("wrong_ticker")
    if cik != APPROVED_LIVE_SMOKE_CIK:
        issues.append("wrong_cik")
    if not user_agent:
        issues.append("user_agent_missing")
    elif _user_agent_is_malformed(user_agent):
        issues.append("user_agent_malformed")
    return tuple(issues)


def _user_agent_is_malformed(user_agent: str) -> bool:
    stripped = user_agent.strip()
    return not stripped or len(stripped) < 8 or any(
        separator in stripped for separator in ("\n", "\r", "\t")
    )


def _fetch_companyfacts_once(
    endpoint: str,
    user_agent: str,
) -> SecCompanyFactsHttpResponse:
    request = Request(
        endpoint,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="GET",
    )
    with urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
        status_code = int(getattr(response, "status", 0) or 0)
    return SecCompanyFactsHttpResponse(status_code=status_code, body=body)


def _smoke_input_from_companyfacts_payload(
    payload: object,
    *,
    retrieval_timestamp: str,
) -> tuple[SecCompanyFactsSmokeInput | None, tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        return None, ("unexpected_sec_shape",)
    if str(payload.get("cik", "")).zfill(10) != APPROVED_LIVE_SMOKE_CIK:
        return None, ("ticker_cik_mismatch",)
    entity_name = str(payload.get("entityName", "")).strip()
    if not entity_name:
        return None, ("missing_company_identity",)

    concept_entries = _concept_entries(payload)
    if not concept_entries:
        return None, ("unexpected_sec_shape",)

    fiscal_year = _latest_annual_fiscal_year(concept_entries)
    if not fiscal_year:
        return None, ("missing_required_context",)
    prior_fiscal_year = str(int(fiscal_year) - 1)

    current_facts, current_issues = _facts_for_year(
        concept_entries,
        fiscal_year=fiscal_year,
    )
    prior_facts, prior_issues = _facts_for_year(
        concept_entries,
        fiscal_year=prior_fiscal_year,
    )
    issues = (*current_issues, *prior_issues)
    if current_issues:
        return None, issues

    period_end_date = _first_period_end(current_facts)
    if not period_end_date:
        return None, ("missing_required_context",)

    return (
        SecCompanyFactsSmokeInput(
            ticker=APPROVED_LIVE_SMOKE_TICKER,
            cik=APPROVED_LIVE_SMOKE_CIK,
            company_name=entity_name,
            fiscal_year=fiscal_year,
            fiscal_period="FY",
            period_end_date=period_end_date,
            retrieval_timestamp=retrieval_timestamp,
            facts=current_facts,
            ticker_candidates=(APPROVED_LIVE_SMOKE_TICKER,),
            cik_candidates=(APPROVED_LIVE_SMOKE_CIK,),
            prior_fiscal_year=prior_fiscal_year if prior_facts else "",
            prior_facts=prior_facts or None,
        ),
        tuple(dict.fromkeys(issues)),
    )


def _concept_entries(payload: Mapping[object, object]) -> dict[str, tuple[Mapping, ...]]:
    facts = payload.get("facts")
    if not isinstance(facts, Mapping):
        return {}
    us_gaap = facts.get("us-gaap")
    if not isinstance(us_gaap, Mapping):
        return {}

    entries: dict[str, tuple[Mapping, ...]] = {}
    for concept in SUPPORTED_SEC_CONCEPTS:
        concept_payload = us_gaap.get(concept)
        if not isinstance(concept_payload, Mapping):
            continue
        units = concept_payload.get("units")
        if not isinstance(units, Mapping):
            continue
        concept_entries: list[Mapping] = []
        for unit_name, unit_entries in units.items():
            if unit_name != "USD" or not isinstance(unit_entries, Sequence):
                continue
            concept_entries.extend(
                entry for entry in unit_entries if isinstance(entry, Mapping)
            )
        if concept_entries:
            entries[concept] = tuple(concept_entries)
    return entries


def _latest_annual_fiscal_year(
    concept_entries: Mapping[str, Sequence[Mapping]],
) -> str:
    years = {
        str(entry.get("fy", ""))
        for entries in concept_entries.values()
        for entry in entries
        if _is_annual_fact(entry) and str(entry.get("fy", "")).isdigit()
    }
    return max(years) if years else ""


def _facts_for_year(
    concept_entries: Mapping[str, Sequence[Mapping]],
    *,
    fiscal_year: str,
) -> tuple[dict[str, tuple[SecCompanyFactsFact, ...]], tuple[str, ...]]:
    facts: dict[str, tuple[SecCompanyFactsFact, ...]] = {}
    issues: list[str] = []
    for concept, entries in concept_entries.items():
        candidates = [
            entry
            for entry in entries
            if _is_annual_fact(entry) and str(entry.get("fy", "")) == fiscal_year
        ]
        if not candidates:
            continue

        selected = _select_deterministic_annual_candidates(candidates)
        if len(selected) > 1:
            issues.append(f"ambiguous_facts:{concept}:{fiscal_year}")
            continue

        fact = _fact_from_entry(concept, selected[0], fiscal_year=fiscal_year)
        if fact is None:
            issues.append(f"missing_provenance:{concept}:{fiscal_year}")
            continue
        facts[concept] = (fact,)
    return facts, tuple(issues)


def _is_annual_fact(entry: Mapping) -> bool:
    return str(entry.get("fp", "")) == "FY"


def _select_deterministic_annual_candidates(
    candidates: Sequence[Mapping],
) -> tuple[Mapping, ...]:
    """Select a single annual SEC fact only when ambiguity can be resolved safely."""

    scoped_candidates = _preferred_annual_form_candidates(candidates) or tuple(candidates)
    latest_filed = _latest_filed_candidates(scoped_candidates)
    latest_period_end = _latest_period_end_candidates(latest_filed)
    return _collapse_equivalent_candidates(latest_period_end)


def _preferred_annual_form_candidates(
    candidates: Sequence[Mapping],
) -> tuple[Mapping, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if str(candidate.get("form", "")).upper() in APPROVED_SEC_ANNUAL_FORMS
    )


def _latest_filed_candidates(candidates: Sequence[Mapping]) -> tuple[Mapping, ...]:
    latest_filed = max(str(candidate.get("filed", "")) for candidate in candidates)
    return tuple(
        candidate
        for candidate in candidates
        if str(candidate.get("filed", "")) == latest_filed
    )

def _latest_period_end_candidates(candidates: Sequence[Mapping]) -> tuple[Mapping, ...]:
    latest_period_end = max(str(candidate.get("end", "")) for candidate in candidates)
    return tuple(
        candidate
        for candidate in candidates
        if str(candidate.get("end", "")) == latest_period_end
    )

def _collapse_equivalent_candidates(
    candidates: Sequence[Mapping],
) -> tuple[Mapping, ...]:
    collapsed: dict[tuple[str, ...], Mapping] = {}
    for candidate in candidates:
        collapsed.setdefault(_canonical_fact_identity(candidate), candidate)
    return tuple(collapsed.values())


def _canonical_fact_identity(candidate: Mapping) -> tuple[str, ...]:
    return (
        str(candidate.get("fy", "")),
        str(candidate.get("fp", "")),
        str(candidate.get("start", "")),
        str(candidate.get("end", "")),
        str(candidate.get("val", "")),
        str(candidate.get("accn", "")),
        str(candidate.get("filed", "")),
    )


def _fact_from_entry(
    concept: str,
    entry: Mapping,
    *,
    fiscal_year: str,
) -> SecCompanyFactsFact | None:
    required = ("val", "end", "accn", "filed")
    if any(entry.get(field) in (None, "") for field in required):
        return None
    return SecCompanyFactsFact(
        concept=concept,
        value=entry.get("val"),
        unit="USD",
        currency="USD",
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        period_end_date=str(entry.get("end", "")),
        accession=f"redacted-sec-accession:{str(entry.get('accn', ''))[-6:]}",
        source_reference=(
            "redacted-sec-companyfacts://"
            f"CIK{APPROVED_LIVE_SMOKE_CIK}/{concept}/{fiscal_year}/FY"
        ),
        source_timestamp=str(entry.get("filed", "")),
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
    )


def _first_period_end(
    facts: Mapping[str, Sequence[SecCompanyFactsFact]],
) -> str:
    for entries in facts.values():
        for fact in entries:
            if fact.period_end_date:
                return fact.period_end_date
    return ""


def _result_from_boundary(
    boundary_result: SecCompanyFactsSmokeResult,
    *,
    retrieval_timestamp: str,
    request_executed: bool,
    request_count: int,
    http_status_category: str,
) -> SecCompanyFactsLiveSmokeResult:
    metrics = _metric_map(boundary_result)
    found = tuple(
        metric_name
        for metric_name, record in metrics.items()
        if record.metric_value not in (None, "")
    )
    missing = tuple(
        metric_name
        for metric_name, record in metrics.items()
        if record.metric_value in (None, "")
    )
    readiness_state = ""
    if boundary_result.ingestion_result is not None:
        readiness_state = (
            boundary_result.ingestion_result.readiness_record.readiness_state
        )
    return SecCompanyFactsLiveSmokeResult(
        ticker=boundary_result.ticker,
        cik=boundary_result.cik,
        company_name=boundary_result.company_name,
        source_family=boundary_result.source_family,
        provider_name=boundary_result.provider_name,
        endpoint=SEC_COMPANYFACTS_ENDPOINT,
        status=(
            "passed"
            if boundary_result.smoke_status == "passed"
            else "review_required"
        ),
        request_executed=request_executed,
        request_count=request_count,
        http_status_category=http_status_category,
        failure_category="",
        retrieval_timestamp=retrieval_timestamp,
        fiscal_context_summary=(
            f"{boundary_result.fiscal_period} {boundary_result.fiscal_year}; "
            f"period_end={boundary_result.period_end_date}"
        ),
        canonical_fields_found=found,
        canonical_fields_missing=missing,
        free_cash_flow_status=_free_cash_flow_status(metrics),
        growth_evidence_status=_growth_evidence_status(boundary_result),
        readiness_state=readiness_state,
        missingness_reasons=tuple(boundary_result.warnings),
        provenance_summary=(
            f"{boundary_result.source_family}|{boundary_result.provider_name}|"
            f"{boundary_result.ticker}|CIK{boundary_result.cik}|"
            f"{boundary_result.fiscal_period}|{boundary_result.fiscal_year}"
        ),
        issues=tuple(boundary_result.warnings),
        boundary_result=boundary_result,
    )


def _failed_result(
    *,
    ticker: str,
    cik: str,
    retrieval_timestamp: str,
    failure_category: str,
    issues: tuple[str, ...],
    request_executed: bool,
    request_count: int,
    http_status_category: str = "",
    boundary_result: SecCompanyFactsSmokeResult | None = None,
) -> SecCompanyFactsLiveSmokeResult:
    return SecCompanyFactsLiveSmokeResult(
        ticker=ticker,
        cik=cik,
        company_name=APPROVED_LIVE_SMOKE_COMPANY,
        source_family=SEC_COMPANYFACTS_SOURCE_FAMILY,
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        endpoint=SEC_COMPANYFACTS_ENDPOINT,
        status="smoke_failed",
        request_executed=request_executed,
        request_count=request_count,
        http_status_category=http_status_category,
        failure_category=failure_category,
        retrieval_timestamp=retrieval_timestamp,
        fiscal_context_summary="",
        canonical_fields_found=(),
        canonical_fields_missing=(),
        free_cash_flow_status="not_evaluated",
        growth_evidence_status="not_evaluated",
        readiness_state="review_required",
        missingness_reasons=issues,
        provenance_summary="",
        issues=issues,
        boundary_result=boundary_result,
    )


def _metric_map(
    boundary_result: SecCompanyFactsSmokeResult,
) -> Mapping[str, object]:
    if boundary_result.ingestion_result is None:
        return {}
    return {
        record.metric_name: record
        for record in boundary_result.ingestion_result.normalized_records
    }


def _free_cash_flow_status(metrics: Mapping[str, object]) -> str:
    record = metrics.get("free_cash_flow")
    if record is None:
        return "missing"
    status = getattr(record, "normalization_status", "")
    if status == "source_reported":
        return "direct"
    if status == "source_derived":
        return "derived"
    return "missing"


def _growth_evidence_status(
    boundary_result: SecCompanyFactsSmokeResult,
) -> str:
    if not boundary_result.growth_evidence:
        return "missing"
    if all(
        record.growth_status == "growth_available"
        for record in boundary_result.growth_evidence
    ):
        return "available"
    return "partial"


def _failure_category_for(issues: Sequence[str]) -> str:
    if not issues:
        return "other"
    first = issues[0]
    if first in {
        "wrong_ticker",
        "wrong_cik",
        "ticker_cik_mismatch",
        "multi_ticker_input",
    }:
        return "ticker_cik_mismatch"
    if first in {"user_agent_missing", "user_agent_malformed"}:
        return first
    if first.startswith("ambiguous_facts"):
        return "ambiguous_facts"
    if first.startswith("missing_provenance"):
        return "missing_provenance"
    if first == "missing_required_context":
        return "missing_required_context"
    if first == "unexpected_sec_shape":
        return "unexpected_sec_shape"
    if first == "live_smoke_disabled_by_default":
        return "explicit_invocation_missing"
    return "other"


def _http_status_category(status_code: int) -> str:
    if status_code <= 0:
        return ""
    return f"{status_code // 100}xx"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
