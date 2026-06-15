from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SEC_COMPANYFACTS_PROVIDER_NAME = "SEC_COMPANYFACTS"
SEC_COMPANYFACTS_TAXONOMY_NAMESPACE = "us-gaap"
SEC_COMPANYFACTS_REQUIRED_FIELDS = (
    "revenue",
    "net_income",
    "operating_cash_flow",
    "capital_expenditures",
)
SEC_COMPANYFACTS_APPROVED_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
}
SEC_COMPANYFACTS_APPROVED_FORMS = ("10-K", "10-K/A")
SEC_COMPANYFACTS_APPROVED_UNIT = "USD"


@dataclass(frozen=True)
class SecCompanyFactsMappedField:
    canonical_field_name: str
    sec_tag_selected: str
    provider_name: str
    taxonomy_namespace: str
    unit: str
    raw_value: Any
    fiscal_year: int | None
    fiscal_period: str | None
    filing_form: str | None
    filing_date: str | None
    period_start_date: str | None
    period_end_date: str | None
    accession_number: str | None
    frame: str | None
    selection_reason: str
    fallback_alias_used: str | None = None


def map_sec_companyfacts_fields(
    payload: dict[str, Any],
    canonical_fields: Iterable[str] = SEC_COMPANYFACTS_REQUIRED_FIELDS,
) -> dict[str, SecCompanyFactsMappedField | None]:
    us_gaap_facts = _us_gaap_facts(payload)
    return {
        canonical_field: _map_one_field(canonical_field, us_gaap_facts)
        for canonical_field in tuple(canonical_fields)
    }


def extract_sec_companyfacts_field_values(
    payload: dict[str, Any],
    canonical_fields: Iterable[str] = SEC_COMPANYFACTS_REQUIRED_FIELDS,
) -> dict[str, Any]:
    mapped_fields = map_sec_companyfacts_fields(payload, canonical_fields=canonical_fields)
    return {
        canonical_field: mapped_field.raw_value if mapped_field is not None else None
        for canonical_field, mapped_field in mapped_fields.items()
    }


def _us_gaap_facts(payload: dict[str, Any]) -> dict[str, Any]:
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        return {}
    us_gaap = facts.get(SEC_COMPANYFACTS_TAXONOMY_NAMESPACE, {})
    return us_gaap if isinstance(us_gaap, dict) else {}


def _map_one_field(
    canonical_field: str,
    us_gaap_facts: dict[str, Any],
) -> SecCompanyFactsMappedField | None:
    aliases = SEC_COMPANYFACTS_APPROVED_ALIASES.get(canonical_field, ())
    primary_alias = aliases[0] if aliases else None
    for alias in aliases:
        selected = _select_latest_annual_fact(us_gaap_facts.get(alias))
        if selected is None:
            continue
        return _mapped_field(
            canonical_field=canonical_field,
            sec_tag=alias,
            selected_fact=selected,
            selection_reason=(
                "primary approved tag selected"
                if alias == primary_alias
                else "approved fallback selected after higher-priority tags were unavailable"
            ),
            fallback_alias_used=None if alias == primary_alias else alias,
        )
    return None


def _select_latest_annual_fact(fact_payload: Any) -> dict[str, Any] | None:
    units = fact_payload.get("units", {}) if isinstance(fact_payload, dict) else {}
    values = units.get(SEC_COMPANYFACTS_APPROVED_UNIT, []) if isinstance(units, dict) else []
    candidates = [
        value
        for value in values
        if isinstance(value, dict)
        and value.get("val") is not None
        and value.get("fp") == "FY"
        and value.get("form") in SEC_COMPANYFACTS_APPROVED_FORMS
        and isinstance(value.get("end"), str)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            _sortable_year(value.get("fy")),
            str(value.get("end") or ""),
            str(value.get("filed") or ""),
            str(value.get("accn") or ""),
        ),
    )


def _mapped_field(
    *,
    canonical_field: str,
    sec_tag: str,
    selected_fact: dict[str, Any],
    selection_reason: str,
    fallback_alias_used: str | None,
) -> SecCompanyFactsMappedField:
    return SecCompanyFactsMappedField(
        canonical_field_name=canonical_field,
        sec_tag_selected=sec_tag,
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        taxonomy_namespace=SEC_COMPANYFACTS_TAXONOMY_NAMESPACE,
        unit=SEC_COMPANYFACTS_APPROVED_UNIT,
        raw_value=selected_fact.get("val"),
        fiscal_year=_optional_int(selected_fact.get("fy")),
        fiscal_period=_optional_str(selected_fact.get("fp")),
        filing_form=_optional_str(selected_fact.get("form")),
        filing_date=_optional_str(selected_fact.get("filed")),
        period_start_date=_optional_str(selected_fact.get("start")),
        period_end_date=_optional_str(selected_fact.get("end")),
        accession_number=_optional_str(selected_fact.get("accn")),
        frame=_optional_str(selected_fact.get("frame")),
        selection_reason=selection_reason,
        fallback_alias_used=fallback_alias_used,
    )


def _sortable_year(value: Any) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else -1


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
