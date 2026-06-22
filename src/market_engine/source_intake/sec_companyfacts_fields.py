from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SEC_COMPANYFACTS_PROVIDER_NAME = "SEC_COMPANYFACTS"
SEC_COMPANYFACTS_TAXONOMY_NAMESPACE = "us-gaap"
SEC_COMPANYFACTS_IFRS_TAXONOMY_NAMESPACE = "ifrs-full"
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
SEC_COMPANYFACTS_APPROVED_FOREIGN_ISSUER_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenue",
        "RevenueFromContractsWithCustomers",
    ),
    "net_income": (
        "ProfitLoss",
    ),
    "operating_cash_flow": (
        "CashFlowsFromUsedInOperatingActivities",
    ),
    "capital_expenditures": (
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    ),
}
SEC_COMPANYFACTS_APPROVED_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A")
SEC_COMPANYFACTS_APPROVED_UNIT = "USD"
SEC_COMPANYFACTS_APPROVED_UNITS = ("USD", "EUR")


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
    facts_by_namespace = _facts_by_namespace(payload)
    return {
        canonical_field: _map_one_field(canonical_field, facts_by_namespace)
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


def _facts_by_namespace(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        return {}
    return {
        namespace: namespace_facts
        for namespace, namespace_facts in facts.items()
        if isinstance(namespace, str) and isinstance(namespace_facts, dict)
    }


def _map_one_field(
    canonical_field: str,
    facts_by_namespace: dict[str, dict[str, Any]],
) -> SecCompanyFactsMappedField | None:
    candidates = _candidate_aliases(canonical_field)
    primary_candidate = candidates[0] if candidates else None
    for namespace, alias, selection_reason in candidates:
        namespace_facts = facts_by_namespace.get(namespace, {})
        selected = _select_latest_annual_fact(namespace_facts.get(alias))
        if selected is None:
            continue
        return _mapped_field(
            canonical_field=canonical_field,
            taxonomy_namespace=namespace,
            sec_tag=alias,
            selected_fact=selected,
            selection_reason=selection_reason,
            fallback_alias_used=None if (namespace, alias, selection_reason) == primary_candidate else alias,
        )
    return None


def _candidate_aliases(canonical_field: str) -> tuple[tuple[str, str, str], ...]:
    us_gaap_aliases = SEC_COMPANYFACTS_APPROVED_ALIASES.get(canonical_field, ())
    ifrs_aliases = SEC_COMPANYFACTS_APPROVED_FOREIGN_ISSUER_ALIASES.get(
        canonical_field,
        (),
    )
    candidates: list[tuple[str, str, str]] = []
    for index, alias in enumerate(us_gaap_aliases):
        candidates.append(
            (
                SEC_COMPANYFACTS_TAXONOMY_NAMESPACE,
                alias,
                (
                    "primary approved tag selected"
                    if index == 0
                    else "approved fallback selected after higher-priority tags were unavailable"
                ),
            )
        )
    for alias in ifrs_aliases:
        candidates.append(
            (
                SEC_COMPANYFACTS_IFRS_TAXONOMY_NAMESPACE,
                alias,
                "approved foreign issuer taxonomy tag selected after US GAAP tags were unavailable",
            )
        )
    return tuple(candidates)


def _select_latest_annual_fact(fact_payload: Any) -> dict[str, Any] | None:
    units = fact_payload.get("units", {}) if isinstance(fact_payload, dict) else {}
    values = [
        (unit, value)
        for unit in SEC_COMPANYFACTS_APPROVED_UNITS
        for value in units.get(unit, [])
    ] if isinstance(units, dict) else []
    candidates = [
        _with_unit(value, unit)
        for unit, value in values
        if _is_annual_fact(value)
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


def _is_annual_fact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("val") is not None
        and value.get("fp") == "FY"
        and value.get("form") in SEC_COMPANYFACTS_APPROVED_FORMS
        and isinstance(value.get("end"), str)
    )


def _with_unit(value: dict[str, Any], unit: str) -> dict[str, Any]:
    return {**value, "_unit": unit}


def _mapped_field(
    *,
    canonical_field: str,
    taxonomy_namespace: str,
    sec_tag: str,
    selected_fact: dict[str, Any],
    selection_reason: str,
    fallback_alias_used: str | None,
) -> SecCompanyFactsMappedField:
    return SecCompanyFactsMappedField(
        canonical_field_name=canonical_field,
        sec_tag_selected=sec_tag,
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        taxonomy_namespace=taxonomy_namespace,
        unit=_selected_unit(selected_fact),
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


def _selected_unit(selected_fact: dict[str, Any]) -> str:
    for unit in SEC_COMPANYFACTS_APPROVED_UNITS:
        # The fact is copied from a unit bucket without mutation, so infer the
        # approved unit deterministically from available metadata when absent.
        if selected_fact.get("_unit") == unit:
            return unit
    return str(selected_fact.get("_unit") or SEC_COMPANYFACTS_APPROVED_UNIT)


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
