from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fundamentals.build_history_intake import REQUIRED_COLUMNS
from scripts.fundamentals.sec_ticker_cik_index import normalize_cik, normalize_ticker

SOURCE_NAME = "SEC Company Facts"
DIRECT_FIELD_CANDIDATES = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "gross_profit": ["us-gaap:GrossProfit"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "total_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "us-gaap:PartnersCapital",
    ],
    "diluted_eps": ["us-gaap:EarningsPerShareDiluted"],
}
BLOCKED_DERIVED_FIELDS = ["total_debt", "free_cash_flow"]
DERIVED_COMPONENT_CANDIDATES = {
    "debt_current": ["us-gaap:DebtCurrent", "us-gaap:LongTermDebtCurrent"],
    "debt_noncurrent": ["us-gaap:LongTermDebtNoncurrent"],
    "debt_lease_inclusive_current": ["us-gaap:LongTermDebtAndFinanceLeaseObligationsCurrent"],
    "debt_lease_inclusive_noncurrent": ["us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    "short_term_borrowings": ["us-gaap:ShortTermBorrowings"],
    "finance_lease_current": ["us-gaap:FinanceLeaseLiabilityCurrent"],
    "finance_lease_noncurrent": ["us-gaap:FinanceLeaseLiabilityNoncurrent"],
    "operating_cash_flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditures": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"],
}
COMPONENT_TAGS = {
    tag: component
    for component, tags in DERIVED_COMPONENT_CANDIDATES.items()
    for tag in tags
}
FISCAL_PERIOD_ORDER = {"FY": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
FIELD_CANDIDATE_INDEX = {
    field: {tag: index for index, tag in enumerate(tags)}
    for field, tags in DIRECT_FIELD_CANDIDATES.items()
}


@dataclass(frozen=True)
class SelectedFact:
    field: str
    tag: str
    unit: str
    value: Any
    fy: int
    fp: str
    end: str
    filed: str
    form: str
    frame: str
    accn: str


@dataclass(frozen=True)
class FactSelectionOutcome:
    fact: SelectedFact | None
    notes: list[str]
    review_evidence: list[dict[str, Any]]


def load_companyfacts_json(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"SEC Company Facts JSON not found: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SEC Company Facts JSON must contain an object.")
    return payload


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_fiscal_year(value: Any, *, tag: str) -> int:
    text = _clean_text(value)
    if text == "":
        raise ValueError(f"SEC fact is missing fiscal year for tag {tag}.")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"SEC fact fiscal year is invalid for tag {tag}: {text}") from exc


def _normalize_fiscal_period(value: Any, *, tag: str) -> str:
    fp = _clean_text(value).upper()
    if fp not in FISCAL_PERIOD_ORDER:
        raise ValueError(f"SEC fact fiscal period is unsupported for tag {tag}: {fp or '<missing>'}")
    return fp


def _fact_period_key(fact: dict[str, Any], *, tag: str) -> tuple[int, str, str]:
    fy = _parse_fiscal_year(fact.get("fy"), tag=tag)
    fp = _normalize_fiscal_period(fact.get("fp"), tag=tag)
    end = _clean_text(fact.get("end"))
    if end == "":
        raise ValueError(f"SEC fact is missing period end date for tag {tag}.")
    return fy, fp, end


def _is_eps_unit(unit: str) -> bool:
    normalized = unit.strip().lower()
    return normalized in {"usd/shares", "usd/share", "shares", "per-share"} or "/shares" in normalized


def _is_monetary_unit(unit: str) -> bool:
    normalized = unit.strip().lower()
    return normalized != "" and not _is_eps_unit(unit) and "shares" not in normalized


def _validate_unit_for_field(field: str, unit: str) -> None:
    if field == "diluted_eps":
        if not _is_eps_unit(unit):
            raise ValueError(f"diluted_eps requires a per-share unit, got: {unit}")
        return
    if not _is_monetary_unit(unit):
        raise ValueError(f"{field} requires a monetary unit, got: {unit}")


def _skipped_fact_evidence(*, tag: str, unit: str, fact: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source_tag": tag,
        "unit": _clean_text(unit),
        "value": _format_value(fact.get("val")),
        "fiscal_year": _clean_text(fact.get("fy")),
        "fiscal_period": _clean_text(fact.get("fp")),
        "period_end_date": _clean_text(fact.get("end")),
        "filed": _clean_text(fact.get("filed")),
        "form": _clean_text(fact.get("form")),
        "frame": _clean_text(fact.get("frame")),
        "accession": _clean_text(fact.get("accn")),
        "reason": reason,
    }


def _format_skipped_fact_note(skipped_fact: dict[str, Any]) -> str:
    return (
        "skipped fact: "
        f"{skipped_fact['source_tag']} "
        f"{skipped_fact['unit'] or '<missing-unit>'} "
        f"fy={skipped_fact['fiscal_year'] or '<missing>'} "
        f"fp={skipped_fact['fiscal_period'] or '<missing>'} "
        f"end={skipped_fact['period_end_date'] or '<missing>'} "
        f"reason={skipped_fact['reason']}"
    )


def _iter_tag_facts(
    payload: dict[str, Any],
    tags: list[str],
    *,
    field: str,
    skipped_facts: list[dict[str, Any]],
) -> list[SelectedFact]:
    facts_root = payload.get("facts", {})
    if not isinstance(facts_root, dict):
        raise ValueError("SEC Company Facts payload is missing a facts object.")

    selected: list[SelectedFact] = []
    for tag in tags:
        namespace, tag_name = tag.split(":", 1)
        tag_payload = facts_root.get(namespace, {}).get(tag_name, {})
        if not isinstance(tag_payload, dict):
            continue
        units = tag_payload.get("units", {})
        if not isinstance(units, dict):
            continue
        for unit, unit_facts in units.items():
            if not isinstance(unit_facts, list):
                continue
            try:
                _validate_unit_for_field(field, unit)
            except ValueError as exc:
                for fact in unit_facts:
                    if isinstance(fact, dict):
                        skipped_facts.append(_skipped_fact_evidence(tag=tag, unit=unit, fact=fact, reason=str(exc)))
                continue
            for fact in unit_facts:
                if not isinstance(fact, dict):
                    continue
                try:
                    fy, fp, end = _fact_period_key(fact, tag=tag)
                except ValueError as exc:
                    skipped_facts.append(_skipped_fact_evidence(tag=tag, unit=unit, fact=fact, reason=str(exc)))
                    continue
                selected.append(
                    SelectedFact(
                        field=field,
                        tag=tag,
                        unit=_clean_text(unit),
                        value=fact.get("val"),
                        fy=fy,
                        fp=fp,
                        end=end,
                        filed=_clean_text(fact.get("filed")),
                        form=_clean_text(fact.get("form")),
                        frame=_clean_text(fact.get("frame")),
                        accn=_clean_text(fact.get("accn")),
                    )
                )
    return selected


def _iter_allowed_facts(payload: dict[str, Any]) -> tuple[list[SelectedFact], list[dict[str, Any]]]:
    selected: list[SelectedFact] = []
    skipped_facts: list[dict[str, Any]] = []
    for field, tags in DIRECT_FIELD_CANDIDATES.items():
        selected.extend(_iter_tag_facts(payload, tags, field=field, skipped_facts=skipped_facts))
    for component, tags in DERIVED_COMPONENT_CANDIDATES.items():
        selected.extend(_iter_tag_facts(payload, tags, field=component, skipped_facts=skipped_facts))
    return selected, skipped_facts


def _canonical_value(value: Any) -> str:
    return _clean_text(value)


def _selection_review_evidence(reason: str, facts: list[SelectedFact]) -> list[dict[str, Any]]:
    return [{**_derived_evidence(fact), "reason": reason} for fact in sorted(facts, key=lambda fact: (fact.tag, fact.unit, fact.filed, fact.form, fact.accn, _canonical_value(fact.value)))]


def _select_one_fact(field: str, facts: list[SelectedFact]) -> FactSelectionOutcome:
    grouped_by_tag_unit: dict[tuple[str, str], list[SelectedFact]] = {}
    for fact in facts:
        grouped_by_tag_unit.setdefault((fact.tag, fact.unit), []).append(fact)

    for (tag, unit), grouped in grouped_by_tag_unit.items():
        distinct_values = {_canonical_value(fact.value) for fact in grouped}
        if len(distinct_values) > 1:
            reason = f"conflicting SEC facts for {field} {tag} {unit} {grouped[0].fy} {grouped[0].fp} {grouped[0].end}"
            return FactSelectionOutcome(None, [f"{field}: {reason}; field left blank for review"], _selection_review_evidence(reason, grouped))

    distinct_units = {fact.unit for fact in facts}
    if len(distinct_units) > 1:
        reason = f"unit conflict for {field} {facts[0].fy} {facts[0].fp} {facts[0].end}: {sorted(distinct_units)}"
        return FactSelectionOutcome(None, [f"{field}: {reason}; field left blank for review"], _selection_review_evidence(reason, facts))

    ordered = sorted(
        facts,
        key=lambda fact: (
            FIELD_CANDIDATE_INDEX[field][fact.tag],
            fact.filed,
            fact.form,
            fact.accn,
            _canonical_value(fact.value),
        ),
    )
    chosen = ordered[0]
    alternate_tags = sorted({fact.tag for fact in ordered if fact.tag != chosen.tag})
    notes = []
    if alternate_tags:
        notes.append(f"{field}: selected {chosen.tag} by candidate order; alternates present: {'|'.join(alternate_tags)}")
    duplicate_count = len([fact for fact in ordered if fact.tag == chosen.tag and fact.unit == chosen.unit])
    if duplicate_count > 1:
        notes.append(f"{field}: duplicate same-value facts present for {chosen.tag}; deterministic first filed/form/accession order used")
    if field == "diluted_eps":
        notes.append("diluted_eps: review-required per-share field preserved with explicit unit and period evidence")
    return FactSelectionOutcome(chosen, notes, [])


def _select_component_fact(component: str, facts: list[SelectedFact]) -> FactSelectionOutcome:
    grouped_by_tag_unit: dict[tuple[str, str], list[SelectedFact]] = {}
    for fact in facts:
        grouped_by_tag_unit.setdefault((fact.tag, fact.unit), []).append(fact)

    for (tag, unit), grouped in grouped_by_tag_unit.items():
        distinct_values = {_canonical_value(fact.value) for fact in grouped}
        if len(distinct_values) > 1:
            reason = f"conflicting SEC facts for {component} {tag} {unit} {grouped[0].fy} {grouped[0].fp} {grouped[0].end}"
            return FactSelectionOutcome(None, [f"{component}: {reason}; component left blank for review"], _selection_review_evidence(reason, grouped))

    distinct_units = {fact.unit for fact in facts}
    if len(distinct_units) > 1:
        reason = f"unit conflict for {component} {facts[0].fy} {facts[0].fp} {facts[0].end}: {sorted(distinct_units)}"
        return FactSelectionOutcome(None, [f"{component}: {reason}; component left blank for review"], _selection_review_evidence(reason, facts))

    distinct_tags = {fact.tag for fact in facts}
    if component == "debt_current" and len(distinct_tags) > 1:
        reason = f"component overlap for total_debt current debt {facts[0].fy} {facts[0].fp} {facts[0].end}: {sorted(distinct_tags)}"
        return FactSelectionOutcome(None, [f"{component}: {reason}; component left blank for review"], _selection_review_evidence(reason, facts))

    ordered = sorted(
        facts,
        key=lambda fact: (
            fact.tag,
            fact.filed,
            fact.form,
            fact.accn,
            _canonical_value(fact.value),
        ),
    )
    chosen = ordered[0]
    notes = []
    duplicate_count = len([fact for fact in ordered if fact.tag == chosen.tag and fact.unit == chosen.unit])
    if duplicate_count > 1:
        notes.append(f"{component}: duplicate same-value facts present for {chosen.tag}; deterministic first filed/form/accession order used")
    return FactSelectionOutcome(chosen, notes, [])


def _fact_to_float(fact: SelectedFact) -> float:
    try:
        return float(fact.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SEC fact value is not numeric for {fact.field} {fact.tag}: {_clean_text(fact.value)}") from exc


def _derived_evidence(fact: SelectedFact) -> dict[str, Any]:
    return {
        "source_tag": fact.tag,
        "unit": fact.unit,
        "value": _format_value(fact.value),
        "fiscal_year": fact.fy,
        "fiscal_period": fact.fp,
        "period_end_date": fact.end,
        "filed": fact.filed,
        "form": fact.form,
        "frame": fact.frame,
        "accession": fact.accn,
    }


def _format_derived_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def _source_reference(cik: str, fact: SelectedFact | None = None) -> str:
    base = f"sec-companyfacts:CIK{cik}"
    if fact is None or fact.accn == "":
        return base
    return f"{base}:{fact.accn}"


def _format_value(value: Any) -> str:
    text = _clean_text(value)
    return text


def _row_notes(evidence: dict[str, Any], review_notes: list[str]) -> str:
    blocked_fields = {}
    if "total_debt" not in evidence:
        blocked_fields["total_debt"] = "not derived in SEC-6C unless clean non-overlapping components are present"
    if "free_cash_flow" not in evidence:
        blocked_fields["free_cash_flow"] = "not derived in SEC-6C unless operating cash flow and capex are present"
    return json.dumps(
        {
            "blocked_fields": blocked_fields,
            "evidence": evidence,
            "review_notes": review_notes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _derive_total_debt(
    field_facts: dict[str, list[SelectedFact]],
    evidence: dict[str, Any],
    review_notes: list[str],
) -> str:
    review_components = [
        "short_term_borrowings",
        "finance_lease_current",
        "finance_lease_noncurrent",
    ]
    present_review_components = [component for component in review_components if field_facts.get(component)]
    if present_review_components:
        review_notes.append(
            f"total_debt: review-required components present and not mixed automatically: {'|'.join(sorted(present_review_components))}"
        )
        return ""

    simple_current = field_facts.get("debt_current", [])
    simple_noncurrent = field_facts.get("debt_noncurrent", [])
    lease_current = field_facts.get("debt_lease_inclusive_current", [])
    lease_noncurrent = field_facts.get("debt_lease_inclusive_noncurrent", [])
    simple_present = bool(simple_current or simple_noncurrent)
    lease_present = bool(lease_current or lease_noncurrent)

    if simple_present and lease_present:
        review_notes.append("total_debt: blocked because simple and lease-inclusive debt component families overlap")
        return ""

    if simple_present:
        if not simple_current or not simple_noncurrent:
            review_notes.append("total_debt: missing current or noncurrent simple debt component; value not inferred")
            return ""
        current = _select_component_fact("debt_current", simple_current)
        noncurrent = _select_component_fact("debt_noncurrent", simple_noncurrent)
        review_notes.extend(current.notes)
        review_notes.extend(noncurrent.notes)
        if current.review_evidence or noncurrent.review_evidence:
            evidence.setdefault("review_required_derived_components", {}).setdefault("total_debt", []).extend(
                current.review_evidence + noncurrent.review_evidence
            )
        if current.fact is None or noncurrent.fact is None:
            review_notes.append("total_debt: blocked because one or more simple debt components require review")
            return ""
        try:
            total = _fact_to_float(current.fact) + _fact_to_float(noncurrent.fact)
        except ValueError as exc:
            review_notes.append(f"total_debt: {exc}; value left blank for review")
            return ""
        evidence["total_debt"] = {
            "formula": "current_debt + noncurrent_debt",
            "formula_version": "SEC-6C_TOTAL_DEBT_SIMPLE_V1",
            "components": {
                "current_debt": _derived_evidence(current.fact),
                "noncurrent_debt": _derived_evidence(noncurrent.fact),
            },
        }
        review_notes.append("total_debt: derived from clean simple current and noncurrent debt components")
        return _format_derived_value(total)

    if lease_present:
        if not lease_current or not lease_noncurrent:
            review_notes.append("total_debt: missing current or noncurrent lease-inclusive debt component; value not inferred")
            return ""
        current = _select_component_fact("debt_lease_inclusive_current", lease_current)
        noncurrent = _select_component_fact("debt_lease_inclusive_noncurrent", lease_noncurrent)
        review_notes.extend(current.notes)
        review_notes.extend(noncurrent.notes)
        if current.review_evidence or noncurrent.review_evidence:
            evidence.setdefault("review_required_derived_components", {}).setdefault("total_debt", []).extend(
                current.review_evidence + noncurrent.review_evidence
            )
        if current.fact is None or noncurrent.fact is None:
            review_notes.append("total_debt: blocked because one or more lease-inclusive debt components require review")
            return ""
        try:
            total = _fact_to_float(current.fact) + _fact_to_float(noncurrent.fact)
        except ValueError as exc:
            review_notes.append(f"total_debt: {exc}; value left blank for review")
            return ""
        evidence["total_debt"] = {
            "formula": "lease_inclusive_current_debt + lease_inclusive_noncurrent_debt",
            "formula_version": "SEC-6C_TOTAL_DEBT_LEASE_INCLUSIVE_V1",
            "components": {
                "lease_inclusive_current_debt": _derived_evidence(current.fact),
                "lease_inclusive_noncurrent_debt": _derived_evidence(noncurrent.fact),
            },
        }
        review_notes.append("total_debt: derived from clean lease-inclusive current and noncurrent debt components")
        return _format_derived_value(total)

    review_notes.append("total_debt: missing source-supported debt components; value not inferred")
    return ""


def _derive_free_cash_flow(
    field_facts: dict[str, list[SelectedFact]],
    evidence: dict[str, Any],
    review_notes: list[str],
) -> str:
    operating_cash_flow_facts = field_facts.get("operating_cash_flow", [])
    capex_facts = field_facts.get("capital_expenditures", [])
    if not operating_cash_flow_facts:
        review_notes.append("free_cash_flow: missing operating cash flow component; value not inferred")
        return ""
    if not capex_facts:
        review_notes.append("free_cash_flow: missing capital expenditure component; value not inferred")
        return ""

    operating_cash_flow = _select_component_fact("operating_cash_flow", operating_cash_flow_facts)
    capex = _select_component_fact("capital_expenditures", capex_facts)
    review_notes.extend(operating_cash_flow.notes)
    review_notes.extend(capex.notes)
    if operating_cash_flow.review_evidence or capex.review_evidence:
        evidence.setdefault("review_required_derived_components", {}).setdefault("free_cash_flow", []).extend(
            operating_cash_flow.review_evidence + capex.review_evidence
        )
    if operating_cash_flow.fact is None or capex.fact is None:
        review_notes.append("free_cash_flow: blocked because one or more components require review")
        return ""

    try:
        operating_value = _fact_to_float(operating_cash_flow.fact)
        capex_value = _fact_to_float(capex.fact)
    except ValueError as exc:
        review_notes.append(f"free_cash_flow: {exc}; value left blank for review")
        return ""
    if capex_value >= 0:
        free_cash_flow = operating_value - capex_value
        formula = "operating_cash_flow - positive_capex_outflow"
        review_notes.append("free_cash_flow: derived by subtracting positive capex outflow")
    else:
        free_cash_flow = operating_value + capex_value
        formula = "operating_cash_flow + already_signed_negative_capex"
        review_notes.append("free_cash_flow: derived by adding already signed negative capex")

    evidence["free_cash_flow"] = {
        "formula": formula,
        "formula_version": "SEC-6C_FREE_CASH_FLOW_V1",
        "components": {
            "operating_cash_flow": _derived_evidence(operating_cash_flow.fact),
            "capital_expenditures": _derived_evidence(capex.fact),
        },
    }
    return _format_derived_value(free_cash_flow)


def transform_companyfacts_payload(
    payload: dict[str, Any],
    *,
    ticker: str,
    cik: str | int,
    source_freshness_date: str,
    extraction_date: str,
) -> pd.DataFrame:
    normalized_ticker = normalize_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("ticker is required.")
    normalized_cik = normalize_cik(cik)
    selected_facts, skipped_facts = _iter_allowed_facts(payload)

    by_period: dict[tuple[int, str, str], dict[str, list[SelectedFact]]] = {}
    for fact in selected_facts:
        by_period.setdefault((fact.fy, fact.fp, fact.end), {}).setdefault(fact.field, []).append(fact)

    rows: list[dict[str, Any]] = []
    for (fy, fp, end), field_facts in sorted(
        by_period.items(),
        key=lambda item: (item[0][0], FISCAL_PERIOD_ORDER[item[0][1]], item[0][2]),
    ):
        row = {column: "" for column in REQUIRED_COLUMNS}
        row.update(
            {
                "ticker": normalized_ticker,
                "fiscal_year": str(fy),
                "fiscal_period": fp,
                "period_end_date": end,
                "source_name": SOURCE_NAME,
                "source_freshness_date": source_freshness_date,
                "extraction_date": extraction_date,
                "total_debt": "",
                "free_cash_flow": "",
            }
        )

        evidence: dict[str, Any] = {}
        review_notes: list[str] = [_format_skipped_fact_note(skipped_fact) for skipped_fact in skipped_facts]
        if skipped_facts:
            evidence["skipped_facts"] = skipped_facts
        monetary_units: set[str] = set()
        first_fact: SelectedFact | None = None
        report_dates: list[str] = []

        for field in DIRECT_FIELD_CANDIDATES:
            facts = field_facts.get(field, [])
            if not facts:
                if field in {"gross_profit", "diluted_eps"}:
                    review_notes.append(f"{field}: missing optional or review-required direct field")
                continue
            selection = _select_one_fact(field, facts)
            review_notes.extend(selection.notes)
            if selection.review_evidence:
                evidence.setdefault("review_required_fields", {})[field] = selection.review_evidence
            if selection.fact is None:
                continue
            chosen = selection.fact
            first_fact = first_fact or chosen
            row[field] = _format_value(chosen.value)
            if field != "diluted_eps":
                monetary_units.add(chosen.unit)
            if chosen.filed:
                report_dates.append(chosen.filed)
            evidence[field] = {
                "source_tag": chosen.tag,
                "unit": chosen.unit,
                "fiscal_year": chosen.fy,
                "fiscal_period": chosen.fp,
                "period_end_date": chosen.end,
                "filed": chosen.filed,
                "form": chosen.form,
                "frame": chosen.frame,
                "accession": chosen.accn,
            }

        row["total_debt"] = _derive_total_debt(field_facts, evidence, review_notes)
        row["free_cash_flow"] = _derive_free_cash_flow(field_facts, evidence, review_notes)

        if len(monetary_units) > 1:
            review_notes.append(f"currency unit conflict for {normalized_ticker} {fy} {fp} {end}: {sorted(monetary_units)}")
            evidence["currency_unit_conflict"] = sorted(monetary_units)
            row["currency"] = ""
        else:
            row["currency"] = sorted(monetary_units)[0] if monetary_units else ""
        row["report_date"] = max(report_dates) if report_dates else ""
        row["source_reference"] = _source_reference(normalized_cik, first_fact)
        row["notes"] = _row_notes(evidence, sorted(set(review_notes)))
        rows.append(row)

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def transform_companyfacts_file(
    companyfacts_json: str | Path,
    *,
    ticker: str,
    cik: str | int,
    source_freshness_date: str,
    extraction_date: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    df = transform_companyfacts_payload(
        load_companyfacts_json(companyfacts_json),
        ticker=ticker,
        cik=cik,
        source_freshness_date=source_freshness_date,
        extraction_date=extraction_date,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
    return df


def summarize_transform(df: pd.DataFrame, output_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "status": "VALID",
        "row_count": int(len(df)),
        "output_path": str(output_path) if output_path is not None else "",
        "blocked_fields": BLOCKED_DERIVED_FIELDS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transform local SEC Company Facts JSON into raw fundamentals history rows.")
    parser.add_argument("--companyfacts-json", required=True, type=Path, help="Local SEC Company Facts-like JSON file.")
    parser.add_argument("--ticker", required=True, help="Ticker context for the transformed rows.")
    parser.add_argument("--cik", required=True, help="CIK context for source references.")
    parser.add_argument("--output", type=Path, help="Optional generated fundamentals history CSV output path.")
    parser.add_argument("--source-freshness-date", required=True, help="Source freshness date to preserve in output rows.")
    parser.add_argument("--extraction-date", required=True, help="Extraction date to preserve in output rows.")
    parser.add_argument("--validate-only", action="store_true", help="Validate and summarize without writing output.")
    args = parser.parse_args(argv)

    output_path = None if args.validate_only else args.output
    df = transform_companyfacts_file(
        args.companyfacts_json,
        ticker=args.ticker,
        cik=args.cik,
        source_freshness_date=args.source_freshness_date,
        extraction_date=args.extraction_date,
        output_path=output_path,
    )
    print(json.dumps(summarize_transform(df, output_path), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
