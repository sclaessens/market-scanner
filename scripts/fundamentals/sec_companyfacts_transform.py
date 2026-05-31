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


def _iter_allowed_facts(payload: dict[str, Any]) -> list[SelectedFact]:
    facts_root = payload.get("facts", {})
    if not isinstance(facts_root, dict):
        raise ValueError("SEC Company Facts payload is missing a facts object.")

    selected: list[SelectedFact] = []
    for field, tags in DIRECT_FIELD_CANDIDATES.items():
        for tag in tags:
            namespace, tag_name = tag.split(":", 1)
            tag_payload = facts_root.get(namespace, {}).get(tag_name, {})
            if not isinstance(tag_payload, dict):
                continue
            units = tag_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            for unit, unit_facts in units.items():
                _validate_unit_for_field(field, unit)
                if not isinstance(unit_facts, list):
                    continue
                for fact in unit_facts:
                    if not isinstance(fact, dict):
                        continue
                    fy, fp, end = _fact_period_key(fact, tag=tag)
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


def _canonical_value(value: Any) -> str:
    return _clean_text(value)


def _select_one_fact(field: str, facts: list[SelectedFact]) -> tuple[SelectedFact, list[str]]:
    grouped_by_tag_unit: dict[tuple[str, str], list[SelectedFact]] = {}
    for fact in facts:
        grouped_by_tag_unit.setdefault((fact.tag, fact.unit), []).append(fact)

    for (tag, unit), grouped in grouped_by_tag_unit.items():
        distinct_values = {_canonical_value(fact.value) for fact in grouped}
        if len(distinct_values) > 1:
            raise ValueError(f"conflicting SEC facts for {field} {tag} {unit} {grouped[0].fy} {grouped[0].fp} {grouped[0].end}")

    distinct_units = {fact.unit for fact in facts}
    if len(distinct_units) > 1:
        raise ValueError(f"unit conflict for {field} {facts[0].fy} {facts[0].fp} {facts[0].end}: {sorted(distinct_units)}")

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
    return chosen, notes


def _source_reference(cik: str, fact: SelectedFact | None = None) -> str:
    base = f"sec-companyfacts:CIK{cik}"
    if fact is None or fact.accn == "":
        return base
    return f"{base}:{fact.accn}"


def _format_value(value: Any) -> str:
    text = _clean_text(value)
    return text


def _row_notes(evidence: dict[str, Any], review_notes: list[str]) -> str:
    return json.dumps(
        {
            "blocked_fields": {
                "total_debt": "blocked in SEC-6A; derivation rules not approved",
                "free_cash_flow": "blocked in SEC-6A; derivation rules not approved",
            },
            "evidence": evidence,
            "review_notes": review_notes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


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
    selected_facts = _iter_allowed_facts(payload)

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
        review_notes: list[str] = []
        monetary_units: set[str] = set()
        first_fact: SelectedFact | None = None
        report_dates: list[str] = []

        for field in DIRECT_FIELD_CANDIDATES:
            facts = field_facts.get(field, [])
            if not facts:
                if field in {"gross_profit", "diluted_eps"}:
                    review_notes.append(f"{field}: missing optional or review-required direct field")
                continue
            chosen, notes = _select_one_fact(field, facts)
            first_fact = first_fact or chosen
            row[field] = _format_value(chosen.value)
            if field != "diluted_eps":
                monetary_units.add(chosen.unit)
            if chosen.filed:
                report_dates.append(chosen.filed)
            review_notes.extend(notes)
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

        if len(monetary_units) > 1:
            raise ValueError(f"currency unit conflict for {normalized_ticker} {fy} {fp} {end}: {sorted(monetary_units)}")
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
