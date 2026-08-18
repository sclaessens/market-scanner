from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from market_engine.data.primary_source_metric_derivation import (
    FACT_PACKAGE_SCHEMA_VERSION,
    derive_primary_source_metrics,
    load_strict_json,
)
from market_engine.data.data11_governance import (
    DEFAULT_DOWNSTREAM_AUTHORITY,
    DEFAULT_RUN30_AUTHORITY,
    Data11GovernanceError,
    build_downstream_measurement,
    canonical_utc_text,
    duration_metadata,
    effective_freshness,
    load_downstream_prestate,
    metric_comparability,
    persist_approval_bundle,
    select_duration_facts,
    validate_authoritative_run30,
    validate_temporal_boundary,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_provider_error,
    persist_sec_companyfacts_raw_snapshot,
)


SCHEMA_VERSION = "market-engine-data11-targeted-diversified-fundamental-derivation-v1"
FUNNEL_SCHEMA_VERSION = "market-engine-data11-technical-candidate-funnel-v1"
INVENTORY_SCHEMA_VERSION = "market-engine-data11-source-inventory-v1"
COHORT_SCHEMA_VERSION = "market-engine-data11-pilot-cohort-v1"
RESULTS_SCHEMA_VERSION = "market-engine-data11-derivation-results-v1"
COMPARISON_SCHEMA_VERSION = "market-engine-data11-metric-comparison-v2"
DOWNSTREAM_SCHEMA_VERSION = "market-engine-data11-downstream-readiness-delta-v2"
DEFAULT_RANKING = Path(
    "artifacts/market_engine/universe_analysis_runs/"
    "me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z/candidate_ranking.json"
)
DEFAULT_MANIFEST = DEFAULT_RANKING.with_name("manifest.json")
DEFAULT_UNIVERSE = Path("config/market_engine/universes/canonical_universe.json")
DEFAULT_FORMULA_CATALOG = Path("config/market_engine/data10_fundamental_metric_formula_catalog.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/run_evidence")
DEFAULT_SOURCE_ROOT = Path("data/market_engine/source_snapshots/sec_companyfacts")
SEC_TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = "market-scanner-data11-pilot governance@example.com"
TOP_LIMIT = 25
MIN_COHORT = 8
MAX_COHORT = 12

_CONCEPT_TAGS: Mapping[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
}

JsonFetcher = Callable[[str], Mapping[str, Any]]


class TargetedDerivationError(ValueError):
    pass


def load_read_only_replay_manifest(
    manifest_path: str | Path,
    *,
    _clock: Callable[[], datetime],
) -> dict[str, Any]:
    """Validate historical evidence without granting acquisition or mutation authority."""
    path = Path(manifest_path).resolve()
    manifest = load_strict_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TargetedDerivationError("replay manifest schema is invalid")
    trusted_now = canonical_utc_text(_clock())
    validate_temporal_boundary(
        generated_at=str(manifest.get("generated_at")),
        acquired_at=str(manifest.get("trusted_now")),
        source_publication_date=str(manifest.get("generated_at"))[:10],
        trusted_now=trusted_now,
    )
    return {
        "mode": "read_only_historical_replay",
        "manifest_path": path.as_posix(),
        "manifest_sha256": _sha256(path),
        "run_id": manifest.get("run_id"),
        "replay_time": manifest.get("trusted_now"),
        "provider_acquisition_allowed": False,
        "approval_allowed": False,
        "downstream_mutation_allowed": False,
    }


def run_targeted_derivation(
    *,
    run_id: str,
    _generated_at: str | None = None,
    ranking_path: str | Path = DEFAULT_RANKING,
    ranking_manifest_path: str | Path = DEFAULT_MANIFEST,
    universe_path: str | Path = DEFAULT_UNIVERSE,
    formula_catalog_path: str | Path = DEFAULT_FORMULA_CATALOG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    cohort_size: int = 10,
    fetch_json: JsonFetcher | None = None,
    run30_authority_path: str | Path = DEFAULT_RUN30_AUTHORITY,
    downstream_authority_path: str | Path = DEFAULT_DOWNSTREAM_AUTHORITY,
    _clock: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    if not MIN_COHORT <= cohort_size <= MAX_COHORT:
        raise TargetedDerivationError(f"cohort_size must be between {MIN_COHORT} and {MAX_COHORT}")
    trusted_now = canonical_utc_text((_clock or (lambda: datetime.now(UTC)))())
    generated_at = _generated_at or trusted_now
    validate_temporal_boundary(
        generated_at=generated_at,
        acquired_at=generated_at,
        source_publication_date=generated_at[:10],
        trusted_now=trusted_now,
    )
    authority = validate_authoritative_run30(authority_path=run30_authority_path)
    ranking_path = Path(authority["artifact_bindings"]["run30_ranking"]["path"])
    ranking_manifest_path = Path(authority["artifact_bindings"]["run30_manifest"]["path"])
    universe_path = Path(authority["artifact_bindings"]["canonical_universe"]["path"])
    formula_catalog_path = Path(formula_catalog_path)
    output_dir = Path(output_root) / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DATA11 evidence: {output_dir}")

    ranking = authority["ranking"]
    ranking_manifest = authority["manifest"]
    universe = authority["canonical_universe"]
    formula_catalog = load_strict_json(formula_catalog_path)
    funnel = build_candidate_funnel(
        ranking,
        ranking_manifest,
        universe,
        ranking_path=ranking_path,
        ranking_manifest_path=ranking_manifest_path,
        universe_path=universe_path,
        authority=authority,
    )
    cohort = select_pilot_cohort(funnel, cohort_size=cohort_size)
    fetcher = fetch_json or _fetch_json
    ticker_index = _ticker_index(fetcher(SEC_TICKER_INDEX_URL))
    source_inventory, fact_packages = acquire_and_extract(
        cohort,
        ticker_index=ticker_index,
        fetch_json=fetcher,
        source_root=Path(source_root),
        run_id=run_id,
        generated_at=generated_at,
        acquired_at=trusted_now,
        trusted_now=trusted_now,
    )
    derivation_results = derive_cohort_metrics(
        cohort,
        fact_packages=fact_packages,
        formula_catalog=formula_catalog,
        generated_at=generated_at,
    )
    comparison = build_metric_comparison(
        funnel, cohort, source_inventory, derivation_results, trusted_now=trusted_now
    )
    prestate = load_downstream_prestate(authority_path=downstream_authority_path)
    downstream = build_downstream_measurement(
        cohort, derivation_results, prestate, downstream_executed=False
    )
    summary = build_summary(cohort, source_inventory, derivation_results, downstream)
    derivation_results["pilot_summary"] = summary

    artifacts: dict[str, Any] = {
        "candidate_funnel.json": funnel,
        "source_inventory.json": source_inventory,
        "cohort_selection.json": cohort,
        "derivation_summary.json": derivation_results,
        "fundamental_comparison_matrix.json": comparison,
        "downstream_readiness_delta.json": downstream,
    }
    output_dir.mkdir(parents=True)
    for name, payload in artifacts.items():
        _write_json(output_dir / name, payload)
    approval_bundles = persist_approval_bundles(
        output_dir=output_dir,
        run_id=run_id,
        inventory=source_inventory,
        fact_packages=fact_packages,
        formula_catalog=formula_catalog,
        derivation_results=derivation_results,
    )
    report = render_report(summary, comparison)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = build_manifest(
        run_id=run_id,
        generated_at=generated_at,
        ranking_path=ranking_path,
        ranking_manifest_path=ranking_manifest_path,
        universe_path=universe_path,
        formula_catalog_path=formula_catalog_path,
        source_root=Path(source_root),
        output_dir=output_dir,
        summary=summary,
        authority=authority,
        downstream_prestate=prestate,
        trusted_now=trusted_now,
        approval_bundles=approval_bundles,
    )
    _write_json(output_dir / "manifest.json", manifest)
    checksum_index = {
        "schema_version": "market-engine-data11-checksum-index-v1",
        "run_id": run_id,
        "files": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "checksum_index.json"
        },
    }
    _write_json(output_dir / "checksum_index.json", checksum_index)
    artifacts["report.md"] = report
    artifacts["manifest.json"] = manifest
    artifacts["checksum_index.json"] = checksum_index
    return artifacts, output_dir


def build_candidate_funnel(
    ranking: Mapping[str, Any],
    manifest: Mapping[str, Any],
    universe: Mapping[str, Any],
    *,
    ranking_path: Path,
    ranking_manifest_path: Path,
    universe_path: Path,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if ranking.get("schema_version") != "market-engine-run30-candidate-ranking-v1":
        raise TargetedDerivationError("authoritative ranking schema is invalid")
    if manifest.get("schema_version") != "market-engine-run30-full-canonical-universe-analysis-v1":
        raise TargetedDerivationError("authoritative ranking manifest schema is invalid")
    if ranking.get("run_id") != manifest.get("run_id"):
        raise TargetedDerivationError("ranking and manifest run identities do not match")
    policy = ranking.get("ranking_policy")
    if not isinstance(policy, Mapping) or policy.get("ranking_scope") != "technical_setup_screening":
        raise TargetedDerivationError("ranking is not the authoritative technical screening funnel")
    candidates = ranking.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < TOP_LIMIT:
        raise TargetedDerivationError("authoritative ranking contains fewer than 25 candidates")
    if manifest.get("input", {}).get("universe_version") != universe.get("universe_version"):
        raise TargetedDerivationError("ranking manifest and canonical universe versions do not match")
    machine_rows = [row for row in candidates if isinstance(row, Mapping)]
    if len(machine_rows) != len(candidates):
        raise TargetedDerivationError("candidate ranking contains a non-object row")
    ordered = sorted(machine_rows, key=lambda row: (int(row.get("rank", 10**9)), str(row.get("symbol")), str(row.get("instrument_id"))))
    if ordered != machine_rows:
        raise TargetedDerivationError("candidate ranking order is not deterministic")
    eligible = [row for row in ordered if row.get("ranking_eligible") is True]
    caller_top = eligible[:TOP_LIMIT]
    if len(caller_top) != TOP_LIMIT:
        raise TargetedDerivationError("authoritative ranking contains fewer than 25 ranking-eligible candidates")
    instrument_ids = [str(row.get("instrument_id") or "") for row in caller_top]
    if len(instrument_ids) != len(set(instrument_ids)):
        raise TargetedDerivationError("top-25 funnel contains duplicate instrument identities")
    try:
        trusted = authority or validate_authoritative_run30()
    except Data11GovernanceError as exc:
        raise TargetedDerivationError(str(exc)) from exc
    if any(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        != json.dumps(trusted_value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for value, trusted_value in (
            (ranking, trusted["ranking"]),
            (manifest, trusted["manifest"]),
            (universe, trusted["canonical_universe"]),
        )
    ):
        raise TargetedDerivationError("caller-supplied RUN30 input differs from tracked authority")
    top = trusted["top_candidates"]
    rows = [
        {
            "rank": row["rank"],
            "ticker": row["symbol"],
            "instrument_id": row["instrument_id"],
            "source_symbol": row["source_symbol"],
            "asset_type": str(row["instrument_id"]).split(":", 1)[0],
            "accounting_framework_status": "unknown_not_inspected",
            "candidate_score": row["candidate_score"],
            "ranking_scope": row["ranking_scope"],
            "full_advice_ready": row["full_advice_ready"],
            "missing_evidence": row.get("missing_evidence") or [],
        }
        for row in top
    ]
    return {
        "schema_version": FUNNEL_SCHEMA_VERSION,
        "source_run_id": ranking["run_id"],
        "source_status": manifest.get("status"),
        "top_limit": TOP_LIMIT,
        "candidate_count": len(rows),
        "accounting_framework_inventory": {
            "equities_framework_unknown_not_inspected": sum(row["asset_type"] == "equity" for row in rows),
            "source_validated_us_gaap": 0,
            "source_validated_ifrs": 0,
            "unsupported_or_missing_framework": 0,
            "non_equity_candidates": sum(row["asset_type"] != "equity" for row in rows),
            "note": "Framework is unknown before primary-source namespace inspection and is never inferred from asset type.",
        },
        "source_bindings": {
            "ranking_path": ranking_path.as_posix(),
            "ranking_sha256": _sha256(ranking_path),
            "manifest_path": ranking_manifest_path.as_posix(),
            "manifest_sha256": _sha256(ranking_manifest_path),
            "canonical_universe_path": universe_path.as_posix(),
            "canonical_universe_sha256": _sha256(universe_path),
            "canonical_universe_version": universe.get("universe_version"),
            "cutoff_date": manifest.get("input", {}).get("cutoff_date"),
            "authority_path": trusted["authority_path"],
            "authority_sha256": trusted["authority_sha256"],
            "run30_universe_index_path": trusted["artifact_bindings"]["run30_universe_index"]["path"],
            "run30_universe_index_sha256": trusted["artifact_bindings"]["run30_universe_index"]["sha256"],
        },
        "candidates": rows,
    }


def select_pilot_cohort(funnel: Mapping[str, Any], *, cohort_size: int) -> dict[str, Any]:
    eligible = [row for row in funnel["candidates"] if row["asset_type"] == "equity"]
    selected = eligible[:cohort_size]
    selected_ids = {row["instrument_id"] for row in selected}
    rows = []
    for row in funnel["candidates"]:
        chosen = row["instrument_id"] in selected_ids
        reason = "SELECTED_RANK_PRIORITY" if chosen else (
            "BLOCKED_APPLICABILITY_UNPROVEN" if row["asset_type"] != "equity" else "NOT_SELECTED_COHORT_LIMIT"
        )
        rows.append({**row, "selected": chosen, "selection_reason": reason})
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_run_id": funnel["source_run_id"],
        "cohort_size": len(selected),
        "minimum_safe_processing_target": 6,
        "selection_policy": {
            "primary_order": "authoritative candidate rank ascending",
            "supported_scope": "equity candidates with official SEC CompanyFacts identity",
            "framework_diversity": "reported only for the inspected cohort; top-25 framework remains unknown before source inspection",
            "ticker_specific_runtime_branches": False,
        },
        "selected_tickers": [row["ticker"] for row in selected],
        "candidates": rows,
    }


def acquire_and_extract(
    cohort: Mapping[str, Any],
    *,
    ticker_index: Mapping[str, Mapping[str, str]],
    fetch_json: JsonFetcher,
    source_root: Path,
    run_id: str,
    generated_at: str,
    acquired_at: str,
    trusted_now: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    fact_packages: dict[str, Mapping[str, Any]] = {}
    for candidate in (row for row in cohort["candidates"] if row["selected"]):
        ticker = candidate["ticker"]
        identity = ticker_index.get(ticker)
        if identity is None:
            inventory.append(_blocked_inventory(candidate, "SEC_TICKER_IDENTITY_NOT_FOUND"))
            continue
        cik = identity["cik"]
        source_url = SEC_COMPANYFACTS_URL.format(cik=cik)
        try:
            payload = dict(fetch_json(source_url))
            snapshot_path = persist_sec_companyfacts_raw_snapshot(
                raw_payload=payload,
                ticker=ticker,
                cik=cik,
                run_id=run_id,
                fetched_at=acquired_at,
                root_dir=source_root,
            )
            checksum = _sha256(snapshot_path)
            fact_package, extraction = build_fact_package(
                candidate,
                payload=payload,
                source_url=source_url,
                source_checksum=checksum,
                generated_at=generated_at,
                _acquired_at=acquired_at,
                run_id=run_id,
                trusted_now=trusted_now,
            )
            if fact_package is not None:
                fact_packages[ticker] = fact_package
            inventory.append({
                "rank": candidate["rank"],
                "ticker": ticker,
                "instrument_id": candidate["instrument_id"],
                "company_identity": payload.get("entityName") or identity["title"],
                "accounting_framework": extraction.get("accounting_framework", "unknown_or_unsupported"),
                "cik": cik,
                "source_family": "official_sec_companyfacts",
                "source_url": source_url,
                "source_snapshot_path": snapshot_path.as_posix(),
                "source_snapshot_sha256": checksum,
                "source_acquisition_status": "acquired",
                "acquired_at": acquired_at,
                **extraction,
            })
        except Exception as exc:
            persist_sec_companyfacts_provider_error(
                ticker=ticker,
                cik=cik,
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                root_dir=source_root,
            )
            inventory.append(_blocked_inventory(candidate, f"SOURCE_ACQUISITION_FAILED:{type(exc).__name__}", cik=cik))
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "provider": "official_sec_companyfacts",
        "provider_calls_performed": len([row for row in inventory if row.get("cik")]),
        "accounting_framework_inventory": {
            "equities_framework_unknown_not_inspected": sum(
                row.get("accounting_framework") == "unknown_not_inspected" for row in inventory
            ),
            "source_validated_us_gaap": sum(
                row.get("accounting_framework") == "us_gaap" for row in inventory
            ),
            "source_validated_ifrs": sum(
                row.get("accounting_framework") == "ifrs" for row in inventory
            ),
            "unsupported_or_missing_framework": sum(
                row.get("accounting_framework") in {"unknown_or_unsupported", "ambiguous"}
                for row in inventory
            ),
            "non_equity_candidates": 0,
        },
        "raw_sources_committed": False,
        "raw_source_retention": "local checksum-bound snapshots under the declared source root",
        "status_counts": dict(sorted(Counter(row["source_acquisition_status"] for row in inventory).items())),
        "instruments": inventory,
    }, fact_packages


def build_fact_package(
    candidate: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    source_url: str,
    source_checksum: str,
    generated_at: str,
    run_id: str,
    trusted_now: str | None = None,
    _acquired_at: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ticker = candidate["ticker"]
    company = str(payload.get("entityName") or ticker)
    namespaces = payload.get("facts")
    if not isinstance(namespaces, Mapping):
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "unknown_or_unsupported", "reason_codes": ["SUPPORTED_ACCOUNTING_FRAMEWORK_MISSING"]}
    has_us_gaap = isinstance(namespaces.get("us-gaap"), Mapping) and bool(namespaces.get("us-gaap"))
    has_ifrs = isinstance(namespaces.get("ifrs-full"), Mapping) and bool(namespaces.get("ifrs-full"))
    if has_us_gaap and has_ifrs:
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "ambiguous", "reason_codes": ["ACCOUNTING_FRAMEWORK_NAMESPACE_AMBIGUOUS"]}
    if has_ifrs:
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "ifrs", "reason_codes": ["IFRS_FRAMEWORK_UNSUPPORTED"]}
    if not has_us_gaap:
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "unknown_or_unsupported", "reason_codes": ["SUPPORTED_ACCOUNTING_FRAMEWORK_MISSING"]}
    facts_root = namespaces["us-gaap"]
    if not isinstance(facts_root, Mapping):
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "us_gaap", "reason_codes": ["US_GAAP_FACTS_MISSING"]}
    observations: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for concept, tags in _CONCEPT_TAGS.items():
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for tag in tags:
            tag_payload = facts_root.get(tag)
            if not isinstance(tag_payload, Mapping):
                continue
            units = tag_payload.get("units")
            for row in units.get("USD", []) if isinstance(units, Mapping) else []:
                if _eligible_duration_fact(row):
                    entries.append((tag, row))
        observations[concept] = entries
    try:
        aligned, period_selection = select_duration_facts(observations)
    except Data11GovernanceError as exc:
        return None, {
            "fact_extraction_status": "blocked",
            "fact_count": 0,
            "accounting_framework": "us_gaap",
            "reason_codes": ["CONFLICTING_DURATION_FACTS"],
            "error": str(exc),
        }
    if "revenue" not in aligned:
        return None, {"fact_extraction_status": "blocked", "fact_count": 0, "accounting_framework": "us_gaap", "reason_codes": ["ALIGNED_REVENUE_FACT_MISSING"]}
    facts = [
        _canonical_fact(
            candidate,
            company=company,
            concept=concept,
            raw_tag=tag,
            observation=row,
            source_url=source_url,
            source_checksum=source_checksum,
            generated_at=generated_at,
            acquired_at=_acquired_at or generated_at,
            run_id=run_id,
            trusted_now=trusted_now or generated_at,
        )
        for concept, (tag, row) in sorted(aligned.items())
    ]
    revenue_id = next(row["fact_id"] for row in facts if row["canonical_concept"] == "revenue")
    period = next(row for row in facts if row["canonical_concept"] == "revenue")
    requests = []
    for metric, numerator_concept in (("gross_margin", "gross_profit"), ("operating_margin", "operating_income")):
        numerator = next((row["fact_id"] for row in facts if row["canonical_concept"] == numerator_concept), None)
        requests.append({
            "request_id": f"{ticker.lower()}-{period['fiscal_year']}-{period['fiscal_period'].lower()}-{metric.replace('_', '-')}",
            "ticker": ticker,
            "canonical_metric": metric,
            "formula_id": metric,
            "formula_version": "2.0.0",
            "fiscal_year": period["fiscal_year"],
            "fiscal_period": period["fiscal_period"],
            "numerator_fact_ids": [numerator] if numerator else [f"missing-{numerator_concept}"],
            "denominator_fact_ids": [revenue_id],
            "component_fact_ids": [],
            "applicability": {
                "status": "applicable" if numerator else "not_applicable",
                "approval_reference": f"{run_id}-{ticker.lower()}-mapping-review-candidate",
            },
        })
    package = {
        "schema_version": FACT_PACKAGE_SCHEMA_VERSION,
        "package_id": f"{run_id}-{ticker.lower()}-primary-facts",
        "derivation_timestamp": generated_at,
        "derivation_approval_reference": f"{run_id}-{ticker.lower()}-derivation-approval-candidate",
        "facts": facts,
        "derivation_requests": requests,
    }
    return package, {
        "fact_extraction_status": "candidate_ready",
        "accounting_framework": "us_gaap",
        "fact_count": len(facts),
        "reporting_period": f"{period['fiscal_year']}-{period['fiscal_period']}",
        "source_publication_date": period["source_publication_date"],
        "period_start": period["period_start"],
        "period_end": period["period_end"],
        "duration_days": period["duration_days"],
        "duration_class": period["duration_class"],
        "fiscal_year": period["fiscal_year"],
        "fiscal_period": period["fiscal_period"],
        "source_accession": period["source_accession"],
        "period_selection": period_selection,
        "raw_source_tags": sorted(row["raw_source_concept"] for row in facts),
        "reason_codes": ["SEPARATE_OPERATOR_APPROVAL_REQUIRED"],
    }


def derive_cohort_metrics(
    cohort: Mapping[str, Any],
    *,
    fact_packages: Mapping[str, Mapping[str, Any]],
    formula_catalog: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    instruments = []
    for candidate in (row for row in cohort["candidates"] if row["selected"]):
        ticker = candidate["ticker"]
        package = fact_packages.get(ticker)
        if package is None:
            instruments.append({
                "rank": candidate["rank"], "ticker": ticker, "instrument_id": candidate["instrument_id"],
                "status": "blocked", "reason_codes": ["PRIMARY_FACT_PACKAGE_UNAVAILABLE"], "derivations": [],
            })
            continue
        derived, validation = derive_primary_source_metrics(package, formula_catalog)
        if derived is None:
            instruments.append({
                "rank": candidate["rank"], "ticker": ticker, "instrument_id": candidate["instrument_id"],
                "status": "failed", "reason_codes": sorted({row["code"] for row in validation.get("issues", [])}),
                "derivations": [],
            })
            continue
        rows = derived["derivations"]
        successful = [row for row in rows if row["status"] == "derived"]
        status = "pending_approval" if successful else "blocked"
        reason_codes = ["CHECKSUM_BOUND_DERIVATION_APPROVAL_REQUIRED"] if successful else sorted({code for row in rows for code in row.get("reason_codes", [])})
        instruments.append({
            "rank": candidate["rank"],
            "ticker": ticker,
            "instrument_id": candidate["instrument_id"],
            "status": status,
            "reason_codes": reason_codes,
            "fact_package_checksum": derived["fact_package_checksum"],
            "formula_catalog_checksum": derived["formula_catalog_checksum"],
            "approval_decision_reference": derived["approval_decision_reference"],
            "approval_state": "pending_no_authority",
            "successful_metric_count": len(successful),
            "blocked_metric_count": len(rows) - len(successful),
            "derivations": rows,
        })
    counts = Counter(row["status"] for row in instruments)
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "engine": "market-engine-data10-primary-source-metric-derivation-engine-v2",
        "approval_boundary": "Pending derivation candidates grant no DATA07, DATA06, or RUN31 authority.",
        "status_counts": dict(sorted(counts.items())),
        "instruments": instruments,
    }


def build_metric_comparison(
    funnel: Mapping[str, Any],
    cohort: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    derivation_results: Mapping[str, Any],
    *,
    trusted_now: str | None = None,
) -> dict[str, Any]:
    selected = set(cohort["selected_tickers"])
    inventory = {row["ticker"]: row for row in source_inventory["instruments"]}
    results = {row["ticker"]: row for row in derivation_results["instruments"]}
    rows = []
    for candidate in funnel["candidates"]:
        ticker = candidate["ticker"]
        result = results.get(ticker)
        derived = {
            row["canonical_metric"]: row
            for row in (result or {}).get("derivations", [])
            if row.get("status") == "derived"
        }
        inventory_row = inventory.get(ticker) or {}
        period = inventory_row.get("reporting_period")
        publication_date = inventory_row.get("source_publication_date")
        freshness_detail = _freshness_detail(
            publication_date,
            inventory_row.get("acquired_at"),
            trusted_now or inventory_row.get("acquired_at"),
        )
        freshness = freshness_detail["effective_freshness"]
        rows.append({
            "rank": candidate["rank"],
            "ticker": ticker,
            "instrument_id": candidate["instrument_id"],
            "selected_for_pilot": ticker in selected,
            "cohort_selection_reason_codes": [
                next(row["selection_reason"] for row in cohort["candidates"] if row["ticker"] == ticker)
            ],
            "technical_artifact_reference": funnel["source_bindings"]["ranking_path"],
            "technical_as_of_date": funnel["source_bindings"]["cutoff_date"],
            "issuer": inventory_row.get("company_identity"),
            "accounting_framework": inventory_row.get("accounting_framework"),
            "fiscal_period": period,
            "evidence_freshness": freshness,
            "fundamental_context_status": (
                "stale" if freshness == "stale" else (
                    "candidate_partial_pending_approval" if derived else ("blocked" if ticker in selected else "missing")
                )
            ),
            "metrics": {
                "revenue_growth_yoy": _missing_metric("direct", "DIRECT_APPROVED_EVIDENCE_MISSING"),
                "eps_growth_yoy": _missing_metric("direct", "DIRECT_APPROVED_EVIDENCE_MISSING"),
                **{
                    metric: {
                        "value": format(row["calculation_result"], ".12f"),
                        "status": "pending_approval",
                        "evidence_type": "derived",
                        "formula_id": row["formula_id"],
                        "formula_version": row["formula_version"],
                        "reporting_period": row["reporting_period"],
                        "period_start": (row.get("period") or {}).get("period_start"),
                        "period_end": (row.get("period") or {}).get("period_end"),
                        "duration_days": inventory_row.get("duration_days"),
                        "duration_class": inventory_row.get("duration_class"),
                        "fiscal_year": row.get("fiscal_year"),
                        "fiscal_period": row.get("fiscal_period"),
                        "accounting_framework": inventory_row.get("accounting_framework"),
                        "source_concepts": sorted(
                            fact.get("raw_source_concept") for fact in row.get("input_facts") or []
                        ),
                        "source_date": publication_date,
                        "source_publication_date": publication_date,
                        "source_reference": inventory_row.get("source_url"),
                        "lineage_checksum": row["calculation_checksum"],
                        "freshness_status": freshness,
                        "artifact_freshness_status": freshness_detail["artifact_freshness"],
                        "approval_status": "pending_no_authority",
                        "comparability_status": "pending_comparability_evaluation",
                        "comparability_reason_codes": [],
                    }
                    for metric, row in sorted(derived.items())
                },
            },
            "missing_metrics": sorted({"revenue_growth_yoy", "eps_growth_yoy", "gross_margin", "operating_margin", "debt_to_equity"} - set(derived)),
            "blocked_metrics": sorted(
                row.get("canonical_metric")
                for row in (result or {}).get("derivations", [])
                if row.get("status") == "blocked"
            ),
            "approval_status": (result or {}).get("approval_state", "not_applicable"),
            "downstream_eligible": False,
            "downstream_blockers": (
                (result or {}).get("reason_codes")
                if ticker in selected else ["NOT_SELECTED_FOR_BOUNDED_PILOT"]
            ),
        })
    comparability = metric_comparability(rows)
    for candidate in rows:
        for metric, evidence in candidate["metrics"].items():
            status, reasons = comparability[(candidate["ticker"], metric)]
            evidence["comparability_status"] = status
            evidence["comparability_reason_codes"] = reasons
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "candidate_count": len(rows),
        "boundary": "Comparison is evidence inventory only and is not ranking, advice, allocation, or execution logic.",
        "candidates": rows,
    }


def build_downstream_index(cohort: Mapping[str, Any], derivation_results: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper that now uses the checksum-bound authoritative prestate."""
    return build_downstream_measurement(
        cohort,
        derivation_results,
        load_downstream_prestate(),
        downstream_executed=False,
    )


def persist_approval_bundles(
    *,
    output_dir: Path,
    run_id: str,
    inventory: Mapping[str, Any],
    fact_packages: Mapping[str, Mapping[str, Any]],
    formula_catalog: Mapping[str, Any],
    derivation_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory_by_ticker = {row["ticker"]: row for row in inventory["instruments"]}
    bundles = []
    for result in derivation_results["instruments"]:
        if result["status"] != "pending_approval":
            continue
        ticker = result["ticker"]
        fact_package = fact_packages[ticker]
        derived, validation = derive_primary_source_metrics(fact_package, formula_catalog)
        if derived is None:
            raise TargetedDerivationError(f"persisted replay unexpectedly failed for {ticker}")
        source = inventory_by_ticker[ticker]
        bundles.append(
            persist_approval_bundle(
                bundle_dir=output_dir / "approval_candidates" / ticker,
                run_id=run_id,
                ticker=ticker,
                cik=source["cik"],
                source_url=source["source_url"],
                full_source_snapshot_sha256=source["source_snapshot_sha256"],
                fact_package=fact_package,
                formula_catalog=formula_catalog,
                derived_package=derived,
                derivation_validation=validation,
            )
        )
    return bundles


def build_summary(
    cohort: Mapping[str, Any],
    inventory: Mapping[str, Any],
    results: Mapping[str, Any],
    downstream: Mapping[str, Any],
) -> dict[str, Any]:
    acquired = sum(row["source_acquisition_status"] == "acquired" for row in inventory["instruments"])
    pending = sum(row["status"] == "pending_approval" for row in results["instruments"])
    blocked = sum(row["status"] == "blocked" for row in results["instruments"])
    failed = sum(row["status"] == "failed" for row in results["instruments"])
    successful_metrics = sum(row.get("successful_metric_count", 0) for row in results["instruments"])
    framework_counts = Counter(row.get("accounting_framework") for row in inventory["instruments"])
    reason_counts = Counter(code for row in results["instruments"] for code in row["reason_codes"])
    status = "completed" if pending >= cohort["minimum_safe_processing_target"] and not blocked and not failed else "completed_with_blockers"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": status,
        "cohort_size": cohort["cohort_size"],
        "attempted_count": len(results["instruments"]),
        "source_acquired_count": acquired,
        "safely_processed_candidate_count": pending,
        "minimum_safe_processing_target": cohort["minimum_safe_processing_target"],
        "minimum_safe_processing_target_met": pending >= cohort["minimum_safe_processing_target"],
        "approved_import_count": 0,
        "derived_success_count": pending,
        "candidate_partial_pending_approval_count": pending,
        "pending_approval_count": pending,
        "blocked_instrument_count": blocked,
        "failed_instrument_count": failed,
        "derived_candidate_metric_count": successful_metrics,
        "direct_metric_count": 0,
        "us_gaap_count": framework_counts.get("us_gaap", 0),
        "ifrs_count": framework_counts.get("ifrs", 0),
        "blockers_by_category": dict(sorted(reason_counts.items())),
        "downstream_status": {
            "data07": "not_executed_no_approved_evidence",
            "data06": "not_executed",
            "run31": "not_executed",
        },
        "guardrails": {
            "decision_engine_changed": False,
            "allocation_logic_added": False,
            "broker_execution_performed": False,
            "portfolio_or_watchlist_mutation_performed": False,
            "publication_performed": False,
            "market_data_mutation_performed": False,
        },
    }


def render_report(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    rows = [
        "# ME-DATA11 Targeted Diversified Fundamental Derivation Pilot",
        "",
        f"Run status: `{summary['run_status']}`",
        "",
        "## Outcome",
        "",
        f"- Cohort: {summary['cohort_size']}",
        f"- Official primary sources acquired: {summary['source_acquired_count']}",
        f"- Safely processed derivation candidates: {summary['safely_processed_candidate_count']}",
        f"- Derived metric candidates: {summary['derived_candidate_metric_count']}",
        f"- Approved downstream imports: {summary['approved_import_count']}",
        "",
        "All derived results remain pending, checksum-bound candidates with no downstream authority. "
        "No DATA07, DATA06, or RUN31 execution occurred because no separate operator approval was supplied.",
        "",
        "## Top-25 comparison",
        "",
        "| Rank | Ticker | Pilot | Derived candidates | Downstream eligible |",
        "|---:|---|---|---:|---|",
    ]
    for row in comparison["candidates"]:
        rows.append(
            f"| {row['rank']} | {row['ticker']} | {'yes' if row['selected_for_pilot'] else 'no'} | "
            f"{sum(metric['evidence_type'] == 'derived' for metric in row['metrics'].values())} | no |"
        )
    rows.extend([
        "",
        "This artifact is evidence inventory only. It does not rank fundamentals, recommend instruments, "
        "determine tradeability, allocate capital, or authorize execution.",
        "",
    ])
    return "\n".join(rows)


def build_manifest(
    *,
    run_id: str,
    generated_at: str,
    ranking_path: Path,
    ranking_manifest_path: Path,
    universe_path: Path,
    formula_catalog_path: Path,
    source_root: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
    downstream_prestate: Mapping[str, Any] | None = None,
    trusted_now: str | None = None,
    approval_bundles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    outputs = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    checksums = {name: _sha256(output_dir / name) for name in outputs}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "trusted_now": trusted_now or generated_at,
        "status": summary["run_status"],
        "inputs": {
            "candidate_ranking": {"path": ranking_path.as_posix(), "sha256": _sha256(ranking_path)},
            "ranking_manifest": {"path": ranking_manifest_path.as_posix(), "sha256": _sha256(ranking_manifest_path)},
            "canonical_universe": {"path": universe_path.as_posix(), "sha256": _sha256(universe_path)},
            "formula_catalog": {"path": formula_catalog_path.as_posix(), "sha256": _sha256(formula_catalog_path)},
            "run30_authority": {
                "path": (authority or {}).get("authority_path"),
                "sha256": (authority or {}).get("authority_sha256"),
            },
            "downstream_prestate_authority": {
                "path": (downstream_prestate or {}).get("authority_path"),
                "sha256": (downstream_prestate or {}).get("authority_sha256"),
            },
        },
        "source_snapshot_root": source_root.as_posix(),
        "outputs": outputs,
        "output_checksums": checksums,
        "approval_candidates": list(approval_bundles),
        "raw_sources_committed": False,
        "guardrails": summary["guardrails"],
    }


def _ticker_index(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    rows = payload.values() if isinstance(payload, Mapping) else []
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik = str(row.get("cik_str") or "")
        if ticker and cik.isdigit():
            result[ticker] = {"cik": cik.zfill(10), "title": str(row.get("title") or ticker)}
    if not result:
        raise TargetedDerivationError("official SEC ticker index is empty or malformed")
    return result


def _eligible_duration_fact(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and isinstance(row.get("val"), (int, float))
        and not isinstance(row.get("val"), bool)
        and row.get("form") in {"10-Q", "10-K", "10-Q/A", "10-K/A"}
        and row.get("fp") in {"Q1", "Q2", "Q3", "Q4", "FY"}
        and all(isinstance(row.get(key), str) and row.get(key) for key in ("start", "end", "filed", "accn"))
    )


def _latest_aligned_observations(
    observations: Mapping[str, list[tuple[str, Mapping[str, Any]]]],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    revenue = sorted(observations.get("revenue", []), key=lambda item: _fact_sort_key(item[1]), reverse=True)
    if not revenue:
        return {}
    revenue_fact = revenue[0]
    key = _alignment_key(revenue_fact[1])
    aligned = {"revenue": revenue_fact}
    for concept in ("gross_profit", "operating_income"):
        match = next((item for item in observations.get(concept, []) if _alignment_key(item[1]) == key), None)
        if match is not None:
            aligned[concept] = match
    return aligned


def _alignment_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return row.get("start"), row.get("end"), row.get("fy"), row.get("fp"), row.get("accn")


def _fact_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("filed")), str(row.get("end")), str(row.get("accn"))


def _canonical_fact(
    candidate: Mapping[str, Any],
    *,
    company: str,
    concept: str,
    raw_tag: str,
    observation: Mapping[str, Any],
    source_url: str,
    source_checksum: str,
    generated_at: str,
    acquired_at: str,
    run_id: str,
    trusted_now: str,
) -> dict[str, Any]:
    ticker = candidate["ticker"]
    fiscal_year = int(observation["fy"])
    fiscal_period = str(observation["fp"])
    duration = duration_metadata(observation)
    validate_temporal_boundary(
        generated_at=generated_at,
        acquired_at=acquired_at,
        source_publication_date=str(observation["filed"]),
        trusted_now=trusted_now,
    )
    return {
        "fact_id": f"{ticker.lower()}-{concept.replace('_', '-')}-{fiscal_year}-{fiscal_period.lower()}-{observation['end']}",
        "ticker": ticker,
        "instrument_id": candidate["instrument_id"],
        "company_identity": company,
        "accounting_framework": "us_gaap",
        "canonical_concept": concept,
        "raw_source_concept": f"us-gaap:{raw_tag}",
        "value": observation["val"],
        "unit": "USD",
        "currency": "USD",
        "scale": 0,
        "period_type": "duration",
        "period_start": observation["start"],
        "period_end": observation["end"],
        "duration_days": duration["duration_days"],
        "duration_class": duration["duration_class"],
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "source_name": "SEC CompanyFacts",
        "source_reference": source_url,
        "source_accession": observation["accn"],
        "source_document_checksum": source_checksum,
        "source_publication_date": observation["filed"],
        "observed_at": acquired_at,
        "acquired_at": acquired_at,
        "parser_version": "market-engine-data11-sec-companyfacts-canonical-extraction-v1",
        "source_approval_reference": f"{run_id}-{ticker.lower()}-source-approval-candidate",
        "canonical_mapping_approval_reference": f"{run_id}-{ticker.lower()}-mapping-review-candidate",
    }


def _blocked_inventory(candidate: Mapping[str, Any], reason: str, *, cik: str | None = None) -> dict[str, Any]:
    return {
        "rank": candidate["rank"],
        "ticker": candidate["ticker"],
        "instrument_id": candidate["instrument_id"],
        "accounting_framework": "unknown_not_inspected",
        "cik": cik,
        "source_family": "official_sec_companyfacts",
        "source_acquisition_status": "blocked",
        "fact_extraction_status": "blocked",
        "fact_count": 0,
        "reason_codes": [reason],
    }


def _missing_metric(evidence_type: str, reason: str) -> dict[str, Any]:
    return {
        "value": None,
        "status": "missing",
        "evidence_type": evidence_type,
        "formula_id": None,
        "formula_version": None,
        "reporting_period": None,
        "source_date": None,
        "source_reference": None,
        "lineage_checksum": None,
        "comparability_status": "not_comparable_missing_evidence",
        "comparability_blockers": [reason],
    }


def _freshness_detail(source_date: Any, acquired_at: Any, trusted_now: Any) -> dict[str, Any]:
    if not all(isinstance(value, str) for value in (source_date, acquired_at, trusted_now)):
        return {"artifact_freshness": "missing", "effective_freshness": "missing"}
    try:
        return effective_freshness(
            source_publication_date=source_date,
            acquired_at=acquired_at,
            trusted_now=trusted_now,
        )
    except Data11GovernanceError:
        return {"artifact_freshness": "invalid", "effective_freshness": "invalid"}


def _freshness_status(source_date: Any, acquired_at: Any) -> str:
    return _freshness_detail(source_date, acquired_at, acquired_at)["artifact_freshness"]


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TargetedDerivationError(f"official JSON response is not an object: {url}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded ME-DATA11 targeted derivation pilot.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--ranking-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--canonical-universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--formula-catalog", type=Path, default=DEFAULT_FORMULA_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--cohort-size", type=int, default=10)
    args = parser.parse_args(argv)
    _, output_dir = run_targeted_derivation(
        run_id=args.run_id,
        ranking_path=args.ranking,
        ranking_manifest_path=args.ranking_manifest,
        universe_path=args.canonical_universe,
        formula_catalog_path=args.formula_catalog,
        output_root=args.output_root,
        source_root=args.source_root,
        cohort_size=args.cohort_size,
    )
    with (output_dir / "manifest.json").open(encoding="utf-8") as handle:
        summary = {"run_status": json.load(handle)["status"]}
    writer = csv.writer(__import__("sys").stdout)
    writer.writerow(("run_status", summary["run_status"]))
    writer.writerow(("evidence_path", output_dir.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
