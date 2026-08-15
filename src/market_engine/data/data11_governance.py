from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from market_engine.data.data11_execution import (
    _BundleArtifact,
    _ExecutionContext,
    _execute_bound_stage_chain,
    validated_after_payload,
)
from market_engine.data.primary_source_metric_derivation import (
    DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION,
    DERIVED_PACKAGE_SCHEMA_VERSION,
    ENGINE_VERSION,
    FORMULA_CATALOG_SCHEMA_VERSION,
    derive_primary_source_metrics,
)


RUN30_AUTHORITY_SCHEMA = "market-engine-data11-run30-input-authority-v1"
DOWNSTREAM_AUTHORITY_SCHEMA = "market-engine-data11-downstream-prestate-authority-v1"
SOURCE_EVIDENCE_SCHEMA = "market-engine-data11-minimal-primary-source-evidence-v1"
MAPPING_REVIEW_SCHEMA = "market-engine-data11-canonical-mapping-review-v1"
APPROVAL_DECISION_SCHEMA = "market-engine-data11-approval-decision-v1"
APPROVAL_VALIDATION_SCHEMA = "market-engine-data11-approval-validation-v1"
GOVERNED_CANDIDATE_SCHEMA = DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION
DEFAULT_RUN30_AUTHORITY = Path("config/market_engine/data11_run30_input_authority.json")
DEFAULT_DOWNSTREAM_AUTHORITY = Path("config/market_engine/data11_downstream_prestate_authority.json")
FRESHNESS_MAX_AGE_DAYS = 120
REQUIRED_REVIEWER_ROLES = ("Operator", "Data Steward", "Governance Auditor")
REQUIRED_REVIEWS = (
    "source_authenticity",
    "primary_source_status",
    "permitted_local_use",
    "issuer_cik_instrument_identity",
    "reporting_period",
    "duration_classification",
    "raw_tag_to_canonical_mapping",
    "formula_applicability",
    "formula_version",
    "unit_currency_scale",
    "freshness",
    "calculation_replay",
    "publication_boundary",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class Data11GovernanceError(ValueError):
    pass


@dataclass(frozen=True)
class _ValidatedApprovalMaterial:
    decision_path: str
    decision_sha256: str
    decision_content: bytes
    artifacts: tuple[_BundleArtifact, ...]


def canonical_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or _CANONICAL_UTC.fullmatch(value) is None:
        raise Data11GovernanceError(f"{field} must be canonical timezone-aware UTC with trailing Z")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise Data11GovernanceError(f"{field} is not a valid UTC timestamp") from exc
    return parsed


def canonical_utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Data11GovernanceError("timestamp must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_temporal_boundary(
    *,
    generated_at: str,
    acquired_at: str,
    source_publication_date: str,
    trusted_now: str,
) -> dict[str, Any]:
    generated = canonical_utc(generated_at, field="generated_at")
    acquired = canonical_utc(acquired_at, field="acquired_at")
    trusted = canonical_utc(trusted_now, field="trusted_now")
    try:
        publication = date.fromisoformat(source_publication_date)
    except (TypeError, ValueError) as exc:
        raise Data11GovernanceError("source_publication_date must be a valid ISO date") from exc
    if generated > trusted:
        raise Data11GovernanceError("generated_at must not be after trusted_now")
    if acquired > trusted:
        raise Data11GovernanceError("acquired_at must not be after trusted_now")
    if publication > acquired.date():
        raise Data11GovernanceError("source_publication_date must not be after acquired_at")
    age_days = (acquired.date() - publication).days
    return {
        "artifact_freshness": "stale" if age_days > FRESHNESS_MAX_AGE_DAYS else "current",
        "artifact_age_days": age_days,
        "generated_at": canonical_utc_text(generated),
        "acquired_at": canonical_utc_text(acquired),
        "trusted_now": canonical_utc_text(trusted),
    }


def effective_freshness(*, source_publication_date: str, acquired_at: str, trusted_now: str) -> dict[str, Any]:
    temporal = validate_temporal_boundary(
        generated_at=acquired_at,
        acquired_at=acquired_at,
        source_publication_date=source_publication_date,
        trusted_now=trusted_now,
    )
    trusted = canonical_utc(trusted_now, field="trusted_now").date()
    publication = date.fromisoformat(source_publication_date)
    effective_age = (trusted - publication).days
    temporal["effective_age_days"] = effective_age
    temporal["effective_freshness"] = "stale" if effective_age > FRESHNESS_MAX_AGE_DAYS else "current"
    return temporal


def validate_authoritative_run30(
    *,
    repository_root: str | Path = ".",
    authority_path: str | Path = DEFAULT_RUN30_AUTHORITY,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    expected_authority = (root / DEFAULT_RUN30_AUTHORITY).resolve()
    supplied_authority = _trusted_path(root, authority_path, label="RUN30 authority binding")
    if supplied_authority != expected_authority:
        raise Data11GovernanceError("RUN30 authority binding must be the tracked canonical contract")
    authority = _strict_json(supplied_authority)
    if authority.get("schema_version") != RUN30_AUTHORITY_SCHEMA:
        raise Data11GovernanceError("RUN30 authority binding schema is invalid")
    artifacts = _load_bound_artifacts(root, authority)
    ranking = artifacts["run30_ranking"]
    manifest = artifacts["run30_manifest"]
    index = artifacts["run30_universe_index"]
    universe = artifacts["canonical_universe"]
    expected_run = authority.get("run_id")
    if {ranking.get("run_id"), manifest.get("run_id"), index.get("run_id")} != {expected_run}:
        raise Data11GovernanceError("RUN30 bound run identities do not reconcile")
    expected_version = authority.get("canonical_universe_version")
    if universe.get("universe_version") != expected_version or manifest.get("input", {}).get("universe_version") != expected_version:
        raise Data11GovernanceError("RUN30 canonical universe binding is inconsistent")
    if ranking.get("ranking_policy", {}).get("ranking_scope") != authority.get("ranking_scope"):
        raise Data11GovernanceError("RUN30 ranking scope is not authoritative")
    index_rows = index.get("instruments")
    if not isinstance(index_rows, list) or len(index_rows) != authority.get("canonical_universe_size"):
        raise Data11GovernanceError("RUN30 universe index size is invalid")
    by_instrument: dict[str, Mapping[str, Any]] = {}
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in index_rows:
        if not isinstance(row, Mapping):
            raise Data11GovernanceError("RUN30 universe index contains a non-object instrument")
        instrument_id = str(row.get("instrument_id") or "")
        symbol = str(row.get("symbol") or "")
        if not instrument_id or instrument_id in by_instrument:
            raise Data11GovernanceError("RUN30 universe index has duplicate or missing instrument identity")
        by_instrument[instrument_id] = row
        by_symbol.setdefault(symbol, []).append(row)
    ambiguous = sorted(symbol for symbol, rows in by_symbol.items() if len(rows) != 1)
    if ambiguous:
        raise Data11GovernanceError("RUN30 universe index contains ambiguous symbols")
    candidates = ranking.get("candidates")
    if not isinstance(candidates, list):
        raise Data11GovernanceError("RUN30 ranking candidates are missing")
    eligible = [row for row in candidates if isinstance(row, Mapping) and row.get("ranking_eligible") is True]
    top = eligible[:25]
    if len(top) != 25:
        raise Data11GovernanceError("RUN30 ranking has fewer than 25 eligible candidates")
    seen: set[str] = set()
    for candidate in top:
        _reconcile_candidate(candidate, by_instrument, seen)
    return {
        "authority": authority,
        "authority_path": supplied_authority.relative_to(root).as_posix(),
        "authority_sha256": _sha256(supplied_authority),
        "ranking": ranking,
        "manifest": manifest,
        "universe_index": index,
        "canonical_universe": universe,
        "top_candidates": top,
        "artifact_bindings": {
            name: {"path": item["path"], "sha256": item["sha256"]}
            for name, item in authority["trusted_artifacts"].items()
        },
    }


def _reconcile_candidate(
    candidate: Mapping[str, Any],
    by_instrument: Mapping[str, Mapping[str, Any]],
    seen: set[str],
) -> None:
    instrument_id = str(candidate.get("instrument_id") or "")
    if instrument_id in seen:
        raise Data11GovernanceError("RUN30 top-25 contains a duplicate instrument")
    seen.add(instrument_id)
    indexed = by_instrument.get(instrument_id)
    if indexed is None:
        raise Data11GovernanceError("RUN30 candidate is not present in the bound canonical universe index")
    for field in ("instrument_id", "symbol", "source_symbol", "ranking_eligible", "ranking_scope"):
        if candidate.get(field) != indexed.get(field):
            raise Data11GovernanceError(f"RUN30 candidate {field} does not match canonical authority")
    if instrument_id.split(":", 1)[0] != indexed.get("asset_type"):
        raise Data11GovernanceError("RUN30 candidate asset type does not match canonical authority")
    if indexed.get("final_processing_status") != "eligible_analyzed" or indexed.get("analysis_status") != "analysed":
        raise Data11GovernanceError("RUN30 candidate is not analysis eligible")
    trace = candidate.get("traceability")
    price = indexed.get("price_history")
    if not isinstance(trace, Mapping) or not isinstance(price, Mapping):
        raise Data11GovernanceError("RUN30 candidate price-history authority is missing")
    source_symbol = str(indexed["source_symbol"])
    expected_path = f"data/processed/{source_symbol}.csv"
    if price.get("artifactpath") != expected_path or trace.get("price_history_path") != expected_path:
        raise Data11GovernanceError("RUN30 candidate price-history path is not canonical")
    path = Path(expected_path)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "processed"):
        raise Data11GovernanceError("RUN30 candidate price-history path escapes the canonical root")
    if not isinstance(price.get("checksum"), str) or _SHA256.fullmatch(str(price["checksum"])) is None:
        raise Data11GovernanceError("RUN30 candidate price-history checksum is invalid")
    for trace_field, price_field in (("start_date", "start_date"), ("end_date", "end_date"), ("row_count", "row_count")):
        if trace.get(trace_field) != price.get(price_field):
            raise Data11GovernanceError("RUN30 candidate traceability does not match canonical price history")


def duration_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        start = date.fromisoformat(str(row["start"]))
        end = date.fromisoformat(str(row["end"]))
    except (KeyError, ValueError) as exc:
        raise Data11GovernanceError("duration fact start/end is invalid") from exc
    days = (end - start).days + 1
    if days <= 0:
        raise Data11GovernanceError("duration fact has a non-positive duration")
    fiscal_period = str(row.get("fp") or "")
    if fiscal_period == "FY":
        duration_class = "annual" if 300 <= days <= 430 else "invalid"
    elif fiscal_period == "Q1":
        duration_class = "discrete_quarter" if 60 <= days <= 130 else "invalid"
    elif fiscal_period in {"Q2", "Q3"}:
        duration_class = "discrete_quarter" if 60 <= days <= 130 else ("year_to_date" if 131 <= days <= 310 else "invalid")
    elif fiscal_period == "Q4":
        duration_class = "discrete_quarter" if 60 <= days <= 130 else ("annual" if 300 <= days <= 430 else "invalid")
    else:
        duration_class = "invalid"
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "duration_days": days,
        "duration_class": duration_class,
        "fiscal_year": row.get("fy"),
        "fiscal_period": fiscal_period,
        "accession": row.get("accn"),
        "source_publication_date": row.get("filed"),
    }


def select_duration_facts(
    observations: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
) -> tuple[dict[str, tuple[str, Mapping[str, Any]]], dict[str, Any]]:
    normalized: dict[str, list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = {}
    for concept, rows in observations.items():
        semantic: dict[tuple[Any, ...], tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
        for raw_tag, row in rows:
            metadata = duration_metadata(row)
            if metadata["duration_class"] == "invalid":
                continue
            key = (
                metadata["period_start"],
                metadata["period_end"],
                metadata["duration_class"],
                metadata["fiscal_year"],
                metadata["fiscal_period"],
                metadata["accession"],
                metadata["source_publication_date"],
            )
            existing = semantic.get(key)
            if existing is not None and existing[1].get("val") != row.get("val"):
                raise Data11GovernanceError("conflicting duration facts share one semantic identity")
            if existing is None or raw_tag < existing[0]:
                semantic[key] = (raw_tag, row, metadata)
        normalized[concept] = list(semantic.values())
    revenue = normalized.get("revenue") or []
    if not revenue:
        return {}, {"status": "blocked", "reason_codes": ["ALIGNED_REVENUE_FACT_MISSING"]}
    reporting_identity = max(
        (
            str(meta["period_end"]),
            str(meta["source_publication_date"]),
            int(meta["fiscal_year"]),
            str(meta["fiscal_period"]),
            str(meta["accession"]),
        )
        for _, _, meta in revenue
    )
    reporting_rows = [item for item in revenue if (
        str(item[2]["period_end"]), str(item[2]["source_publication_date"]), int(item[2]["fiscal_year"]),
        str(item[2]["fiscal_period"]), str(item[2]["accession"]),
    ) == reporting_identity]
    fiscal_period = reporting_identity[3]
    preferred = "annual" if fiscal_period in {"FY", "Q4"} else "discrete_quarter"
    reporting_rows.sort(
        key=lambda item: (
            item[2]["duration_class"] != preferred,
            item[2]["duration_days"],
            item[2]["period_start"],
            item[0],
            json.dumps(item[1], sort_keys=True),
        )
    )
    selected_revenue = reporting_rows[0]
    selected_meta = selected_revenue[2]
    aligned: dict[str, tuple[str, Mapping[str, Any]]] = {"revenue": (selected_revenue[0], selected_revenue[1])}
    alignment = (
        selected_meta["period_start"], selected_meta["period_end"], selected_meta["duration_class"],
        selected_meta["fiscal_year"], selected_meta["fiscal_period"], selected_meta["accession"],
    )
    for concept in sorted(set(normalized) - {"revenue"}):
        matches = [item for item in normalized[concept] if (
            item[2]["period_start"], item[2]["period_end"], item[2]["duration_class"],
            item[2]["fiscal_year"], item[2]["fiscal_period"], item[2]["accession"],
        ) == alignment]
        if matches:
            matches.sort(key=lambda item: (item[0], json.dumps(item[1], sort_keys=True)))
            aligned[concept] = (matches[0][0], matches[0][1])
    return aligned, {"status": "selected", **selected_meta, "selection_policy": "latest_reporting_identity_then_discrete_quarter_or_annual"}


def persist_approval_bundle(
    *,
    bundle_dir: str | Path,
    run_id: str,
    ticker: str,
    cik: str,
    source_url: str,
    full_source_snapshot_sha256: str,
    fact_package: Mapping[str, Any],
    formula_catalog: Mapping[str, Any],
    derived_package: Mapping[str, Any],
    derivation_validation: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(bundle_dir)
    if root.exists():
        raise FileExistsError(f"approval bundle already exists: {root}")
    root.mkdir(parents=True)
    successful = [row for row in derived_package.get("derivations") or [] if row.get("status") == "derived"]
    if not successful:
        raise Data11GovernanceError("approval bundle requires at least one successful derivation")
    source_evidence = {
        "schema_version": SOURCE_EVIDENCE_SCHEMA,
        "ticker": ticker,
        "instrument_id": successful[0]["instrument_id"],
        "issuer_identity": successful[0]["company_identity"],
        "cik": cik,
        "official_source_url": source_url,
        "full_source_snapshot_sha256": full_source_snapshot_sha256,
        "accounting_framework": "us_gaap",
        "observations": [
            {
                key: fact.get(key)
                for key in (
                    "fact_id", "canonical_concept", "raw_source_concept", "value", "unit", "currency", "scale",
                    "period_type", "period_start", "period_end", "duration_days", "duration_class", "fiscal_year",
                    "fiscal_period", "source_accession", "source_publication_date", "source_reference",
                )
            }
            for fact in fact_package["facts"]
        ],
        "boundary": "Minimal checksum-bound review evidence; human source-authenticity approval is still required.",
    }
    mapping_review = {
        "schema_version": MAPPING_REVIEW_SCHEMA,
        "decision": "pending",
        "ticker": ticker,
        "mappings": sorted(
            {
                (fact["raw_source_concept"], fact["canonical_concept"], fact["accounting_framework"])
                for fact in fact_package["facts"]
            }
        ),
        "limitations": ["This pending mapping review grants no canonical mapping authority."],
    }
    governed = _build_derived_only_governed_candidate(derived_package, fact_package, run_id=run_id, ticker=ticker)
    payloads = {
        "source_evidence.json": source_evidence,
        "mapping_review.json": mapping_review,
        "fact_package.json": dict(fact_package),
        "formula_catalog.json": dict(formula_catalog),
        "derived_package.json": dict(derived_package),
        "derivation_validation.json": dict(derivation_validation),
        "governed_package_candidate.json": governed,
    }
    for name, payload in payloads.items():
        _write_json(root / name, payload)
    decision_id = str(fact_package["derivation_approval_reference"])
    bindings = {
        name.removesuffix(".json"): {"path": name, "sha256": _sha256(root / name)}
        for name in sorted(payloads)
    }
    approval = {
        "schema_version": APPROVAL_DECISION_SCHEMA,
        "decision_id": decision_id,
        "decision": "pending",
        "ticker": ticker,
        "instrument_id": successful[0]["instrument_id"],
        "issuer_identity": successful[0]["company_identity"],
        "cik": cik,
        "reporting_period": successful[0]["reporting_period"],
        "reviewer_roles": list(REQUIRED_REVIEWER_ROLES),
        "reviews": {name: {"status": "pending"} for name in REQUIRED_REVIEWS},
        "approved_derived_metrics": sorted(row["canonical_metric"] for row in successful),
        "approved_calculation_checksums": sorted(row["calculation_checksum"] for row in successful),
        "artifact_bindings": bindings,
        "limitations": [
            "This candidate grants no authority while the decision or any review is pending.",
            "Approval must be an explicit human decision over these exact checksum-bound artifacts.",
        ],
    }
    _write_json(root / "approval_candidate.json", approval)
    return {"ticker": ticker, "bundle_path": root.as_posix(), "approval_candidate_path": (root / "approval_candidate.json").as_posix(), "approval_candidate_sha256": _sha256(root / "approval_candidate.json")}


def validate_approval_decision(
    decision_path: str | Path | None,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    validation, _ = _validate_approval_decision_with_material(
        decision_path,
        repository_root=repository_root,
    )
    return validation


def _validate_approval_decision_with_material(
    decision_path: str | Path | None,
    *,
    repository_root: str | Path = ".",
) -> tuple[dict[str, Any], _ValidatedApprovalMaterial | None]:
    issues: list[str] = []
    if not isinstance(decision_path, (str, Path)) or not Path(decision_path).is_file():
        return _approval_result(["APPROVAL_DECISION_MISSING"], None), None
    decision_file = Path(decision_path).resolve()
    try:
        decision, decision_content = _strict_json_with_bytes(decision_file)
    except Data11GovernanceError:
        return _approval_result(["APPROVAL_DECISION_MALFORMED"], decision_path), None
    decision_sha256 = _bytes_sha256(decision_content)
    if decision.get("schema_version") != APPROVAL_DECISION_SCHEMA:
        issues.append("APPROVAL_SCHEMA_INVALID")
    status = decision.get("decision")
    if status != "approved":
        issues.append({"pending": "APPROVAL_PENDING", "rejected": "APPROVAL_REJECTED", "blocked": "APPROVAL_BLOCKED"}.get(str(status), "APPROVAL_STATUS_INVALID"))
    if decision.get("reviewer_roles") != list(REQUIRED_REVIEWER_ROLES):
        issues.append("REVIEWER_ROLES_INVALID")
    reviews = decision.get("reviews") if isinstance(decision.get("reviews"), Mapping) else {}
    if any((reviews.get(name) or {}).get("status") != "approved" for name in REQUIRED_REVIEWS):
        issues.append("REQUIRED_REVIEWS_NOT_APPROVED")
    bindings = decision.get("artifact_bindings") if isinstance(decision.get("artifact_bindings"), Mapping) else {}
    payloads: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    artifact_material: dict[str, _BundleArtifact] = {}
    for name in (
        "source_evidence", "mapping_review", "fact_package", "formula_catalog", "derived_package",
        "derivation_validation", "governed_package_candidate",
    ):
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            issues.append(f"{name.upper()}_BINDING_MISSING")
            continue
        try:
            path = _decision_bound_path(decision_file.parent, str(binding.get("path") or ""), label=name)
            payload, content = _strict_json_with_bytes(path)
            checksum = _bytes_sha256(content)
            if binding.get("sha256") != checksum:
                issues.append(f"{name.upper()}_CHECKSUM_MISMATCH")
            payloads[name] = payload
            paths[name] = path
            artifact_material[name] = _BundleArtifact(
                name=name,
                filename=path.name,
                original_path=path.as_posix(),
                sha256=checksum,
                content=content,
            )
        except Data11GovernanceError:
            issues.append(f"{name.upper()}_ARTIFACT_INVALID")
    fact = payloads.get("fact_package")
    catalog = payloads.get("formula_catalog")
    derived = payloads.get("derived_package")
    validation = payloads.get("derivation_validation")
    mapping = payloads.get("mapping_review")
    source = payloads.get("source_evidence")
    governed = payloads.get("governed_package_candidate")
    if isinstance(source, Mapping) and source.get("schema_version") != SOURCE_EVIDENCE_SCHEMA:
        issues.append("SOURCE_EVIDENCE_SCHEMA_INVALID")
    if isinstance(mapping, Mapping) and mapping.get("schema_version") != MAPPING_REVIEW_SCHEMA:
        issues.append("MAPPING_REVIEW_SCHEMA_INVALID")
    if isinstance(fact, Mapping) and fact.get("schema_version") is None:
        issues.append("FACT_PACKAGE_SCHEMA_INVALID")
    if isinstance(catalog, Mapping) and catalog.get("schema_version") != FORMULA_CATALOG_SCHEMA_VERSION:
        issues.append("FORMULA_CATALOG_SCHEMA_INVALID")
    if isinstance(derived, Mapping) and derived.get("schema_version") != DERIVED_PACKAGE_SCHEMA_VERSION:
        issues.append("DERIVED_PACKAGE_SCHEMA_INVALID")
    if isinstance(governed, Mapping) and governed.get("schema_version") != GOVERNED_CANDIDATE_SCHEMA:
        issues.append("GOVERNED_PACKAGE_SCHEMA_INVALID")
    if all(isinstance(item, Mapping) for item in (fact, catalog, derived, validation)):
        try:
            replayed, replay_validation = derive_primary_source_metrics(fact, catalog)
            if replayed is None or _canonical_bytes(replayed) != _canonical_bytes(derived):
                issues.append("DERIVED_REPLAY_MISMATCH")
            if _canonical_bytes(replay_validation) != _canonical_bytes(validation):
                issues.append("DERIVATION_VALIDATION_REPLAY_MISMATCH")
        except (KeyError, TypeError, ValueError):
            issues.append("DERIVATION_REPLAY_INVALID")
    if isinstance(mapping, Mapping) and mapping.get("decision") != "approved":
        issues.append("CANONICAL_MAPPING_NOT_APPROVED")
    if isinstance(source, Mapping):
        if source.get("ticker") != decision.get("ticker") or source.get("cik") != decision.get("cik"):
            issues.append("SOURCE_IDENTITY_MISMATCH")
    if isinstance(source, Mapping) and isinstance(fact, Mapping):
        expected_observations = [
            {
                key: row.get(key)
                for key in (
                    "fact_id", "canonical_concept", "raw_source_concept", "value", "unit", "currency", "scale",
                    "period_type", "period_start", "period_end", "duration_days", "duration_class", "fiscal_year",
                    "fiscal_period", "source_accession", "source_publication_date", "source_reference",
                )
            }
            for row in fact.get("facts") or []
        ]
        if source.get("observations") != expected_observations:
            issues.append("SOURCE_EVIDENCE_FACT_RECONCILIATION_FAILED")
        fact_checksums = {row.get("source_document_checksum") for row in fact.get("facts") or []}
        if fact_checksums != {source.get("full_source_snapshot_sha256")}:
            issues.append("SOURCE_CHECKSUM_FACT_RECONCILIATION_FAILED")
    if isinstance(mapping, Mapping) and isinstance(fact, Mapping):
        expected_mappings = sorted(
            [
                [row.get("raw_source_concept"), row.get("canonical_concept"), row.get("accounting_framework")]
                for row in fact.get("facts") or []
            ]
        )
        if mapping.get("ticker") != decision.get("ticker") or mapping.get("mappings") != expected_mappings:
            issues.append("CANONICAL_MAPPING_FACT_RECONCILIATION_FAILED")
    if isinstance(fact, Mapping) and fact.get("derivation_approval_reference") != decision.get("decision_id"):
        issues.append("DECISION_REFERENCE_MISMATCH")
    if isinstance(derived, Mapping):
        successful = [row for row in derived.get("derivations") or [] if row.get("status") == "derived"]
        metrics = sorted(row["canonical_metric"] for row in successful)
        calculations = sorted(row["calculation_checksum"] for row in successful)
        periods = {row.get("reporting_period") for row in successful}
        issuers = {row.get("company_identity") for row in successful}
        instruments = {row.get("instrument_id") for row in successful}
        if decision.get("approved_derived_metrics") != metrics:
            issues.append("PARTIAL_OR_WRONG_METRIC_APPROVAL")
        if decision.get("approved_calculation_checksums") != calculations:
            issues.append("CALCULATION_CHECKSUM_SET_MISMATCH")
        if periods != {decision.get("reporting_period")}:
            issues.append("REPORTING_PERIOD_MISMATCH")
        if issuers != {decision.get("issuer_identity")} or instruments != {decision.get("instrument_id")}:
            issues.append("APPROVAL_IDENTITY_MISMATCH")
        if isinstance(fact, Mapping) and isinstance(catalog, Mapping):
            try:
                ticker = str(decision.get("ticker"))
                suffix = f"-{ticker.lower()}-primary-facts"
                fact_package_id = str(fact.get("package_id") or "")
                if not fact_package_id.endswith(suffix):
                    raise ValueError("fact package id does not bind the governed package identity")
                run_id = fact_package_id.removesuffix(suffix)
                rebuilt = _build_derived_only_governed_candidate(derived, fact, run_id=run_id, ticker=ticker)
                if isinstance(governed, Mapping):
                    if _canonical_bytes(rebuilt) != _canonical_bytes(governed):
                        issues.append("GOVERNED_PACKAGE_REPLAY_MISMATCH")
            except (IndexError, KeyError, TypeError, ValueError):
                issues.append("GOVERNED_PACKAGE_REPLAY_INVALID")
    if isinstance(governed, Mapping):
        records = governed.get("records")
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
            issues.append("GOVERNED_PACKAGE_RECORD_SET_INVALID")
        else:
            record = records[0]
            if record.get("ticker") != decision.get("ticker") or record.get("instrument_id") != decision.get("instrument_id"):
                issues.append("GOVERNED_PACKAGE_IDENTITY_MISMATCH")
            if record.get("approval_decision_reference") != decision.get("decision_id"):
                issues.append("GOVERNED_PACKAGE_DECISION_REFERENCE_MISMATCH")
            if governed.get("approval_decision_reference") != decision.get("decision_id"):
                issues.append("GOVERNED_PACKAGE_DECISION_REFERENCE_MISMATCH")
            record_metrics = record.get("metrics")
            if not isinstance(record_metrics, Mapping):
                issues.append("GOVERNED_PACKAGE_METRIC_SET_MISMATCH")
            else:
                metrics = sorted(record_metrics)
                if metrics != decision.get("approved_derived_metrics"):
                    issues.append("GOVERNED_PACKAGE_METRIC_SET_MISMATCH")
                checksums = [
                    metric.get("derivation_lineage", {}).get("calculation_checksum")
                    for metric in record_metrics.values()
                    if isinstance(metric, Mapping)
                ]
                if (
                    any(not isinstance(value, str) for value in checksums)
                    or sorted(checksums) != decision.get("approved_calculation_checksums")
                ):
                    issues.append("GOVERNED_PACKAGE_CALCULATION_SET_MISMATCH")
    execution_binding = None
    material = None
    governed_path = paths.get("governed_package_candidate")
    governed_material = artifact_material.get("governed_package_candidate")
    if not issues and isinstance(governed, Mapping) and governed_path is not None and governed_material is not None:
        execution_binding = {
            "schema_version": "market-engine-data11-approved-execution-binding-v2",
            "decision_id": decision["decision_id"],
            "decision_path": decision_file.as_posix(),
            "decision_sha256": decision_sha256,
            "ticker": decision["ticker"],
            "instrument_id": decision["instrument_id"],
            "bundle_root": decision_file.parent.as_posix(),
            "governed_package_path": governed_path.as_posix(),
            "governed_package_sha256": governed_material.sha256,
            "governed_package_id": governed["package_id"],
            "approved_metrics": decision["approved_derived_metrics"],
            "calculation_checksums": decision["approved_calculation_checksums"],
            "approval_artifact_bindings": {
                name: {"path": item.original_path, "sha256": item.sha256}
                for name, item in sorted(artifact_material.items())
            },
        }
        material = _ValidatedApprovalMaterial(
            decision_path=decision_file.as_posix(),
            decision_sha256=decision_sha256,
            decision_content=decision_content,
            artifacts=tuple(artifact_material[name] for name in sorted(artifact_material)),
        )
    result = _approval_result(
        sorted(set(issues)),
        decision_path,
        decision_id=decision.get("decision_id"),
        ticker=decision.get("ticker"),
        execution_binding=execution_binding,
    )
    return result, material


def execute_approved_candidate(
    decision_path: str | Path | None,
    *,
    data07_runner: Callable[..., Any],
    data07_operational_kwargs: Mapping[str, Any] | None = None,
    data06_runner: Callable[..., Any] | None = None,
    data06_operational_kwargs: Mapping[str, Any] | None = None,
    run31_runner: Callable[..., Any] | None = None,
    run31_operational_kwargs: Mapping[str, Any] | None = None,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    validation, material = _validate_approval_decision_with_material(
        decision_path,
        repository_root=repository_root,
    )
    context = None
    if material is not None and validation.get("validation_status") == "approved":
        try:
            runtime_settings = _execution_runtime_settings(repository_root)
            binding = validation["execution_binding"]
            context = _ExecutionContext(
                decision_id=binding["decision_id"],
                decision_path=material.decision_path,
                decision_sha256=material.decision_sha256,
                decision_content=material.decision_content,
                ticker=binding["ticker"],
                instrument_id=binding["instrument_id"],
                governed_package_path=binding["governed_package_path"],
                governed_package_sha256=binding["governed_package_sha256"],
                governed_package_id=binding["governed_package_id"],
                approved_metrics=tuple(binding["approved_metrics"]),
                calculation_checksums=tuple(binding["calculation_checksums"]),
                artifacts=material.artifacts,
                approval_validation=validation,
                runtime_settings=runtime_settings,
                repository_root=Path(repository_root).resolve().as_posix(),
            )
        except (Data11GovernanceError, KeyError, TypeError, ValueError):
            validation = {
                **validation,
                "validation_status": "blocked",
                "concrete_package_source_approved": False,
                "reason_codes": sorted(set(validation.get("reason_codes") or []) | {"EXECUTION_RUNTIME_AUTHORITY_INVALID"}),
            }
    return _execute_bound_stage_chain(
        context=context,
        approval_validation=validation,
        data07_runner=data07_runner,
        data07_operational_kwargs=data07_operational_kwargs,
        data06_runner=data06_runner,
        data06_operational_kwargs=data06_operational_kwargs,
        run31_runner=run31_runner,
        run31_operational_kwargs=run31_operational_kwargs,
    )


def _execution_runtime_settings(repository_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    run30 = validate_authoritative_run30(repository_root=root)
    prestate = load_downstream_prestate(repository_root=root)
    if prestate.get("measurement_status") != "measured":
        raise Data11GovernanceError("downstream prestate authority is invalid")
    canonical_universe = (root / run30["artifact_bindings"]["canonical_universe"]["path"]).resolve()
    data06_manifest = (root / prestate["artifact_bindings"]["data06_manifest"]["path"]).resolve()
    run31_index = (root / prestate["artifact_bindings"]["run31_compact_index"]["path"]).resolve()
    cutoff_date = run30["manifest"].get("input", {}).get("cutoff_date")
    if not isinstance(cutoff_date, str):
        raise Data11GovernanceError("RUN30 cutoff date is missing")
    price_history_root = (root / "data" / "processed").resolve().as_posix()
    return {
        "data07": {
            "batch_tier": "pilot",
            "canonical_universe": canonical_universe.as_posix(),
            "price_history_root": price_history_root,
            "baseline_data06_run": data06_manifest.parent.as_posix(),
            "baseline_run31_evidence": run31_index.parent.as_posix(),
            "raw_snapshot_root": (root / "data/market_engine/source_snapshots/fundamental_metrics").resolve().as_posix(),
            "as_of_date": cutoff_date,
        },
        "data06": {
            "canonical_universe": canonical_universe.as_posix(),
            "price_history_root": price_history_root,
            "as_of_date": cutoff_date,
        },
        "run31": {
            "canonical_universe": canonical_universe.as_posix(),
            "price_history_root": price_history_root,
            "compact_evidence_root": run31_index.parent.as_posix(),
            "freshness_reference_date": cutoff_date,
        },
    }


def load_downstream_prestate(
    *,
    repository_root: str | Path = ".",
    authority_path: str | Path = DEFAULT_DOWNSTREAM_AUTHORITY,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    expected = (root / DEFAULT_DOWNSTREAM_AUTHORITY).resolve()
    try:
        supplied = _trusted_path(root, authority_path, label="downstream authority binding")
    except Data11GovernanceError:
        return _unknown_prestate("DOWNSTREAM_AUTHORITY_MISSING_OR_INVALID")
    if supplied != expected:
        return _unknown_prestate("DOWNSTREAM_AUTHORITY_PATH_UNTRUSTED")
    try:
        authority = _strict_json(supplied)
        if authority.get("schema_version") != DOWNSTREAM_AUTHORITY_SCHEMA:
            return _unknown_prestate("DOWNSTREAM_AUTHORITY_SCHEMA_INVALID")
        artifacts = _load_bound_artifacts(root, authority)
        manifest = artifacts["data06_manifest"]
        summary = artifacts["data06_summary"]
        per_ticker = artifacts["data06_per_ticker"]
        run31 = artifacts["run31_compact_index"]
        run31_summary = artifacts["run31_evidence_summary"]
        top_checksums = artifacts["run31_top_level_checksums"]
        if top_checksums.get("evidence_coverage_summary.json") != authority["trusted_artifacts"]["run31_evidence_summary"]["sha256"]:
            return _unknown_prestate("RUN31_CHECKSUM_INDEX_MISMATCH")
        if manifest.get("canonical_universe", {}).get("universe_version") != authority.get("canonical_universe_version"):
            return _unknown_prestate("DATA06_UNIVERSE_MISMATCH")
        rows = per_ticker.get("tickers")
        if not isinstance(rows, list) or len(rows) != authority.get("canonical_universe_size"):
            return _unknown_prestate("DATA06_PER_TICKER_RECONCILIATION_FAILED")
        by_ticker = {str(row.get("ticker")): row for row in rows if isinstance(row, Mapping)}
        if len(by_ticker) != len(rows):
            return _unknown_prestate("DATA06_PER_TICKER_IDENTITY_INVALID")
        declared = summary.get("after") or {}
        counts = Counter(str(row.get("overall_fundamental_status") or "missing") for row in rows)
        before = {
            "fundamental_complete": counts["complete"],
            "fundamental_partial": counts["partial"],
            "fundamental_missing": counts["missing"],
            "invalid_stale_conflicting": sum(counts[key] for key in ("invalid", "stale", "conflicting")),
            "advice_input_ready": run31.get("metrics", {}).get("canonical_advice_input_ready"),
            "full_advice_ready": run31.get("metrics", {}).get("full_advice_ready"),
            "unable_to_advise": run31.get("advice_counts", {}).get("unable_to_advise"),
        }
        expected_counts = {
            "fundamental_complete": declared.get("fundamental_complete"),
            "fundamental_partial": declared.get("fundamental_partial"),
            "fundamental_missing": declared.get("fundamental_missing"),
            "invalid_stale_conflicting": declared.get("invalid_stale_conflicting"),
            "advice_input_ready": declared.get("canonical_advice_input_ready"),
            "full_advice_ready": declared.get("full_advice_ready"),
            "unable_to_advise": declared.get("unable_to_advise"),
        }
        if (
            before != expected_counts
            or run31_summary.get("summary", {}).get("fundamental_counts", {}).get("available")
            != before["fundamental_complete"]
        ):
            return _unknown_prestate("DOWNSTREAM_PRESTATE_COUNT_MISMATCH")
        return {
            "measurement_status": "measured",
            "reason_codes": [],
            "authority_path": supplied.relative_to(root).as_posix(),
            "authority_sha256": _sha256(supplied),
            "artifact_bindings": authority["trusted_artifacts"],
            "before": before,
            "by_ticker": by_ticker,
            "data06_run_id": manifest.get("run_id"),
            "run31_run_id": run31.get("run_id"),
        }
    except (KeyError, Data11GovernanceError):
        return _unknown_prestate("DOWNSTREAM_PRESTATE_INVALID")


def build_downstream_measurement(
    cohort: Mapping[str, Any],
    derivation_results: Mapping[str, Any],
    prestate: Mapping[str, Any],
    *,
    downstream_executed: bool,
    authoritative_after: Any = None,
) -> dict[str, Any]:
    if prestate.get("measurement_status") != "measured":
        return {
            "schema_version": "market-engine-data11-downstream-readiness-delta-v2",
            "measurement_status": "unknown_not_measured",
            "reason_codes": prestate.get("reason_codes") or ["DOWNSTREAM_PRESTATE_UNKNOWN"],
            "before": None,
            "after_authoritative": None,
            "regressions_outside_selected_cohort": "unknown_not_measured",
            "rows": [],
            "downstream_executed": downstream_executed,
        }
    after_payload = validated_after_payload(authoritative_after) if downstream_executed else prestate
    if downstream_executed and (
        after_payload is None
        or after_payload.get("measurement_status") != "measured"
        or not _fundamental_totals_reconcile(after_payload)
    ):
        return {
            "schema_version": "market-engine-data11-downstream-readiness-delta-v2",
            "measurement_status": "unknown_not_measured",
            "reason_codes": ["AUTHORITATIVE_AFTER_STATE_UNKNOWN"],
            "before": prestate["before"],
            "after_authoritative": None,
            "regressions_outside_selected_cohort": "unknown_not_measured",
            "rows": [],
            "downstream_executed": True,
        }
    selected = set(cohort["selected_tickers"])
    by_result = {row["ticker"]: row for row in derivation_results["instruments"]}
    rows = []
    for ticker in cohort["selected_tickers"]:
        baseline = prestate["by_ticker"][ticker]
        current = after_payload["by_ticker"][ticker]
        result = by_result[ticker]
        before_ready = baseline.get("canonical_advice_input_ready")
        after_ready = current.get("canonical_advice_input_ready")
        rows.append({
            "ticker": ticker,
            "instrument_id": baseline["instrument_id"],
            "before_fundamental_status": baseline["overall_fundamental_status"],
            "after_authoritative_fundamental_status": current["overall_fundamental_status"],
            "candidate_only_status": "candidate_partial_pending_approval" if result["status"] == "pending_approval" else "blocked",
            "advice_readiness_transition": (
                f"{before_ready}->{after_ready}"
                if before_ready is not None and after_ready is not None else "unknown_not_measured_per_ticker"
            ),
            "reason_codes": result["reason_codes"],
        })
    before = dict(prestate["before"])
    after = dict(before) if not downstream_executed else dict(after_payload["before"])
    outside_changes: list[dict[str, Any]] = []
    outside_regressions: list[str] = []
    status_order = {"complete": 3, "partial": 2, "missing": 1, "invalid": 0, "stale": 0, "conflicting": 0}
    if downstream_executed:
        for ticker, baseline in prestate["by_ticker"].items():
            if ticker in selected:
                continue
            current = after_payload["by_ticker"].get(ticker)
            if current is None:
                outside_changes.append({"ticker": ticker, "transition": "missing_after_record"})
                outside_regressions.append(ticker)
                continue
            old_status = str(baseline.get("overall_fundamental_status"))
            new_status = str(current.get("overall_fundamental_status"))
            old_ready = baseline.get("canonical_advice_input_ready")
            new_ready = current.get("canonical_advice_input_ready")
            if (old_status, old_ready) != (new_status, new_ready):
                outside_changes.append({
                    "ticker": ticker,
                    "fundamental_status": f"{old_status}->{new_status}",
                    "advice_input_ready": f"{old_ready}->{new_ready}",
                })
            if status_order.get(new_status, -1) < status_order.get(old_status, -1) or (old_ready is True and new_ready is not True):
                outside_regressions.append(ticker)
    return {
        "schema_version": "market-engine-data11-downstream-readiness-delta-v2",
        "measurement_status": "measured",
        "reason_codes": ["NO_APPROVED_IMPORT_AUTHORITATIVE_ARTIFACTS_UNCHANGED"] if not downstream_executed else [],
        "prestate": {
            "data06_run_id": prestate["data06_run_id"],
            "run31_run_id": prestate["run31_run_id"],
            "authority_path": prestate["authority_path"],
            "authority_sha256": prestate["authority_sha256"],
            "artifact_bindings": prestate["artifact_bindings"],
        },
        "before": before,
        "after_authoritative": after,
        "absolute_delta": {key: after[key] - before[key] for key in before},
        "candidate_only_non_authoritative": {
            "status": "candidate_partial_pending_approval",
            "ticker_count": sum(row["status"] == "pending_approval" for row in derivation_results["instruments"]),
        },
        "rows": rows,
        "changes_outside_selected_cohort": outside_changes,
        "regressions_outside_selected_cohort": 0 if not downstream_executed else len(outside_regressions),
        "outside_cohort_regression_tickers": outside_regressions,
        "regressions_outside_selected_cohort_basis": (
            "Identical checksum-bound authoritative prestate and after-state because no downstream runner executed."
            if not downstream_executed else "Measured comparison of checksum-validated before and after per-ticker artifacts."
        ),
        "outside_cohort_size": len(prestate["by_ticker"]) - len(selected),
        "downstream_executed": downstream_executed,
    }


def metric_comparability(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], tuple[str, list[str]]]:
    available: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for metric, evidence in (row.get("metrics") or {}).items():
            if isinstance(evidence, Mapping) and evidence.get("value") is not None:
                available.setdefault(metric, []).append({"ticker": row["ticker"], **evidence, "accounting_framework": row.get("accounting_framework")})
    result: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for row in rows:
        ticker = str(row["ticker"])
        for metric, evidence in (row.get("metrics") or {}).items():
            if not isinstance(evidence, Mapping) or evidence.get("value") is None:
                result[(ticker, metric)] = ("not_comparable_missing_evidence", ["MISSING_METRIC_EVIDENCE"])
                continue
            reasons: list[str] = []
            if evidence.get("freshness_status") != "current":
                reasons.append("STALE_OR_INVALID_EVIDENCE")
            peers = [peer for peer in available.get(metric, []) if peer["ticker"] != ticker]
            if not peers:
                reasons.append("NO_COMPARABLE_PEER")
            same_class = [peer for peer in peers if peer.get("duration_class") == evidence.get("duration_class")]
            if peers and not same_class:
                reasons.append("DURATION_CLASS_MISMATCH")
            same_period = [peer for peer in same_class if peer.get("fiscal_period") == evidence.get("fiscal_period")]
            if same_class and not same_period:
                reasons.append("FISCAL_PERIOD_MISMATCH")
            if any(peer.get("accounting_framework") != row.get("accounting_framework") for peer in same_period):
                reasons.append("ACCOUNTING_FRAMEWORK_DIFFERENCE")
            if reasons:
                status = "comparable_limited" if set(reasons) <= {"ACCOUNTING_FRAMEWORK_DIFFERENCE"} else "not_comparable"
            elif any(peer.get("period_start") != evidence.get("period_start") or peer.get("period_end") != evidence.get("period_end") for peer in same_period):
                status = "comparable_limited"
                reasons.append("FISCAL_CALENDAR_DIFFERENCE")
            else:
                status = "comparable"
                reasons.append("MATCHED_DEFINITION_DURATION_PERIOD_FRAMEWORK_AND_FRESHNESS")
            result[(ticker, metric)] = (status, reasons)
    return result


def _fundamental_totals_reconcile(state: Mapping[str, Any]) -> bool:
    totals = state.get("before")
    rows = state.get("by_ticker")
    if not isinstance(totals, Mapping) or not isinstance(rows, Mapping):
        return False
    counts = Counter(
        str(row.get("overall_fundamental_status") or "missing")
        for row in rows.values()
        if isinstance(row, Mapping)
    )
    return (
        totals.get("fundamental_complete") == counts["complete"]
        and totals.get("fundamental_partial") == counts["partial"]
        and totals.get("fundamental_missing") == counts["missing"]
        and totals.get("invalid_stale_conflicting")
        == sum(counts[name] for name in ("invalid", "stale", "conflicting"))
    )


def _build_derived_only_governed_candidate(
    derived: Mapping[str, Any], fact: Mapping[str, Any], *, run_id: str, ticker: str
) -> dict[str, Any]:
    rows = [row for row in derived.get("derivations") or [] if row.get("status") == "derived"]
    first = rows[0]
    source_dates = sorted({value for row in rows for value in row.get("source_publication_dates") or []})
    fact_by_id = {row["fact_id"]: row for row in fact["facts"]}
    first_fact = fact_by_id[first["input_facts"][0]["fact_id"]]
    package_id = f"{run_id}-{ticker.lower()}-governed-package-candidate"
    metrics = {
        row["canonical_metric"]: {
            "evidence_type": "derived",
            "value": row["calculation_result"],
            "unit": "ratio",
            "reporting_period": row["reporting_period"],
            "derivation_lineage": row,
        }
        for row in sorted(rows, key=lambda item: item["canonical_metric"])
    }
    return {
        "schema_version": GOVERNED_CANDIDATE_SCHEMA,
        "package_id": package_id,
        "package_schema_version": DERIVED_PACKAGE_SCHEMA_VERSION,
        "approval_decision_reference": derived["approval_decision_reference"],
        "source_packages": {"direct": None, "derived": {"schema_version": derived["schema_version"], "package_id": derived["package_id"], "package_checksum": _canonical_checksum(derived), "approval_reference": derived["approval_decision_reference"]}},
        "records": [{
            "ticker": ticker,
            "instrument_id": first["instrument_id"],
            "approval_decision_reference": derived["approval_decision_reference"],
            "company_name": first["company_identity"],
            "provider_symbol": ticker,
            "provider": "governed_primary_source_derivation_candidate",
            "source_date": max(source_dates),
            "reporting_period": first["reporting_period"],
            "period_type": first_fact["period_type"],
            "period_start": first_fact.get("period_start"),
            "period_end": first_fact["period_end"],
            "fiscal_year": first["fiscal_year"],
            "fiscal_period": first["fiscal_period"],
            "source_reference": f"governed-evidence://{package_id}/{ticker}/{first['reporting_period']}",
            "parser_version": ENGINE_VERSION,
            "snapshot_id": package_id,
            "acquired_at": first_fact["acquired_at"],
            "observed_at": first_fact["observed_at"],
            "metrics": metrics,
        }],
        "boundary": "Derived-only candidate; DATA07 use requires explicit DATA11 human approval validation.",
    }


def _load_bound_artifacts(root: Path, authority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    trusted = authority.get("trusted_artifacts")
    if not isinstance(trusted, Mapping) or not trusted:
        raise Data11GovernanceError("authority binding has no trusted artifacts")
    result = {}
    for name, binding in trusted.items():
        if not isinstance(binding, Mapping):
            raise Data11GovernanceError("authority artifact binding is invalid")
        path = _trusted_path(root, str(binding.get("path") or ""), label=str(name))
        expected = binding.get("sha256")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None or _sha256(path) != expected:
            raise Data11GovernanceError(f"authority checksum mismatch: {name}")
        payload = _strict_json(path)
        schema = binding.get("schema_version")
        if schema is not None and payload.get("schema_version") != schema:
            raise Data11GovernanceError(f"authority schema mismatch: {name}")
        result[str(name)] = payload
    return result


def _trusted_path(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise Data11GovernanceError(f"{label} path must be repository-relative without traversal")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Data11GovernanceError(f"{label} path escapes repository root") from exc
    if not resolved.is_file():
        raise Data11GovernanceError(f"{label} file is missing")
    return resolved


def _decision_bound_path(bundle_root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise Data11GovernanceError(f"{label} approval artifact must be a file in the decision bundle")
    resolved = (bundle_root / candidate).resolve()
    if resolved.parent != bundle_root.resolve() or not resolved.is_file():
        raise Data11GovernanceError(f"{label} approval artifact is missing or escapes its bundle")
    return resolved


def _strict_json(path: Path) -> Mapping[str, Any]:
    value, _ = _strict_json_with_bytes(path)
    return value


def _strict_json_with_bytes(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Data11GovernanceError(f"strict JSON is invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise Data11GovernanceError(f"JSON artifact must be an object: {path}")
    return value, content


def _approval_result(issues: Sequence[str], decision_path: str | Path | None, **identity: Any) -> dict[str, Any]:
    approved = not issues
    return {
        "schema_version": APPROVAL_VALIDATION_SCHEMA,
        "validation_status": "approved" if approved else "blocked",
        "concrete_package_source_approved": approved,
        "decision_path": Path(decision_path).as_posix() if decision_path else None,
        **identity,
        "reason_codes": sorted(set(issues)),
    }


def _unknown_prestate(reason: str) -> dict[str, Any]:
    return {"measurement_status": "unknown_not_measured", "reason_codes": [reason], "before": None, "by_ticker": {}}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
