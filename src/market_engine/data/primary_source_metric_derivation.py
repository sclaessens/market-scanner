from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FACT_PACKAGE_SCHEMA_VERSION = "market-engine-data10-primary-source-fact-package-v1"
FORMULA_CATALOG_SCHEMA_VERSION = "market-engine-data10-fundamental-metric-formula-catalog-v1"
DERIVED_PACKAGE_SCHEMA_VERSION = "market-engine-data10-derived-fundamental-metrics-v1"
DERIVATION_VALIDATION_SCHEMA_VERSION = "market-engine-data10-derivation-validation-v1"
DERIVATION_APPROVAL_DECISION_SCHEMA_VERSION = "market-engine-data10-derivation-approval-decision-v1"
DERIVATION_APPROVAL_VALIDATION_SCHEMA_VERSION = "market-engine-data10-derivation-approval-validation-v1"
DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION = "market-engine-data07-governed-fundamental-metrics-v2"
ENGINE_VERSION = "market-engine-data10-primary-source-metric-derivation-engine-v1"
APPROVED_SCOPE = "bounded_governed_primary_source_metric_derivation_pilot"
SUPPORTED_ACCOUNTING_FRAMEWORKS = frozenset({"us_gaap", "ifrs"})
SUPPORTED_PERIOD_TYPES = frozenset({"duration", "instant"})
SUPPORTED_OPERATIONS = frozenset({"ratio", "component_sum_ratio"})
CANONICAL_CONCEPTS = frozenset(
    {
        "revenue",
        "gross_profit",
        "operating_income",
        "commercial_paper",
        "short_term_borrowings",
        "current_term_debt",
        "noncurrent_term_debt",
        "total_interest_bearing_debt",
        "total_equity",
    }
)
REQUIRED_REVIEWER_ROLES = ("Operator", "Data Steward", "Governance Auditor")
REQUIRED_REVIEW_DIMENSIONS = (
    "source_authenticity",
    "primary_source_status",
    "permitted_local_use",
    "publication_boundary",
    "ticker_instrument_company_mapping",
    "accounting_framework",
    "canonical_concept_mapping",
    "raw_source_tag_mapping",
    "formula_identity_and_version",
    "numerator_selection",
    "denominator_selection",
    "debt_component_selection",
    "reporting_period_alignment",
    "unit_currency_scale_compatibility",
    "denominator_safety",
    "direct_versus_derived_classification",
    "freshness",
    "package_checksum",
    "fact_package_checksum",
    "formula_catalog_checksum",
    "calculation_checksum",
)

_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_FISCAL_PERIOD = re.compile(r"^(FY|Q[1-4])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESULT_QUANTUM = Decimal("0.000000000001")


class PrimarySourceMetricDerivationError(ValueError):
    pass


def load_strict_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PrimarySourceMetricDerivationError(f"strict JSON input is invalid: {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PrimarySourceMetricDerivationError(f"JSON input must be an object: {source}")
    return value


def derive_primary_source_metrics(
    fact_package: Mapping[str, Any],
    formula_catalog: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    issues: list[dict[str, str]] = []
    facts = _validate_fact_package(fact_package, issues)
    formulas = _validate_formula_catalog(formula_catalog, issues)
    if issues:
        return None, _validation_result(fact_package, issues, [])

    derivations: list[dict[str, Any]] = []
    requests = fact_package["derivation_requests"]
    for index, request in enumerate(
        sorted(requests, key=lambda row: str(row.get("request_id") or "") if isinstance(row, Mapping) else "")
    ):
        derivations.append(
            _derive_request(
                request,
                request_path=f"$.derivation_requests[{index}]",
                facts=facts,
                formulas=formulas,
                fact_package=fact_package,
            )
        )
    successful = [row for row in derivations if row["status"] == "derived"]
    blocked = [row for row in derivations if row["status"] == "blocked"]
    package = {
        "schema_version": DERIVED_PACKAGE_SCHEMA_VERSION,
        "package_id": f"{fact_package['package_id']}-derived",
        "fact_package_id": fact_package["package_id"],
        "formula_catalog_id": formula_catalog["catalog_id"],
        "derivation_engine_version": ENGINE_VERSION,
        "derivation_timestamp": fact_package["derivation_timestamp"],
        "approval_decision_reference": fact_package["derivation_approval_reference"],
        "fact_package_checksum": _fact_package_checksum(fact_package),
        "formula_catalog_checksum": _canonical_checksum(formula_catalog),
        "successful_metric_count": len(successful),
        "blocked_metric_count": len(blocked),
        "derivations": derivations,
        "boundary": (
            "Every emitted metric is derived evidence. The source documents did not publish these ratios; "
            "DATA07 consumption requires a separate checksum-bound derivation approval."
        ),
    }
    validation = _validation_result(fact_package, [], derivations)
    validation["derived_package_checksum"] = _canonical_checksum(package)
    return package, validation


def build_data07_governed_package(
    direct_package: Mapping[str, Any],
    derived_package: Mapping[str, Any],
    *,
    package_id: str,
    direct_approval_reference: str,
    direct_approval_checksum: str,
    direct_package_checksum: str,
) -> dict[str, Any]:
    if derived_package.get("schema_version") != DERIVED_PACKAGE_SCHEMA_VERSION:
        raise PrimarySourceMetricDerivationError("derived package schema is not supported")
    direct_records = direct_package.get("records")
    if not isinstance(direct_records, list) or len(direct_records) != 1 or not isinstance(direct_records[0], Mapping):
        raise PrimarySourceMetricDerivationError("bounded governed merge requires one direct record")
    direct = direct_records[0]
    ticker = _required_text(direct, "ticker", "direct.ticker")
    period = _required_text(direct, "reporting_period", "direct.reporting_period")
    derived_rows = [row for row in derived_package.get("derivations") or [] if row.get("status") == "derived"]
    if not derived_rows:
        raise PrimarySourceMetricDerivationError("at least one approved derivation candidate is required")
    if any(row.get("ticker") != ticker or row.get("reporting_period") != period for row in derived_rows):
        raise PrimarySourceMetricDerivationError("direct and derived ticker/reporting-period identity must match")
    if any(
        row.get("instrument_id") != direct.get("instrument_id")
        or row.get("company_identity") != direct.get("company_name")
        or row.get("fiscal_year") != direct.get("fiscal_year")
        or row.get("fiscal_period") != direct.get("fiscal_period")
        for row in derived_rows
    ):
        raise PrimarySourceMetricDerivationError(
            "direct and derived instrument/company/fiscal identity must match"
        )

    direct_metrics = direct.get("metrics")
    if not isinstance(direct_metrics, Mapping) or not direct_metrics:
        raise PrimarySourceMetricDerivationError("direct package metrics are missing")
    metrics: dict[str, Any] = {}
    for metric in sorted(direct_metrics):
        evidence = direct_metrics[metric]
        if not isinstance(evidence, Mapping):
            raise PrimarySourceMetricDerivationError("direct metric evidence must be an object")
        metrics[metric] = {
            "evidence_type": "direct",
            "value": evidence.get("value"),
            "unit": evidence.get("unit"),
            "reporting_period": evidence.get("reporting_period") or period,
            "direct_lineage": {
                "raw_source_field": evidence.get("raw_source_field"),
                "source_approval_reference": direct_approval_reference,
                "source_approval_checksum": direct_approval_checksum,
                "source_package_checksum": direct_package_checksum,
            },
        }
    for row in sorted(derived_rows, key=lambda item: item["canonical_metric"]):
        metric = row["canonical_metric"]
        if metric in metrics:
            raise PrimarySourceMetricDerivationError(f"direct/derived metric conflict: {metric}")
        metrics[metric] = {
            "evidence_type": "derived",
            "value": row["calculation_result"],
            "unit": "ratio",
            "reporting_period": period,
            "derivation_lineage": row,
        }
    metric_order = (
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "gross_margin",
        "operating_margin",
        "debt_to_equity",
    )
    metrics = {metric: metrics[metric] for metric in metric_order if metric in metrics}
    source_dates = [str(direct.get("source_date") or "")]
    source_dates.extend(
        str(value)
        for row in derived_rows
        for value in row.get("source_publication_dates") or []
        if value
    )
    record = {
        "ticker": ticker,
        "instrument_id": direct.get("instrument_id"),
        "company_name": direct.get("company_name"),
        "provider_symbol": direct.get("provider_symbol"),
        "provider": "governed_primary_source_merge",
        "source_date": max(source_dates),
        "reporting_period": period,
        "period_type": direct.get("period_type"),
        "period_start": direct.get("period_start"),
        "period_end": direct.get("period_end"),
        "fiscal_year": direct.get("fiscal_year"),
        "fiscal_period": direct.get("fiscal_period"),
        "source_reference": f"governed-evidence://{package_id}/{ticker}/{period}",
        "parser_version": ENGINE_VERSION,
        "snapshot_id": package_id,
        "acquired_at": direct.get("acquired_at"),
        "observed_at": direct.get("observed_at"),
        "metrics": metrics,
    }
    return {
        "schema_version": DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "package_schema_version": DERIVED_PACKAGE_SCHEMA_VERSION,
        "approval_decision_reference": derived_package["approval_decision_reference"],
        "source_packages": {
            "direct": {
                "schema_version": direct_package.get("schema_version"),
                "package_id": direct_package.get("package_id"),
                "package_checksum": direct_package_checksum,
                "approval_reference": direct_approval_reference,
                "approval_checksum": direct_approval_checksum,
            },
            "derived": {
                "schema_version": derived_package["schema_version"],
                "package_id": derived_package["package_id"],
                "package_checksum": _canonical_checksum(derived_package),
                "approval_reference": derived_package["approval_decision_reference"],
            },
        },
        "records": [record],
        "boundary": "Direct and derived evidence remain explicitly classified; this v2 package requires derivation approval before DATA07 import.",
    }


def validate_derivation_approval_decision(
    decision_path: str | Path | None,
    governed_package_path: str | Path,
    *,
    source_document_root: str | Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    package_path = Path(governed_package_path)
    if decision_path is None or not Path(decision_path).is_file():
        _issue(issues, "DERIVATION_APPROVAL_DECISION_MISSING", "$.decision", "explicit derivation approval is required")
        return _approval_result(issues, None, package_path)
    decision_file = Path(decision_path)
    try:
        decision = load_strict_json(decision_file)
        package = load_strict_json(package_path)
    except PrimarySourceMetricDerivationError:
        _issue(issues, "DERIVATION_APPROVAL_MALFORMED", "$", "approval decision and package must be strict JSON objects")
        return _approval_result(issues, decision_file, package_path)

    if decision.get("schema_version") != DERIVATION_APPROVAL_DECISION_SCHEMA_VERSION:
        _issue(issues, "DERIVATION_APPROVAL_SCHEMA_MISMATCH", "$.schema_version", "derivation approval schema is invalid")
    if decision.get("decision") != "approved":
        value = decision.get("decision")
        code = "DERIVATION_APPROVAL_REJECTED" if value == "rejected" else "DERIVATION_APPROVAL_BLOCKED"
        _issue(issues, code, "$.decision", "derivation approval must be approved")
    if decision.get("scope") != APPROVED_SCOPE:
        _issue(issues, "DERIVATION_APPROVAL_SCOPE_MISMATCH", "$.scope", "derivation approval scope is invalid")
    if package.get("schema_version") != DATA07_GOVERNED_PACKAGE_SCHEMA_VERSION:
        _issue(issues, "GOVERNED_PACKAGE_SCHEMA_MISMATCH", "$.package", "DATA07 governed package schema is invalid")
    if package.get("approval_decision_reference") != decision.get("decision_id"):
        _issue(issues, "DERIVATION_APPROVAL_REFERENCE_MISMATCH", "$.decision_id", "package approval reference does not match decision")

    roles = decision.get("reviewer_roles")
    if roles != list(REQUIRED_REVIEWER_ROLES):
        _issue(issues, "DERIVATION_REVIEWER_ROLES_INVALID", "$.reviewer_roles", "all required reviewer roles must be present in deterministic order")
    reviews = decision.get("reviews") if isinstance(decision.get("reviews"), Mapping) else {}
    for dimension in REQUIRED_REVIEW_DIMENSIONS:
        review = reviews.get(dimension)
        if not isinstance(review, Mapping):
            _issue(issues, "DERIVATION_REVIEW_DIMENSION_MISSING", f"$.reviews.{dimension}", "required review dimension is missing")
        elif review.get("status") != "approved":
            _issue(issues, "DERIVATION_REVIEW_DIMENSION_NOT_APPROVED", f"$.reviews.{dimension}.status", "review dimension must be approved")

    bindings = decision.get("artifact_bindings") if isinstance(decision.get("artifact_bindings"), Mapping) else {}
    bound_governed_path = _binding_path(bindings, "governed_package_path", issues)
    if bound_governed_path is not None and bound_governed_path.resolve() != package_path.resolve():
        _issue(
            issues,
            "DERIVATION_GOVERNED_PACKAGE_PATH_MISMATCH",
            "$.artifact_bindings.governed_package_path",
            "approval must bind the exact governed package presented to DATA07",
        )
    _binding_checksum(
        bindings,
        "governed_package_path",
        "governed_package_sha256",
        bound_governed_path,
        issues,
    )
    bound_payloads: dict[str, Mapping[str, Any]] = {}
    for stem in ("fact_package", "formula_catalog", "derived_package", "direct_package", "direct_approval"):
        path = _binding_path(bindings, f"{stem}_path", issues)
        _binding_checksum(bindings, f"{stem}_path", f"{stem}_sha256", path, issues)
        if path is not None and path.is_file() and stem != "direct_approval":
            try:
                bound_payloads[stem] = load_strict_json(path)
            except PrimarySourceMetricDerivationError:
                _issue(issues, "BOUND_ARTIFACT_MALFORMED", f"$.artifact_bindings.{stem}_path", "bound artifact must be strict JSON")

    derived = bound_payloads.get("derived_package", {})
    fact_package = bound_payloads.get("fact_package", {})
    formula_catalog = bound_payloads.get("formula_catalog", {})
    if derived.get("schema_version") != DERIVED_PACKAGE_SCHEMA_VERSION:
        _issue(issues, "DERIVED_PACKAGE_SCHEMA_MISMATCH", "$.artifact_bindings.derived_package_path", "derived package schema is invalid")
    if fact_package.get("schema_version") != FACT_PACKAGE_SCHEMA_VERSION:
        _issue(issues, "FACT_PACKAGE_SCHEMA_MISMATCH", "$.artifact_bindings.fact_package_path", "fact package schema is invalid")
    if formula_catalog.get("schema_version") != FORMULA_CATALOG_SCHEMA_VERSION:
        _issue(issues, "FORMULA_CATALOG_SCHEMA_MISMATCH", "$.artifact_bindings.formula_catalog_path", "formula catalog schema is invalid")

    successful_metrics = sorted(row.get("canonical_metric") for row in derived.get("derivations") or [] if row.get("status") == "derived")
    blocked_metrics = sorted(row.get("canonical_metric") for row in derived.get("derivations") or [] if row.get("status") == "blocked")
    if decision.get("approved_derived_metrics") != successful_metrics:
        _issue(issues, "APPROVED_DERIVED_METRIC_SET_MISMATCH", "$.approved_derived_metrics", "approved derived metrics must exactly match successful derivations")
    if decision.get("explicitly_blocked_metrics") != blocked_metrics:
        _issue(issues, "BLOCKED_DERIVATION_SET_MISMATCH", "$.explicitly_blocked_metrics", "blocked metrics must exactly match blocked derivations")
    package_tickers = sorted({row.get("ticker") for row in package.get("records") or [] if isinstance(row, Mapping) and isinstance(row.get("ticker"), str)})
    if decision.get("approved_tickers") != package_tickers or len(package_tickers) != 1:
        _issue(issues, "DERIVATION_APPROVED_TICKER_SET_MISMATCH", "$.approved_tickers", "bounded approval requires the exact one-ticker package set")
    expected_calculations = sorted(row.get("calculation_checksum") for row in derived.get("derivations") or [] if row.get("status") == "derived")
    if decision.get("approved_calculation_checksums") != expected_calculations:
        _issue(issues, "CALCULATION_CHECKSUM_SET_MISMATCH", "$.approved_calculation_checksums", "approved calculation checksums must exactly match successful derivations")

    _validate_source_documents(decision.get("source_documents"), source_document_root, issues)
    return _approval_result(
        issues,
        decision_file,
        package_path,
        decision_id=decision.get("decision_id"),
        package_id=package.get("package_id"),
    )


def execute_approved_derivation_import(
    *,
    fact_package_path: str | Path,
    formula_catalog_path: str | Path,
    derived_package_path: str | Path,
    governed_package_path: str | Path,
    approval_decision_path: str | Path | None,
    source_document_root: str | Path | None,
    data07_runner: Callable[..., Any],
    data07_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        expected, validation = derive_primary_source_metrics(
            load_strict_json(fact_package_path),
            load_strict_json(formula_catalog_path),
        )
        committed_derived = load_strict_json(derived_package_path)
    except PrimarySourceMetricDerivationError:
        return {"status": "blocked", "reason_codes": ["DERIVATION_INPUT_MALFORMED"], "data07_executed": False}
    if expected is None or not any(row.get("status") == "derived" for row in validation.get("derivations") or []):
        return {"status": "blocked", "reason_codes": validation["reason_codes"] or ["NO_SUCCESSFUL_DERIVATION"], "data07_executed": False}
    if _canonical_bytes(expected) != _canonical_bytes(committed_derived):
        return {"status": "blocked", "reason_codes": ["DERIVED_PACKAGE_REPRODUCTION_MISMATCH"], "data07_executed": False}
    approval = validate_derivation_approval_decision(
        approval_decision_path,
        governed_package_path,
        source_document_root=source_document_root,
    )
    if not approval["concrete_package_source_approved"]:
        return {"status": "blocked", "reason_codes": approval["reason_codes"], "data07_executed": False}
    result = data07_runner(**dict(data07_kwargs))
    return {"status": "completed", "reason_codes": [], "data07_executed": True, "data07_result": result}


def persist_derivation_candidate(
    *,
    fact_package_path: str | Path,
    formula_catalog_path: str | Path,
    direct_package_path: str | Path,
    direct_approval_path: str | Path,
    derived_output_path: str | Path,
    validation_output_path: str | Path,
    governed_output_path: str | Path,
    governed_package_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outputs = [Path(derived_output_path), Path(validation_output_path), Path(governed_output_path)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("ME-DATA10 candidate output paths must not already exist")
    fact_package = load_strict_json(fact_package_path)
    formula_catalog = load_strict_json(formula_catalog_path)
    direct_package = load_strict_json(direct_package_path)
    direct_approval = load_strict_json(direct_approval_path)
    derived, validation = derive_primary_source_metrics(fact_package, formula_catalog)
    if derived is None or not any(row.get("status") == "derived" for row in validation["derivations"]):
        raise PrimarySourceMetricDerivationError("candidate package has no successful governed derivation")
    governed = build_data07_governed_package(
        direct_package,
        derived,
        package_id=governed_package_id,
        direct_approval_reference=_required_text(direct_approval, "decision_id", "direct_approval.decision_id"),
        direct_approval_checksum=_sha256(Path(direct_approval_path)),
        direct_package_checksum=_sha256(Path(direct_package_path)),
    )
    for path, payload in zip(outputs, (derived, validation, governed), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return derived, validation, governed


def _validate_fact_package(value: Mapping[str, Any], issues: list[dict[str, str]]) -> dict[str, Mapping[str, Any]]:
    if value.get("schema_version") != FACT_PACKAGE_SCHEMA_VERSION:
        _issue(issues, "FACT_PACKAGE_SCHEMA_MISMATCH", "$.schema_version", "fact package schema is invalid")
    for key in ("package_id", "derivation_timestamp", "derivation_approval_reference"):
        _validate_required_text(value, key, f"$.{key}", issues)
    _validate_timestamp(value.get("derivation_timestamp"), "$.derivation_timestamp", issues)
    records = value.get("facts")
    requests = value.get("derivation_requests")
    if not isinstance(records, list) or not records:
        _issue(issues, "FACTS_MISSING", "$.facts", "at least one primary-source fact is required")
        records = []
    if not isinstance(requests, list) or not requests:
        _issue(issues, "DERIVATION_REQUESTS_MISSING", "$.derivation_requests", "at least one derivation request is required")
    facts: dict[str, Mapping[str, Any]] = {}
    semantic: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    required_fields = (
        "fact_id", "ticker", "instrument_id", "company_identity", "accounting_framework",
        "canonical_concept", "raw_source_concept", "value", "unit", "currency", "scale",
        "period_type", "period_end", "fiscal_year", "fiscal_period", "source_name",
        "source_reference", "source_document_checksum", "source_publication_date", "observed_at",
        "acquired_at", "parser_version", "source_approval_reference", "canonical_mapping_approval_reference",
    )
    for index, fact in enumerate(records):
        path = f"$.facts[{index}]"
        if not isinstance(fact, Mapping):
            _issue(issues, "FACT_NOT_OBJECT", path, "fact must be an object")
            continue
        for key in required_fields:
            if key == "value":
                continue
            _validate_required_text(fact, key, f"{path}.{key}", issues) if key not in {"scale", "fiscal_year"} else None
        fact_id = fact.get("fact_id")
        ticker = fact.get("ticker")
        if isinstance(ticker, str) and not _TICKER.fullmatch(ticker):
            _issue(issues, "TICKER_INVALID", f"{path}.ticker", "ticker must be canonical uppercase text")
        expected_instrument = f"equity:{str(ticker).lower()}" if isinstance(ticker, str) else None
        if fact.get("instrument_id") != expected_instrument:
            _issue(issues, "INSTRUMENT_IDENTITY_MISMATCH", f"{path}.instrument_id", "instrument ID must match ticker")
        framework = fact.get("accounting_framework")
        if framework not in SUPPORTED_ACCOUNTING_FRAMEWORKS:
            _issue(issues, "ACCOUNTING_FRAMEWORK_UNSUPPORTED", f"{path}.accounting_framework", "canonical mapping is not approved for this accounting framework")
        concept = fact.get("canonical_concept")
        if concept not in CANONICAL_CONCEPTS:
            _issue(issues, "CANONICAL_CONCEPT_UNSUPPORTED", f"{path}.canonical_concept", "canonical financial concept is unsupported")
        raw_value = fact.get("value")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            _issue(issues, "FACT_VALUE_NOT_NUMERIC", f"{path}.value", "fact value must be a JSON number")
        elif not math.isfinite(float(raw_value)):
            _issue(issues, "FACT_VALUE_NOT_FINITE", f"{path}.value", "NaN and Infinity are forbidden")
        scale = fact.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, int) or not -12 <= scale <= 12:
            _issue(issues, "FACT_SCALE_INVALID", f"{path}.scale", "scale must be an unambiguous integer exponent")
        period_type = fact.get("period_type")
        if period_type not in SUPPORTED_PERIOD_TYPES:
            _issue(issues, "PERIOD_TYPE_UNSUPPORTED", f"{path}.period_type", "period type must be duration or instant")
        start = fact.get("period_start")
        if period_type == "duration":
            _validate_date(start, f"{path}.period_start", issues)
        elif start is not None:
            _issue(issues, "INSTANT_PERIOD_START_FORBIDDEN", f"{path}.period_start", "instant facts must not have a period start")
        _validate_date(fact.get("period_end"), f"{path}.period_end", issues)
        _validate_date(fact.get("source_publication_date"), f"{path}.source_publication_date", issues)
        _validate_timestamp(fact.get("observed_at"), f"{path}.observed_at", issues)
        _validate_timestamp(fact.get("acquired_at"), f"{path}.acquired_at", issues)
        fiscal_year = fact.get("fiscal_year")
        if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or not 1900 <= fiscal_year <= 2200:
            _issue(issues, "FISCAL_YEAR_INVALID", f"{path}.fiscal_year", "fiscal year is invalid")
        fiscal_period = fact.get("fiscal_period")
        if not isinstance(fiscal_period, str) or not _FISCAL_PERIOD.fullmatch(fiscal_period):
            _issue(issues, "FISCAL_PERIOD_INVALID", f"{path}.fiscal_period", "fiscal period is invalid")
        checksum = fact.get("source_document_checksum")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            _issue(issues, "SOURCE_DOCUMENT_CHECKSUM_INVALID", f"{path}.source_document_checksum", "source checksum must be lowercase SHA-256")
        if isinstance(fact_id, str):
            if fact_id in facts:
                _issue(issues, "DUPLICATE_FACT_ID", f"{path}.fact_id", "fact ID is duplicated")
            facts[fact_id] = fact
        key = (
            ticker, concept, period_type, start, fact.get("period_end"), fiscal_year,
            fiscal_period, fact.get("unit"), fact.get("currency"), scale,
        )
        if key in semantic:
            code = "DUPLICATE_FACT" if semantic[key].get("value") == raw_value else "CONFLICTING_FACT"
            _issue(issues, code, path, "canonical fact identity is duplicated")
        semantic[key] = fact
    return facts


def _validate_formula_catalog(value: Mapping[str, Any], issues: list[dict[str, str]]) -> dict[str, Mapping[str, Any]]:
    if value.get("schema_version") != FORMULA_CATALOG_SCHEMA_VERSION:
        _issue(issues, "FORMULA_CATALOG_SCHEMA_MISMATCH", "$.schema_version", "formula catalog schema is invalid")
    _validate_required_text(value, "catalog_id", "$.catalog_id", issues)
    formulas_value = value.get("formulas")
    if not isinstance(formulas_value, list) or not formulas_value:
        _issue(issues, "FORMULA_CATALOG_EMPTY", "$.formulas", "formula catalog must be non-empty")
        return {}
    formulas: dict[str, Mapping[str, Any]] = {}
    for index, formula in enumerate(formulas_value):
        path = f"$.formulas[{index}]"
        if not isinstance(formula, Mapping):
            _issue(issues, "FORMULA_NOT_OBJECT", path, "formula must be an object")
            continue
        for key in ("formula_id", "version", "canonical_metric", "canonical_expression", "operation", "period_type"):
            _validate_required_text(formula, key, f"{path}.{key}", issues)
        formula_id = formula.get("formula_id")
        if formula.get("operation") not in SUPPORTED_OPERATIONS:
            _issue(issues, "FORMULA_OPERATION_UNSUPPORTED", f"{path}.operation", "formula operation is unsupported")
        if formula.get("period_type") not in SUPPORTED_PERIOD_TYPES:
            _issue(issues, "FORMULA_PERIOD_TYPE_INVALID", f"{path}.period_type", "formula period type is invalid")
        concepts = formula.get("allowed_component_concepts")
        if formula.get("operation") == "component_sum_ratio" and (not isinstance(concepts, list) or not concepts):
            _issue(issues, "FORMULA_COMPONENT_ALLOWLIST_MISSING", f"{path}.allowed_component_concepts", "component formula requires an allowlist")
        if isinstance(formula_id, str):
            if formula_id in formulas:
                _issue(issues, "DUPLICATE_FORMULA_ID", f"{path}.formula_id", "formula ID is duplicated")
            formulas[formula_id] = formula
    return formulas


def _derive_request(
    request: Any,
    *,
    request_path: str,
    facts: Mapping[str, Mapping[str, Any]],
    formulas: Mapping[str, Mapping[str, Any]],
    fact_package: Mapping[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    if not isinstance(request, Mapping):
        return {
            "request_id": None,
            "ticker": None,
            "canonical_metric": None,
            "formula_id": None,
            "formula_version": None,
            "reporting_period": None,
            "approval_decision_reference": fact_package.get("derivation_approval_reference"),
            "status": "blocked",
            "reason_codes": ["DERIVATION_REQUEST_INVALID"],
        }
    formula_id = request.get("formula_id")
    formula = formulas.get(str(formula_id))
    base = {
        "request_id": request.get("request_id"),
        "ticker": request.get("ticker"),
        "canonical_metric": request.get("canonical_metric"),
        "formula_id": formula_id,
        "formula_version": request.get("formula_version"),
        "reporting_period": f"{request.get('fiscal_year')}-{request.get('fiscal_period')}",
        "approval_decision_reference": fact_package.get("derivation_approval_reference"),
    }
    if formula is None:
        return {**base, "status": "blocked", "reason_codes": ["FORMULA_ID_UNSUPPORTED"]}
    applicability = request.get("applicability")
    if not isinstance(applicability, Mapping) or not isinstance(applicability.get("approval_reference"), str):
        problems.append("APPLICABILITY_EVIDENCE_MISSING")
    elif applicability.get("status") == "not_applicable":
        problems.append("FORMULA_NOT_APPLICABLE")
    elif applicability.get("status") != "applicable":
        problems.append("APPLICABILITY_STATUS_INVALID")
    if request.get("formula_version") != formula.get("version"):
        problems.append("FORMULA_VERSION_MISMATCH")
    if request.get("canonical_metric") != formula.get("canonical_metric"):
        problems.append("FORMULA_METRIC_MISMATCH")
    ticker = request.get("ticker")
    if not isinstance(ticker, str) or not _TICKER.fullmatch(ticker):
        problems.append("DERIVATION_TICKER_INVALID")
    operation = formula.get("operation")
    numerator_ids = _string_list(
        request.get("numerator_fact_ids"),
        "NUMERATOR_FACT_IDS_INVALID",
        problems,
        allow_empty=operation == "component_sum_ratio",
    )
    denominator_ids = _string_list(request.get("denominator_fact_ids"), "DENOMINATOR_FACT_IDS_INVALID", problems)
    component_ids = _string_list(request.get("component_fact_ids"), "COMPONENT_FACT_IDS_INVALID", problems, allow_empty=True)
    numerator_facts = _resolve_facts(numerator_ids, facts, "NUMERATOR_MISSING", problems)
    denominator_facts = _resolve_facts(denominator_ids, facts, "DENOMINATOR_MISSING", problems)
    component_facts = _resolve_facts(component_ids, facts, "DEBT_COMPONENT_MISSING", problems)
    if operation == "ratio" and (len(numerator_facts) != 1 or len(denominator_facts) != 1 or component_facts):
        problems.append("FORMULA_INPUT_CARDINALITY_INVALID")
    if operation == "component_sum_ratio":
        required = request.get("required_component_concepts")
        if not isinstance(required, list) or not required or any(not isinstance(item, str) or not item for item in required):
            problems.append("DEBT_COMPONENT_SET_NOT_APPROVED")
            required = []
        selected = [str(fact.get("canonical_concept")) for fact in component_facts]
        if sorted(selected) != sorted(required):
            problems.append("DEBT_COMPONENT_MISSING")
        allowed = formula.get("allowed_component_concepts") or []
        if any(concept not in allowed for concept in selected):
            problems.append("DEBT_COMPONENT_UNSUPPORTED")
        if numerator_facts or len(denominator_facts) != 1 or not component_facts:
            problems.append("FORMULA_INPUT_CARDINALITY_INVALID")
    selected_facts = numerator_facts + denominator_facts + component_facts
    if selected_facts:
        if any(fact.get("ticker") != ticker for fact in selected_facts):
            problems.append("FACT_TICKER_MISMATCH")
        if len({fact.get("instrument_id") for fact in selected_facts}) != 1:
            problems.append("FACT_INSTRUMENT_MISMATCH")
        if len({fact.get("company_identity") for fact in selected_facts}) != 1:
            problems.append("FACT_COMPANY_MISMATCH")
        if len({fact.get("accounting_framework") for fact in selected_facts}) != 1:
            problems.append("ACCOUNTING_FRAMEWORK_MISMATCH")
        _validate_alignment(selected_facts, formula, request, problems)
    denominator = denominator_facts[0] if len(denominator_facts) == 1 else None
    if denominator is not None and _decimal(denominator.get("value")) <= 0:
        problems.append("DENOMINATOR_NOT_POSITIVE")
    if (
        operation == "ratio"
        and numerator_facts
        and denominator_facts
        and formula.get("canonical_metric") in {"gross_margin", "operating_margin"}
    ):
        if _decimal(denominator_facts[0].get("value")) <= 0:
            problems.append("REVENUE_NOT_POSITIVE")
    if operation == "component_sum_ratio" and any(_decimal(fact.get("value")) < 0 for fact in component_facts):
        problems.append("NEGATIVE_DEBT_COMPONENT")
    problems = sorted(set(problems))
    if problems:
        return {
            **base,
            "status": "blocked",
            "reason_codes": problems,
            "numerator_fact_ids": sorted(numerator_ids),
            "denominator_fact_ids": sorted(denominator_ids),
            "component_fact_ids": sorted(component_ids),
        }

    numerator_value = (
        sum((_normalized_decimal(fact) for fact in component_facts), Decimal(0))
        if operation == "component_sum_ratio"
        else _normalized_decimal(numerator_facts[0])
    )
    denominator_value = _normalized_decimal(denominator_facts[0])
    if denominator_value <= 0:
        return {**base, "status": "blocked", "reason_codes": ["DENOMINATOR_NOT_POSITIVE"]}
    result = (numerator_value / denominator_value).quantize(_RESULT_QUANTUM, rounding=ROUND_HALF_EVEN)
    inputs = [_lineage_fact(fact) for fact in sorted(selected_facts, key=lambda row: str(row["fact_id"]))]
    calculation = {
        **base,
        "instrument_id": selected_facts[0]["instrument_id"],
        "company_identity": selected_facts[0]["company_identity"],
        "fiscal_year": request.get("fiscal_year"),
        "fiscal_period": request.get("fiscal_period"),
        "status": "derived",
        "evidence_type": "derived",
        "formula": {
            "formula_id": formula["formula_id"],
            "version": formula["version"],
            "canonical_expression": formula["canonical_expression"],
            "operation": operation,
        },
        "numerator_fact_ids": sorted(numerator_ids),
        "denominator_fact_ids": sorted(denominator_ids),
        "component_fact_ids": sorted(component_ids),
        "input_facts": inputs,
        "normalized_numerator": _decimal_text(numerator_value),
        "normalized_denominator": _decimal_text(denominator_value),
        "calculation_result": float(result),
        "result_unit": "ratio",
        "period": {
            "period_type": formula["period_type"],
            "period_start": selected_facts[0].get("period_start"),
            "period_end": selected_facts[0].get("period_end"),
            "fiscal_year": request.get("fiscal_year"),
            "fiscal_period": request.get("fiscal_period"),
        },
        "validation": {
            "period_alignment": "passed",
            "unit_compatibility": "passed",
            "currency_compatibility": "passed",
            "scale_compatibility": "passed",
            "denominator_safety": "passed",
        },
        "source_document_checksums": sorted({str(fact["source_document_checksum"]) for fact in selected_facts}),
        "source_publication_dates": sorted({str(fact["source_publication_date"]) for fact in selected_facts}),
        "derivation_timestamp": fact_package["derivation_timestamp"],
        "derivation_engine_version": ENGINE_VERSION,
        "limitations": [
            "The ratio is calculated from approved canonical primary-source facts; the source did not publish the ratio itself.",
            "No estimate, interpolation, annualization, fallback metric, or missing-value substitution was used.",
        ],
    }
    calculation["calculation_checksum"] = _canonical_checksum(calculation)
    return calculation


def _validate_alignment(
    facts: Sequence[Mapping[str, Any]],
    formula: Mapping[str, Any],
    request: Mapping[str, Any],
    problems: list[str],
) -> None:
    expected_period_type = formula.get("period_type")
    if any(fact.get("period_type") != expected_period_type for fact in facts):
        problems.append("PERIOD_TYPE_MISMATCH")
    if len({fact.get("period_end") for fact in facts}) != 1:
        problems.append("PERIOD_END_MISMATCH")
    if expected_period_type == "duration" and len({fact.get("period_start") for fact in facts}) != 1:
        problems.append("PERIOD_START_MISMATCH")
    if any(fact.get("fiscal_year") != request.get("fiscal_year") for fact in facts):
        problems.append("FISCAL_YEAR_MISMATCH")
    if any(fact.get("fiscal_period") != request.get("fiscal_period") for fact in facts):
        problems.append("FISCAL_PERIOD_MISMATCH")
    if len({fact.get("currency") for fact in facts}) != 1:
        problems.append("CURRENCY_MISMATCH")
    if len({fact.get("unit") for fact in facts}) != 1:
        problems.append("UNIT_MISMATCH")
    if len({fact.get("scale") for fact in facts}) != 1:
        problems.append("SCALE_MISMATCH")


def _validation_result(
    fact_package: Mapping[str, Any],
    issues: Sequence[Mapping[str, str]],
    derivations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    package_issues = sorted(issues, key=lambda row: (row["path"], row["reason_code"]))
    blocked_codes = sorted({code for row in derivations if row.get("status") == "blocked" for code in row.get("reason_codes") or []})
    successful = sum(row.get("status") == "derived" for row in derivations)
    blocked = sum(row.get("status") == "blocked" for row in derivations)
    status = "rejected" if package_issues else ("passed_with_blocked_derivations" if blocked else "passed")
    return {
        "schema_version": DERIVATION_VALIDATION_SCHEMA_VERSION,
        "validator_version": ENGINE_VERSION,
        "fact_package_id": fact_package.get("package_id"),
        "validation_status": status,
        "data07_consumable_candidate": not package_issues and successful > 0,
        "successful_derivation_count": successful,
        "blocked_derivation_count": blocked,
        "reason_codes": sorted({row["reason_code"] for row in package_issues} | set(blocked_codes)),
        "issues": package_issues,
        "derivations": list(derivations),
        "boundary": "Successful derived evidence remains blocked from DATA07 until an explicit checksum-bound derivation approval passes.",
    }


def _lineage_fact(fact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact["fact_id"],
        "canonical_concept": fact["canonical_concept"],
        "raw_source_concept": fact["raw_source_concept"],
        "input_value": fact["value"],
        "input_unit": fact["unit"],
        "currency": fact["currency"],
        "scale": fact["scale"],
        "normalized_value": _decimal_text(_normalized_decimal(fact)),
        "period_type": fact["period_type"],
        "period_start": fact.get("period_start"),
        "period_end": fact["period_end"],
        "source_reference": fact["source_reference"],
        "source_document_checksum": fact["source_document_checksum"],
        "source_approval_reference": fact["source_approval_reference"],
        "canonical_mapping_approval_reference": fact["canonical_mapping_approval_reference"],
    }


def _normalized_decimal(fact: Mapping[str, Any]) -> Decimal:
    return _decimal(fact.get("value")) * (Decimal(10) ** int(fact.get("scale", 0)))


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PrimarySourceMetricDerivationError(f"numeric fact is invalid: {value}") from exc
    if not result.is_finite():
        raise PrimarySourceMetricDerivationError("numeric fact must be finite")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _resolve_facts(
    fact_ids: Sequence[str],
    facts: Mapping[str, Mapping[str, Any]],
    missing_code: str,
    problems: list[str],
) -> list[Mapping[str, Any]]:
    resolved = []
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is None:
            problems.append(missing_code)
        else:
            resolved.append(fact)
    return resolved


def _string_list(value: Any, code: str, problems: list[str], *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        problems.append(code)
        return []
    if any(not isinstance(item, str) or not item for item in value) or len(set(value)) != len(value):
        problems.append(code)
        return []
    return value


def _validate_source_documents(value: Any, source_root: str | Path | None, issues: list[dict[str, str]]) -> None:
    if not isinstance(value, list) or not value:
        _issue(issues, "DERIVATION_SOURCE_DOCUMENTS_MISSING", "$.source_documents", "source document bindings are required")
        return
    if source_root is None:
        _issue(issues, "DERIVATION_SOURCE_DOCUMENT_ROOT_MISSING", "$.source_documents", "source document root is required")
        return
    root = Path(source_root).resolve()
    for index, document in enumerate(value):
        path = f"$.source_documents[{index}]"
        if not isinstance(document, Mapping):
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_INVALID", path, "source document binding must be an object")
            continue
        relative_value = document.get("relative_path")
        if not isinstance(relative_value, str) or not relative_value:
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_PATH_INVALID", f"{path}.relative_path", "relative source document path is required")
            continue
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_PATH_ESCAPE", f"{path}.relative_path", "source document path must remain below the source root")
            continue
        local = (root / relative).resolve()
        try:
            local.relative_to(root)
        except ValueError:
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_PATH_ESCAPE", f"{path}.relative_path", "resolved source document path escapes the source root")
            continue
        if not local.is_file():
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_MISSING", f"{path}.relative_path", "source document is missing")
        elif document.get("sha256") != _sha256(local):
            _issue(issues, "DERIVATION_SOURCE_DOCUMENT_CHECKSUM_MISMATCH", f"{path}.sha256", "source document checksum does not match")


def _approval_result(
    issues: list[dict[str, str]],
    decision_path: Path | None,
    package_path: Path,
    **identity: Any,
) -> dict[str, Any]:
    issues.sort(key=lambda row: (row["path"], row["reason_code"]))
    approved = not issues
    return {
        "schema_version": DERIVATION_APPROVAL_VALIDATION_SCHEMA_VERSION,
        "validation_status": "approved" if approved else "blocked",
        "concrete_package_source_approved": approved,
        "decision_path": decision_path.as_posix() if decision_path else None,
        "package_path": package_path.as_posix(),
        **identity,
        "reason_codes": sorted({row["reason_code"] for row in issues}),
        "issues": issues,
    }


def _binding_path(bindings: Mapping[str, Any], key: str, issues: list[dict[str, str]]) -> Path | None:
    value = bindings.get(key)
    if not isinstance(value, str) or not value:
        _issue(issues, "DERIVATION_ARTIFACT_BINDING_MISSING", f"$.artifact_bindings.{key}", "artifact path binding is required")
        return None
    return Path(value)


def _binding_checksum(
    bindings: Mapping[str, Any],
    path_key: str,
    checksum_key: str,
    path: Path | None,
    issues: list[dict[str, str]],
) -> None:
    target = path if path is not None else _binding_path(bindings, path_key, issues)
    if target is None or not target.is_file() or bindings.get(checksum_key) != _sha256(target):
        _issue(issues, "DERIVATION_ARTIFACT_CHECKSUM_MISMATCH", f"$.artifact_bindings.{checksum_key}", "artifact checksum does not match bound content")


def _validate_required_text(value: Mapping[str, Any], key: str, path: str, issues: list[dict[str, str]]) -> None:
    if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
        _issue(issues, "REQUIRED_FIELD_MISSING", path, f"{key} must be a non-empty string")


def _required_text(value: Mapping[str, Any], key: str, path: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise PrimarySourceMetricDerivationError(f"required text is missing: {path}")
    return result.strip()


def _validate_date(value: Any, path: str, issues: list[dict[str, str]]) -> None:
    try:
        if not isinstance(value, str):
            raise ValueError
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        _issue(issues, "DATE_INVALID", path, "date must use YYYY-MM-DD")


def _validate_timestamp(value: Any, path: str, issues: list[dict[str, str]]) -> None:
    try:
        if not isinstance(value, str):
            raise ValueError
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except ValueError:
        _issue(issues, "TIMESTAMP_INVALID", path, "timestamp must be timezone-aware ISO-8601")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fact_package_checksum(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    facts = value.get("facts")
    requests = value.get("derivation_requests")
    if isinstance(facts, list):
        normalized["facts"] = sorted(
            facts,
            key=lambda row: str(row.get("fact_id") or "") if isinstance(row, Mapping) else "",
        )
    if isinstance(requests, list):
        normalized["derivation_requests"] = sorted(
            requests,
            key=lambda row: str(row.get("request_id") or "") if isinstance(row, Mapping) else "",
        )
    return _canonical_checksum(normalized)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"forbidden JSON numeric constant: {value}")


def _issue(issues: list[dict[str, str]], code: str, path: str, message: str) -> None:
    issues.append({"reason_code": code, "path": path, "message": message})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic governed derived fundamental metric candidates.")
    parser.add_argument("--fact-package", required=True)
    parser.add_argument("--formula-catalog", required=True)
    parser.add_argument("--direct-package", required=True)
    parser.add_argument("--direct-approval", required=True)
    parser.add_argument("--derived-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--governed-output", required=True)
    parser.add_argument("--governed-package-id", required=True)
    args = parser.parse_args(argv)
    try:
        persist_derivation_candidate(
            fact_package_path=args.fact_package,
            formula_catalog_path=args.formula_catalog,
            direct_package_path=args.direct_package,
            direct_approval_path=args.direct_approval,
            derived_output_path=args.derived_output,
            validation_output_path=args.validation_output,
            governed_output_path=args.governed_output,
            governed_package_id=args.governed_package_id,
        )
    except (FileExistsError, PrimarySourceMetricDerivationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
