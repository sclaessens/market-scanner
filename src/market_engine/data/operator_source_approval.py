from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from market_engine.data.fundamental_evidence_coverage import MVP_METRIC_FIELDS
DECISION_SCHEMA_VERSION = "market-engine-data09-source-approval-decision-v1"
VALIDATION_SCHEMA_VERSION = "market-engine-data09-source-approval-validation-v1"
OPERATOR_IMPORT_SCHEMA_VERSION = "market-engine-data07-operator-fundamental-metrics-v1"
INPUT_SCHEMA_VERSION = "market-engine-data08-operator-fundamental-metric-input-v1"
REPORT_SCHEMA_VERSION = "market-engine-data08-operator-fundamental-metric-validation-report-v1"
VALIDATOR_VERSION = "market-engine-data08-operator-fundamental-metric-validator-v1"
APPROVED_SCOPE = "bounded_operator_fundamental_metric_pilot"
BOUNDED_PILOT_MAX_TICKERS = 1
REQUIRED_REVIEWER_ROLES = ("Operator", "Data Steward", "Governance Auditor")
REQUIRED_REVIEW_DIMENSIONS = (
    "source_authenticity",
    "primary_source",
    "identity_crosscheck",
    "ticker_instrument_company_mapping",
    "metric_lineage",
    "reporting_period",
    "fiscal_context",
    "unit_scale",
    "freshness",
    "permitted_local_use",
    "publication_boundary",
)


def validate_source_approval_decision(
    decision_path: str | Path | None,
    package_path: str | Path,
    *,
    source_document_root: str | Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    decision_file = Path(decision_path) if decision_path else None
    package_file = Path(package_path)
    if decision_file is None or not decision_file.is_file():
        _issue(issues, "SOURCE_APPROVAL_DECISION_MISSING", "$.decision", "explicit source approval decision is required")
        return _result(issues, None, package_file)
    try:
        decision = json.loads(decision_file.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError):
        _issue(issues, "SOURCE_APPROVAL_MALFORMED", "$", "source approval decision must be valid strict JSON")
        return _result(issues, decision_file, package_file)
    if not isinstance(decision, Mapping):
        _issue(issues, "SOURCE_APPROVAL_MALFORMED", "$", "source approval decision must be an object")
        return _result(issues, decision_file, package_file)

    if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
        _issue(issues, "UNSUPPORTED_SOURCE_APPROVAL_SCHEMA", "$.schema_version", "unsupported source approval schema")
    status = decision.get("decision")
    if status == "blocked":
        _issue(issues, "SOURCE_APPROVAL_BLOCKED", "$.decision", "source approval is blocked")
    elif status == "rejected":
        _issue(issues, "SOURCE_APPROVAL_REJECTED", "$.decision", "source approval is rejected")
    elif status != "approved":
        _issue(issues, "UNKNOWN_SOURCE_APPROVAL_DECISION", "$.decision", "source approval decision is unknown")
    if decision.get("scope") != APPROVED_SCOPE:
        _issue(issues, "SOURCE_APPROVAL_SCOPE_MISMATCH", "$.scope", "decision scope is not the bounded operator pilot")
    if tuple(decision.get("reviewer_roles") or ()) != REQUIRED_REVIEWER_ROLES:
        _issue(issues, "REVIEWER_ROLES_MISMATCH", "$.reviewer_roles", "all required reviewer roles must be recorded in order")

    bindings = decision.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        bindings = {}
        _issue(issues, "ARTIFACT_BINDINGS_MISSING", "$.artifact_bindings", "checksum-bound artifact identities are required")
    package = _read_json(package_file, issues, "PACKAGE_MALFORMED", "$.artifact_bindings.package")
    input_file = _bound_path(bindings, "input_path", issues)
    report_file = _bound_path(bindings, "validation_report_path", issues)
    input_payload = _read_json(input_file, issues, "INPUT_ARTIFACT_MALFORMED", "$.artifact_bindings.input_path") if input_file else None
    report = _read_json(report_file, issues, "VALIDATION_REPORT_MALFORMED", "$.artifact_bindings.validation_report_path") if report_file else None

    _checksum_binding(bindings, "input_sha256", input_file, issues, "INPUT_CHECKSUM_MISMATCH")
    _checksum_binding(bindings, "package_sha256", package_file, issues, "PACKAGE_CHECKSUM_MISMATCH")
    _checksum_binding(bindings, "validation_report_sha256", report_file, issues, "REPORT_CHECKSUM_MISMATCH")

    package_id = decision.get("package_id")
    for payload, path in ((package, "$.package_id"), (input_payload, "$.package_id"), (report, "$.package_id")):
        if isinstance(payload, Mapping) and payload.get("package_id") != package_id:
            _issue(issues, "PACKAGE_ID_MISMATCH", path, "package identity does not match the approval decision")
    if isinstance(package, Mapping):
        if package.get("schema_version") != OPERATOR_IMPORT_SCHEMA_VERSION or package.get("package_schema_version") != INPUT_SCHEMA_VERSION:
            _issue(issues, "DATA08_CONTRACT_MISMATCH", "$.artifact_bindings.package", "accepted package contract identity is invalid")
    if isinstance(input_payload, Mapping) and input_payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        _issue(issues, "DATA08_CONTRACT_MISMATCH", "$.artifact_bindings.input_path", "DATA08 input contract identity is invalid")
    if isinstance(report, Mapping):
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            _issue(issues, "DATA08_CONTRACT_MISMATCH", "$.artifact_bindings.validation_report_path", "DATA08 report contract identity is invalid")
        if report.get("validator_version") != VALIDATOR_VERSION:
            _issue(issues, "DATA08_VALIDATOR_VERSION_MISMATCH", "$.validator_version", "DATA08 validator version is invalid")
        if report.get("status") != "accepted":
            _issue(issues, "DATA08_REPORT_NOT_ACCEPTED", "$.status", "DATA08 report must be accepted")
        if report.get("downstream_consumability") != "structurally_valid_for_explicit_source_approval_review":
            _issue(issues, "DATA08_DOWNSTREAM_STATE_MISMATCH", "$.downstream_consumability", "DATA08 downstream state is invalid")
        if input_file and report.get("input_sha256") != _sha256(input_file):
            _issue(issues, "INPUT_CHECKSUM_MISMATCH", "$.input_sha256", "DATA08 report input checksum is invalid")

    reviews = decision.get("reviews")
    reviews = reviews if isinstance(reviews, Mapping) else {}
    for dimension in REQUIRED_REVIEW_DIMENSIONS:
        review = reviews.get(dimension)
        if not isinstance(review, Mapping):
            _issue(issues, "REVIEW_DIMENSION_MISSING", f"$.reviews.{dimension}", "required review dimension is missing")
        elif review.get("status") != "approved":
            _issue(issues, "REVIEW_DIMENSION_NOT_APPROVED", f"$.reviews.{dimension}.status", "review dimension must be approved")

    package_metrics = set()
    package_tickers = set()
    if isinstance(package, Mapping):
        for record in package.get("records") or []:
            if isinstance(record, Mapping):
                package_metrics.update((record.get("metrics") or {}).keys())
                ticker = record.get("ticker")
                if isinstance(ticker, str) and ticker:
                    package_tickers.add(ticker)
    approved_tickers = decision.get("approved_tickers")
    if not isinstance(approved_tickers, list) or not approved_tickers:
        _issue(issues, "APPROVED_TICKERS_MISSING", "$.approved_tickers", "approved tickers must be a non-empty list")
    else:
        valid_tickers = all(isinstance(ticker, str) and ticker and ticker == ticker.upper() for ticker in approved_tickers)
        if not valid_tickers or len(set(approved_tickers)) != len(approved_tickers):
            _issue(issues, "APPROVED_TICKERS_INVALID", "$.approved_tickers", "approved tickers must be unique canonical ticker strings")
        if approved_tickers != sorted(approved_tickers):
            _issue(issues, "APPROVED_TICKERS_NOT_SORTED", "$.approved_tickers", "approved tickers must be deterministically sorted")
        if set(approved_tickers) != package_tickers:
            _issue(issues, "APPROVED_TICKER_SET_MISMATCH", "$.approved_tickers", "approved tickers must exactly match package tickers")
    if not package_tickers:
        _issue(issues, "PACKAGE_TICKER_SET_EMPTY", "$.artifact_bindings.package", "accepted package must contain a ticker")
    elif len(package_tickers) > BOUNDED_PILOT_MAX_TICKERS:
        _issue(issues, "BOUNDED_PILOT_TICKER_LIMIT_EXCEEDED", "$.artifact_bindings.package", "package exceeds the bounded pilot ticker limit")
    approved_metrics = decision.get("approved_metrics")
    if not isinstance(approved_metrics, list) or not approved_metrics or set(approved_metrics) != package_metrics:
        _issue(issues, "APPROVED_METRIC_SET_MISMATCH", "$.approved_metrics", "approved metrics must exactly match package metrics")
    missing_metrics = decision.get("explicitly_missing_metrics")
    if not isinstance(missing_metrics, list) or set(missing_metrics) != set(MVP_METRIC_FIELDS) - package_metrics:
        _issue(issues, "EXPLICIT_MISSING_METRIC_SET_MISMATCH", "$.explicitly_missing_metrics", "explicitly missing metrics must be the exact allowlist complement")

    documents = decision.get("source_documents")
    if not isinstance(documents, list) or not documents:
        _issue(issues, "SOURCE_DOCUMENTS_MISSING", "$.source_documents", "source document bindings are required")
    elif source_document_root is None:
        _issue(issues, "SOURCE_DOCUMENT_ROOT_MISSING", "$.source_documents", "an operator-supplied source document root is required")
    else:
        root = Path(source_document_root).resolve()
        for index, document in enumerate(documents):
            if not isinstance(document, Mapping):
                _issue(issues, "SOURCE_DOCUMENT_BINDING_INVALID", f"$.source_documents[{index}]", "source document binding must be an object")
                continue
            relative_value = document.get("relative_path")
            if not isinstance(relative_value, str) or not relative_value:
                _issue(issues, "SOURCE_DOCUMENT_PATH_INVALID", f"$.source_documents[{index}].relative_path", "relative source document path is required")
                continue
            relative_path = Path(relative_value)
            if relative_path.is_absolute():
                _issue(issues, "SOURCE_DOCUMENT_PATH_ABSOLUTE", f"$.source_documents[{index}].relative_path", "source document path must be relative")
                continue
            if ".." in relative_path.parts:
                _issue(issues, "SOURCE_DOCUMENT_PATH_ESCAPE", f"$.source_documents[{index}].relative_path", "source document path must not traverse outside the source root")
                continue
            local_path = (root / relative_path).resolve()
            try:
                local_path.relative_to(root)
            except ValueError:
                _issue(issues, "SOURCE_DOCUMENT_PATH_ESCAPE", f"$.source_documents[{index}].relative_path", "resolved source document path escapes the source root")
                continue
            expected = document.get("sha256")
            if not local_path.is_file():
                _issue(issues, "SOURCE_DOCUMENT_MISSING", f"$.source_documents[{index}].relative_path", "source document is missing from the supplied root")
            elif expected != _sha256(local_path):
                _issue(issues, "SOURCE_DOCUMENT_CHECKSUM_MISMATCH", f"$.source_documents[{index}].sha256", "source document checksum does not match local evidence")

    return _result(issues, decision_file, package_file, decision_id=decision.get("decision_id"), package_id=package_id)


def _result(issues: list[dict[str, str]], decision_path: Path | None, package_path: Path, **identity: Any) -> dict[str, Any]:
    issues.sort(key=lambda row: (row["path"], row["reason_code"]))
    approved = not issues
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validation_status": "approved" if approved else "blocked",
        "concrete_package_source_approved": approved,
        "decision_path": decision_path.as_posix() if decision_path else None,
        "package_path": package_path.as_posix(),
        **identity,
        "reason_codes": sorted({row["reason_code"] for row in issues}),
        "issues": issues,
    }


def _bound_path(bindings: Mapping[str, Any], key: str, issues: list[dict[str, str]]) -> Path | None:
    value = bindings.get(key)
    if not isinstance(value, str) or not value:
        _issue(issues, "ARTIFACT_BINDING_MISSING", f"$.artifact_bindings.{key}", "artifact path binding is required")
        return None
    return Path(value)


def _checksum_binding(bindings: Mapping[str, Any], key: str, path: Path | None, issues: list[dict[str, str]], code: str) -> None:
    if path is None or not path.is_file() or bindings.get(key) != _sha256(path):
        _issue(issues, code, f"$.artifact_bindings.{key}", "artifact checksum does not match bound content")


def _read_json(path: Path, issues: list[dict[str, str]], code: str, issue_path: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError):
        _issue(issues, code, issue_path, "bound artifact must be readable strict JSON")
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"forbidden JSON numeric constant: {value}")


def _issue(issues: list[dict[str, str]], code: str, path: str, message: str) -> None:
    issues.append({"reason_code": code, "path": path, "message": message})
