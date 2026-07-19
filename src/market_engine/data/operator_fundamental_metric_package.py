from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.data.fundamental_evidence_coverage import MVP_METRIC_FIELDS
from market_engine.data.validated_fundamental_metric_sourcing import (
    ALLOWED_METRIC_UNITS,
    OPERATOR_IMPORT_SCHEMA_VERSION,
)


INPUT_SCHEMA_VERSION = "market-engine-data08-operator-fundamental-metric-input-v1"
VALIDATOR_VERSION = "market-engine-data08-operator-fundamental-metric-validator-v1"
REPORT_SCHEMA_VERSION = "market-engine-data08-operator-fundamental-metric-validation-report-v1"
PERIOD_TYPES = frozenset({"quarter", "annual"})
AUTHORITY_FIELDS = frozenset(
    {
        "recommendation",
        "decision",
        "tradeable",
        "conviction",
        "urgency",
        "ranking",
        "allocation",
        "position_size",
        "target_price",
        "expected_return",
        "buy",
        "sell",
        "hold",
        "execution_gating",
        "order",
    }
)
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_FISCAL_PERIOD = re.compile(r"^(FY|Q[1-4])$")


class OperatorPackageInputError(ValueError):
    pass


class OperatorPackageConfigurationError(ValueError):
    reason_code = "OUTPUT_PATH_COLLISION"


def prepare_operator_fundamental_metric_package(
    input_path: str | Path,
    *,
    package_output_path: str | Path,
    report_output_path: str | Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = Path(input_path)
    package_path = Path(package_output_path)
    report_path = Path(report_output_path)
    if package_path.resolve(strict=False) == report_path.resolve(strict=False):
        raise OperatorPackageConfigurationError(
            "OUTPUT_PATH_COLLISION: package and validation report outputs resolve to the same path"
        )
    if package_path.exists() or report_path.exists():
        raise FileExistsError("ME-DATA08 output paths must not already exist")
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise OperatorPackageInputError(f"unable to read operator input: {exc}") from exc
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise OperatorPackageInputError(f"malformed operator JSON: {exc}") from exc

    package, report = validate_and_normalize_operator_input(payload, input_sha256=_sha256_bytes(raw.encode("utf-8")))
    report["artifacts"] = {
        "accepted_package": package_path.as_posix() if package is not None else None,
        "validation_report": report_path.as_posix(),
    }
    if package is not None:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(package_path, package)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    return package, report


def validate_and_normalize_operator_input(
    payload: object,
    *,
    input_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    deferred_normalization_count = 0
    records_out: list[dict[str, Any]] = []
    input_count = 0
    package_id: str | None = None

    if not isinstance(payload, Mapping):
        _issue(errors, "PACKAGE_NOT_OBJECT", "$", "operator input must be a JSON object")
    else:
        _unknown_fields(payload, {"schema_version", "package_id", "created_at", "records"}, "$", errors)
        package_id = _required_text(payload, "package_id", "$.package_id", errors)
        if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
            _issue(errors, "UNSUPPORTED_SCHEMA_VERSION", "$.schema_version", f"expected {INPUT_SCHEMA_VERSION}")
        _timestamp(payload.get("created_at"), "$.created_at", errors)
        _authority_issues(payload, "$", errors)
        records = payload.get("records")
        if not isinstance(records, list):
            _issue(errors, "RECORDS_MISSING", "$.records", "records must be a non-empty array")
            records = []
        input_count = len(records)
        if not records:
            _issue(errors, "EMPTY_METRIC_SET", "$.records", "at least one metric record is required")
        seen: dict[tuple[str, str, str], str] = {}
        record_groups: dict[tuple[str, str], dict[str, Any]] = {}
        for index, record in enumerate(records):
            path = f"$.records[{index}]"
            if not isinstance(record, Mapping):
                _issue(errors, "RECORD_NOT_OBJECT", path, "metric record must be an object")
                continue
            _unknown_fields(
                record,
                {"ticker", "company_identity", "canonical_metric", "value", "unit", "currency", "period_type", "period_start", "period_end", "instant_date", "fiscal_year", "fiscal_period", "provenance"},
                path,
                errors,
            )
            before = len(errors)
            ticker = _required_text(record, "ticker", f"{path}.ticker", errors).upper()
            if ticker and not _TICKER.fullmatch(ticker):
                _issue(errors, "INVALID_TICKER", f"{path}.ticker", "ticker format is invalid")
            company = record.get("company_identity")
            if not isinstance(company, Mapping):
                _issue(errors, "COMPANY_IDENTITY_MISSING", f"{path}.company_identity", "company identity is required")
                company = {}
            _unknown_fields(company, {"name", "instrument_id"}, f"{path}.company_identity", errors)
            company_name = _required_text(company, "name", f"{path}.company_identity.name", errors)
            instrument_id = _required_text(company, "instrument_id", f"{path}.company_identity.instrument_id", errors)
            expected_instrument = f"equity:{ticker.lower()}" if ticker else ""
            if instrument_id and instrument_id != expected_instrument:
                _issue(errors, "COMPANY_TICKER_MISMATCH", f"{path}.company_identity.instrument_id", "instrument identity does not match ticker")

            metric = _required_text(record, "canonical_metric", f"{path}.canonical_metric", errors)
            if metric not in MVP_METRIC_FIELDS:
                code = "AMBIGUOUS_METRIC_ALIAS" if metric in {"revenue_growth", "eps_growth", "margin", "debt_equity"} else "METRIC_NOT_ALLOWLISTED"
                _issue(errors, code, f"{path}.canonical_metric", "metric is not an approved canonical MVP metric")
            period_type = _required_text(record, "period_type", f"{path}.period_type", errors)
            if period_type not in PERIOD_TYPES:
                _issue(errors, "INCOMPATIBLE_PERIOD_TYPE", f"{path}.period_type", "MVP metrics require quarter or annual duration periods")
            fiscal_year = record.get("fiscal_year")
            if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or not 1900 <= fiscal_year <= 2200:
                _issue(errors, "INVALID_FISCAL_CONTEXT", f"{path}.fiscal_year", "fiscal_year must be an integer from 1900 through 2200")
            fiscal_period = _required_text(record, "fiscal_period", f"{path}.fiscal_period", errors)
            if fiscal_period and not _FISCAL_PERIOD.fullmatch(fiscal_period):
                _issue(errors, "INVALID_FISCAL_CONTEXT", f"{path}.fiscal_period", "fiscal_period must be FY or Q1 through Q4")
            if (period_type == "annual" and fiscal_period != "FY") or (period_type == "quarter" and not fiscal_period.startswith("Q")):
                _issue(errors, "INCOMPATIBLE_PERIOD_TYPE", f"{path}.fiscal_period", "fiscal_period is incompatible with period_type")
            start = _date(record.get("period_start"), f"{path}.period_start", errors)
            end = _date(record.get("period_end"), f"{path}.period_end", errors)
            if start and end and start >= end:
                _issue(errors, "INVALID_REPORTING_PERIOD", f"{path}.period_end", "period_end must be after period_start")
            if record.get("instant_date") is not None:
                _issue(errors, "INCOMPATIBLE_PERIOD_TYPE", f"{path}.instant_date", "duration MVP metrics do not accept instant_date")
            reporting_period = f"{fiscal_year}-{fiscal_period}" if isinstance(fiscal_year, int) and fiscal_period else ""

            value = record.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _issue(errors, "VALUE_NOT_NUMERIC", f"{path}.value", "value must be a JSON number")
            elif not math.isfinite(float(value)):
                _issue(errors, "VALUE_NOT_FINITE", f"{path}.value", "NaN and Infinity are forbidden")
            unit = _required_text(record, "unit", f"{path}.unit", errors)
            if unit not in ALLOWED_METRIC_UNITS:
                _issue(errors, "INVALID_UNIT", f"{path}.unit", "unit must be ratio or percent")
            if "scale" in record or "multiplier" in record:
                _issue(errors, "AMBIGUOUS_SCALE", path, "scale and multiplier fields are unsupported")
            if record.get("currency") not in (None, ""):
                _issue(errors, "CURRENCY_NOT_APPLICABLE", f"{path}.currency", "approved MVP ratio metrics do not accept currency")

            provenance = record.get("provenance")
            if not isinstance(provenance, Mapping):
                _issue(errors, "PROVENANCE_INCOMPLETE", f"{path}.provenance", "provenance object is required")
                provenance = {}
            _unknown_fields(provenance, {"source_name", "source_reference", "raw_source_field", "source_date", "observed_at", "acquired_at", "parser_version"}, f"{path}.provenance", errors)
            source_name = _required_text(provenance, "source_name", f"{path}.provenance.source_name", errors, "PROVENANCE_INCOMPLETE")
            source_reference = _required_text(provenance, "source_reference", f"{path}.provenance.source_reference", errors, "PROVENANCE_INCOMPLETE")
            raw_source_field = _required_text(provenance, "raw_source_field", f"{path}.provenance.raw_source_field", errors, "PROVENANCE_INCOMPLETE")
            parser_version = _required_text(provenance, "parser_version", f"{path}.provenance.parser_version", errors, "PROVENANCE_INCOMPLETE")
            source_date = _date(provenance.get("source_date"), f"{path}.provenance.source_date", errors)
            observed_at = _timestamp(provenance.get("observed_at"), f"{path}.provenance.observed_at", errors)
            acquired_at = _timestamp(provenance.get("acquired_at"), f"{path}.provenance.acquired_at", errors)
            if len(errors) != before:
                continue
            raw_value = float(value)
            normalized_value = raw_value / 100.0 if unit == "percent" else raw_value
            if unit == "percent":
                deferred_normalization_count += 1
                warnings.append({"reason_code": "PERCENT_VALIDATED_FOR_DATA07_RATIO_NORMALIZATION", "path": f"{path}.value", "metric_identity": metric, "message": "raw percent value and unit are retained for deferred ME-DATA07 ratio normalization"})
            duplicate_key = (ticker, metric, reporting_period)
            fingerprint = json.dumps({"value": normalized_value, "source": source_reference}, sort_keys=True)
            if duplicate_key in seen:
                code = "DUPLICATE_METRIC_CONFLICT" if seen[duplicate_key] != fingerprint else "DUPLICATE_METRIC_RECORD"
                _issue(errors, code, path, "canonical ticker/metric/period key is duplicated", metric)
                continue
            seen[duplicate_key] = fingerprint
            group_key = (ticker, reporting_period)
            group = record_groups.setdefault(group_key, {
                "ticker": ticker,
                "instrument_id": instrument_id,
                "company_name": company_name,
                "provider_symbol": ticker,
                "provider": source_name,
                "source_date": source_date,
                "reporting_period": reporting_period,
                "period_type": period_type,
                "period_start": start,
                "period_end": end,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "source_reference": source_reference,
                "parser_version": parser_version,
                "snapshot_id": payload.get("package_id"),
                "acquired_at": acquired_at,
                "observed_at": observed_at,
                "metrics": {},
            })
            identity_fields = ("instrument_id", "company_name", "provider", "source_reference", "period_type", "period_start", "period_end")
            candidate = {"instrument_id": instrument_id, "company_name": company_name, "provider": source_name, "source_reference": source_reference, "period_type": period_type, "period_start": start, "period_end": end}
            if any(group[field] != candidate[field] for field in identity_fields):
                _issue(errors, "CONFLICTING_RECORD_CONTEXT", path, "records for one ticker/reporting period have conflicting identity, period, or provenance", metric)
                continue
            group["metrics"][metric] = {
                "value": raw_value,
                "unit": unit,
                "reporting_period": reporting_period,
                "raw_source_field": raw_source_field,
            }

        records_out = sorted(record_groups.values(), key=lambda row: (row["ticker"], row["reporting_period"]))
        for row in records_out:
            row["metrics"] = {key: row["metrics"][key] for key in MVP_METRIC_FIELDS if key in row["metrics"]}

    errors.sort(key=lambda item: (item["path"], item["reason_code"], item.get("metric_identity") or ""))
    warnings.sort(key=lambda item: (item["path"], item["reason_code"]))
    accepted = not errors
    package = None
    if accepted:
        package = {
            "schema_version": OPERATOR_IMPORT_SCHEMA_VERSION,
            "package_id": package_id,
            "package_schema_version": INPUT_SCHEMA_VERSION,
            "records": records_out,
        }
    return package, {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "package_id": package_id,
        "status": "accepted" if accepted else "rejected",
        "downstream_consumability": "structurally_valid_for_explicit_source_approval_review" if accepted else "not_consumable",
        "counts": {
            "input_metrics": input_count,
            "accepted_metrics": input_count if accepted else 0,
            "deferred_normalization_metrics": deferred_normalization_count if accepted else 0,
            "warning_count": len(warnings),
            "rejected_metrics": 0 if accepted else input_count,
            "error_count": len(errors),
        },
        "input_sha256": input_sha256,
        "errors": errors,
        "warnings": warnings,
        "boundary": "Validation proves structural provenance completeness only. Source authenticity and governance approval remain unverified; explicit source approval review is required before ME-DATA07 import. No automatic import, analysis readiness, recommendation, or decision authority is granted.",
    }


def _required_text(value: Mapping[str, Any], key: str, path: str, errors: list[dict[str, Any]], code: str = "REQUIRED_FIELD_MISSING") -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        _issue(errors, code, path, f"{key} must be a non-empty string")
        return ""
    return result.strip()


def _timestamp(value: object, path: str, errors: list[dict[str, Any]]) -> str:
    if not isinstance(value, str):
        _issue(errors, "INVALID_TIMESTAMP", path, "timestamp must be an ISO-8601 UTC string")
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _issue(errors, "INVALID_TIMESTAMP", path, "timestamp must be an ISO-8601 UTC string")
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _issue(errors, "INVALID_TIMESTAMP", path, "timestamp must include a UTC offset")
        return ""
    return value


def _date(value: object, path: str, errors: list[dict[str, Any]]) -> str:
    if not isinstance(value, str):
        _issue(errors, "INVALID_REPORTING_PERIOD", path, "period date must use YYYY-MM-DD")
        return ""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        _issue(errors, "INVALID_REPORTING_PERIOD", path, "period date must use YYYY-MM-DD")
        return ""
    return value


def _authority_issues(value: object, path: str, errors: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in AUTHORITY_FIELDS:
                _issue(errors, "AUTHORITY_FIELD_FORBIDDEN", child_path, "authority-bearing fields are outside the fundamental evidence boundary")
            _authority_issues(child, child_path, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _authority_issues(child, f"{path}[{index}]", errors)


def _unknown_fields(value: Mapping[str, Any], allowed: set[str], path: str, errors: list[dict[str, Any]]) -> None:
    for key in sorted(set(value) - allowed):
        _issue(errors, "UNKNOWN_FIELD", f"{path}.{key}", "field is outside the versioned package contract")


def _issue(errors: list[dict[str, Any]], code: str, path: str, message: str, metric_identity: str | None = None) -> None:
    issue = {"reason_code": code, "path": path, "message": message}
    if metric_identity:
        issue["metric_identity"] = metric_identity
    errors.append(issue)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"forbidden JSON numeric constant: {value}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
