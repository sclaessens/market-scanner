"""Controlled v2 synthetic persistence boundary for fundamentals records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence


class PersistenceIssueCode(StrEnum):
    """Source/data-focused persistence validation issue codes."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_FIELD_TYPE = "invalid_field_type"
    MISSING_PROVENANCE = "missing_provenance"
    MISSING_RAW_EVIDENCE_LINK = "missing_raw_evidence_link"
    MISSING_NORMALIZED_RECORD_SET_LINK = "missing_normalized_record_set_link"
    MISSING_VALUE_ZERO_SUBSTITUTION = "missing_value_zero_substitution"
    FORBIDDEN_FIELD = "forbidden_field"
    FORBIDDEN_OUTPUT_PATH = "forbidden_output_path"
    FAIL_CLOSED_SOURCE_DATA_STATE = "fail_closed_source_data_state"


@dataclass(frozen=True)
class PersistenceIssue:
    """Explicit persistence issue without investment interpretation."""

    record_family: str
    record_id: str
    field_name: str
    issue_code: PersistenceIssueCode
    observed_value: object


@dataclass(frozen=True)
class PersistenceBatch:
    """Validated in-memory persistence batch with separated record families."""

    raw_records: tuple[Mapping[str, object], ...]
    normalized_records: tuple[Mapping[str, object], ...]
    readiness_records: tuple[Mapping[str, object], ...]
    issues: tuple[PersistenceIssue, ...]
    batch_status: str


@dataclass(frozen=True)
class PersistenceWriteRecord:
    """Deterministic metadata for one synthetic write."""

    record_family: str
    record_id: str
    output_path: str
    byte_count: int


@dataclass(frozen=True)
class PersistenceWriteResult:
    """Deterministic metadata returned by synthetic persistence writes."""

    output_root: str
    batch_status: str
    raw_record_count: int
    normalized_record_count: int
    readiness_record_count: int
    write_records: tuple[PersistenceWriteRecord, ...]
    issues: tuple[PersistenceIssue, ...]


RAW_REQUIRED_FIELDS: tuple[str, ...] = (
    "raw_evidence_id",
    "provider_name",
    "provider_category",
    "provider_record_id",
    "original_source_reference",
    "ticker",
    "symbol",
    "entity_identifier",
    "source_timestamp",
    "retrieval_timestamp",
    "reported_period",
    "fiscal_year",
    "fiscal_quarter",
    "currency",
    "unit",
    "raw_fields",
    "missing_field_evidence",
    "provenance_metadata",
    "raw_payload_hash",
    "capture_version",
    "validation_warnings",
)

NORMALIZED_REQUIRED_FIELDS: tuple[str, ...] = (
    "normalized_record_id",
    "raw_evidence_id",
    "provider_name",
    "original_source_reference",
    "ticker",
    "entity_identifier",
    "metric_name",
    "metric_value",
    "metric_value_status",
    "currency",
    "currency_status",
    "unit",
    "unit_status",
    "reported_period",
    "fiscal_year",
    "fiscal_quarter",
    "source_timestamp",
    "retrieval_timestamp",
    "normalization_version",
    "validation_warnings",
)

READINESS_REQUIRED_FIELDS: tuple[str, ...] = (
    "readiness_record_id",
    "raw_evidence_id",
    "normalized_record_set_id",
    "ticker",
    "provider_name",
    "readiness_state",
    "source_data_status",
    "missing_fundamentals_count",
    "partial_data_count",
    "stale_data_count",
    "invalid_data_count",
    "provenance_status",
    "parseability_status",
    "consistency_status",
    "freshness_status",
    "readiness_warnings",
    "readiness_version",
)

OPTIONAL_EMPTY_FIELDS: tuple[str, ...] = (
    "fiscal_quarter",
    "validation_warnings",
    "missing_field_evidence",
)

MISSING_VALUE_STATUSES: tuple[str, ...] = (
    "missing",
    "missing_source_field",
    "not_reported",
    "unavailable",
    "invalid",
    "invalid_unparseable",
    "not_parseable",
    "blocked_by_provenance_gap",
)

ZERO_LIKE_MISSING_VALUES: tuple[object, ...] = (0, 0.0, "0", False, "")

FORBIDDEN_OUTPUT_FIELDS: tuple[str, ...] = (
    "allocation",
    "conviction",
    "portfolio_action",
    "recommendation",
    "target_price",
    "tradeability",
    "urgency",
    "valuation_attractiveness",
    "valuation_score",
)

FORBIDDEN_OUTPUT_WORDS: tuple[str, ...] = (
    "buy",
    "sell",
    "hold",
)

FORBIDDEN_OUTPUT_ROOT_PARTS: tuple[tuple[str, ...], ...] = (
    ("data",),
    ("reports",),
    ("reports", "daily"),
    ("reports", "daily", "tele" + "gram_message.txt"),
    (".github", "workflows"),
)

WRITE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("raw", "raw_source_evidence", "raw_evidence_id"),
    ("normalized", "normalized_fundamentals", "normalized_record_id"),
    ("readiness", "source_data_readiness", "readiness_record_id"),
)


def validate_raw_evidence_record(
    record: Mapping[str, object],
) -> tuple[PersistenceIssue, ...]:
    """Validate one raw evidence mapping without file IO or source access."""

    issues = _required_field_issues(
        "raw",
        record,
        record_id_field="raw_evidence_id",
        required_fields=RAW_REQUIRED_FIELDS,
    )
    issues.extend(_mapping_type_issue("raw", record, "raw_fields"))
    issues.extend(_mapping_type_issue("raw", record, "provenance_metadata"))
    issues.extend(_sequence_type_issue("raw", record, "missing_field_evidence"))
    issues.extend(_sequence_type_issue("raw", record, "validation_warnings"))
    issues.extend(_provenance_issues("raw", record))
    issues.extend(_forbidden_semantic_issues("raw", record))
    issues.extend(_raw_missing_to_zero_issues(record))
    return tuple(issues)


def validate_normalized_fundamental_record(
    record: Mapping[str, object],
) -> tuple[PersistenceIssue, ...]:
    """Validate one normalized fundamentals mapping."""

    issues = _required_field_issues(
        "normalized",
        record,
        record_id_field="normalized_record_id",
        required_fields=NORMALIZED_REQUIRED_FIELDS,
    )
    issues.extend(_sequence_type_issue("normalized", record, "validation_warnings"))
    issues.extend(_provenance_issues("normalized", record))
    issues.extend(_normalized_missing_to_zero_issues(record))
    issues.extend(_forbidden_semantic_issues("normalized", record))
    return tuple(issues)


def validate_readiness_record(
    record: Mapping[str, object],
) -> tuple[PersistenceIssue, ...]:
    """Validate one neutral source-data readiness mapping."""

    issues = _required_field_issues(
        "readiness",
        record,
        record_id_field="readiness_record_id",
        required_fields=READINESS_REQUIRED_FIELDS,
    )
    issues.extend(_sequence_type_issue("readiness", record, "readiness_warnings"))
    issues.extend(_readiness_fail_closed_issues(record))
    if not record.get("raw_evidence_id"):
        issues.append(
            _issue(
                "readiness",
                record,
                "raw_evidence_id",
                PersistenceIssueCode.MISSING_RAW_EVIDENCE_LINK,
            )
        )
    if not record.get("normalized_record_set_id"):
        issues.append(
            _issue(
                "readiness",
                record,
                "normalized_record_set_id",
                PersistenceIssueCode.MISSING_NORMALIZED_RECORD_SET_LINK,
            )
        )
    issues.extend(_forbidden_semantic_issues("readiness", record))
    return tuple(issues)


def prepare_persistence_batch(
    raw_records: Sequence[Mapping[str, object]],
    normalized_records: Sequence[Mapping[str, object]],
    readiness_records: Sequence[Mapping[str, object]],
) -> PersistenceBatch:
    """Prepare a deterministic batch while preserving record family separation."""

    raw_records_tuple = tuple(raw_records)
    normalized_records_tuple = tuple(normalized_records)
    readiness_records_tuple = tuple(readiness_records)
    issues: list[PersistenceIssue] = []

    raw_ids = {
        str(record.get("raw_evidence_id"))
        for record in raw_records_tuple
        if record.get("raw_evidence_id")
    }
    normalized_set_ids = {
        str(record.get("normalized_record_set_id"))
        for record in readiness_records_tuple
        if record.get("normalized_record_set_id")
    }

    for record in raw_records_tuple:
        issues.extend(validate_raw_evidence_record(record))
    for record in normalized_records_tuple:
        issues.extend(validate_normalized_fundamental_record(record))
        raw_evidence_id = record.get("raw_evidence_id")
        if raw_evidence_id not in raw_ids:
            issues.append(
                _issue(
                    "normalized",
                    record,
                    "raw_evidence_id",
                    PersistenceIssueCode.MISSING_RAW_EVIDENCE_LINK,
                )
            )
    for record in readiness_records_tuple:
        issues.extend(validate_readiness_record(record))
        if record.get("raw_evidence_id") not in raw_ids:
            issues.append(
                _issue(
                    "readiness",
                    record,
                    "raw_evidence_id",
                    PersistenceIssueCode.MISSING_RAW_EVIDENCE_LINK,
                )
            )
    if normalized_records_tuple and not normalized_set_ids:
        issues.append(
            PersistenceIssue(
                record_family="batch",
                record_id="",
                field_name="normalized_record_set_id",
                issue_code=PersistenceIssueCode.MISSING_NORMALIZED_RECORD_SET_LINK,
                observed_value=None,
            )
        )

    return PersistenceBatch(
        raw_records=raw_records_tuple,
        normalized_records=normalized_records_tuple,
        readiness_records=readiness_records_tuple,
        issues=tuple(issues),
        batch_status="valid" if not issues else "fail_closed",
    )


def write_synthetic_persistence_batch(
    batch: PersistenceBatch,
    output_root: str | Path,
) -> PersistenceWriteResult:
    """Write separated synthetic records under a caller-provided temp root only."""

    output_root_path = Path(output_root)
    path_issues = validate_synthetic_output_root(output_root_path)
    issues = (*batch.issues, *path_issues)
    if issues:
        return PersistenceWriteResult(
            output_root=str(output_root_path),
            batch_status="fail_closed",
            raw_record_count=len(batch.raw_records),
            normalized_record_count=len(batch.normalized_records),
            readiness_record_count=len(batch.readiness_records),
            write_records=(),
            issues=issues,
        )

    write_records: list[PersistenceWriteRecord] = []
    for record_family, directory_name, record_id_field in WRITE_FAMILIES:
        records = _records_for_family(batch, record_family)
        family_dir = output_root_path / directory_name
        family_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            record_id = str(record[record_id_field])
            output_path = family_dir / f"{_safe_file_stem(record_id)}.json"
            serialized = _stable_json(record)
            output_path.write_text(serialized, encoding="utf-8")
            write_records.append(
                PersistenceWriteRecord(
                    record_family=record_family,
                    record_id=record_id,
                    output_path=str(output_path),
                    byte_count=len(serialized.encode("utf-8")),
                )
            )

    return PersistenceWriteResult(
        output_root=str(output_root_path),
        batch_status="written",
        raw_record_count=len(batch.raw_records),
        normalized_record_count=len(batch.normalized_records),
        readiness_record_count=len(batch.readiness_records),
        write_records=tuple(write_records),
        issues=(),
    )


def validate_synthetic_output_root(
    output_root: str | Path,
) -> tuple[PersistenceIssue, ...]:
    """Reject production, report, Telegram, and workflow output roots."""

    path = Path(output_root)
    normalized_parts = tuple(part.lower() for part in path.parts)
    issues: list[PersistenceIssue] = []

    for forbidden_parts in FORBIDDEN_OUTPUT_ROOT_PARTS:
        if _contains_path_parts(normalized_parts, forbidden_parts):
            issues.append(
                PersistenceIssue(
                    record_family="write",
                    record_id="",
                    field_name="output_root",
                    issue_code=PersistenceIssueCode.FORBIDDEN_OUTPUT_PATH,
                    observed_value=str(path),
                )
            )
            break

    return tuple(issues)


def _required_field_issues(
    record_family: str,
    record: Mapping[str, object],
    *,
    record_id_field: str,
    required_fields: Sequence[str],
) -> list[PersistenceIssue]:
    issues: list[PersistenceIssue] = []
    for field_name in required_fields:
        if field_name not in record:
            issues.append(
                PersistenceIssue(
                    record_family=record_family,
                    record_id=str(record.get(record_id_field, "")),
                    field_name=field_name,
                    issue_code=PersistenceIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue
        if _is_missing_required_value(field_name, record[field_name]):
            issues.append(
                _issue(
                    record_family,
                    record,
                    field_name,
                    PersistenceIssueCode.MISSING_REQUIRED_VALUE,
                )
            )
    return issues


def _mapping_type_issue(
    record_family: str,
    record: Mapping[str, object],
    field_name: str,
) -> tuple[PersistenceIssue, ...]:
    if field_name in record and not isinstance(record[field_name], Mapping):
        return (
            _issue(
                record_family,
                record,
                field_name,
                PersistenceIssueCode.INVALID_FIELD_TYPE,
            ),
        )
    return ()


def _sequence_type_issue(
    record_family: str,
    record: Mapping[str, object],
    field_name: str,
) -> tuple[PersistenceIssue, ...]:
    if field_name in record and not isinstance(record[field_name], list | tuple):
        return (
            _issue(
                record_family,
                record,
                field_name,
                PersistenceIssueCode.INVALID_FIELD_TYPE,
            ),
        )
    return ()


def _provenance_issues(
    record_family: str,
    record: Mapping[str, object],
) -> list[PersistenceIssue]:
    issues: list[PersistenceIssue] = []
    for field_name in (
        "provider_name",
        "original_source_reference",
        "source_timestamp",
        "retrieval_timestamp",
    ):
        if field_name in record and not record.get(field_name):
            issues.append(
                _issue(
                    record_family,
                    record,
                    field_name,
                    PersistenceIssueCode.MISSING_PROVENANCE,
                )
            )
    return issues


def _raw_missing_to_zero_issues(
    record: Mapping[str, object],
) -> list[PersistenceIssue]:
    missing_fields = {
        str(evidence.get("field_name"))
        for evidence in record.get("missing_field_evidence", ())
        if isinstance(evidence, Mapping) and evidence.get("field_name")
    }
    raw_fields = record.get("raw_fields", {})
    if not isinstance(raw_fields, Mapping):
        return []
    return [
        _issue(
            "raw",
            record,
            f"raw_fields.{field_name}",
            PersistenceIssueCode.MISSING_VALUE_ZERO_SUBSTITUTION,
        )
        for field_name in sorted(missing_fields)
        if raw_fields.get(field_name) in ZERO_LIKE_MISSING_VALUES
        and raw_fields.get(field_name) is not None
    ]


def _normalized_missing_to_zero_issues(
    record: Mapping[str, object],
) -> tuple[PersistenceIssue, ...]:
    value_status = str(record.get("metric_value_status", ""))
    if (
        value_status in MISSING_VALUE_STATUSES
        and record.get("metric_value") in ZERO_LIKE_MISSING_VALUES
        and record.get("metric_value") is not None
    ):
        return (
            _issue(
                "normalized",
                record,
                "metric_value",
                PersistenceIssueCode.MISSING_VALUE_ZERO_SUBSTITUTION,
            ),
        )
    return ()


def _forbidden_semantic_issues(
    record_family: str,
    record: Mapping[str, object],
) -> list[PersistenceIssue]:
    issues: list[PersistenceIssue] = []
    for field_name in record:
        lowered = str(field_name).lower()
        if lowered in FORBIDDEN_OUTPUT_FIELDS or lowered in FORBIDDEN_OUTPUT_WORDS:
            issues.append(
                _issue(
                    record_family,
                    record,
                    str(field_name),
                    PersistenceIssueCode.FORBIDDEN_FIELD,
                )
            )
    return issues


def _readiness_fail_closed_issues(
    record: Mapping[str, object],
) -> tuple[PersistenceIssue, ...]:
    if record.get("source_data_status") in {
        "controlled_forbidden_input_present",
        "invalid",
        "provenance_gap",
    }:
        return (
            _issue(
                "readiness",
                record,
                "source_data_status",
                PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE,
            ),
        )
    if record.get("readiness_state") == "fail_closed":
        return (
            _issue(
                "readiness",
                record,
                "readiness_state",
                PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE,
            ),
        )
    return ()


def _is_missing_required_value(field_name: str, value: object) -> bool:
    if field_name in OPTIONAL_EMPTY_FIELDS:
        return value is None
    if field_name in {"metric_value"}:
        return False
    if isinstance(value, Mapping | list | tuple):
        return False
    return value is None or value == ""


def _issue(
    record_family: str,
    record: Mapping[str, object],
    field_name: str,
    issue_code: PersistenceIssueCode,
) -> PersistenceIssue:
    record_id = str(
        record.get("raw_evidence_id")
        or record.get("normalized_record_id")
        or record.get("readiness_record_id")
        or ""
    )
    return PersistenceIssue(
        record_family=record_family,
        record_id=record_id,
        field_name=field_name,
        issue_code=issue_code,
        observed_value=record.get(field_name),
    )


def _records_for_family(
    batch: PersistenceBatch,
    record_family: str,
) -> tuple[Mapping[str, object], ...]:
    if record_family == "raw":
        return batch.raw_records
    if record_family == "normalized":
        return batch.normalized_records
    return batch.readiness_records


def _stable_json(record: Mapping[str, object]) -> str:
    return json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _safe_file_stem(record_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in record_id
    )


def _contains_path_parts(
    path_parts: tuple[str, ...],
    forbidden_parts: tuple[str, ...],
) -> bool:
    if len(forbidden_parts) > len(path_parts):
        return False
    for index in range(0, len(path_parts) - len(forbidden_parts) + 1):
        if path_parts[index : index + len(forbidden_parts)] == forbidden_parts:
            return True
    return False
