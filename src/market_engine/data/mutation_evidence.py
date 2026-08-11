from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.data.observation_receipts import canonical_row_sha256


MUTATION_SCHEMA_VERSION = "market-engine-canonical-mutation-ledger-v2"
SESSION_LEDGER_SCHEMA_VERSION = "market-engine-session-resolution-ledger-v1"
MUTATION_TYPES = frozenset(
    {"row_added", "row_modified", "row_deleted", "row_unchanged"}
)
SESSION_STATES = frozenset(
    {
        "observed_primary",
        "observed_fallback",
        "explained_absent",
        "unresolved",
        "not_expected",
    }
)


class MutationEvidenceError(ValueError):
    pass


def derive_canonical_mutations(
    *,
    baseline_path: str | Path | None,
    staged_path: str | Path,
    instrument_id: str,
    ticker: str,
    exchange: str,
    currency: str,
    include_unchanged: bool = False,
) -> list[dict[str, Any]]:
    baseline = _load_rows(Path(baseline_path)) if baseline_path is not None else {}
    staged = _load_rows(Path(staged_path))
    mutations: list[dict[str, Any]] = []
    for session in sorted(set(baseline).union(staged)):
        previous = baseline.get(session)
        current = staged.get(session)
        previous_digest = (
            _row_digest(previous, instrument_id=instrument_id, currency=currency)
            if previous is not None
            else None
        )
        current_digest = (
            _row_digest(current, instrument_id=instrument_id, currency=currency)
            if current is not None
            else None
        )
        mutation_type = (
            "row_added"
            if previous is None
            else "row_deleted"
            if current is None
            else "row_unchanged"
            if previous_digest == current_digest
            else "row_modified"
        )
        if mutation_type == "row_unchanged" and not include_unchanged:
            continue
        field_diff = {}
        if previous is not None and current is not None:
            field_diff = {
                field: {"previous": previous[field], "current": current[field]}
                for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
                if previous[field] != current[field]
            }
        mutations.append(
            {
                "schema_version": MUTATION_SCHEMA_VERSION,
                "instrument_id": instrument_id,
                "ticker": ticker,
                "exchange": exchange,
                "session_date": session,
                "mutation_type": mutation_type,
                "previous_canonical_row_sha256": previous_digest,
                "new_canonical_row_sha256": current_digest,
                "field_diff": field_diff,
                "previous_values": _diagnostic_values(previous),
                "new_values": _diagnostic_values(current),
            }
        )
    return mutations


def mutation_evidence_diagnostics(
    mutations: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    artifact_replay_failures: Sequence[str] = (),
) -> dict[str, Any]:
    """Return deterministic reconciliation diagnostics without discarding mutations."""
    mutation_rows = sorted(
        [dict(row) for row in mutations],
        key=lambda row: (
            str(row.get("instrument_id")),
            str(row.get("session_date")),
            str(row.get("mutation_type")),
        ),
    )
    receipt_rows = sorted(
        [dict(row) for row in receipts],
        key=lambda row: (
            str(row.get("instrument_id")), str(row.get("session_date"))
        ),
    )
    required_rows = [
        row
        for row in mutation_rows
        if row.get("mutation_type") in {"row_added", "row_modified"}
    ]
    required = {
        (str(row.get("instrument_id")), str(row.get("session_date"))): row
        for row in required_rows
    }
    provided = {
        (str(row.get("instrument_id")), str(row.get("session_date"))): row
        for row in receipt_rows
    }
    duplicate_mutation_count = len(required_rows) - len(required)
    duplicate_receipt_count = len(receipt_rows) - len(provided)
    missing = sorted(set(required) - set(provided))
    excess = sorted(set(provided) - set(required))
    identity_mismatches: list[tuple[str, str]] = []
    digest_mismatches: list[tuple[str, str]] = []
    for identity in sorted(set(required).intersection(provided)):
        mutation = required[identity]
        receipt = provided[identity]
        if (
            mutation.get("ticker") != receipt.get("ticker")
            or mutation.get("exchange") != receipt.get("exchange")
        ):
            identity_mismatches.append(identity)
        if mutation.get("new_canonical_row_sha256") != receipt.get(
            "canonical_row_sha256"
        ):
            digest_mismatches.append(identity)
    correction_blockers = [
        row
        for row in mutation_rows
        if row.get("mutation_type") in {"row_modified", "row_deleted"}
    ]
    failures = sorted(str(value) for value in artifact_replay_failures)
    diagnostic_rows: list[dict[str, Any]] = []
    for mutation in mutation_rows:
        identity = (
            str(mutation.get("instrument_id")),
            str(mutation.get("session_date")),
        )
        mutation_type = str(mutation.get("mutation_type"))
        reason = None
        blocker = None
        receipt_status = "not_required"
        if mutation_type in {"row_added", "row_modified"}:
            receipt_status = "missing" if identity in missing else "present"
        if identity in identity_mismatches:
            receipt_status = "identity_mismatch"
            reason = "receipt_identity_mismatch"
            blocker = "MUTATION_EVIDENCE_IDENTITY_MISMATCH"
        elif identity in digest_mismatches:
            receipt_status = "canonical_digest_mismatch"
            reason = "receipt_canonical_digest_mismatch"
            blocker = "MUTATION_EVIDENCE_DIGEST_MISMATCH"
        elif identity in missing:
            reason = "mutation_without_receipt"
            blocker = "MUTATION_EVIDENCE_MISSING"
        if mutation_type == "row_modified":
            reason = reason or "correction_contract_not_available"
            blocker = blocker or "HISTORICAL_MODIFICATION_UNAPPROVED"
        elif mutation_type == "row_deleted":
            reason = "canonical_deletion_not_supported"
            blocker = "CANONICAL_DELETION_UNSUPPORTED"
        diagnostic_rows.append(
            {
                **mutation,
                "receipt_status": receipt_status,
                "artifact_status": "replay_failed" if failures else "replayed",
                "evidence_failure_reason": reason,
                "correction_policy_status": (
                    "not_available"
                    if mutation_type in {"row_modified", "row_deleted"}
                    else "not_required"
                ),
                "publication_blocker_code": blocker,
            }
        )
    modified_instruments = {
        str(row.get("instrument_id"))
        for row in mutation_rows
        if row.get("mutation_type") == "row_modified"
    }
    valid = not any(
        (
            duplicate_mutation_count,
            duplicate_receipt_count,
            missing,
            excess,
            identity_mismatches,
            digest_mismatches,
            correction_blockers,
            failures,
        )
    )
    return {
        "schema_version": MUTATION_SCHEMA_VERSION,
        "status": "valid" if valid else "invalid",
        "affected_instrument_count": len(
            {
                str(row.get("instrument_id"))
                for row in mutation_rows
                if row.get("mutation_type") != "row_unchanged"
            }
        ),
        "added_row_count": sum(row.get("mutation_type") == "row_added" for row in mutation_rows),
        "modified_instrument_count": len(modified_instruments),
        "modified_row_count": sum(row.get("mutation_type") == "row_modified" for row in mutation_rows),
        "deleted_row_count": sum(row.get("mutation_type") == "row_deleted" for row in mutation_rows),
        "unchanged_overlap_row_count": sum(row.get("mutation_type") == "row_unchanged" for row in mutation_rows),
        "mutations_without_receipt_count": len(missing),
        "receipts_without_mutation_count": len(excess),
        "artifact_replay_failure_count": len(failures),
        "identity_mismatch_count": len(identity_mismatches),
        "canonical_digest_mismatch_count": len(digest_mismatches),
        "correction_contract_blocker_count": len(correction_blockers),
        "duplicate_mutation_count": duplicate_mutation_count,
        "duplicate_receipt_count": duplicate_receipt_count,
        "artifact_replay_failures": failures,
        "mutations_without_receipt": [list(value) for value in missing],
        "receipts_without_mutation": [list(value) for value in excess],
        "identity_mismatches": [list(value) for value in identity_mismatches],
        "canonical_digest_mismatches": [list(value) for value in digest_mismatches],
        "diagnostic_rows": diagnostic_rows,
        "mutation_root": mutation_root(mutation_rows, receipt_rows),
    }


def reconcile_mutation_evidence(
    mutations: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostics = mutation_evidence_diagnostics(mutations, receipts)
    if diagnostics["deleted_row_count"]:
        raise MutationEvidenceError("canonical row deletion is not supported")
    if diagnostics["modified_row_count"]:
        raise MutationEvidenceError(
            "canonical historical modification requires an unsupported correction contract"
        )
    if diagnostics["duplicate_mutation_count"]:
        raise MutationEvidenceError("mutation ledger contains duplicate sessions")
    if diagnostics["duplicate_receipt_count"]:
        raise MutationEvidenceError("observation receipts contain duplicate sessions")
    if diagnostics["mutations_without_receipt_count"] or diagnostics[
        "receipts_without_mutation_count"
    ]:
        raise MutationEvidenceError(
            "publisher-derived mutations do not equal replayed observations"
        )
    if diagnostics["identity_mismatch_count"] or diagnostics[
        "canonical_digest_mismatch_count"
    ]:
        raise MutationEvidenceError(
            "observation receipt does not match publisher-derived mutation"
        )
    return {
        "schema_version": MUTATION_SCHEMA_VERSION,
        "evidence_required_mutation_count": diagnostics["added_row_count"],
        "added_count": diagnostics["added_row_count"],
        "modified_count": diagnostics["modified_row_count"],
        "deleted_count": diagnostics["deleted_row_count"],
        "mutation_root": diagnostics["mutation_root"],
    }


def mutation_root(
    mutations: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> str:
    receipts_by_identity = {
        (str(row.get("instrument_id")), str(row.get("session_date"))): row
        for row in receipts
    }
    leaves = []
    for mutation in mutations:
        if mutation.get("mutation_type") not in {"row_added", "row_modified"}:
            continue
        identity = (
            str(mutation.get("instrument_id")),
            str(mutation.get("session_date")),
        )
        receipt = receipts_by_identity.get(identity, {})
        leaves.append(
            {
                "exchange": mutation.get("exchange"),
                "instrument_id": identity[0],
                "session_date": identity[1],
                "canonical_row_sha256": mutation.get(
                    "new_canonical_row_sha256"
                ),
                "receipt_sha256": receipt.get("receipt_sha256"),
            }
        )
    return hashlib.sha256(_canonical_json(sorted(leaves, key=_leaf_key))).hexdigest()


def derive_session_resolution(
    *,
    expected_sessions: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    absence_attestations: Sequence[Mapping[str, Any]],
    canonical_mutation_sessions: Sequence[str],
    fallback_exhausted_sessions: Sequence[str] = (),
    not_expected_sessions: Sequence[str] = (),
) -> dict[str, Any]:
    expected = _unique_set(expected_sessions, "expected sessions")
    not_expected = _unique_set(not_expected_sessions, "not-expected sessions")
    if expected.intersection(not_expected):
        raise MutationEvidenceError(
            "session cannot be expected and not expected"
        )
    exhausted = _unique_set(fallback_exhausted_sessions, "fallback exhaustion")
    if not exhausted.issubset(expected):
        raise MutationEvidenceError(
            "fallback exhaustion is outside expected sessions"
        )
    mutation_sessions = _unique_set(
        canonical_mutation_sessions, "canonical mutation sessions"
    )
    observed: dict[str, str] = {}
    for receipt in receipts:
        session = str(receipt.get("session_date"))
        route = str(receipt.get("acquisition_route"))
        state = (
            "observed_primary"
            if route in {"primary", "primary_replay"}
            else "observed_fallback"
            if route == "fallback"
            else None
        )
        if state is None or session in observed:
            raise MutationEvidenceError("observed session route is invalid or duplicate")
        observed[session] = state
    absent = _unique_set(
        [str(row.get("session_date")) for row in absence_attestations],
        "absence attestations",
    )
    if set(observed).intersection(absent):
        raise MutationEvidenceError(
            "session cannot be observed and explained absent"
        )
    if not set(observed).issubset(expected) or not absent.issubset(expected):
        raise MutationEvidenceError("session evidence is outside expected sessions")
    if not mutation_sessions.issubset(expected):
        raise MutationEvidenceError("canonical mutation exists in not-expected session")
    if mutation_sessions != set(observed):
        raise MutationEvidenceError(
            "canonical mutation sessions do not equal observed receipt sessions"
        )
    unresolved = expected - set(observed) - absent
    partition = [
        {
            "session_date": session,
            "state": (
                "not_expected"
                if session in not_expected
                else observed.get(session)
                or ("explained_absent" if session in absent else "unresolved")
            ),
        }
        for session in sorted(expected.union(not_expected))
    ]
    if {row["state"] for row in partition} - SESSION_STATES:
        raise MutationEvidenceError("session partition contains unknown state")
    return {
        "schema_version": SESSION_LEDGER_SCHEMA_VERSION,
        "partition": partition,
        "observed_sessions": sorted(observed),
        "explained_absence_sessions": sorted(absent),
        "unresolved_sessions": sorted(unresolved),
        "fallback_candidates": sorted(unresolved - exhausted),
        "not_expected_sessions": sorted(not_expected),
    }


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if set(reader.fieldnames or ()) != required:
                raise MutationEvidenceError("canonical CSV columns are invalid")
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                session = str(row["Date"])
                if session in rows:
                    raise MutationEvidenceError("canonical CSV contains duplicate sessions")
                rows[session] = {key: str(value) for key, value in row.items()}
            return rows
    except (OSError, csv.Error, TypeError) as exc:
        raise MutationEvidenceError("canonical CSV cannot be read") from exc


def _row_digest(
    row: Mapping[str, str], *, instrument_id: str, currency: str
) -> str:
    return canonical_row_sha256(
        instrument_id=instrument_id,
        session_date=str(row["Date"]),
        open_value=row["Open"],
        high=row["High"],
        low=row["Low"],
        close=row["Close"],
        adj_close=row["Adj Close"],
        volume=row["Volume"],
        currency=currency,
    )


def _diagnostic_values(row: Mapping[str, str] | None) -> dict[str, str] | None:
    if row is None:
        return None
    return {
        key: str(row[key])
        for key in ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume")
    }


def _unique_set(values: Sequence[str], label: str) -> set[str]:
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise MutationEvidenceError(f"{label} contain duplicates")
    try:
        for value in normalized:
            date.fromisoformat(value)
    except ValueError as exc:
        raise MutationEvidenceError(f"{label} contain invalid dates") from exc
    return set(normalized)


def _leaf_key(value: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(
        str(value.get(key))
        for key in (
            "exchange",
            "instrument_id",
            "session_date",
            "canonical_row_sha256",
            "receipt_sha256",
        )
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
