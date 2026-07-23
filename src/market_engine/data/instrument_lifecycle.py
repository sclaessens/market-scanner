from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


LIFECYCLE_SCHEMA_VERSION = "market-engine-instrument-lifecycle-registry-v1"
DEFAULT_LIFECYCLE_REGISTRY = Path(
    "config/market_engine/universes/instrument_lifecycle.json"
)
LIFECYCLE_STATUSES = frozenset({"active", "inactive"})
EFFECTIVE_LIFECYCLE_STATUSES = frozenset({"active", "inactive", "pending"})
SOURCE_AUTHORITIES = frozenset({"sec", "issuer", "acquirer", "exchange"})


class InstrumentLifecycleError(ValueError):
    pass


def load_lifecycle_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InstrumentLifecycleError(
            f"lifecycle registry is missing or malformed: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise InstrumentLifecycleError("lifecycle registry must be a JSON object")
    if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise InstrumentLifecycleError("lifecycle registry schema is unsupported")
    records = payload.get("records")
    if not isinstance(records, list):
        raise InstrumentLifecycleError("lifecycle registry records must be a list")

    normalized: list[dict[str, Any]] = []
    seen_instrument_ids: set[str] = set()
    for index, value in enumerate(records):
        if not isinstance(value, Mapping):
            raise InstrumentLifecycleError(
                f"lifecycle record {index} must be an object"
            )
        record = _validate_record(value, index=index)
        instrument_id = record["instrument_id"]
        if instrument_id in seen_instrument_ids:
            raise InstrumentLifecycleError(
                f"duplicate lifecycle instrument ID: {instrument_id}"
            )
        seen_instrument_ids.add(instrument_id)
        normalized.append(record)

    normalized.sort(key=lambda row: (row["instrument_id"], row["ticker"]))
    canonical_payload = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "records": normalized,
    }
    return {
        **canonical_payload,
        "registry_checksum": canonical_checksum(canonical_payload),
        "records_by_instrument_id": {
            row["instrument_id"]: row for row in normalized
        },
    }


def apply_lifecycle_registry(
    instruments: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    *,
    as_of: date,
) -> dict[str, Any]:
    records_by_id = registry.get("records_by_instrument_id")
    if not isinstance(records_by_id, Mapping):
        raise InstrumentLifecycleError("normalized lifecycle registry is required")

    enriched: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for raw in instruments:
        instrument_id = _required_text(raw, "instrument_id")
        ticker = _required_text(raw, "symbol")
        known_ids.add(instrument_id)
        record = records_by_id.get(instrument_id)
        if record is not None:
            if record["ticker"] != ticker:
                raise InstrumentLifecycleError(
                    f"lifecycle ticker mismatch for {instrument_id}"
                )
            base_exchange = str(raw.get("exchange") or "").upper()
            if (
                base_exchange not in {"", "UNKNOWN"}
                and _normalized_exchange(base_exchange)
                != _normalized_exchange(record["exchange"])
            ):
                raise InstrumentLifecycleError(
                    f"lifecycle exchange mismatch for {instrument_id}"
                )
        enriched.append(_apply_record(raw, record=record, as_of=as_of))

    unknown_ids = sorted(set(records_by_id) - known_ids)
    if unknown_ids:
        raise InstrumentLifecycleError(
            "lifecycle registry references unknown instruments: "
            + ", ".join(unknown_ids)
        )

    enriched.sort(key=lambda row: (row["instrument_id"], row["symbol"]))
    active = [row for row in enriched if row["lifecycle_status"] == "active"]
    inactive = [row for row in enriched if row["lifecycle_status"] == "inactive"]
    pending = [row for row in enriched if row["lifecycle_status"] == "pending"]
    active_binding = [
        _instrument_binding(row)
        for row in active
    ]
    full_binding = [
        _instrument_binding(row)
        for row in enriched
    ]
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "as_of_date": as_of.isoformat(),
        "registry_checksum": registry["registry_checksum"],
        "active_universe_checksum": canonical_checksum(active_binding),
        "governed_universe_checksum": canonical_checksum(full_binding),
        "active_universe_size": len(active),
        "inactive_retained_instrument_count": len(inactive),
        "pending_instrument_count": len(pending),
        "instruments": enriched,
        "active_instruments": active,
        "inactive_instruments": inactive,
        "pending_instruments": pending,
    }


def canonical_checksum(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def record_provenance_checksum(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("provenance_checksum", None)
    return canonical_checksum(payload)


def _validate_record(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    record = dict(value)
    instrument_id = _required_text(record, "instrument_id")
    ticker = _required_text(record, "ticker").upper()
    issuer_name = _required_text(record, "issuer_name")
    exchange = _required_text(record, "exchange").upper()
    lifecycle_status = _required_text(record, "lifecycle_status")
    if lifecycle_status not in LIFECYCLE_STATUSES:
        raise InstrumentLifecycleError(
            f"unknown lifecycle status for {instrument_id}: {lifecycle_status}"
        )
    status_effective_date = _required_date(record, "status_effective_date")
    listing_start_date = _optional_date(record, "listing_start_date")
    regular_way_listing_date = _optional_date(
        record, "regular_way_listing_date"
    )
    delisting_end_date = _optional_date(record, "delisting_end_date")
    lifecycle_reason = _required_text(record, "lifecycle_reason")
    corporate_action_type = _required_text(record, "corporate_action_type")
    successor_or_acquirer = record.get("successor_or_acquirer")
    if successor_or_acquirer is not None:
        if not isinstance(successor_or_acquirer, Mapping):
            raise InstrumentLifecycleError(
                f"successor/acquirer must be an object for {instrument_id}"
            )
        successor_or_acquirer = {
            "name": _required_text(successor_or_acquirer, "name"),
            "ticker": (
                str(successor_or_acquirer["ticker"]).upper()
                if successor_or_acquirer.get("ticker")
                else None
            ),
        }

    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise InstrumentLifecycleError(
            f"lifecycle evidence is required for {instrument_id}"
        )
    normalized_evidence = [
        _validate_evidence(entry, instrument_id=instrument_id, index=evidence_index)
        for evidence_index, entry in enumerate(evidence)
    ]
    normalized_evidence.sort(
        key=lambda row: (
            row["source_publication_date"],
            row["source_url"],
        )
    )

    if lifecycle_status == "active":
        if listing_start_date is None or regular_way_listing_date is None:
            raise InstrumentLifecycleError(
                f"active governed listing dates are required for {instrument_id}"
            )
        if status_effective_date != listing_start_date:
            raise InstrumentLifecycleError(
                f"active status effective date must equal listing start for {instrument_id}"
            )
        if listing_start_date > regular_way_listing_date:
            raise InstrumentLifecycleError(
                f"listing dates are contradictory for {instrument_id}"
            )
        if delisting_end_date is not None:
            raise InstrumentLifecycleError(
                f"active lifecycle record cannot have a delisting date for {instrument_id}"
            )
    else:
        if delisting_end_date is None:
            raise InstrumentLifecycleError(
                f"inactive lifecycle record requires a delisting end date for {instrument_id}"
            )
        if status_effective_date <= delisting_end_date:
            raise InstrumentLifecycleError(
                f"inactive effective date must follow the final trading date for {instrument_id}"
            )

    normalized = {
        "instrument_id": instrument_id,
        "ticker": ticker,
        "issuer_name": issuer_name,
        "exchange": exchange,
        "lifecycle_status": lifecycle_status,
        "status_effective_date": status_effective_date.isoformat(),
        "listing_start_date": (
            listing_start_date.isoformat() if listing_start_date else None
        ),
        "regular_way_listing_date": (
            regular_way_listing_date.isoformat()
            if regular_way_listing_date
            else None
        ),
        "delisting_end_date": (
            delisting_end_date.isoformat() if delisting_end_date else None
        ),
        "lifecycle_reason": lifecycle_reason,
        "corporate_action_type": corporate_action_type,
        "successor_or_acquirer": successor_or_acquirer,
        "evidence": normalized_evidence,
        "provenance_checksum": record.get("provenance_checksum"),
    }
    checksum = normalized.get("provenance_checksum")
    if not isinstance(checksum, str) or checksum != record_provenance_checksum(
        normalized
    ):
        raise InstrumentLifecycleError(
            f"lifecycle provenance checksum mismatch for {instrument_id}"
        )
    return normalized


def _validate_evidence(
    value: Any,
    *,
    instrument_id: str,
    index: int,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise InstrumentLifecycleError(
            f"lifecycle evidence {index} must be an object for {instrument_id}"
        )
    authority = _required_text(value, "source_authority")
    if authority not in SOURCE_AUTHORITIES:
        raise InstrumentLifecycleError(
            f"unsupported evidence authority for {instrument_id}: {authority}"
        )
    source_type = _required_text(value, "source_type")
    source_url = _required_text(value, "source_url")
    if not source_url.startswith("https://"):
        raise InstrumentLifecycleError(
            f"lifecycle evidence URL must use HTTPS for {instrument_id}"
        )
    publication_date = _required_date(value, "source_publication_date")
    retrieved_at = _required_timestamp(value, "evidence_retrieved_at")
    if publication_date > retrieved_at.date():
        raise InstrumentLifecycleError(
            f"lifecycle evidence publication date is in the future for {instrument_id}"
        )
    return {
        "source_authority": authority,
        "source_type": source_type,
        "source_url": source_url,
        "source_publication_date": publication_date.isoformat(),
        "evidence_retrieved_at": _utc_text(retrieved_at),
    }


def _apply_record(
    instrument: Mapping[str, Any],
    *,
    record: Mapping[str, Any] | None,
    as_of: date,
) -> dict[str, Any]:
    base = dict(instrument)
    base_active = bool(base.get("active", True))
    if record is None:
        lifecycle_status = "active" if base_active else "inactive"
        return {
            **base,
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "lifecycle_status": lifecycle_status,
            "lifecycle_record_status": lifecycle_status,
            "lifecycle_status_effective_date": None,
            "listing_start_date": None,
            "regular_way_listing_date": None,
            "delisting_end_date": None,
            "lifecycle_reason": "canonical_universe_active"
            if lifecycle_status == "active"
            else "canonical_universe_inactive",
            "corporate_action_type": "none",
            "successor_or_acquirer": None,
            "lifecycle_provenance_checksum": None,
            "lifecycle_evidence": [],
            "active": lifecycle_status == "active",
            "analysis_eligible": bool(base.get("analysis_eligible", True))
            and lifecycle_status == "active",
        }

    effective_date = date.fromisoformat(record["status_effective_date"])
    record_status = record["lifecycle_status"]
    if record_status == "active":
        current_status = "active" if as_of >= effective_date else "pending"
        current_reason = (
            record["lifecycle_reason"]
            if current_status == "active"
            else "pre_listing_not_expected"
        )
    else:
        current_status = "inactive" if as_of >= effective_date else "active"
        current_reason = (
            record["lifecycle_reason"]
            if current_status == "inactive"
            else "future_inactive_event_not_yet_effective"
        )

    return {
        **base,
        "exchange": record["exchange"],
        "name": record["issuer_name"],
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_status": current_status,
        "lifecycle_record_status": record_status,
        "lifecycle_status_effective_date": record["status_effective_date"],
        "listing_start_date": record["listing_start_date"],
        "regular_way_listing_date": record["regular_way_listing_date"],
        "delisting_end_date": record["delisting_end_date"],
        "lifecycle_reason": current_reason,
        "corporate_action_type": record["corporate_action_type"],
        "successor_or_acquirer": record["successor_or_acquirer"],
        "lifecycle_provenance_checksum": record["provenance_checksum"],
        "lifecycle_evidence": record["evidence"],
        "active": current_status == "active",
        "analysis_eligible": bool(base.get("analysis_eligible", True))
        and current_status == "active",
    }


def _instrument_binding(instrument: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instrument_id": instrument["instrument_id"],
        "ticker": instrument["symbol"],
        "source_symbol": instrument["source_symbol"],
        "exchange": instrument.get("exchange"),
        "lifecycle_status": instrument["lifecycle_status"],
        "lifecycle_record_status": instrument["lifecycle_record_status"],
        "lifecycle_status_effective_date": instrument[
            "lifecycle_status_effective_date"
        ],
        "listing_start_date": instrument["listing_start_date"],
        "regular_way_listing_date": instrument["regular_way_listing_date"],
        "delisting_end_date": instrument["delisting_end_date"],
        "lifecycle_provenance_checksum": instrument[
            "lifecycle_provenance_checksum"
        ],
    }


def _required_text(value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise InstrumentLifecycleError(f"required lifecycle field is missing: {key}")
    return raw.strip()


def _required_date(value: Mapping[str, Any], key: str) -> date:
    raw = _required_text(value, key)
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise InstrumentLifecycleError(
            f"invalid lifecycle date for {key}: {raw}"
        ) from exc


def _optional_date(value: Mapping[str, Any], key: str) -> date | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise InstrumentLifecycleError(f"lifecycle date must be text: {key}")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise InstrumentLifecycleError(
            f"invalid lifecycle date for {key}: {raw}"
        ) from exc


def _required_timestamp(value: Mapping[str, Any], key: str) -> datetime:
    raw = _required_text(value, key)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InstrumentLifecycleError(
            f"invalid lifecycle timestamp for {key}: {raw}"
        ) from exc
    if parsed.tzinfo is None:
        raise InstrumentLifecycleError(
            f"lifecycle timestamp must include a timezone: {key}"
        )
    return parsed.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _normalized_exchange(value: str) -> str:
    aliases = {
        "NASDAQ": "XNAS",
        "NYSE": "XNYS",
        "XNAS": "XNAS",
        "XNYS": "XNYS",
    }
    return aliases.get(value.upper(), value.upper())


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")
