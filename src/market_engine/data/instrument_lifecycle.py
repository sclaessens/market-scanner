from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


LIFECYCLE_SCHEMA_VERSION = "market-engine-instrument-lifecycle-registry-v2"
DEFAULT_LIFECYCLE_REGISTRY = Path(
    "config/market_engine/universes/instrument_lifecycle.json"
)
LIFECYCLE_STATUSES = frozenset({"active", "inactive"})
EFFECTIVE_LIFECYCLE_STATUSES = frozenset({"active", "inactive", "pending"})
SOURCE_AUTHORITIES = frozenset({"sec", "issuer", "acquirer", "exchange"})
LIFECYCLE_REASONS = frozenset(
    {
        "active_recent_listing",
        "inactive_after_completed_corporate_action",
    }
)
CORPORATE_ACTION_TYPES = frozenset(
    {
        "cash_acquisition",
        "cash_and_stock_acquisition",
        "spin_off_listing",
        "take_private_acquisition",
    }
)
SOURCE_TYPES = frozenset(
    {
        "completion_release",
        "distribution_timing_release",
        "exchange_notice",
        "form_8_k",
    }
)
SUPPORTED_EXCHANGES = frozenset({"NASDAQ", "NYSE"})
EVIDENCE_SUPPORT_TYPES = frozenset(
    {
        "corporate_action_completion",
        "listing_completion",
        "listing_schedule",
        "trading_termination",
    }
)
SOURCE_TYPE_SUPPORT = {
    "completion_release": frozenset(
        {"corporate_action_completion", "listing_completion", "trading_termination"}
    ),
    "distribution_timing_release": frozenset({"listing_schedule"}),
    "exchange_notice": frozenset({"listing_schedule", "trading_termination"}),
    "form_8_k": EVIDENCE_SUPPORT_TYPES,
}
AUTHORITY_SOURCE_TYPES = {
    "sec": frozenset({"form_8_k"}),
    "issuer": frozenset({"completion_release", "distribution_timing_release"}),
    "acquirer": frozenset({"completion_release"}),
    "exchange": frozenset({"exchange_notice"}),
}
EXCHANGE_EVIDENCE_HOSTS = {
    "NASDAQ": frozenset({"nasdaq.com", "www.nasdaq.com"}),
    "NYSE": frozenset({"nyse.com", "www.nyse.com"}),
}
SEC_EVIDENCE_HOSTS = frozenset({"sec.gov", "www.sec.gov"})
EVIDENCE_AUTHORITY_ORDER = {
    "sec": 0,
    "exchange": 1,
    "issuer": 2,
    "acquirer": 3,
}


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
    if lifecycle_reason not in LIFECYCLE_REASONS:
        raise InstrumentLifecycleError(
            f"unknown lifecycle reason for {instrument_id}: {lifecycle_reason}"
        )
    corporate_action_type = _required_text(record, "corporate_action_type")
    if corporate_action_type not in CORPORATE_ACTION_TYPES:
        raise InstrumentLifecycleError(
            "unknown corporate action type for "
            f"{instrument_id}: {corporate_action_type}"
        )
    if exchange not in SUPPORTED_EXCHANGES:
        raise InstrumentLifecycleError(
            f"unsupported lifecycle exchange for {instrument_id}: {exchange}"
        )
    if (
        lifecycle_status == "active"
        and (
            lifecycle_reason != "active_recent_listing"
            or corporate_action_type != "spin_off_listing"
        )
    ) or (
        lifecycle_status == "inactive"
        and (
            lifecycle_reason != "inactive_after_completed_corporate_action"
            or corporate_action_type == "spin_off_listing"
        )
    ):
        raise InstrumentLifecycleError(
            f"lifecycle transition semantics are contradictory for {instrument_id}"
        )
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

    official_source_hosts = _validate_official_source_hosts(
        record.get("official_source_hosts"),
        instrument_id=instrument_id,
    )
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise InstrumentLifecycleError(
            f"lifecycle evidence is required for {instrument_id}"
        )
    normalized_evidence = [
        _validate_evidence(
            entry,
            instrument_id=instrument_id,
            ticker=ticker,
            exchange=exchange,
            official_source_hosts=official_source_hosts,
            index=evidence_index,
        )
        for evidence_index, entry in enumerate(evidence)
    ]
    normalized_evidence.sort(
        key=lambda row: (
            row["source_publication_date"],
            EVIDENCE_AUTHORITY_ORDER[row["source_authority"]],
            row["source_url"],
        )
    )
    governed_host_authorities = {
        entry["source_authority"]
        for entry in normalized_evidence
        if entry["source_authority"] in {"issuer", "acquirer"}
    }
    if set(official_source_hosts) != governed_host_authorities:
        raise InstrumentLifecycleError(
            f"official source host bindings are incomplete or unused for {instrument_id}"
        )
    if any(
        entry["source_authority"] == "acquirer"
        for entry in normalized_evidence
    ) and successor_or_acquirer is None:
        raise InstrumentLifecycleError(
            f"acquirer evidence is not identity-bound for {instrument_id}"
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
        if successor_or_acquirer is not None:
            raise InstrumentLifecycleError(
                f"active listing cannot declare an acquirer for {instrument_id}"
            )
        _validate_active_evidence(
            normalized_evidence,
            instrument_id=instrument_id,
            regular_way_listing_date=regular_way_listing_date,
        )
    else:
        if successor_or_acquirer is None:
            raise InstrumentLifecycleError(
                f"inactive acquisition requires an acquirer for {instrument_id}"
            )
        if delisting_end_date is None:
            raise InstrumentLifecycleError(
                f"inactive lifecycle record requires a delisting end date for {instrument_id}"
            )
        if status_effective_date <= delisting_end_date:
            raise InstrumentLifecycleError(
                f"inactive effective date must follow the final trading date for {instrument_id}"
            )
        _validate_inactive_evidence(
            normalized_evidence,
            instrument_id=instrument_id,
            delisting_end_date=delisting_end_date,
            status_effective_date=status_effective_date,
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
        "official_source_hosts": official_source_hosts,
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
    ticker: str,
    exchange: str,
    official_source_hosts: Mapping[str, Sequence[str]],
    index: int,
) -> dict[str, Any]:
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
    if source_type not in SOURCE_TYPES:
        raise InstrumentLifecycleError(
            f"unsupported evidence source type for {instrument_id}: {source_type}"
        )
    if source_type not in AUTHORITY_SOURCE_TYPES[authority]:
        raise InstrumentLifecycleError(
            f"evidence authority/source type mismatch for {instrument_id}"
        )
    source_url = _required_text(value, "source_url")
    parsed_url = urlsplit(source_url)
    try:
        source_port = parsed_url.port
    except ValueError as exc:
        raise InstrumentLifecycleError(
            f"lifecycle evidence URL is invalid for {instrument_id}"
        ) from exc
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or source_port not in {None, 443}
    ):
        raise InstrumentLifecycleError(
            f"lifecycle evidence URL must use HTTPS for {instrument_id}"
        )
    source_host = parsed_url.hostname.lower().rstrip(".")
    if authority == "sec":
        allowed_hosts = SEC_EVIDENCE_HOSTS
    elif authority == "exchange":
        allowed_hosts = EXCHANGE_EVIDENCE_HOSTS[exchange]
    else:
        allowed_hosts = frozenset(official_source_hosts.get(authority, ()))
    if source_host not in allowed_hosts:
        raise InstrumentLifecycleError(
            f"lifecycle evidence host is not authorized for {instrument_id}"
        )
    subject_instrument_id = _required_text(value, "subject_instrument_id")
    subject_ticker = _required_text(value, "subject_ticker").upper()
    subject_exchange = _required_text(value, "subject_exchange").upper()
    if (
        subject_instrument_id != instrument_id
        or subject_ticker != ticker
        or subject_exchange != exchange
    ):
        raise InstrumentLifecycleError(
            f"lifecycle evidence identity mismatch for {instrument_id}"
        )
    supports = value.get("transition_support")
    if (
        not isinstance(supports, list)
        or not supports
        or any(not isinstance(item, str) for item in supports)
    ):
        raise InstrumentLifecycleError(
            f"lifecycle evidence transition support is required for {instrument_id}"
        )
    normalized_supports = sorted(set(supports))
    if (
        len(normalized_supports) != len(supports)
        or any(item not in EVIDENCE_SUPPORT_TYPES for item in normalized_supports)
        or any(
            item not in SOURCE_TYPE_SUPPORT[source_type]
            for item in normalized_supports
        )
    ):
        raise InstrumentLifecycleError(
            f"lifecycle evidence transition support is invalid for {instrument_id}"
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
        "source_host": source_host,
        "subject_instrument_id": subject_instrument_id,
        "subject_ticker": subject_ticker,
        "subject_exchange": subject_exchange,
        "transition_support": normalized_supports,
        "source_publication_date": publication_date.isoformat(),
        "evidence_retrieved_at": _utc_text(retrieved_at),
    }


def _validate_official_source_hosts(
    value: Any,
    *,
    instrument_id: str,
) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise InstrumentLifecycleError(
            f"official source hosts must be an object for {instrument_id}"
        )
    normalized: dict[str, list[str]] = {}
    for authority, hosts in value.items():
        if authority not in {"issuer", "acquirer"}:
            raise InstrumentLifecycleError(
                f"official source host authority is invalid for {instrument_id}"
            )
        if (
            not isinstance(hosts, list)
            or not hosts
            or any(not isinstance(host, str) for host in hosts)
        ):
            raise InstrumentLifecycleError(
                f"official source hosts are invalid for {instrument_id}"
            )
        normalized_hosts = sorted(
            {host.lower().rstrip(".") for host in hosts}
        )
        if (
            len(normalized_hosts) != len(hosts)
            or any(
                not host
                or "://" in host
                or "/" in host
                or ":" in host
                or "." not in host
                for host in normalized_hosts
            )
        ):
            raise InstrumentLifecycleError(
                f"official source hosts are invalid for {instrument_id}"
            )
        normalized[authority] = normalized_hosts
    return dict(sorted(normalized.items()))


def _validate_active_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    instrument_id: str,
    regular_way_listing_date: date,
) -> None:
    support = {
        item
        for entry in evidence
        for item in entry["transition_support"]
    }
    if "listing_schedule" not in support:
        raise InstrumentLifecycleError(
            f"listing schedule evidence is required for {instrument_id}"
        )
    latest_retrieval = max(
        datetime.fromisoformat(
            str(entry["evidence_retrieved_at"]).replace("Z", "+00:00")
        ).date()
        for entry in evidence
    )
    if (
        latest_retrieval >= regular_way_listing_date
        and "listing_completion" not in support
    ):
        raise InstrumentLifecycleError(
            f"listing completion evidence is required for {instrument_id}"
        )
    for entry in evidence:
        publication = date.fromisoformat(entry["source_publication_date"])
        entry_support = set(entry["transition_support"])
        if (
            "listing_schedule" in entry_support
            and publication > regular_way_listing_date
        ):
            raise InstrumentLifecycleError(
                f"listing schedule evidence is too late for {instrument_id}"
            )
        if (
            "listing_completion" in entry_support
            and publication < regular_way_listing_date
        ):
            raise InstrumentLifecycleError(
                "LISTING_COMPLETION_BEFORE_REGULAR_WAY: "
                "listing completion evidence predates regular-way listing for "
                f"{instrument_id}"
            )


def _validate_inactive_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    instrument_id: str,
    delisting_end_date: date,
    status_effective_date: date,
) -> None:
    support = {
        item
        for entry in evidence
        for item in entry["transition_support"]
    }
    required = {"corporate_action_completion", "trading_termination"}
    if not required.issubset(support):
        raise InstrumentLifecycleError(
            f"completion and trading termination evidence are required for {instrument_id}"
        )
    for entry in evidence:
        relevant = required.intersection(entry["transition_support"])
        if not relevant:
            continue
        publication = date.fromisoformat(entry["source_publication_date"])
        if not delisting_end_date <= publication <= status_effective_date:
            raise InstrumentLifecycleError(
                f"inactive transition evidence date is invalid for {instrument_id}"
            )


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
        regular_way_date = date.fromisoformat(
            record["regular_way_listing_date"]
        )
        evidence_support = {
            support
            for entry in record["evidence"]
            for support in entry["transition_support"]
        }
        if (
            as_of >= regular_way_date
            and "listing_completion" not in evidence_support
        ):
            raise InstrumentLifecycleError(
                "listing completion evidence is required before active "
                f"projection for {record['instrument_id']}"
            )
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
