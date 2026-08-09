from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


SCHEMA_VERSION = "market-engine-verified-daily-ohlcv-evidence-v1"
DEFAULT_REGISTRY = Path(
    "config/market_engine/universes/verified_daily_ohlcv_evidence.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VerifiedPriceObservationError(ValueError):
    pass


def canonical_checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def evidence_checksum(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("evidence_checksum", None)
    return canonical_checksum(payload)


def load_verified_price_observations(
    path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedPriceObservationError(
            f"verified price-observation evidence is missing or malformed: {source}"
        ) from exc
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(
        payload.get("records"), list
    ):
        raise VerifiedPriceObservationError(
            "verified price-observation evidence schema is unsupported"
        )
    records = [_validate_record(value, index=index) for index, value in enumerate(payload["records"])]
    records.sort(key=lambda row: (row["instrument_id"], row["session_date"]))
    if len({(row["instrument_id"], row["session_date"]) for row in records}) != len(records):
        raise VerifiedPriceObservationError(
            "verified price-observation evidence contains duplicate sessions"
        )
    normalized = {"schema_version": SCHEMA_VERSION, "records": records}
    return {
        **normalized,
        "registry_checksum": canonical_checksum(normalized),
        "records_by_instrument_id": {
            instrument_id: [
                row for row in records if row["instrument_id"] == instrument_id
            ]
            for instrument_id in sorted({row["instrument_id"] for row in records})
        },
    }


def _validate_record(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedPriceObservationError(f"observation record {index} must be an object")
    record = dict(value)
    required_text = {}
    for key in (
        "instrument_id",
        "ticker",
        "session_date",
        "source_identity",
        "source_url",
        "retrieved_at",
        "daily_ohlcv_validation_status",
    ):
        raw = record.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise VerifiedPriceObservationError(
                f"observation record {index} requires {key}"
            )
        required_text[key] = raw.strip()
    try:
        session = date.fromisoformat(required_text["session_date"])
        retrieved = datetime.fromisoformat(
            required_text["retrieved_at"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise VerifiedPriceObservationError(
            f"observation record {index} has an invalid date"
        ) from exc
    if retrieved.tzinfo is None:
        raise VerifiedPriceObservationError(
            f"observation record {index} retrieval timestamp needs a timezone"
        )
    parsed_url = urlsplit(required_text["source_url"])
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise VerifiedPriceObservationError(
            f"observation record {index} source URL must use HTTPS"
        )
    numbers: dict[str, float] = {}
    for key in ("open", "high", "low", "close", "adj_close", "volume"):
        raw = record.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise VerifiedPriceObservationError(
                f"observation record {index} requires numeric {key}"
            )
        numbers[key] = int(raw) if key == "volume" else float(raw)
        if not math.isfinite(numbers[key]):
            raise VerifiedPriceObservationError(
                f"observation record {index} has non-finite {key}"
            )
    if (
        numbers["high"] < max(numbers["open"], numbers["close"])
        or numbers["low"] > min(numbers["open"], numbers["close"])
        or numbers["volume"] < 0
        or required_text["daily_ohlcv_validation_status"]
        != "complete_daily_ohlcv_observation"
    ):
        raise VerifiedPriceObservationError(
            f"observation record {index} fails daily OHLCV validation"
        )
    normalized = {
        "instrument_id": required_text["instrument_id"],
        "ticker": required_text["ticker"].upper(),
        "session_date": session.isoformat(),
        **numbers,
        "source_identity": required_text["source_identity"],
        "source_url": required_text["source_url"],
        "retrieved_at": retrieved.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "daily_ohlcv_validation_status": required_text[
            "daily_ohlcv_validation_status"
        ],
        "evidence_checksum": record.get("evidence_checksum"),
    }
    if not isinstance(normalized["evidence_checksum"], str) or not SHA256.fullmatch(
        normalized["evidence_checksum"]
    ) or normalized["evidence_checksum"] != evidence_checksum(normalized):
        raise VerifiedPriceObservationError(
            f"observation record {index} evidence checksum is invalid"
        )
    return normalized
