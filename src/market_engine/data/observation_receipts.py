from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = "market-engine-market-price-source-policy-v1"
RECEIPT_SCHEMA_VERSION = "market-engine-observation-receipt-v1"
PARSER_NAME = "canonical-json-daily-ohlcv"
PARSER_VERSION = "v1"
SERIALIZATION_VERSION = "market-engine-canonical-row-v1"
DEFAULT_SOURCE_POLICY = Path(
    "config/market_engine/source_policies/market_price_sources.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ObservationReceiptError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_source_policy(path: str | Path = DEFAULT_SOURCE_POLICY) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("market-price source policy is missing or malformed") from exc
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION or not isinstance(
        payload.get("providers"), list
    ):
        raise ObservationReceiptError("market-price source policy schema is unsupported")
    providers: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["providers"]):
        if not isinstance(raw, Mapping):
            raise ObservationReceiptError(f"source policy provider {index} must be an object")
        provider = dict(raw)
        required = (
            "provider_id",
            "approval_id",
            "data_type",
            "retention_classification",
            "redistribution_classification",
        )
        if any(not isinstance(provider.get(key), str) or not provider[key] for key in required):
            raise ObservationReceiptError(f"source policy provider {index} is incomplete")
        if not PROVIDER_ID.fullmatch(provider["provider_id"]):
            raise ObservationReceiptError(
                f"source policy provider {index} has invalid provider ID"
            )
        if provider["data_type"] != "daily_ohlcv":
            raise ObservationReceiptError(
                f"source policy provider {index} has unsupported data type"
            )
        for key in (
            "approved_for_acquisition",
            "approved_for_raw_storage",
            "approved_for_canonical_publication",
        ):
            if not isinstance(provider.get(key), bool):
                raise ObservationReceiptError(
                    f"source policy provider {index} requires boolean {key}"
                )
        exchanges = provider.get("exchanges")
        if not isinstance(exchanges, list) or not exchanges or any(
            not isinstance(value, str) or not value for value in exchanges
        ):
            raise ObservationReceiptError(f"source policy provider {index} requires exchanges")
        providers.append({**provider, "exchanges": sorted(set(exchanges))})
    if len({row["provider_id"] for row in providers}) != len(providers):
        raise ObservationReceiptError("market-price source policy contains duplicate providers")
    providers.sort(key=lambda row: row["provider_id"])
    normalized = {"schema_version": POLICY_SCHEMA_VERSION, "providers": providers}
    return {
        **normalized,
        "policy_checksum": sha256_bytes(_canonical_json(normalized)),
        "providers_by_id": {row["provider_id"]: row for row in providers},
    }


def approved_fallback_policy(
    policy: Mapping[str, Any],
    *,
    provider_id: str,
    exchange: str,
) -> dict[str, Any]:
    provider = (policy.get("providers_by_id") or {}).get(provider_id)
    if not isinstance(provider, Mapping):
        raise ObservationReceiptError(f"unknown market-price fallback provider: {provider_id}")
    if not all(
        provider.get(key) is True
        for key in (
            "approved_for_acquisition",
            "approved_for_raw_storage",
            "approved_for_canonical_publication",
        )
    ):
        raise ObservationReceiptError(
            "market-price fallback provider is not approved for canonical "
            f"publication: {provider_id}"
        )
    if exchange not in provider.get("exchanges", []):
        raise ObservationReceiptError(
            f"market-price fallback provider is not approved for exchange: {provider_id}/{exchange}"
        )
    if not provider.get("approval_id"):
        raise ObservationReceiptError(
            f"market-price fallback approval ID is missing: {provider_id}"
        )
    return dict(provider)


def preserve_raw_artifact(
    payload: bytes,
    *,
    artifact_root: str | Path,
    provider_id: str,
    content_type: str,
) -> dict[str, str]:
    if not PROVIDER_ID.fullmatch(provider_id):
        raise ObservationReceiptError("raw market-price artifact provider ID is invalid")
    if content_type != "application/json":
        raise ObservationReceiptError("raw market-price artifact content type is unsupported")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("raw market-price artifact is malformed") from exc
    if _contains_secret_material(decoded):
        raise ObservationReceiptError("raw market-price artifact contains credential material")
    digest = sha256_bytes(payload)
    root = Path(artifact_root)
    relative = Path("evidence") / "market_price" / provider_id / f"{digest}.json"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise ObservationReceiptError("raw market-price artifact checksum collision")
    target.write_bytes(payload)
    return {
        "raw_artifact_locator": relative.as_posix(),
        "raw_artifact_sha256": digest,
        "content_type": content_type,
    }


def parse_raw_daily_ohlcv(payload: bytes) -> list[dict[str, str | int]]:
    try:
        raw = json.loads(payload.decode("utf-8"), parse_float=str, parse_int=str)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("raw market-price artifact is malformed") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("bars"), list):
        raise ObservationReceiptError("raw market-price artifact has unsupported schema")
    rows: list[dict[str, str | int]] = []
    for index, value in enumerate(raw["bars"]):
        if not isinstance(value, Mapping):
            raise ObservationReceiptError(f"raw market-price bar {index} must be an object")
        try:
            session = date.fromisoformat(str(value["session_date"]))
            prices = {
                key: _decimal_text(value[key])
                for key in ("open", "high", "low", "close", "adj_close")
            }
            volume_decimal = Decimal(str(value["volume"]))
        except (KeyError, ValueError, InvalidOperation) as exc:
            raise ObservationReceiptError(f"raw market-price bar {index} is invalid") from exc
        if volume_decimal != volume_decimal.to_integral_value() or volume_decimal < 0:
            raise ObservationReceiptError(f"raw market-price bar {index} has invalid volume")
        if Decimal(prices["high"]) < max(Decimal(prices["open"]), Decimal(prices["close"])):
            raise ObservationReceiptError(f"raw market-price bar {index} has invalid high")
        if Decimal(prices["low"]) > min(Decimal(prices["open"]), Decimal(prices["close"])):
            raise ObservationReceiptError(f"raw market-price bar {index} has invalid low")
        rows.append(
            {
                "session_date": session.isoformat(),
                **prices,
                "volume": int(volume_decimal),
            }
        )
    rows.sort(key=lambda row: str(row["session_date"]))
    if len({row["session_date"] for row in rows}) != len(rows):
        raise ObservationReceiptError("raw market-price artifact contains duplicate sessions")
    return rows


def build_observation_receipts(
    payload: bytes,
    *,
    policy: Mapping[str, Any],
    provider_id: str,
    instrument_id: str,
    ticker: str,
    exchange: str,
    currency: str,
    retrieved_at: str,
    request_start: str,
    request_end_exclusive: str,
    raw_artifact_locator: str,
    raw_artifact_sha256: str,
    response_status: int,
    content_type: str,
) -> list[dict[str, Any]]:
    approved = approved_fallback_policy(
        policy, provider_id=provider_id, exchange=exchange
    )
    _utc_timestamp(retrieved_at)
    try:
        start = date.fromisoformat(request_start)
        end = date.fromisoformat(request_end_exclusive)
    except (TypeError, ValueError) as exc:
        raise ObservationReceiptError(
            "market-price receipt request window is invalid"
        ) from exc
    if start >= end:
        raise ObservationReceiptError(
            "market-price receipt request window is empty"
        )
    if (
        not isinstance(response_status, int)
        or isinstance(response_status, bool)
        or response_status < 200
        or response_status >= 300
    ):
        raise ObservationReceiptError("raw market-price response status is unsuccessful")
    if content_type != "application/json":
        raise ObservationReceiptError("raw market-price artifact content type is unsupported")
    if sha256_bytes(payload) != raw_artifact_sha256:
        raise ObservationReceiptError("raw market-price artifact checksum mismatch")
    receipts: list[dict[str, Any]] = []
    for row in parse_raw_daily_ohlcv(payload):
        session = date.fromisoformat(str(row["session_date"]))
        if not start <= session < end:
            raise ObservationReceiptError("raw market-price observation is outside request window")
        canonical_values = {
            "instrument_id": instrument_id,
            "session_date": session.isoformat(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "adj_close": row["adj_close"],
            "volume": row["volume"],
            "currency": currency,
        }
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "instrument_id": instrument_id,
            "ticker": ticker,
            "exchange": exchange,
            "session_date": session.isoformat(),
            "provider_id": provider_id,
            "source_approval_id": approved["approval_id"],
            "retrieved_at": retrieved_at,
            "request_start": request_start,
            "request_end_exclusive": request_end_exclusive,
            "raw_artifact_locator": raw_artifact_locator,
            "raw_artifact_sha256": raw_artifact_sha256,
            "response_status": response_status,
            "content_type": content_type,
            "retention_classification": approved["retention_classification"],
            "redistribution_classification": approved[
                "redistribution_classification"
            ],
            "parser_name": PARSER_NAME,
            "parser_version": PARSER_VERSION,
            **{key: row[key] for key in ("open", "high", "low", "close", "adj_close", "volume")},
            "currency": currency,
            "canonical_row_serialization_version": SERIALIZATION_VERSION,
            "canonical_row_sha256": sha256_bytes(_canonical_json(canonical_values)),
        }
        receipt["receipt_sha256"] = sha256_bytes(_canonical_json(receipt))
        receipts.append(receipt)
    return receipts


def observation_receipt_root(receipts: Sequence[Mapping[str, Any]]) -> str:
    leaves = sorted(str(row.get("canonical_row_sha256")) for row in receipts)
    if any(not SHA256.fullmatch(value) for value in leaves) or len(set(leaves)) != len(leaves):
        raise ObservationReceiptError("observation receipt leaves are invalid or duplicated")
    return sha256_bytes(_canonical_json(leaves))


def replay_observation_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise ObservationReceiptError("observation receipts must be a sequence")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ObservationReceiptError("observation receipt must be an object")
        locator = receipt.get("raw_artifact_locator")
        raw_sha = receipt.get("raw_artifact_sha256")
        if not isinstance(locator, str) or not isinstance(raw_sha, str):
            raise ObservationReceiptError("observation receipt raw artifact binding is missing")
        groups.setdefault((locator, raw_sha), []).append(receipt)
    replayed: list[dict[str, Any]] = []
    for (locator, raw_sha), declared in groups.items():
        exemplar = declared[0]
        relative = Path(locator)
        expected_prefix = (
            "evidence",
            "market_price",
            str(exemplar.get("provider_id")),
        )
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:3] != expected_prefix
            or len(relative.parts) != 4
            or relative.name != f"{raw_sha}.json"
            or not SHA256.fullmatch(raw_sha)
        ):
            raise ObservationReceiptError("raw artifact locator is invalid")
        raw_path = Path(artifact_root) / relative
        if not raw_path.is_file():
            raise ObservationReceiptError("raw market-price artifact is missing")
        payload = raw_path.read_bytes()
        if sha256_bytes(payload) != raw_sha:
            raise ObservationReceiptError("raw market-price artifact checksum mismatch")
        if (
            exemplar.get("parser_name") != PARSER_NAME
            or exemplar.get("parser_version") != PARSER_VERSION
        ):
            raise ObservationReceiptError("observation receipt parser is unsupported")
        rebuilt = build_observation_receipts(
            payload,
            policy=policy,
            provider_id=str(exemplar.get("provider_id")),
            instrument_id=str(exemplar.get("instrument_id")),
            ticker=str(exemplar.get("ticker")),
            exchange=str(exemplar.get("exchange")),
            currency=str(exemplar.get("currency")),
            retrieved_at=str(exemplar.get("retrieved_at")),
            request_start=str(exemplar.get("request_start")),
            request_end_exclusive=str(exemplar.get("request_end_exclusive")),
            raw_artifact_locator=locator,
            raw_artifact_sha256=raw_sha,
            response_status=exemplar.get("response_status"),
            content_type=str(exemplar.get("content_type")),
        )
        if sorted(declared, key=lambda row: str(row.get("session_date"))) != rebuilt:
            raise ObservationReceiptError("observation receipt does not replay from raw artifact")
        replayed.extend(rebuilt)
    replayed.sort(key=lambda row: (row["instrument_id"], row["session_date"]))
    if len({(row["instrument_id"], row["session_date"]) for row in replayed}) != len(replayed):
        raise ObservationReceiptError("observation receipts contain duplicate instrument sessions")
    return replayed


def _decimal_text(value: Any) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise InvalidOperation
    normalized = format(decimal.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def _utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ObservationReceiptError("retrieval timestamp must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservationReceiptError("retrieval timestamp must be UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ObservationReceiptError("retrieval timestamp must be UTC")
    return parsed


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "api_key",
                "apikey",
                "access_token",
                "authorization",
                "password",
                "secret",
            }:
                return True
            if _contains_secret_material(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    return False
