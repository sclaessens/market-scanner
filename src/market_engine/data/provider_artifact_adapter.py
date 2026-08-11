from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

from market_engine.data.observation_receipts import (
    ObservationReceiptError,
    approved_source_policy,
)


ADAPTER_ENVELOPE_SCHEMA_VERSION = "market-engine-provider-artifact-envelope-v1"
PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION = (
    "market-engine-provider-artifact-reference-v1"
)
WINDOW_SEMANTICS = "start_inclusive_end_exclusive"


def capture_provider_artifact(
    raw_response: bytes,
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
    provider_id: str,
    adapter_id: str,
    adapter_version: str,
    instrument_id: str,
    canonical_ticker: str,
    provider_symbol: str,
    exchange: str,
    currency: str,
    acquisition_route: str,
    request_method_id: str,
    request_parameters: Mapping[str, Any],
    request_start: str,
    request_end_exclusive: str,
    timezone: str,
    pagination: Mapping[str, Any],
    retrieved_at: str,
    response_status: int,
    response_content_type: str,
    provider_request_id: str | None = None,
) -> dict[str, str]:
    """Capture the earliest adapter-returned bytes with their request identity.

    This is a code-level trust boundary, not a cryptographic signature. The
    publisher still verifies every digest, policy binding, parser, identity,
    and canonical row independently.
    """
    approved = approved_source_policy(
        policy,
        provider_id=provider_id,
        exchange=exchange,
        acquisition_route=acquisition_route,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
    )
    retrieved = _utc_timestamp(retrieved_at)
    try:
        start = date.fromisoformat(request_start)
        end = date.fromisoformat(request_end_exclusive)
    except (TypeError, ValueError) as exc:
        raise ObservationReceiptError("provider artifact request window is invalid") from exc
    if start >= end:
        raise ObservationReceiptError("provider artifact request window is empty")
    if (
        not isinstance(response_status, int)
        or isinstance(response_status, bool)
        or not 200 <= response_status < 300
    ):
        raise ObservationReceiptError("provider artifact response status is unsuccessful")
    if response_content_type != "application/json":
        raise ObservationReceiptError("provider artifact content type is unsupported")
    if not _safe_identifier(request_method_id):
        raise ObservationReceiptError("provider request method ID is unsafe")
    try:
        decoded_response = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("provider artifact raw response is malformed") from exc
    if _contains_secret_material(decoded_response) or _contains_secret_material(
        request_parameters
    ):
        raise ObservationReceiptError("provider artifact contains credential material")
    if provider_request_id is not None and not _safe_identifier(provider_request_id):
        raise ObservationReceiptError("provider request ID is unsafe")
    raw_digest = _sha256(raw_response)
    envelope = {
        "schema_version": ADAPTER_ENVELOPE_SCHEMA_VERSION,
        "provider_id": provider_id,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "parser_name": approved["parser_name"],
        "parser_version": approved["parser_version"],
        "instrument_id": instrument_id,
        "canonical_ticker": canonical_ticker,
        "provider_symbol": provider_symbol,
        "exchange": exchange,
        "currency": currency,
        "acquisition_route": acquisition_route,
        "request_method_id": request_method_id,
        "request_parameters": dict(request_parameters),
        "request_start": start.isoformat(),
        "request_end_exclusive": end.isoformat(),
        "window_semantics": WINDOW_SEMANTICS,
        "timezone": timezone,
        "pagination": dict(pagination),
        "retrieved_at": _utc_text(retrieved),
        "response_status": response_status,
        "response_content_type": response_content_type,
        "provider_request_id": provider_request_id,
        "raw_response_base64": base64.b64encode(raw_response).decode("ascii"),
        "raw_response_sha256": raw_digest,
        "source_policy_id": approved["approval_id"],
    }
    envelope["envelope_sha256"] = _sha256(_canonical_json(envelope))
    encoded = _canonical_json(envelope)
    artifact_digest = _sha256(encoded)
    relative = (
        Path("evidence")
        / "market_price"
        / provider_id
        / f"{artifact_digest}.json"
    )
    target = Path(artifact_root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != encoded:
        raise ObservationReceiptError("provider artifact checksum collision")
    target.write_bytes(encoded)
    return {
        "schema_version": PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "artifact_locator": relative.as_posix(),
        "artifact_sha256": artifact_digest,
        "envelope_sha256": envelope["envelope_sha256"],
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "api_key",
                "apikey",
                "access_token",
                "authorization",
                "cookie",
                "password",
                "secret",
                "token",
            }:
                return True
            if _contains_secret_material(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(
            marker in lowered
            for marker in (
                "api_key=", "apikey=", "access_token=", "token=",
                "password=", "authorization=", "://user:",
            )
        )
    return False


def _safe_identifier(value: str) -> bool:
    lowered = value.lower()
    return (
        len(value) <= 200
        and not any(term in lowered for term in ("token", "secret", "password", "key="))
        and "://" not in value
    )


def _utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ObservationReceiptError("provider retrieval timestamp must be UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ObservationReceiptError("provider retrieval timestamp must be UTC")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
