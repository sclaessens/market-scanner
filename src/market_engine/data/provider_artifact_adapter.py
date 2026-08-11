from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

from market_engine.data.observation_receipts import (
    ObservationReceiptError,
    approved_source_policy,
)


ADAPTER_ENVELOPE_SCHEMA_VERSION = "market-engine-provider-artifact-envelope-v2"
PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION = "market-engine-provider-artifact-reference-v2"
ACQUISITION_RUN_MANIFEST_SCHEMA_VERSION = "market-engine-market-price-acquisition-run-v1"
WINDOW_SEMANTICS = "start_inclusive_end_exclusive"


@dataclass(frozen=True)
class AcquisitionRequest:
    method_id: str
    parameters: Mapping[str, Any]
    start: str
    end_exclusive: str
    timezone: str
    pagination: Mapping[str, Any]


class RegisteredMarketPriceAdapter:
    """Policy-registered request/response boundary for retained market-price bytes.

    Identity is derived from the selected route and authoritative instrument
    record. Callers cannot independently supply envelope identity fields.
    """

    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        instrument: Mapping[str, Any],
        provider_id: str,
        acquisition_route: str,
    ) -> None:
        provider = approved_source_policy(
            policy,
            provider_id=provider_id,
            exchange=str(instrument.get("exchange")),
            acquisition_route=acquisition_route,
        )
        required = ("instrument_id", "symbol", "source_symbol", "exchange", "currency")
        if any(not isinstance(instrument.get(key), str) or not instrument[key] for key in required):
            raise ObservationReceiptError("registered instrument identity is incomplete")
        if instrument.get("source_mapping_status") != "mapped":
            raise ObservationReceiptError("instrument provider-symbol mapping is not approved")
        self._policy = policy
        self._provider = provider
        self._instrument = {key: str(instrument[key]) for key in required}
        self._provider_id = provider_id
        self._route = acquisition_route

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def adapter_id(self) -> str:
        return str(self._provider["adapter_id"])

    def request(
        self,
        *,
        method_id: str,
        start: str,
        end_exclusive: str,
        timezone: str,
        pagination: Mapping[str, Any],
        extra_parameters: Mapping[str, Any] | None = None,
    ) -> AcquisitionRequest:
        parameters = {
            "symbol": self._instrument["source_symbol"],
            "start": start,
            "end": end_exclusive,
            **dict(extra_parameters or {}),
        }
        if parameters["symbol"] != self._instrument["source_symbol"]:
            raise ObservationReceiptError("provider request symbol differs from registered mapping")
        return AcquisitionRequest(method_id, parameters, start, end_exclusive, timezone, dict(pagination))

    def capture_response(
        self,
        raw_response: bytes,
        *,
        request: AcquisitionRequest,
        artifact_root: str | Path,
        acquisition_run_id: str,
        retrieved_at: str,
        response_status: int,
        response_content_type: str,
        provider_request_id: str | None = None,
    ) -> dict[str, str]:
        return _capture_provider_artifact(
            raw_response,
            artifact_root=artifact_root,
            policy=self._policy,
            provider=self._provider,
            instrument=self._instrument,
            provider_id=self._provider_id,
            acquisition_route=self._route,
            request=request,
            acquisition_run_id=acquisition_run_id,
            retrieved_at=retrieved_at,
            response_status=response_status,
            response_content_type=response_content_type,
            provider_request_id=provider_request_id,
        )


def _capture_provider_artifact(
    raw_response: bytes,
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
    provider: Mapping[str, Any],
    instrument: Mapping[str, str],
    provider_id: str,
    acquisition_route: str,
    request: AcquisitionRequest,
    acquisition_run_id: str,
    retrieved_at: str,
    response_status: int,
    response_content_type: str,
    provider_request_id: str | None,
) -> dict[str, str]:
    if not _safe_identifier(acquisition_run_id):
        raise ObservationReceiptError("acquisition run ID is unsafe")
    approved = approved_source_policy(
        policy,
        provider_id=provider_id,
        exchange=instrument["exchange"],
        acquisition_route=acquisition_route,
        adapter_id=str(provider["adapter_id"]),
        adapter_version=str(provider["adapter_version"]),
    )
    retrieved = _utc_timestamp(retrieved_at)
    try:
        start = date.fromisoformat(request.start)
        end = date.fromisoformat(request.end_exclusive)
    except (TypeError, ValueError) as exc:
        raise ObservationReceiptError("provider artifact request window is invalid") from exc
    if start >= end:
        raise ObservationReceiptError("provider artifact request window is empty")
    expected_parameters = {
        "symbol": instrument["source_symbol"], "start": start.isoformat(), "end": end.isoformat()
    }
    if any(request.parameters.get(key) != value for key, value in expected_parameters.items()):
        raise ObservationReceiptError("provider request parameters differ from constructed request")
    if not isinstance(response_status, int) or isinstance(response_status, bool) or not 200 <= response_status < 300:
        raise ObservationReceiptError("provider artifact response status is unsuccessful")
    if response_content_type != "application/json":
        raise ObservationReceiptError("provider artifact content type is unsupported")
    if not _safe_identifier(request.method_id):
        raise ObservationReceiptError("provider request method ID is unsafe")
    try:
        decoded_response = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("provider artifact raw response is malformed") from exc
    if _contains_secret_material(decoded_response) or _contains_secret_material(request.parameters):
        raise ObservationReceiptError("provider artifact contains credential material")
    if provider_request_id is not None and not _safe_identifier(provider_request_id):
        raise ObservationReceiptError("provider request ID is unsafe")
    raw_digest = _sha256(raw_response)
    request_identity = {
        "request_method_id": request.method_id,
        "request_parameters": dict(request.parameters),
        "request_start": start.isoformat(),
        "request_end_exclusive": end.isoformat(),
        "window_semantics": WINDOW_SEMANTICS,
        "timezone": request.timezone,
        "pagination": dict(request.pagination),
    }
    request_digest = _sha256(_canonical_json(request_identity))
    envelope = {
        "schema_version": ADAPTER_ENVELOPE_SCHEMA_VERSION,
        "acquisition_run_id": acquisition_run_id,
        "provider_id": provider_id,
        "adapter_id": approved["adapter_id"],
        "adapter_version": approved["adapter_version"],
        "parser_name": approved["parser_name"],
        "parser_version": approved["parser_version"],
        "instrument_id": instrument["instrument_id"],
        "canonical_ticker": instrument["symbol"],
        "provider_symbol": instrument["source_symbol"],
        "exchange": instrument["exchange"],
        "currency": instrument["currency"],
        "acquisition_route": acquisition_route,
        **request_identity,
        "request_sha256": request_digest,
        "retrieved_at": _utc_text(retrieved),
        "response_status": response_status,
        "response_content_type": response_content_type,
        "provider_request_id": provider_request_id,
        "raw_response_base64": base64.b64encode(raw_response).decode("ascii"),
        "raw_response_sha256": raw_digest,
        "source_policy_id": approved["approval_id"],
        "producer_component_id": "market_engine.data.provider_artifact_adapter.RegisteredMarketPriceAdapter",
    }
    envelope["envelope_sha256"] = _sha256(_canonical_json(envelope))
    encoded = _canonical_json(envelope)
    artifact_digest = _sha256(encoded)
    relative = Path("evidence") / "market_price" / provider_id / f"{artifact_digest}.json"
    _write_immutable(Path(artifact_root) / relative, encoded, "provider artifact")
    manifest = {
        "schema_version": ACQUISITION_RUN_MANIFEST_SCHEMA_VERSION,
        "acquisition_run_id": acquisition_run_id,
        "adapter_id": approved["adapter_id"],
        "adapter_version": approved["adapter_version"],
        "provider_id": provider_id,
        "acquisition_route": acquisition_route,
        "instrument_id": instrument["instrument_id"],
        "canonical_ticker": instrument["symbol"],
        "provider_symbol": instrument["source_symbol"],
        "exchange": instrument["exchange"],
        "request_sha256": request_digest,
        "artifact_sha256": artifact_digest,
        "raw_response_sha256": raw_digest,
        "retrieved_at": _utc_text(retrieved),
        "source_policy_id": approved["approval_id"],
        "artifact_locator": relative.as_posix(),
        "producer_component_id": envelope["producer_component_id"],
    }
    manifest_encoded = _canonical_json(manifest)
    manifest_sha = _sha256(manifest_encoded)
    manifest_relative = Path("evidence") / "market_price" / "acquisition_runs" / acquisition_run_id / f"{manifest_sha}.json"
    _write_immutable(Path(artifact_root) / manifest_relative, manifest_encoded, "acquisition run manifest")
    return {
        "schema_version": PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "acquisition_run_id": acquisition_run_id,
        "acquisition_manifest_locator": manifest_relative.as_posix(),
        "acquisition_manifest_sha256": manifest_sha,
        "artifact_locator": relative.as_posix(),
        "artifact_sha256": artifact_digest,
        "envelope_sha256": envelope["envelope_sha256"],
    }


def _write_immutable(path: Path, encoded: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise ObservationReceiptError(f"{label} checksum collision")
    path.write_bytes(encoded)


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_key", "apikey", "access_token", "authorization", "cookie", "password", "secret", "token"}:
                return True
            if _contains_secret_material(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in ("api_key=", "apikey=", "access_token=", "token=", "password=", "authorization=", "://user:"))
    return False


def _safe_identifier(value: str) -> bool:
    return bool(value) and len(value) <= 200 and not any(term in value.lower() for term in ("token", "secret", "password", "key=")) and "://" not in value and "/" not in value and ".." not in value


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
