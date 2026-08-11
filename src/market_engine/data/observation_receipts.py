from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = "market-engine-market-price-source-policy-v3"
PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION = "market-engine-provider-artifact-reference-v2"
ADAPTER_ENVELOPE_SCHEMA_VERSION = "market-engine-provider-artifact-envelope-v2"
ACQUISITION_RUN_MANIFEST_SCHEMA_VERSION = "market-engine-market-price-acquisition-run-v1"
REPLAYED_OBSERVATION_SCHEMA_VERSION = "market-engine-replayed-observation-v1"
RECEIPT_SCHEMA_VERSION = "market-engine-observation-receipt-v3"
ABSENCE_ATTESTATION_SCHEMA_VERSION = (
    "market-engine-observation-absence-attestation-v2"
)
PARSER_NAME = "canonical-json-daily-ohlcv"
PARSER_VERSION = "v1"
SERIALIZATION_VERSION = "market-engine-canonical-row-v1"
WINDOW_SEMANTICS = "start_inclusive_end_exclusive"
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
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError(
            "market-price source policy is missing or malformed"
        ) from exc
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION or not isinstance(
        payload.get("providers"), list
    ):
        raise ObservationReceiptError("market-price source policy schema is unsupported")
    providers: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["providers"]):
        if not isinstance(raw, Mapping):
            raise ObservationReceiptError(
                f"source policy provider {index} must be an object"
            )
        provider = dict(raw)
        required = (
            "provider_id",
            "approval_id",
            "data_type",
            "adapter_id",
            "adapter_version",
            "parser_name",
            "parser_version",
            "retention_classification",
            "redistribution_classification",
        )
        if any(
            not isinstance(provider.get(key), str) or not provider[key]
            for key in required
        ):
            raise ObservationReceiptError(
                f"source policy provider {index} is incomplete"
            )
        if not PROVIDER_ID.fullmatch(provider["provider_id"]):
            raise ObservationReceiptError(
                f"source policy provider {index} has invalid provider ID"
            )
        if provider["data_type"] != "daily_ohlcv":
            raise ObservationReceiptError(
                f"source policy provider {index} has unsupported data type"
            )
        if (
            provider["parser_name"] != PARSER_NAME
            or provider["parser_version"] != PARSER_VERSION
        ):
            raise ObservationReceiptError(
                f"source policy provider {index} has unsupported parser"
            )
        for key in (
            "approved_for_acquisition",
            "approved_for_retention",
            "approved_for_replay",
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
            raise ObservationReceiptError(
                f"source policy provider {index} requires exchanges"
            )
        routes = provider.get("acquisition_routes")
        if not isinstance(routes, list) or not routes or any(
            value not in {"primary", "primary_replay", "fallback"}
            for value in routes
        ):
            raise ObservationReceiptError(
                f"source policy provider {index} requires acquisition routes"
            )
        providers.append(
            {
                **provider,
                "exchanges": sorted(set(exchanges)),
                "acquisition_routes": sorted(set(routes)),
            }
        )
    if len({row["provider_id"] for row in providers}) != len(providers):
        raise ObservationReceiptError(
            "market-price source policy contains duplicate providers"
        )
    if len({row["approval_id"] for row in providers}) != len(providers):
        raise ObservationReceiptError(
            "market-price source policy contains duplicate approval IDs"
        )
    providers.sort(key=lambda row: row["provider_id"])
    normalized = {"schema_version": POLICY_SCHEMA_VERSION, "providers": providers}
    return {
        **normalized,
        "policy_checksum": sha256_bytes(_canonical_json(normalized)),
        "providers_by_id": {row["provider_id"]: row for row in providers},
    }


def approved_source_policy(
    policy: Mapping[str, Any],
    *,
    provider_id: str,
    exchange: str,
    acquisition_route: str,
    adapter_id: str | None = None,
    adapter_version: str | None = None,
) -> dict[str, Any]:
    provider = (policy.get("providers_by_id") or {}).get(provider_id)
    if not isinstance(provider, Mapping):
        raise ObservationReceiptError(f"unknown market-price provider: {provider_id}")
    if not all(
        provider.get(key) is True
        for key in (
            "approved_for_acquisition",
            "approved_for_retention",
            "approved_for_replay",
            "approved_for_canonical_publication",
        )
    ):
        raise ObservationReceiptError(
            "market-price provider is not approved for canonical publication: "
            f"{provider_id}"
        )
    if exchange not in provider.get("exchanges", []):
        raise ObservationReceiptError(
            f"market-price provider is not approved for exchange: {provider_id}/{exchange}"
        )
    if acquisition_route not in provider.get("acquisition_routes", []):
        raise ObservationReceiptError(
            "market-price provider is not approved for acquisition route: "
            f"{provider_id}/{acquisition_route}"
        )
    if adapter_id is not None and provider.get("adapter_id") != adapter_id:
        raise ObservationReceiptError("provider artifact adapter identity is not approved")
    if adapter_version is not None and provider.get("adapter_version") != adapter_version:
        raise ObservationReceiptError("provider artifact adapter version is not approved")
    return dict(provider)


def replay_provider_artifacts(
    references: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    artifact_ids: set[tuple[str, str]] = set()
    for reference in references:
        loaded = load_provider_artifact(
            reference,
            artifact_root=artifact_root,
            policy=policy,
        )
        identity = (
            loaded["reference"]["artifact_locator"],
            loaded["reference"]["artifact_sha256"],
        )
        if identity in artifact_ids:
            raise ObservationReceiptError("provider artifact references are duplicated")
        artifact_ids.add(identity)
        observations.extend(loaded["observations"])
    observations.sort(
        key=lambda row: (
            str(row["instrument_id"]),
            str(row["session_date"]),
            str(row["artifact_sha256"]),
        )
    )
    identities = [
        (str(row["instrument_id"]), str(row["session_date"]))
        for row in observations
    ]
    if len(identities) != len(set(identities)):
        raise ObservationReceiptError(
            "provider artifacts contain duplicate or conflicting instrument sessions"
        )
    return observations


def load_provider_artifact(
    reference: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "schema_version",
        "acquisition_run_id",
        "acquisition_manifest_locator",
        "acquisition_manifest_sha256",
        "artifact_locator",
        "artifact_sha256",
        "envelope_sha256",
    }:
        raise ObservationReceiptError("provider artifact reference is invalid")
    if reference.get("schema_version") != PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION:
        raise ObservationReceiptError("provider artifact reference schema is unsupported")
    locator = reference.get("artifact_locator")
    run_id = reference.get("acquisition_run_id")
    manifest_locator = reference.get("acquisition_manifest_locator")
    manifest_sha = reference.get("acquisition_manifest_sha256")
    artifact_sha = reference.get("artifact_sha256")
    envelope_sha = reference.get("envelope_sha256")
    if not all(
        isinstance(value, str)
        for value in (locator, artifact_sha, envelope_sha, run_id, manifest_locator, manifest_sha)
    ) or not SHA256.fullmatch(str(artifact_sha)) or not SHA256.fullmatch(
        str(envelope_sha)
    ) or not SHA256.fullmatch(str(manifest_sha)):
        raise ObservationReceiptError("provider artifact reference digest is invalid")
    relative = Path(str(locator))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("evidence", "market_price")
        or len(relative.parts) != 4
        or relative.name != f"{artifact_sha}.json"
    ):
        raise ObservationReceiptError("provider artifact locator is invalid")
    manifest_relative = Path(str(manifest_locator))
    if (
        manifest_relative.is_absolute()
        or ".." in manifest_relative.parts
        or manifest_relative.parts[:3] != ("evidence", "market_price", "acquisition_runs")
        or len(manifest_relative.parts) != 5
        or manifest_relative.parts[3] != run_id
        or manifest_relative.name != f"{manifest_sha}.json"
    ):
        raise ObservationReceiptError("trusted acquisition run reference is invalid")
    manifest_path = Path(artifact_root) / manifest_relative
    if not manifest_path.is_file():
        raise ObservationReceiptError("provider artifact is absent from trusted acquisition run metadata")
    manifest_encoded = manifest_path.read_bytes()
    if sha256_bytes(manifest_encoded) != manifest_sha:
        raise ObservationReceiptError("trusted acquisition run manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("trusted acquisition run manifest is malformed") from exc
    path = Path(artifact_root) / relative
    if not path.is_file():
        raise ObservationReceiptError("provider artifact is missing")
    encoded = path.read_bytes()
    if sha256_bytes(encoded) != artifact_sha:
        raise ObservationReceiptError("provider artifact checksum mismatch")
    try:
        envelope = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("provider artifact envelope is malformed") from exc
    if not isinstance(envelope, Mapping):
        raise ObservationReceiptError("provider artifact envelope must be an object")
    required_envelope_fields = {
        "schema_version", "acquisition_run_id", "provider_id", "adapter_id", "adapter_version",
        "parser_name", "parser_version", "instrument_id", "canonical_ticker",
        "provider_symbol", "exchange", "currency", "acquisition_route",
        "request_method_id", "request_parameters", "request_start",
        "request_end_exclusive", "window_semantics", "timezone", "pagination",
        "retrieved_at", "response_status", "response_content_type",
        "provider_request_id", "raw_response_base64", "raw_response_sha256",
        "source_policy_id", "request_sha256", "producer_component_id", "envelope_sha256",
    }
    if set(envelope) != required_envelope_fields:
        raise ObservationReceiptError("provider artifact envelope fields are invalid")
    declared_envelope_sha = envelope.get("envelope_sha256")
    unsigned = dict(envelope)
    unsigned.pop("envelope_sha256", None)
    if (
        declared_envelope_sha != envelope_sha
        or sha256_bytes(_canonical_json(unsigned)) != envelope_sha
    ):
        raise ObservationReceiptError("provider artifact envelope digest mismatch")
    if envelope.get("schema_version") != ADAPTER_ENVELOPE_SCHEMA_VERSION:
        raise ObservationReceiptError("provider artifact envelope schema is unsupported")
    provider_id = str(envelope.get("provider_id"))
    if relative.parts[2] != provider_id:
        raise ObservationReceiptError("provider artifact locator provider mismatch")
    approved = approved_source_policy(
        policy,
        provider_id=provider_id,
        exchange=str(envelope.get("exchange")),
        acquisition_route=str(envelope.get("acquisition_route")),
        adapter_id=str(envelope.get("adapter_id")),
        adapter_version=str(envelope.get("adapter_version")),
    )
    if (
        envelope.get("source_policy_id") != approved["approval_id"]
        or envelope.get("parser_name") != approved["parser_name"]
        or envelope.get("parser_version") != approved["parser_version"]
        or envelope.get("window_semantics") != WINDOW_SEMANTICS
    ):
        raise ObservationReceiptError("provider artifact policy or parser binding mismatch")
    request_identity = {
        key: envelope[key]
        for key in (
            "request_method_id", "request_parameters", "request_start",
            "request_end_exclusive", "window_semantics", "timezone", "pagination",
        )
    }
    if sha256_bytes(_canonical_json(request_identity)) != envelope.get("request_sha256"):
        raise ObservationReceiptError("provider artifact request digest mismatch")
    expected_manifest = {
        "schema_version": ACQUISITION_RUN_MANIFEST_SCHEMA_VERSION,
        "acquisition_run_id": envelope["acquisition_run_id"],
        "adapter_id": envelope["adapter_id"],
        "adapter_version": envelope["adapter_version"],
        "provider_id": envelope["provider_id"],
        "acquisition_route": envelope["acquisition_route"],
        "instrument_id": envelope["instrument_id"],
        "canonical_ticker": envelope["canonical_ticker"],
        "provider_symbol": envelope["provider_symbol"],
        "exchange": envelope["exchange"],
        "request_sha256": envelope["request_sha256"],
        "artifact_sha256": artifact_sha,
        "raw_response_sha256": envelope["raw_response_sha256"],
        "retrieved_at": envelope["retrieved_at"],
        "source_policy_id": envelope["source_policy_id"],
        "artifact_locator": locator,
        "producer_component_id": envelope["producer_component_id"],
    }
    if manifest != expected_manifest or run_id != envelope["acquisition_run_id"]:
        raise ObservationReceiptError("provider artifact trusted acquisition identity mismatch")
    text_fields = (
        "provider_id", "adapter_id", "adapter_version", "instrument_id",
        "canonical_ticker", "provider_symbol", "exchange", "currency",
        "acquisition_route", "request_method_id", "timezone",
        "response_content_type", "raw_response_base64", "raw_response_sha256",
        "source_policy_id", "request_sha256", "producer_component_id", "acquisition_run_id",
    )
    if any(
        not isinstance(envelope.get(key), str) or not envelope[key]
        for key in text_fields
    ):
        raise ObservationReceiptError("provider artifact identity is incomplete")
    if not isinstance(envelope.get("request_parameters"), Mapping) or not isinstance(
        envelope.get("pagination"), Mapping
    ):
        raise ObservationReceiptError("provider artifact request metadata is invalid")
    if (
        envelope["request_parameters"].get("symbol") != envelope["provider_symbol"]
        or envelope["request_parameters"].get("start") != envelope["request_start"]
        or envelope["request_parameters"].get("end")
        != envelope["request_end_exclusive"]
    ):
        raise ObservationReceiptError("provider artifact request identity mismatch")
    response_status = envelope.get("response_status")
    if (
        not isinstance(response_status, int)
        or isinstance(response_status, bool)
        or not 200 <= response_status < 300
        or envelope.get("response_content_type") != "application/json"
    ):
        raise ObservationReceiptError("provider artifact response metadata is invalid")
    _utc_timestamp(str(envelope.get("retrieved_at")))
    try:
        start = date.fromisoformat(str(envelope.get("request_start")))
        end = date.fromisoformat(str(envelope.get("request_end_exclusive")))
        raw_response = base64.b64decode(
            str(envelope.get("raw_response_base64")), validate=True
        )
    except (TypeError, ValueError) as exc:
        raise ObservationReceiptError("provider artifact request or response is invalid") from exc
    if start >= end or sha256_bytes(raw_response) != envelope.get("raw_response_sha256"):
        raise ObservationReceiptError("provider artifact raw response digest mismatch")
    try:
        decoded_response = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("provider artifact raw response is malformed") from exc
    if (
        _contains_secret_material(envelope.get("request_parameters"))
        or _contains_secret_material(envelope.get("request_method_id"))
        or _contains_secret_material(envelope.get("provider_request_id"))
        or "://" in str(envelope.get("request_method_id"))
    ):
        raise ObservationReceiptError("provider artifact contains credential material")
    if _contains_secret_material(decoded_response):
        raise ObservationReceiptError("provider artifact contains credential material")
    observations = _replay_envelope_observations(
        envelope,
        raw_response,
        reference=reference,
    )
    return {
        "reference": dict(reference),
        "envelope": dict(envelope),
        "raw_response": raw_response,
        "observations": observations,
    }


def select_mutation_observations(
    mutations: Sequence[Mapping[str, Any]],
    replayed_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mutation_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for mutation in mutations:
        identity = (
            str(mutation.get("instrument_id")),
            str(mutation.get("session_date")),
        )
        if identity in mutation_by_identity:
            raise ObservationReceiptError("mutation ledger contains duplicate sessions")
        mutation_by_identity[identity] = mutation
    accepted: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for observation in replayed_observations:
        identity = (
            str(observation.get("instrument_id")),
            str(observation.get("session_date")),
        )
        mutation = mutation_by_identity.get(identity)
        if mutation is None:
            raise ObservationReceiptError(
                "replayed provider observation has no canonical row"
            )
        mutation_type = mutation.get("mutation_type")
        expected_digest = mutation.get("new_canonical_row_sha256")
        if expected_digest != observation.get("canonical_row_sha256"):
            raise ObservationReceiptError(
                "replayed provider observation conflicts with canonical row"
            )
        if mutation_type == "row_unchanged":
            unchanged.append(dict(observation))
        elif mutation_type in {"row_added", "row_modified"}:
            accepted.append(dict(observation))
        else:
            raise ObservationReceiptError(
                "replayed provider observation targets a deleted canonical row"
            )
    required = {
        (str(row.get("instrument_id")), str(row.get("session_date")))
        for row in mutations
        if row.get("mutation_type") in {"row_added", "row_modified"}
    }
    accepted_identities = {
        (str(row["instrument_id"]), str(row["session_date"])) for row in accepted
    }
    if required != accepted_identities:
        raise ObservationReceiptError(
            "publisher-derived mutations do not equal accepted artifact observations"
        )
    receipts = [_observation_receipt(row) for row in accepted]
    receipts.sort(key=lambda row: (row["instrument_id"], row["session_date"]))
    return {
        "replayed_observations": [dict(row) for row in replayed_observations],
        "accepted_mutation_observations": accepted,
        "mutation_receipts": receipts,
        "unchanged_overlap_observations": unchanged,
    }


def validate_declared_receipts(
    declared: Sequence[Mapping[str, Any]],
    publisher_selected: Sequence[Mapping[str, Any]],
) -> None:
    normalized_declared = sorted(
        [dict(row) for row in declared],
        key=lambda row: (str(row.get("instrument_id")), str(row.get("session_date"))),
    )
    if normalized_declared != list(publisher_selected):
        raise ObservationReceiptError(
            "declared receipts do not equal publisher-selected mutation receipts"
        )


def build_absence_attestation(
    *,
    artifact_reference: Mapping[str, Any],
    artifact_root: str | Path,
    policy: Mapping[str, Any],
    session_date: str,
    lifecycle_cutoff: str,
    reason_code: str,
    calendar_expected: bool,
) -> dict[str, Any]:
    loaded = load_provider_artifact(
        artifact_reference,
        artifact_root=artifact_root,
        policy=policy,
    )
    envelope = loaded["envelope"]
    try:
        session = date.fromisoformat(session_date)
        cutoff = date.fromisoformat(lifecycle_cutoff)
        start = date.fromisoformat(str(envelope["request_start"]))
        end = date.fromisoformat(str(envelope["request_end_exclusive"]))
    except (TypeError, ValueError) as exc:
        raise ObservationReceiptError("absence attestation date is invalid") from exc
    if reason_code != "terminal_daily_ohlcv_not_returned":
        raise ObservationReceiptError("absence attestation reason is unsupported")
    if not calendar_expected or session != cutoff or not start <= session < end:
        raise ObservationReceiptError("absence attestation boundary is invalid")
    parsed_sessions = [row["session_date"] for row in loaded["observations"]]
    if session.isoformat() in parsed_sessions:
        raise ObservationReceiptError(
            "absence artifact contains the allegedly absent session"
        )
    attestation = {
        "schema_version": ABSENCE_ATTESTATION_SCHEMA_VERSION,
        "instrument_id": envelope["instrument_id"],
        "ticker": envelope["canonical_ticker"],
        "provider_symbol": envelope["provider_symbol"],
        "exchange": envelope["exchange"],
        "session_date": session.isoformat(),
        "lifecycle_cutoff": cutoff.isoformat(),
        "absence_reason_code": reason_code,
        "calendar_expected": True,
        "provider_id": envelope["provider_id"],
        "adapter_id": envelope["adapter_id"],
        "adapter_version": envelope["adapter_version"],
        "source_policy_id": envelope["source_policy_id"],
        "acquisition_route": envelope["acquisition_route"],
        "request_method_id": envelope["request_method_id"],
        "request_parameters": envelope["request_parameters"],
        "request_start": envelope["request_start"],
        "request_end_exclusive": envelope["request_end_exclusive"],
        "window_semantics": envelope["window_semantics"],
        "timezone": envelope["timezone"],
        "pagination": envelope["pagination"],
        "retrieved_at": envelope["retrieved_at"],
        "response_status": envelope["response_status"],
        "response_content_type": envelope["response_content_type"],
        "provider_request_id": envelope["provider_request_id"],
        "acquisition_run_id": artifact_reference["acquisition_run_id"],
        "acquisition_manifest_locator": artifact_reference["acquisition_manifest_locator"],
        "acquisition_manifest_sha256": artifact_reference["acquisition_manifest_sha256"],
        "artifact_locator": artifact_reference["artifact_locator"],
        "artifact_sha256": artifact_reference["artifact_sha256"],
        "envelope_sha256": artifact_reference["envelope_sha256"],
        "raw_response_sha256": envelope["raw_response_sha256"],
        "parser_name": envelope["parser_name"],
        "parser_version": envelope["parser_version"],
        "parsed_session_dates": parsed_sessions,
    }
    attestation["attestation_sha256"] = sha256_bytes(_canonical_json(attestation))
    return attestation


def replay_absence_attestations(
    attestations: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    replayed: list[dict[str, Any]] = []
    for declared in attestations:
        reference = {
            "schema_version": PROVIDER_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            "acquisition_run_id": declared.get("acquisition_run_id"),
            "acquisition_manifest_locator": declared.get("acquisition_manifest_locator"),
            "acquisition_manifest_sha256": declared.get("acquisition_manifest_sha256"),
            "artifact_locator": declared.get("artifact_locator"),
            "artifact_sha256": declared.get("artifact_sha256"),
            "envelope_sha256": declared.get("envelope_sha256"),
        }
        rebuilt = build_absence_attestation(
            artifact_reference=reference,
            artifact_root=artifact_root,
            policy=policy,
            session_date=str(declared.get("session_date")),
            lifecycle_cutoff=str(declared.get("lifecycle_cutoff")),
            reason_code=str(declared.get("absence_reason_code")),
            calendar_expected=declared.get("calendar_expected") is True,
        )
        if dict(declared) != rebuilt:
            raise ObservationReceiptError(
                "absence attestation does not replay from provider artifact"
            )
        replayed.append(rebuilt)
    replayed.sort(key=lambda row: (row["instrument_id"], row["session_date"]))
    identities = [(row["instrument_id"], row["session_date"]) for row in replayed]
    if len(identities) != len(set(identities)):
        raise ObservationReceiptError("absence attestations contain duplicates")
    return replayed


def bind_absence_evidence_to_consumer(
    attestations: Sequence[Mapping[str, Any]],
    *,
    consumer_identity: Mapping[str, Any],
    expected_sessions: Sequence[str],
    lifecycle_cutoff: str | None,
) -> list[dict[str, Any]]:
    """Validate self-consistent absence evidence against its actual consumer."""
    expected = set(expected_sessions)
    identity_fields = (
        ("instrument_id", "instrument_id"),
        ("ticker", "symbol"),
        ("provider_id", "provider_id"),
        ("provider_symbol", "source_symbol"),
        ("exchange", "exchange"),
        ("source_policy_id", "source_policy_id"),
        ("acquisition_route", "acquisition_route"),
        ("timezone", "timezone"),
    )
    validated: list[dict[str, Any]] = []
    for attestation in attestations:
        if any(
            attestation.get(evidence_key) != consumer_identity.get(consumer_key)
            for evidence_key, consumer_key in identity_fields
        ):
            raise ObservationReceiptError(
                "ABSENCE_EVIDENCE_CONSUMER_IDENTITY_MISMATCH"
            )
        session = str(attestation.get("session_date"))
        if (
            session not in expected
            or attestation.get("lifecycle_cutoff") != lifecycle_cutoff
            or attestation.get("absence_reason_code")
            != "terminal_daily_ohlcv_not_returned"
            or attestation.get("calendar_expected") is not True
            or not str(attestation.get("request_start")) <= session
            < str(attestation.get("request_end_exclusive"))
            or attestation.get("window_semantics") != WINDOW_SEMANTICS
        ):
            raise ObservationReceiptError(
                "ABSENCE_EVIDENCE_CONSUMER_LIFECYCLE_MISMATCH"
            )
        validated.append(dict(attestation))
    return validated


def observation_receipt_root(receipts: Sequence[Mapping[str, Any]]) -> str:
    leaves = sorted(
        "|".join(
            (
                str(row.get("exchange")),
                str(row.get("instrument_id")),
                str(row.get("session_date")),
                str(row.get("canonical_row_sha256")),
                str(row.get("receipt_sha256")),
            )
        )
        for row in receipts
    )
    if any(
        len(value.split("|")) != 5
        or not SHA256.fullmatch(value.split("|")[3])
        or not SHA256.fullmatch(value.split("|")[4])
        for value in leaves
    ) or len(set(leaves)) != len(leaves):
        raise ObservationReceiptError(
            "observation receipt leaves are invalid or duplicated"
        )
    return sha256_bytes(_canonical_json(leaves))


def canonical_row_sha256(
    *,
    instrument_id: str,
    session_date: str,
    open_value: Any,
    high: Any,
    low: Any,
    close: Any,
    adj_close: Any,
    volume: Any,
    currency: str,
) -> str:
    volume_decimal = Decimal(str(volume))
    if volume_decimal != volume_decimal.to_integral_value():
        raise ObservationReceiptError("canonical row volume is not integral")
    payload = {
        "instrument_id": instrument_id,
        "session_date": date.fromisoformat(session_date).isoformat(),
        "open": _decimal_text(open_value),
        "high": _decimal_text(high),
        "low": _decimal_text(low),
        "close": _decimal_text(close),
        "adj_close": _decimal_text(adj_close),
        "volume": int(volume_decimal),
        "currency": currency,
    }
    return sha256_bytes(_canonical_json(payload))


def _replay_envelope_observations(
    envelope: Mapping[str, Any],
    raw_response: bytes,
    *,
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _parse_raw_records(raw_response)
    start = date.fromisoformat(str(envelope["request_start"]))
    end = date.fromisoformat(str(envelope["request_end_exclusive"]))
    observations: list[dict[str, Any]] = []
    for raw_record, parsed in rows:
        session = date.fromisoformat(str(parsed["session_date"]))
        if not start <= session < end:
            raise ObservationReceiptError(
                "provider observation is outside the envelope request window"
            )
        record_sha = sha256_bytes(_canonical_json(raw_record))
        observation = {
            "schema_version": REPLAYED_OBSERVATION_SCHEMA_VERSION,
            "provider_id": envelope["provider_id"],
            "adapter_id": envelope["adapter_id"],
            "adapter_version": envelope["adapter_version"],
            "instrument_id": envelope["instrument_id"],
            "ticker": envelope["canonical_ticker"],
            "provider_symbol": envelope["provider_symbol"],
            "exchange": envelope["exchange"],
            "currency": envelope["currency"],
            "session_date": session.isoformat(),
            "acquisition_route": envelope["acquisition_route"],
            "source_policy_id": envelope["source_policy_id"],
            "request_method_id": envelope["request_method_id"],
            "request_parameters": envelope["request_parameters"],
            "request_start": envelope["request_start"],
            "request_end_exclusive": envelope["request_end_exclusive"],
            "window_semantics": envelope["window_semantics"],
            "timezone": envelope["timezone"],
            "pagination": envelope["pagination"],
            "retrieved_at": envelope["retrieved_at"],
            "response_status": envelope["response_status"],
            "response_content_type": envelope["response_content_type"],
            "provider_request_id": envelope["provider_request_id"],
            "acquisition_run_id": reference["acquisition_run_id"],
            "acquisition_manifest_locator": reference["acquisition_manifest_locator"],
            "acquisition_manifest_sha256": reference["acquisition_manifest_sha256"],
            "artifact_locator": reference["artifact_locator"],
            "artifact_sha256": reference["artifact_sha256"],
            "envelope_sha256": reference["envelope_sha256"],
            "raw_response_sha256": envelope["raw_response_sha256"],
            "raw_record_locator": f"bars/{session.isoformat()}/{record_sha}",
            "raw_record_sha256": record_sha,
            "parser_name": envelope["parser_name"],
            "parser_version": envelope["parser_version"],
            **parsed,
        }
        observation["canonical_row_sha256"] = canonical_row_sha256(
            instrument_id=str(envelope["instrument_id"]),
            session_date=session.isoformat(),
            open_value=parsed["open"],
            high=parsed["high"],
            low=parsed["low"],
            close=parsed["close"],
            adj_close=parsed["adj_close"],
            volume=parsed["volume"],
            currency=str(envelope["currency"]),
        )
        observations.append(observation)
    observations.sort(key=lambda row: (row["session_date"], row["raw_record_sha256"]))
    return observations


def _observation_receipt(observation: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "provider_id",
        "adapter_id",
        "adapter_version",
        "instrument_id",
        "ticker",
        "provider_symbol",
        "exchange",
        "currency",
        "session_date",
        "acquisition_route",
        "source_policy_id",
        "request_method_id",
        "request_parameters",
        "request_start",
        "request_end_exclusive",
        "window_semantics",
        "timezone",
        "retrieved_at",
        "response_status",
        "response_content_type",
        "provider_request_id",
        "acquisition_run_id",
        "acquisition_manifest_locator",
        "acquisition_manifest_sha256",
        "artifact_locator",
        "artifact_sha256",
        "envelope_sha256",
        "raw_response_sha256",
        "raw_record_locator",
        "raw_record_sha256",
        "parser_name",
        "parser_version",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "canonical_row_sha256",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        **{key: observation[key] for key in fields},
        "canonical_row_serialization_version": SERIALIZATION_VERSION,
    }
    receipt["receipt_sha256"] = sha256_bytes(_canonical_json(receipt))
    return receipt


def _parse_raw_records(
    payload: bytes,
) -> list[tuple[dict[str, Any], dict[str, str | int]]]:
    try:
        raw = json.loads(payload.decode("utf-8"), parse_float=str, parse_int=str)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObservationReceiptError("provider raw response is malformed") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("bars"), list):
        raise ObservationReceiptError("provider raw response has unsupported schema")
    rows: list[tuple[dict[str, Any], dict[str, str | int]]] = []
    for index, value in enumerate(raw["bars"]):
        if not isinstance(value, Mapping):
            raise ObservationReceiptError(f"provider raw bar {index} must be an object")
        record = dict(value)
        try:
            session = date.fromisoformat(str(value["session_date"]))
            prices = {
                key: _decimal_text(value[key])
                for key in ("open", "high", "low", "close", "adj_close")
            }
            volume_decimal = Decimal(str(value["volume"]))
        except (KeyError, ValueError, InvalidOperation) as exc:
            raise ObservationReceiptError(f"provider raw bar {index} is invalid") from exc
        if volume_decimal != volume_decimal.to_integral_value() or volume_decimal < 0:
            raise ObservationReceiptError(f"provider raw bar {index} has invalid volume")
        if Decimal(prices["high"]) < max(
            Decimal(prices["open"]), Decimal(prices["close"])
        ):
            raise ObservationReceiptError(f"provider raw bar {index} has invalid high")
        if Decimal(prices["low"]) > min(
            Decimal(prices["open"]), Decimal(prices["close"])
        ):
            raise ObservationReceiptError(f"provider raw bar {index} has invalid low")
        rows.append(
            (
                record,
                {
                    "session_date": session.isoformat(),
                    **prices,
                    "volume": int(volume_decimal),
                },
            )
        )
    sessions = [row[1]["session_date"] for row in rows]
    if len(sessions) != len(set(sessions)):
        raise ObservationReceiptError("provider raw response contains duplicate sessions")
    return rows


def _decimal_text(value: Any) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise InvalidOperation
    normalized = format(decimal.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def _utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
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
