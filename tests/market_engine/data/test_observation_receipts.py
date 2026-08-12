from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from market_engine.data import observation_receipts as receipts
from market_engine.data.provider_artifact_adapter import RegisteredMarketPriceAdapter


def _policy(tmp_path: Path, **overrides: object) -> tuple[Path, dict[str, object]]:
    provider = {
        "provider_id": "approved-test-market-data",
        "approval_id": "approval-test-daily-ohlcv-v1",
        "data_type": "daily_ohlcv",
        "adapter_id": "test-http-adapter",
        "adapter_version": "v1",
        "parser_name": receipts.PARSER_NAME,
        "parser_version": receipts.PARSER_VERSION,
        "approved_for_acquisition": True,
        "approved_for_retention": True,
        "approved_for_replay": True,
        "approved_for_canonical_publication": True,
        "acquisition_routes": ["primary", "primary_replay", "fallback"],
        "exchanges": ["NYSE", "NASDAQ"],
        "retention_classification": "immutable_test_evidence",
        "redistribution_classification": "test_only",
        **overrides,
    }
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {"schema_version": receipts.POLICY_SCHEMA_VERSION, "providers": [provider]}
        ),
        encoding="utf-8",
    )
    return path, provider


def _payload(*sessions: str, close: str = "10.50") -> bytes:
    return json.dumps(
        {
            "bars": [
                {
                    "session_date": session,
                    "open": "10.10",
                    "high": "10.70",
                    "low": "10.00",
                    "close": close,
                    "adj_close": "10.50",
                    "volume": 1234,
                }
                for session in sessions
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _artifact(
    tmp_path: Path,
    *,
    payload: bytes | None = None,
    instrument_id: str = "equity:aaa",
    ticker: str = "AAA",
    provider_symbol: str = "AAA",
    exchange: str = "NYSE",
    route: str = "primary_replay",
    start: str = "2026-07-23",
    end: str = "2026-07-25",
    pagination: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    policy_path, _ = _policy(tmp_path)
    policy = receipts.load_source_policy(policy_path)
    adapter = RegisteredMarketPriceAdapter(
        policy=policy,
        provider_id="approved-test-market-data",
        acquisition_route=route,
        instrument={
            "instrument_id": instrument_id,
            "symbol": ticker,
            "source_symbol": provider_symbol,
            "exchange": exchange,
            "currency": "USD",
            "source_mapping_status": "mapped",
        },
    )
    request = adapter.request(
        method_id="daily-bars",
        start=start,
        end_exclusive=end,
        timezone="America/New_York",
        pagination=pagination or {"page": 1, "terminal": True},
    )
    reference = adapter.capture_response(
        payload if payload is not None else _payload("2026-07-23", "2026-07-24"),
        request=request,
        artifact_root=tmp_path,
        acquisition_run_id="test-acquisition-run",
        retrieved_at="2026-08-09T12:00:00Z",
        response_status=200,
        response_content_type="application/json",
        provider_request_id="request-123",
    )
    return policy, reference


def _mutation(observation: dict[str, object], mutation_type: str) -> dict[str, object]:
    return {
        "instrument_id": observation["instrument_id"],
        "ticker": observation["ticker"],
        "exchange": observation["exchange"],
        "session_date": observation["session_date"],
        "mutation_type": mutation_type,
        "new_canonical_row_sha256": observation["canonical_row_sha256"],
    }


def test_adapter_envelope_replays_before_publisher_selects_overlap(tmp_path: Path) -> None:
    policy, reference = _artifact(tmp_path)
    replayed = receipts.replay_provider_artifacts(
        [reference], artifact_root=tmp_path, policy=policy
    )
    selected = receipts.select_mutation_observations(
        [_mutation(replayed[0], "row_unchanged"), _mutation(replayed[1], "row_added")],
        replayed,
    )

    assert [row["session_date"] for row in replayed] == ["2026-07-23", "2026-07-24"]
    assert [row["session_date"] for row in selected["accepted_mutation_observations"]] == ["2026-07-24"]
    assert [row["session_date"] for row in selected["unchanged_overlap_observations"]] == ["2026-07-23"]
    assert [row["session_date"] for row in selected["mutation_receipts"]] == ["2026-07-24"]
    assert selected["mutation_receipts"][0]["artifact_sha256"] == reference["artifact_sha256"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"approved_for_acquisition": False},
        {"approved_for_retention": False},
        {"approved_for_replay": False},
        {"approved_for_canonical_publication": False},
    ],
)
def test_every_source_policy_right_is_required(tmp_path: Path, overrides: dict[str, object]) -> None:
    path, _ = _policy(tmp_path, **overrides)
    policy = receipts.load_source_policy(path)
    with pytest.raises(receipts.ObservationReceiptError, match="not approved"):
        receipts.approved_source_policy(
            policy,
            provider_id="approved-test-market-data",
            exchange="NYSE",
            acquisition_route="primary",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrument_id", "equity:bbb"),
        ("canonical_ticker", "BBB"),
        ("provider_symbol", "BBB"),
        ("exchange", "NASDAQ"),
        ("request_start", "2026-07-22"),
        ("request_end_exclusive", "2026-07-26"),
        ("window_semantics", "inclusive"),
        ("timezone", "UTC"),
        ("pagination", {"page": 2}),
        ("provider_id", "other-provider"),
        ("adapter_version", "unknown"),
        ("parser_version", "unknown"),
        ("source_policy_id", "other-approval"),
    ],
)
def test_downstream_envelope_relabelling_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    policy, reference = _artifact(tmp_path)
    path = tmp_path / reference["artifact_locator"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope[field] = value
    unsigned = {key: nested for key, nested in envelope.items() if key != "envelope_sha256"}
    envelope["envelope_sha256"] = receipts.sha256_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    )
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(receipts.ObservationReceiptError, match="checksum"):
        receipts.replay_provider_artifacts([reference], artifact_root=tmp_path, policy=policy)


def test_complete_downstream_relabelling_cannot_replace_trusted_acquisition_identity(
    tmp_path: Path,
) -> None:
    policy, reference = _artifact(tmp_path)
    original = tmp_path / reference["artifact_locator"]
    envelope = json.loads(original.read_text(encoding="utf-8"))
    envelope["instrument_id"] = "equity:bbb"
    envelope["canonical_ticker"] = "BBB"
    envelope["provider_symbol"] = "BBB"
    envelope["request_parameters"]["symbol"] = "BBB"
    request_identity = {
        key: envelope[key]
        for key in (
            "request_method_id", "request_parameters", "request_start",
            "request_end_exclusive", "window_semantics", "timezone", "pagination",
        )
    }
    envelope["request_sha256"] = receipts.sha256_bytes(
        json.dumps(request_identity, sort_keys=True, separators=(",", ":")).encode()
    )
    unsigned = {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    envelope["envelope_sha256"] = receipts.sha256_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    )
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    artifact_sha = receipts.sha256_bytes(encoded)
    relabelled = original.with_name(f"{artifact_sha}.json")
    relabelled.write_bytes(encoded)
    rebuilt_reference = {
        **reference,
        "artifact_locator": relabelled.relative_to(tmp_path).as_posix(),
        "artifact_sha256": artifact_sha,
        "envelope_sha256": envelope["envelope_sha256"],
    }

    with pytest.raises(
        receipts.ObservationReceiptError,
        match="trusted acquisition identity mismatch",
    ):
        receipts.replay_provider_artifacts(
            [rebuilt_reference], artifact_root=tmp_path, policy=policy
        )


def test_artifact_without_trusted_acquisition_run_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    policy, reference = _artifact(tmp_path)
    unregistered = {
        **reference,
        "acquisition_manifest_locator": (
            "evidence/market_price/acquisition_runs/test-acquisition-run/"
            f"{'0' * 64}.json"
        ),
        "acquisition_manifest_sha256": "0" * 64,
    }
    with pytest.raises(receipts.ObservationReceiptError, match="absent from trusted"):
        receipts.replay_provider_artifacts(
            [unregistered], artifact_root=tmp_path, policy=policy
        )


def test_receipt_cannot_relabel_independently_replayed_identity(tmp_path: Path) -> None:
    policy, reference = _artifact(tmp_path)
    observation = receipts.replay_provider_artifacts(
        [reference], artifact_root=tmp_path, policy=policy
    )[0]
    selected = receipts.select_mutation_observations(
        [_mutation(observation, "row_added")], [observation]
    )
    declaration = copy.deepcopy(selected["mutation_receipts"])
    declaration[0]["ticker"] = "BBB"
    with pytest.raises(receipts.ObservationReceiptError, match="declared receipts"):
        receipts.validate_declared_receipts(declaration, selected["mutation_receipts"])


def test_raw_response_and_envelope_mutation_fail_replay(tmp_path: Path) -> None:
    policy, reference = _artifact(tmp_path)
    path = tmp_path / reference["artifact_locator"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(receipts.ObservationReceiptError, match="checksum"):
        receipts.replay_provider_artifacts([reference], artifact_root=tmp_path, policy=policy)
    path.write_bytes(original)
    envelope = json.loads(original)
    envelope["raw_response_sha256"] = "0" * 64
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    changed_reference = {**reference, "artifact_sha256": receipts.sha256_bytes(encoded)}
    changed_path = path.with_name(f"{changed_reference['artifact_sha256']}.json")
    path.rename(changed_path)
    changed_reference["artifact_locator"] = changed_path.relative_to(tmp_path).as_posix()
    with pytest.raises(receipts.ObservationReceiptError, match="envelope digest"):
        receipts.replay_provider_artifacts(
            [changed_reference], artifact_root=tmp_path, policy=policy
        )


def test_adapter_rejects_credentials_before_storage(tmp_path: Path) -> None:
    policy_path, _ = _policy(tmp_path)
    policy = receipts.load_source_policy(policy_path)
    with pytest.raises(receipts.ObservationReceiptError, match="credential"):
        adapter = RegisteredMarketPriceAdapter(
            policy=policy,
            provider_id="approved-test-market-data",
            acquisition_route="primary",
            instrument={"instrument_id": "equity:aaa", "symbol": "AAA", "source_symbol": "AAA", "exchange": "NYSE", "currency": "USD", "source_mapping_status": "mapped"},
        )
        request = adapter.request(
            method_id="daily-bars", start="2026-07-24", end_exclusive="2026-07-25",
            timezone="America/New_York",
            pagination={},
        )
        adapter.capture_response(
            b'{"api_key":"must-not-be-stored","bars":[]}',
            request=request,
            artifact_root=tmp_path,
            acquisition_run_id="credential-test-run",
            retrieved_at="2026-08-09T12:00:00Z",
            response_status=200,
            response_content_type="application/json",
        )
    assert not list((tmp_path / "evidence").rglob("*.json"))


def test_registered_adapter_rejects_unapproved_alias_and_request_symbol_override(
    tmp_path: Path,
) -> None:
    policy_path, _ = _policy(tmp_path)
    policy = receipts.load_source_policy(policy_path)
    with pytest.raises(receipts.ObservationReceiptError, match="mapping is not approved"):
        RegisteredMarketPriceAdapter(
            policy=policy,
            provider_id="approved-test-market-data",
            acquisition_route="primary",
            instrument={
                "instrument_id": "equity:legacy", "symbol": "LEGACY",
                "source_symbol": "LEGACY-A", "exchange": "NYSE",
                "currency": "USD", "source_mapping_status": "unsupported",
            },
        )
    adapter = RegisteredMarketPriceAdapter(
        policy=policy,
        provider_id="approved-test-market-data",
        acquisition_route="primary",
        instrument={
            "instrument_id": "equity:aaa", "symbol": "AAA",
            "source_symbol": "AAA", "exchange": "NYSE", "currency": "USD",
            "source_mapping_status": "mapped",
        },
    )
    with pytest.raises(receipts.ObservationReceiptError, match="registered mapping"):
        adapter.request(
            method_id="daily-bars",
            start="2026-07-24",
            end_exclusive="2026-07-25",
            timezone="America/New_York",
            pagination={"page": 1},
            extra_parameters={"symbol": "BBB"},
        )


def test_absence_attestation_is_bound_to_empty_adapter_envelope(tmp_path: Path) -> None:
    policy, reference = _artifact(
        tmp_path, payload=_payload(), start="2026-07-24", end="2026-07-25"
    )
    attestation = receipts.build_absence_attestation(
        artifact_reference=reference,
        artifact_root=tmp_path,
        policy=policy,
        session_date="2026-07-24",
        lifecycle_cutoff="2026-07-24",
        reason_code="terminal_daily_ohlcv_not_returned",
        calendar_expected=True,
    )
    assert receipts.replay_absence_attestations(
        [attestation], artifact_root=tmp_path, policy=policy
    ) == [attestation]
    changed = {**attestation, "instrument_id": "equity:bbb"}
    with pytest.raises(receipts.ObservationReceiptError, match="does not replay"):
        receipts.replay_absence_attestations(
            [changed], artifact_root=tmp_path, policy=policy
        )


@pytest.mark.parametrize("consumer_ticker", ["AAA", "TMHC"])
def test_valid_b_absence_evidence_cannot_explain_another_consumer(
    tmp_path: Path, consumer_ticker: str
) -> None:
    policy, reference = _artifact(
        tmp_path,
        payload=_payload(),
        instrument_id="equity:bbb",
        ticker="BBB",
        provider_symbol="BBB",
        start="2026-07-24",
        end="2026-07-25",
    )
    attestation = receipts.build_absence_attestation(
        artifact_reference=reference,
        artifact_root=tmp_path,
        policy=policy,
        session_date="2026-07-24",
        lifecycle_cutoff="2026-07-24",
        reason_code="terminal_daily_ohlcv_not_returned",
        calendar_expected=True,
    )
    replayed = receipts.replay_absence_attestations(
        [attestation], artifact_root=tmp_path, policy=policy
    )
    with pytest.raises(
        receipts.ObservationReceiptError,
        match="ABSENCE_EVIDENCE_CONSUMER_IDENTITY_MISMATCH",
    ):
        receipts.bind_absence_evidence_to_consumer(
            replayed,
            consumer_identity={
                "instrument_id": f"equity:{consumer_ticker.lower()}",
                "symbol": consumer_ticker,
                "source_symbol": consumer_ticker,
                "exchange": "NYSE",
                "provider_id": "approved-test-market-data",
                "source_policy_id": "approval-test-daily-ohlcv-v1",
                "acquisition_route": "primary_replay",
                "timezone": "America/New_York",
            },
            expected_sessions=["2026-07-24"],
            lifecycle_cutoff="2026-07-24",
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("symbol", "LEGACY", "IDENTITY"),
        ("source_symbol", "BBB-A", "IDENTITY"),
        ("exchange", "NASDAQ", "IDENTITY"),
        ("provider_id", "other-provider", "IDENTITY"),
        ("source_policy_id", "other-policy", "IDENTITY"),
        ("acquisition_route", "fallback", "IDENTITY"),
        ("timezone", "UTC", "IDENTITY"),
    ],
)
def test_absence_consumer_route_fields_are_independently_bound(
    tmp_path: Path, field: str, value: str, reason: str
) -> None:
    policy, reference = _artifact(
        tmp_path, payload=_payload(), start="2026-07-24", end="2026-07-25"
    )
    attestation = receipts.build_absence_attestation(
        artifact_reference=reference,
        artifact_root=tmp_path,
        policy=policy,
        session_date="2026-07-24",
        lifecycle_cutoff="2026-07-24",
        reason_code="terminal_daily_ohlcv_not_returned",
        calendar_expected=True,
    )
    consumer = {
        "instrument_id": "equity:aaa",
        "symbol": "AAA",
        "source_symbol": "AAA",
        "exchange": "NYSE",
        "provider_id": "approved-test-market-data",
        "source_policy_id": "approval-test-daily-ohlcv-v1",
        "acquisition_route": "primary_replay",
        "timezone": "America/New_York",
        field: value,
    }
    with pytest.raises(receipts.ObservationReceiptError, match=f"CONSUMER_{reason}_MISMATCH"):
        receipts.bind_absence_evidence_to_consumer(
            [attestation],
            consumer_identity=consumer,
            expected_sessions=["2026-07-24"],
            lifecycle_cutoff="2026-07-24",
        )


@pytest.mark.parametrize(
    ("expected_sessions", "cutoff"),
    [([], "2026-07-24"), (["2026-07-24"], "2026-07-23")],
)
def test_absence_consumer_session_and_cutoff_are_bound(
    tmp_path: Path, expected_sessions: list[str], cutoff: str
) -> None:
    policy, reference = _artifact(
        tmp_path, payload=_payload(), start="2026-07-24", end="2026-07-25"
    )
    attestation = receipts.build_absence_attestation(
        artifact_reference=reference,
        artifact_root=tmp_path,
        policy=policy,
        session_date="2026-07-24",
        lifecycle_cutoff="2026-07-24",
        reason_code="terminal_daily_ohlcv_not_returned",
        calendar_expected=True,
    )
    with pytest.raises(receipts.ObservationReceiptError, match="CONSUMER_LIFECYCLE_MISMATCH"):
        receipts.bind_absence_evidence_to_consumer(
            [attestation],
            consumer_identity={
                "instrument_id": "equity:aaa", "symbol": "AAA",
                "source_symbol": "AAA", "exchange": "NYSE",
                "provider_id": "approved-test-market-data",
                "source_policy_id": "approval-test-daily-ohlcv-v1",
                "acquisition_route": "primary_replay",
                "timezone": "America/New_York",
            },
            expected_sessions=expected_sessions,
            lifecycle_cutoff=cutoff,
        )


def test_absence_rejects_present_session_and_wrong_cutoff(tmp_path: Path) -> None:
    policy, reference = _artifact(
        tmp_path, payload=_payload("2026-07-24"), start="2026-07-24", end="2026-07-25"
    )
    with pytest.raises(receipts.ObservationReceiptError, match="contains"):
        receipts.build_absence_attestation(
            artifact_reference=reference,
            artifact_root=tmp_path,
            policy=policy,
            session_date="2026-07-24",
            lifecycle_cutoff="2026-07-24",
            reason_code="terminal_daily_ohlcv_not_returned",
            calendar_expected=True,
        )
    empty_policy, empty_reference = _artifact(
        tmp_path, payload=_payload(), start="2026-07-24", end="2026-07-25"
    )
    with pytest.raises(receipts.ObservationReceiptError, match="boundary"):
        receipts.build_absence_attestation(
            artifact_reference=empty_reference,
            artifact_root=tmp_path,
            policy=empty_policy,
            session_date="2026-07-24",
            lifecycle_cutoff="2026-07-23",
            reason_code="terminal_daily_ohlcv_not_returned",
            calendar_expected=True,
        )


def test_duplicate_sessions_across_paginated_artifacts_fail(tmp_path: Path) -> None:
    policy, first = _artifact(
        tmp_path, payload=_payload("2026-07-24"), pagination={"page": 1}
    )
    _, second = _artifact(
        tmp_path, payload=_payload("2026-07-24"), pagination={"page": 2}
    )
    with pytest.raises(receipts.ObservationReceiptError, match="duplicate"):
        receipts.replay_provider_artifacts(
            [first, second], artifact_root=tmp_path, policy=policy
        )


def test_record_outside_bound_window_fails(tmp_path: Path) -> None:
    policy, reference = _artifact(
        tmp_path,
        payload=_payload("2026-07-22"),
        start="2026-07-23",
        end="2026-07-25",
    )
    with pytest.raises(receipts.ObservationReceiptError, match="outside"):
        receipts.replay_provider_artifacts(
            [reference], artifact_root=tmp_path, policy=policy
        )
