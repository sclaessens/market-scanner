from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from market_engine.data import observation_receipts as receipts
from market_engine.data.provider_artifact_adapter import capture_provider_artifact


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
    reference = capture_provider_artifact(
        payload if payload is not None else _payload("2026-07-23", "2026-07-24"),
        artifact_root=tmp_path,
        policy=policy,
        provider_id="approved-test-market-data",
        adapter_id="test-http-adapter",
        adapter_version="v1",
        instrument_id=instrument_id,
        canonical_ticker=ticker,
        provider_symbol=provider_symbol,
        exchange=exchange,
        currency="USD",
        acquisition_route=route,
        request_method_id="daily-bars",
        request_parameters={"symbol": provider_symbol, "start": start, "end": end},
        request_start=start,
        request_end_exclusive=end,
        timezone="America/New_York",
        pagination=pagination or {"page": 1, "terminal": True},
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
        capture_provider_artifact(
            b'{"api_key":"must-not-be-stored","bars":[]}',
            artifact_root=tmp_path,
            policy=policy,
            provider_id="approved-test-market-data",
            adapter_id="test-http-adapter",
            adapter_version="v1",
            instrument_id="equity:aaa",
            canonical_ticker="AAA",
            provider_symbol="AAA",
            exchange="NYSE",
            currency="USD",
            acquisition_route="primary",
            request_method_id="daily-bars",
            request_parameters={},
            request_start="2026-07-24",
            request_end_exclusive="2026-07-25",
            timezone="America/New_York",
            pagination={},
            retrieved_at="2026-08-09T12:00:00Z",
            response_status=200,
            response_content_type="application/json",
        )
    assert not list((tmp_path / "evidence").rglob("*.json"))


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
