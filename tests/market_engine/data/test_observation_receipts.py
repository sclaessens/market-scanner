from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from market_engine.data import observation_receipts as receipts


def _policy(tmp_path: Path, **overrides: object) -> tuple[Path, dict[str, object]]:
    provider = {
        "provider_id": "approved-test-market-data",
        "approval_id": "approval-test-daily-ohlcv-v1",
        "data_type": "daily_ohlcv",
        "approved_for_acquisition": True,
        "approved_for_raw_storage": True,
        "approved_for_canonical_publication": True,
        "exchanges": ["NYSE"],
        "retention_classification": "immutable_test_evidence",
        "redistribution_classification": "test_only",
        **overrides,
    }
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": receipts.POLICY_SCHEMA_VERSION,
                "providers": [provider],
            }
        ),
        encoding="utf-8",
    )
    return path, provider


def _payload(*, close: str = "10.50", volume: int = 1234) -> bytes:
    return json.dumps(
        {
            "bars": [
                {
                    "session_date": "2026-07-24",
                    "open": "10.10",
                    "high": "10.70",
                    "low": "10.00",
                    "close": close,
                    "adj_close": "10.50",
                    "volume": volume,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _receipt_fixture(tmp_path: Path) -> tuple[dict[str, object], bytes, list[dict[str, object]]]:
    policy_path, _ = _policy(tmp_path)
    policy = receipts.load_source_policy(policy_path)
    payload = _payload()
    artifact = receipts.preserve_raw_artifact(
        payload,
        artifact_root=tmp_path,
        provider_id="approved-test-market-data",
        content_type="application/json",
    )
    built = receipts.build_observation_receipts(
        payload,
        policy=policy,
        provider_id="approved-test-market-data",
        instrument_id="equity:aaa",
        ticker="AAA",
        exchange="NYSE",
        currency="USD",
        retrieved_at="2026-08-09T12:00:00Z",
        request_start="2026-07-24",
        request_end_exclusive="2026-07-25",
        raw_artifact_locator=artifact["raw_artifact_locator"],
        raw_artifact_sha256=artifact["raw_artifact_sha256"],
        response_status=200,
        content_type="application/json",
    )
    return policy, payload, built


def test_approved_raw_response_replays_exact_receipt(tmp_path: Path) -> None:
    policy, _, built = _receipt_fixture(tmp_path)

    replayed = receipts.replay_observation_receipts(
        built, artifact_root=tmp_path, policy=policy
    )

    assert replayed == built
    assert receipts.observation_receipt_root(replayed) == receipts.observation_receipt_root(built)
    assert built[0]["canonical_row_sha256"]
    assert built[0]["response_status"] == 200
    assert built[0]["content_type"] == "application/json"
    assert built[0]["retention_classification"] == "immutable_test_evidence"
    assert built[0]["redistribution_classification"] == "test_only"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"approved_for_acquisition": False}, "not approved"),
        ({"approved_for_raw_storage": False}, "not approved"),
        ({"approved_for_canonical_publication": False}, "not approved"),
        ({"approval_id": ""}, "incomplete"),
    ],
)
def test_non_publishable_source_policy_fails_closed(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    path, _ = _policy(tmp_path, **overrides)
    if overrides.get("approval_id") == "":
        with pytest.raises(receipts.ObservationReceiptError, match=message):
            receipts.load_source_policy(path)
        return
    policy = receipts.load_source_policy(path)
    with pytest.raises(receipts.ObservationReceiptError, match=message):
        receipts.approved_fallback_policy(
            policy,
            provider_id="approved-test-market-data",
            exchange="NYSE",
        )


def test_unknown_or_wrong_exchange_provider_fails_closed(tmp_path: Path) -> None:
    path, _ = _policy(tmp_path)
    policy = receipts.load_source_policy(path)
    with pytest.raises(receipts.ObservationReceiptError, match="unknown"):
        receipts.approved_fallback_policy(
            policy, provider_id="reachable-but-unapproved", exchange="NYSE"
        )
    with pytest.raises(receipts.ObservationReceiptError, match="exchange"):
        receipts.approved_fallback_policy(
            policy, provider_id="approved-test-market-data", exchange="NASDAQ"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(parser_version="unknown"), "parser"),
        (lambda row: row.update(ticker="BBB"), "does not replay"),
        (lambda row: row.update(session_date="2026-07-25"), "does not replay"),
        (lambda row: row.update(raw_artifact_sha256="0" * 64), "artifact"),
    ],
)
def test_receipt_mutations_fail_replay(
    tmp_path: Path, mutation, message: str
) -> None:
    policy, _, built = _receipt_fixture(tmp_path)
    changed = copy.deepcopy(built)
    mutation(changed[0])

    with pytest.raises(receipts.ObservationReceiptError, match=message):
        receipts.replay_observation_receipts(
            changed, artifact_root=tmp_path, policy=policy
        )


def test_missing_and_mutated_raw_artifacts_fail_replay(tmp_path: Path) -> None:
    policy, _, built = _receipt_fixture(tmp_path)
    raw = tmp_path / built[0]["raw_artifact_locator"]
    original = raw.read_bytes()
    raw.unlink()
    with pytest.raises(receipts.ObservationReceiptError, match="missing"):
        receipts.replay_observation_receipts(
            built, artifact_root=tmp_path, policy=policy
        )
    raw.write_bytes(original + b" ")
    with pytest.raises(receipts.ObservationReceiptError, match="checksum"):
        receipts.replay_observation_receipts(
            built, artifact_root=tmp_path, policy=policy
        )


def test_receipt_numeric_serialization_and_root_are_deterministic(tmp_path: Path) -> None:
    policy, _, built = _receipt_fixture(tmp_path)
    second_payload = _payload(close="10.5000", volume=1234)
    parsed = receipts.parse_raw_daily_ohlcv(second_payload)

    assert parsed[0]["close"] == "10.5"
    distinct_rows = [
        built[0],
        {**built[0], "canonical_row_sha256": "1" * 64},
    ]
    assert receipts.observation_receipt_root(
        distinct_rows
    ) == receipts.observation_receipt_root(list(reversed(distinct_rows)))
    with pytest.raises(receipts.ObservationReceiptError, match="duplicated"):
        receipts.observation_receipt_root([built[0], built[0]])
    assert policy["policy_checksum"]


def test_raw_artifact_rejects_credential_material(tmp_path: Path) -> None:
    payload = json.dumps({"api_key": "must-not-be-stored", "bars": []}).encode()
    with pytest.raises(receipts.ObservationReceiptError, match="credential") as captured:
        receipts.preserve_raw_artifact(
            payload,
            artifact_root=tmp_path,
            provider_id="approved-test-market-data",
            content_type="application/json",
        )
    assert "must-not-be-stored" not in str(captured.value)
    assert not list(tmp_path.rglob("*.json"))
