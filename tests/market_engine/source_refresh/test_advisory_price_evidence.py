from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from market_engine.data.local_market_data_universe import UNIVERSE_SNAPSHOT_SCHEMA_VERSION
from market_engine.source_refresh import advisory_price_evidence as advisory
from market_engine.source_refresh.advisory_price_evidence import (
    AdvisoryPriceIssue,
    build_advisory_price_artifact,
    consume_advisory_price_context,
    load_advisory_price_artifact,
    validate_observations_payload,
)


RETRIEVED_AT = "2026-08-13T05:30:00Z"
TRUSTED_NOW = "2026-08-13T06:00:00Z"
SOURCE_MAIN_SHA = "a" * 40


def _instrument(instrument_id: str = "equity:aaa", ticker: str = "AAA") -> dict[str, object]:
    return {
        "instrument_id": instrument_id,
        "symbol": ticker,
        "source_symbol": ticker,
        "source_mapping_status": "mapped",
        "currency": "USD",
        "exchange": "US",
        "country": "US",
        "asset_type": "equity",
    }


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    universe = {
        "schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "universe_version": "synthetic-v1",
        "instruments": [_instrument(), _instrument("equity:bbb", "BBB")],
    }
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    policy = json.loads(advisory.DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return {"universe": universe_path, "policy": policy_path, "output": tmp_path / "artifacts"}


def _observation(instrument: dict[str, object], **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "instrument_id": instrument["instrument_id"],
        "canonical_ticker": instrument["symbol"],
        "price": "123.4500",
        "currency": "USD",
        "observation_type": advisory.OBSERVATION_TYPE,
        "observation_timestamp": "2026-08-12T20:00:00Z",
        "source_id": advisory.SOURCE_ID,
    }
    value.update(overrides)
    return value


def _provider(overrides: dict[str, dict[str, object]] | None = None):
    def provider(instruments, _retrieval):
        result = {str(row["instrument_id"]): _observation(dict(row)) for row in instruments}
        for key, values in (overrides or {}).items():
            if values.get("__missing__"):
                result.pop(key, None)
            else:
                result[key].update(values)
        return result

    return provider


def _build(inputs: dict[str, Path], *, provider=None, run_id: str = "me-sr25-synthetic"):
    return build_advisory_price_artifact(
        run_id=run_id,
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=inputs["output"],
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        retrieval_timestamp=RETRIEVED_AT,
        provider=provider or _provider(),
    )


def _assert_code(code: str, function, *args, **kwargs) -> None:
    with pytest.raises(AdvisoryPriceIssue) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def _rewrite_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _rehash_observations(root: Path) -> None:
    observations_path = root / advisory.OBSERVATIONS_FILE
    manifest_path = root / advisory.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["observations_sha256"] = advisory._sha256_file(observations_path)
    integrity = dict(manifest)
    integrity.pop("artifact_sha256")
    manifest["artifact_sha256"] = advisory._sha256(advisory._canonical_json(integrity))
    manifest_path.write_bytes(advisory._canonical_json(manifest) + b"\n")


def test_valid_price_observation_roundtrip_and_exact_decimal(inputs: dict[str, Path]) -> None:
    manifest, root = _build(inputs)
    loaded = load_advisory_price_artifact(
        root, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW
    )
    assert manifest == loaded["manifest"]
    assert loaded["observations"]["records"][0]["price"] == "123.4500"
    assert manifest["status_counts"] == {"attempted": 2, "fresh": 2, "stale": 0, "missing": 0, "invalid": 0}
    assert manifest["canonical_publication_status"] == "not_authorized_advisory_only"


@pytest.mark.parametrize("price", ["0", "0.0", "-1", 1.25, True, "NaN", "Infinity", "1e2"])
def test_invalid_price_domains_are_per_ticker_invalid(inputs: dict[str, Path], price: object) -> None:
    manifest, root = _build(inputs, provider=_provider({"equity:aaa": {"price": price}}))
    assert manifest["status_counts"] == {"attempted": 2, "fresh": 1, "stale": 0, "missing": 0, "invalid": 1}
    rows = load_advisory_price_artifact(
        root, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW
    )["observations"]["records"]
    invalid = next(row for row in rows if row["instrument_id"] == "equity:aaa")
    assert invalid["error_code"] == "PRICE_INVALID"
    assert invalid["price"] is None


def test_invalid_currency_is_fail_closed_per_ticker(inputs: dict[str, Path]) -> None:
    manifest, _root = _build(inputs, provider=_provider({"equity:aaa": {"currency": "XYZ"}}))
    assert manifest["status_counts"]["invalid"] == 1


@pytest.mark.parametrize(
    ("changes", "error_code"),
    [
        ({"instrument_id": "equity:unknown"}, "INSTRUMENT_IDENTITY_MISMATCH"),
        ({"canonical_ticker": "ALIAS"}, "INSTRUMENT_IDENTITY_MISMATCH"),
        ({"source_id": "unapproved-source"}, "SOURCE_ID_INVALID"),
        ({"observation_type": "live_price"}, "OBSERVATION_TYPE_INVALID"),
    ],
)
def test_unknown_identity_alias_source_and_price_type_fail_closed(
    inputs: dict[str, Path], changes: dict[str, object], error_code: str
) -> None:
    _manifest, root = _build(inputs, provider=_provider({"equity:aaa": changes}))
    records = load_advisory_price_artifact(
        root, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW
    )["observations"]["records"]
    assert next(row for row in records if row["instrument_id"] == "equity:aaa")["error_code"] == error_code


@pytest.mark.parametrize(
    "timestamp",
    ["2026-08-12T20:00:00", "2026-08-13T06:00:00Z", "2999-01-01T00:00:00Z"],
)
def test_naive_observation_after_retrieval_and_future_observation_are_invalid(
    inputs: dict[str, Path], timestamp: str
) -> None:
    manifest, _root = _build(inputs, provider=_provider({"equity:aaa": {"observation_timestamp": timestamp}}))
    assert manifest["status_counts"]["invalid"] == 1


def test_future_retrieval_timestamp_is_rejected(inputs: dict[str, Path]) -> None:
    _assert_code(
        "FUTURE_RETRIEVAL_TIMESTAMP",
        build_advisory_price_artifact,
        run_id="future",
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=inputs["output"],
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        retrieval_timestamp="2999-01-01T00:00:00Z",
        provider=_provider(),
    )


def test_naive_retrieval_timestamp_is_rejected(inputs: dict[str, Path]) -> None:
    _assert_code(
        "TIMESTAMP_INVALID",
        build_advisory_price_artifact,
        run_id="naive-retrieval",
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=inputs["output"],
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        retrieval_timestamp="2026-08-13T05:30:00",
        provider=_provider(),
    )


def test_fresh_stale_missing_and_invalid_are_explicit(inputs: dict[str, Path]) -> None:
    universe = json.loads(inputs["universe"].read_text(encoding="utf-8"))
    universe["instruments"].extend([_instrument("equity:ccc", "CCC"), _instrument("equity:ddd", "DDD")])
    inputs["universe"].write_text(json.dumps(universe), encoding="utf-8")
    provider = _provider(
        {
            "equity:bbb": {"observation_timestamp": "2026-08-11T20:00:00Z"},
            "equity:ccc": {"__missing__": True},
            "equity:ddd": {"price": "-1"},
        }
    )
    manifest, _root = _build(inputs, provider=provider)
    assert manifest["status_counts"] == {"attempted": 4, "fresh": 1, "stale": 1, "missing": 1, "invalid": 1}


def test_partial_failure_does_not_discard_valid_ticker(inputs: dict[str, Path]) -> None:
    manifest, root = _build(inputs, provider=_provider({"equity:bbb": {"__missing__": True}}))
    assert manifest["status_counts"]["fresh"] == 1
    assert manifest["status_counts"]["missing"] == 1
    context = consume_advisory_price_context(
        root, instrument_id="equity:aaa", canonical_ticker="AAA",
        universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW,
    )
    assert context["current_price"] == "123.4500"


def test_existing_adapter_uses_bounded_single_ticker_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruments = [_instrument(), _instrument("equity:bbb", "BBB")]
    attempts: list[str] = []
    monkeypatch.setattr(advisory, "download_yfinance_batch", lambda *_args: {})

    def single(symbol: str, _start: str, _end: str) -> pd.DataFrame:
        attempts.append(symbol)
        if symbol == "AAA":
            return pd.DataFrame(
                [{"Date": "2026-08-12", "Close": "123.45"}]
            )
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(advisory, "_download_yfinance_history", single)
    acquired = advisory._acquire_with_existing_adapter(
        instruments,
        advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp"),
    )
    assert acquired["equity:aaa"]["price"] == "123.45"
    assert acquired["equity:bbb"]["acquisition_error_code"] == "ACQUISITION_FAILED"
    assert attempts.count("AAA") == 1
    assert attempts.count("BBB") == 2


def test_existing_adapter_fails_closed_for_unmapped_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = _instrument()
    instrument["source_mapping_status"] = "unmapped"
    monkeypatch.setattr(advisory, "download_yfinance_batch", lambda *_args: pytest.fail("provider must not be called"))
    acquired = advisory._acquire_with_existing_adapter(
        [instrument],
        advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp"),
    )
    assert acquired["equity:aaa"]["acquisition_error_code"] == "SOURCE_MAPPING_UNAUTHORIZED"


def test_existing_adapter_isolates_invalid_provider_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruments = [_instrument(), _instrument("equity:bbb", "BBB")]
    frames = {
        "AAA": pd.DataFrame([{"Date": "2026-08-12", "Close": "123.45"}]),
        "BBB": pd.DataFrame([{"Date": "2026-08-12", "Close": "NaN"}]),
    }
    monkeypatch.setattr(advisory, "download_yfinance_batch", lambda *_args: frames)
    acquired = advisory._acquire_with_existing_adapter(
        instruments,
        advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp"),
    )
    assert acquired["equity:aaa"]["price"] == "123.45"
    assert acquired["equity:bbb"]["acquisition_error_code"] == "PRICE_INVALID"


def test_existing_adapter_rejects_ambiguous_provider_alias_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruments = [_instrument(), _instrument("equity:bbb", "BBB")]
    instruments[1]["source_symbol"] = "AAA"
    monkeypatch.setattr(advisory, "download_yfinance_batch", lambda *_args: pytest.fail("provider must not be called"))
    acquired = advisory._acquire_with_existing_adapter(
        instruments,
        advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp"),
    )
    assert {row["acquisition_error_code"] for row in acquired.values()} == {"SOURCE_MAPPING_AMBIGUOUS"}


def test_total_provider_failure_becomes_per_ticker_invalid(inputs: dict[str, Path]) -> None:
    def failed_provider(_instruments, _retrieval):
        raise TimeoutError("synthetic total failure")

    manifest, _root = _build(inputs, provider=failed_provider)
    assert manifest["status_counts"] == {
        "attempted": 2,
        "fresh": 0,
        "stale": 0,
        "missing": 0,
        "invalid": 2,
    }


def test_full_952_instrument_canonical_universe_is_processed_and_reconciled(
    tmp_path: Path,
) -> None:
    retrieval = advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp")

    def provider(instruments, _retrieval):
        result = {}
        for instrument in instruments:
            profile, expected = advisory.expected_completed_session(instrument, retrieval)
            if profile is None or expected is None:
                result[str(instrument["instrument_id"])] = {
                    "acquisition_error_code": "EXPECTED_SESSION_UNAVAILABLE",
                    "acquisition_error_detail": "Synthetic unsupported market profile.",
                }
                continue
            result[str(instrument["instrument_id"])] = _observation(
                dict(instrument),
                observation_timestamp=advisory._completed_session_close_timestamp(
                    instrument, expected
                ),
                currency=instrument["currency"],
            )
        return result

    manifest, root = build_advisory_price_artifact(
        run_id="me-sr25-full-universe-synthetic",
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=tmp_path,
        retrieval_timestamp=RETRIEVED_AT,
        provider=provider,
    )
    assert manifest["expected_instrument_count"] == 952
    assert manifest["status_counts"]["attempted"] == 952
    assert sum(
        manifest["status_counts"][key]
        for key in ("fresh", "stale", "missing", "invalid")
    ) == 952
    loaded = load_advisory_price_artifact(root, trusted_now=TRUSTED_NOW)
    assert len(loaded["observations"]["records"]) == 952


def test_serialization_and_order_are_deterministic(inputs: dict[str, Path]) -> None:
    first_manifest, first = _build(inputs, run_id="deterministic-a")
    second_manifest, second = _build(inputs, run_id="deterministic-b")
    first_records = json.loads((first / advisory.OBSERVATIONS_FILE).read_text(encoding="utf-8"))["records"]
    second_records = json.loads((second / advisory.OBSERVATIONS_FILE).read_text(encoding="utf-8"))["records"]
    assert [row["instrument_id"] for row in first_records] == ["equity:aaa", "equity:bbb"]
    assert [dict(row, run_id="same") for row in first_records] == [dict(row, run_id="same") for row in second_records]
    assert first_manifest["observations_sha256"] != second_manifest["observations_sha256"]


def test_extra_provider_identity_is_rejected(inputs: dict[str, Path]) -> None:
    def provider(instruments, retrieval):
        values = dict(_provider()(instruments, retrieval))
        values["equity:extra"] = _observation(_instrument("equity:extra", "EXTRA"))
        return values

    _assert_code("UNEXPECTED_INSTRUMENT", _build, inputs, provider=provider)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra"])
def test_duplicate_missing_and_extra_artifact_records_are_rejected(inputs: dict[str, Path], mutation: str) -> None:
    _manifest, root = _build(inputs)
    observations = root / advisory.OBSERVATIONS_FILE
    def mutate(payload):
        if mutation == "duplicate":
            payload["records"].append(deepcopy(payload["records"][0]))
        elif mutation == "missing":
            payload["records"].pop()
        else:
            extra = deepcopy(payload["records"][0])
            extra["instrument_id"] = "equity:extra"
            extra["canonical_ticker"] = "EXTRA"
            payload["records"].append(extra)
    _rewrite_json(observations, mutate)
    if mutation != "duplicate":
        _rehash_observations(root)
    expected = {
        "duplicate": "DUPLICATE_INSTRUMENT",
        "missing": "MISSING_INSTRUMENT",
        "extra": "UNEXPECTED_INSTRUMENT",
    }
    _assert_code(
        expected[mutation],
        load_advisory_price_artifact,
        root,
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now=TRUSTED_NOW,
    )


def test_tampered_observation_and_manifest_are_detected(inputs: dict[str, Path]) -> None:
    _manifest, root = _build(inputs)
    _rewrite_json(root / advisory.OBSERVATIONS_FILE, lambda payload: payload["records"][0].update(price="999"))
    _assert_code("ARTIFACT_INTEGRITY_INVALID", load_advisory_price_artifact, root, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW)

    inputs["output"] = inputs["output"].parent / "second"
    _manifest, second = _build(inputs)
    _rewrite_json(second / advisory.MANIFEST_FILE, lambda payload: payload["status_counts"].update(fresh=1, stale=1))
    _assert_code("ARTIFACT_INTEGRITY_INVALID", load_advisory_price_artifact, second, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW)


def test_rehashed_forged_freshness_is_recomputed_and_rejected(inputs: dict[str, Path]) -> None:
    _manifest, root = _build(inputs)
    observations_path = root / advisory.OBSERVATIONS_FILE
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    observations["records"][0]["observation_timestamp"] = "2026-08-11T20:00:00Z"
    observations_path.write_bytes(advisory._canonical_json(observations) + b"\n")
    _rehash_observations(root)
    _assert_code(
        "FRESHNESS_INVALID",
        load_advisory_price_artifact,
        root,
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now=TRUSTED_NOW,
    )


def test_forged_caller_price_mapping_is_rejected(inputs: dict[str, Path]) -> None:
    _assert_code(
        "FORGED_CALLER_CONTEXT", consume_advisory_price_context,
        {"price": "999999"}, instrument_id="equity:aaa", canonical_ticker="AAA",
        universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW,
    )


def test_wrong_schema_and_artifact_versions_are_rejected(inputs: dict[str, Path]) -> None:
    _manifest, root = _build(inputs)
    _rewrite_json(root / advisory.OBSERVATIONS_FILE, lambda payload: payload.update(schema_version="unknown"))
    _assert_code("ARTIFACT_VERSION_INVALID", load_advisory_price_artifact, root, universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW)


def test_consumer_preserves_fresh_stale_missing_and_invalid_states(inputs: dict[str, Path]) -> None:
    for suffix, override, expected in (
        ("fresh", {}, "fresh"),
        ("stale", {"observation_timestamp": "2026-08-11T20:00:00Z"}, "stale"),
        ("missing", {"__missing__": True}, "missing"),
        ("invalid", {"price": "-1"}, "invalid"),
    ):
        local = dict(inputs)
        local["output"] = inputs["output"].parent / suffix
        _manifest, root = _build(local, provider=_provider({"equity:aaa": override}), run_id=f"state-{suffix}")
        context = consume_advisory_price_context(
            root, instrument_id="equity:aaa", canonical_ticker="AAA",
            universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW,
        )
        assert context["price_context_status"] == expected
        assert (context["current_price"] is not None) is (expected == "fresh")
        assert context["currency"] == "USD"
        assert context["source_id"] == advisory.SOURCE_ID
        assert context["observation_type"] == advisory.OBSERVATION_TYPE
        assert context["advisory_only"] is True


def test_wrong_consumer_identity_fails_closed(inputs: dict[str, Path]) -> None:
    _manifest, root = _build(inputs)
    _assert_code(
        "INSTRUMENT_IDENTITY_MISMATCH", consume_advisory_price_context,
        root, instrument_id="equity:aaa", canonical_ticker="ALIAS",
        universe_path=inputs["universe"], policy_path=inputs["policy"], trusted_now=TRUSTED_NOW,
    )


def test_artifact_has_no_market_data_portfolio_order_notification_or_publication_side_effects() -> None:
    source = Path(advisory.__file__).read_text(encoding="utf-8").lower()
    for marker in (
        "git push", "market-data", "data/portfolio", "manual_transaction_ledger",
        "telegram", "send_notification", "place_order", "broker_api", "decision_engine",
    ):
        assert marker not in source


def test_workflow_is_advisory_only_scheduled_and_retained() -> None:
    repository = Path(__file__).parents[3]
    workflow = (repository / ".github/workflows/advisory-price-evidence.yml").read_text(encoding="utf-8")
    assert 'cron: "30 5 * * *"' in workflow
    assert "market_engine.source_refresh.advisory_price_evidence build" in workflow
    assert "retention-days: 14" in workflow
    for forbidden in ("contents: write", "git push", "market-data", "publish"):
        assert forbidden not in workflow.lower()


def test_json_schemas_validate_real_valid_and_invalid_payloads(inputs: dict[str, Path]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    _manifest, root = _build(inputs)
    repository = Path(__file__).parents[3]
    observation_schema = json.loads((repository / "config/market_engine/advisory_price/advisory_price_observations_v1.schema.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((repository / "config/market_engine/advisory_price/advisory_price_manifest_v1.schema.json").read_text(encoding="utf-8"))
    observations = json.loads((root / advisory.OBSERVATIONS_FILE).read_text(encoding="utf-8"))
    manifest = json.loads((root / advisory.MANIFEST_FILE).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(observation_schema, format_checker=jsonschema.FormatChecker()).validate(observations)
    jsonschema.Draft202012Validator(manifest_schema, format_checker=jsonschema.FormatChecker()).validate(manifest)
    invalid = deepcopy(observations)
    invalid["records"][0]["price"] = "-1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(observation_schema).validate(invalid)


def test_semantically_modified_payload_never_reaches_consumer(inputs: dict[str, Path]) -> None:
    _manifest, root = _build(inputs)
    payload = json.loads((root / advisory.OBSERVATIONS_FILE).read_text(encoding="utf-8"))
    payload["records"][0]["retrieval_timestamp"] = "2026-08-12T19:00:00Z"
    _assert_code("OBSERVATION_AFTER_RETRIEVAL", validate_observations_payload, payload, trusted_now=advisory._timestamp(TRUSTED_NOW, "trusted_now"))
