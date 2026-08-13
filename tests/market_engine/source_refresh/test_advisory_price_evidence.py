from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
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


def _build_at(
    inputs: dict[str, Path],
    *,
    retrieval_timestamp: str,
    observation_timestamp: str,
    run_id: str,
):
    return build_advisory_price_artifact(
        run_id=run_id,
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=inputs["output"],
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        retrieval_timestamp=retrieval_timestamp,
        provider=_provider(
            {
                "equity:aaa": {"observation_timestamp": observation_timestamp},
                "equity:bbb": {"observation_timestamp": observation_timestamp},
            }
        ),
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


def test_default_build_timestamp_is_canonical_utc_and_validated(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(advisory, "_canonical_now", lambda: RETRIEVED_AT)
    manifest, root = build_advisory_price_artifact(
        run_id="default-build-time",
        source_main_sha=SOURCE_MAIN_SHA,
        output_root=inputs["output"],
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        provider=_provider(),
    )
    assert manifest["generated_at"] == RETRIEVED_AT
    assert manifest["generated_at"].endswith("Z")
    records = json.loads((root / advisory.OBSERVATIONS_FILE).read_text(encoding="utf-8"))["records"]
    assert {row["retrieval_timestamp"] for row in records} == {RETRIEVED_AT}
    assert advisory._timestamp(manifest["generated_at"], "generated_at").tzinfo is not None


def test_default_load_and_consume_times_are_canonical_and_safe(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest, root = _build(inputs)
    monkeypatch.setattr(advisory, "_canonical_now", lambda: TRUSTED_NOW)
    loaded = load_advisory_price_artifact(
        root,
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
    )
    assert advisory._utc_text(loaded["trusted_now"]) == TRUSTED_NOW
    context = consume_advisory_price_context(
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
    )
    assert context["evaluated_at"] == TRUSTED_NOW
    assert context["current_price"] == "123.4500"


def test_cli_build_and_consume_use_default_canonical_times(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(advisory, "_canonical_now", lambda: RETRIEVED_AT)
    monkeypatch.setattr(advisory, "_acquire_with_existing_adapter", _provider())
    build_stdout = StringIO()
    assert advisory.run_command(
        [
            "build",
            "--run-id", "cli-default-time",
            "--source-main-sha", SOURCE_MAIN_SHA,
            "--output-root", inputs["output"].as_posix(),
            "--universe", inputs["universe"].as_posix(),
            "--policy", inputs["policy"].as_posix(),
        ],
        stdout=build_stdout,
        stderr=StringIO(),
    ) == 0
    build_result = json.loads(build_stdout.getvalue())
    assert build_result["manifest"]["generated_at"] == RETRIEVED_AT

    monkeypatch.setattr(advisory, "_canonical_now", lambda: TRUSTED_NOW)
    consume_stdout = StringIO()
    assert advisory.run_command(
        [
            "consume",
            "--artifact-root", build_result["artifact_path"],
            "--instrument-id", "equity:aaa",
            "--ticker", "AAA",
            "--universe", inputs["universe"].as_posix(),
            "--policy", inputs["policy"].as_posix(),
        ],
        stdout=consume_stdout,
        stderr=StringIO(),
    ) == 0
    context = json.loads(consume_stdout.getvalue())
    assert context["evaluated_at"] == TRUSTED_NOW
    assert context["current_price"] == "123.4500"


def test_exact_workflow_build_command_path_needs_no_timestamp_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(advisory, "_canonical_now", lambda: RETRIEVED_AT)
    monkeypatch.setattr(advisory, "_acquire_with_existing_adapter", _provider())
    stdout = StringIO()
    assert advisory.run_command(
        [
            "build",
            "--run-id", "me-sr25-advisory-price-workflow-test",
            "--source-main-sha", SOURCE_MAIN_SHA,
            "--output-root", tmp_path.as_posix(),
        ],
        stdout=stdout,
        stderr=StringIO(),
    ) == 0
    result = json.loads(stdout.getvalue())
    assert result["manifest"]["generated_at"] == RETRIEVED_AT
    assert result["manifest"]["expected_instrument_count"] == 952


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
        assert context["schema_version"] == "market-engine-advisory-price-context-v2"
        assert context["artifact_freshness_status"] == expected
        assert context["effective_freshness_status"] == expected
        assert (context["current_price"] is not None) is (expected == "fresh")
        assert context["currency"] == "USD"
        assert context["source_id"] == advisory.SOURCE_ID
        assert context["observation_type"] == advisory.OBSERVATION_TYPE
        assert context["advisory_only"] is True


def test_consumption_freshness_changes_without_mutating_artifact(
    inputs: dict[str, Path],
) -> None:
    manifest, root = _build(inputs)
    manifest_bytes = (root / advisory.MANIFEST_FILE).read_bytes()
    observations_bytes = (root / advisory.OBSERVATIONS_FILE).read_bytes()

    still_fresh = consume_advisory_price_context(
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now="2026-08-13T19:00:00Z",
    )
    assert still_fresh["price_context_status"] == "fresh"
    assert still_fresh["effective_freshness_status"] == "fresh"
    assert still_fresh["effective_observation_age_completed_sessions"] == 0
    assert still_fresh["artifact_freshness_status"] == "fresh"
    assert still_fresh["artifact_observation_age_completed_sessions"] == 0
    assert still_fresh["current_price"] == "123.4500"

    loaded_after_next_session = load_advisory_price_artifact(
        root,
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now="2026-08-13T22:00:00Z",
    )
    assert loaded_after_next_session["effective_freshness"]["equity:aaa"] == {
        "status": "stale",
        "observation_age_completed_sessions": 1,
        "error_code": None,
    }
    assert loaded_after_next_session["manifest"]["status_counts"] == manifest["status_counts"]

    after_next_session = consume_advisory_price_context(
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now="2026-08-13T22:00:00Z",
    )
    assert after_next_session["price_context_status"] == "stale"
    assert after_next_session["effective_freshness_status"] == "stale"
    assert after_next_session["effective_observation_age_completed_sessions"] == 1
    assert after_next_session["artifact_freshness_status"] == "fresh"
    assert after_next_session["artifact_observation_age_completed_sessions"] == 0
    assert after_next_session["current_price"] is None
    assert manifest["status_counts"] == {
        "attempted": 2,
        "fresh": 2,
        "stale": 0,
        "missing": 0,
        "invalid": 0,
    }
    assert (root / advisory.MANIFEST_FILE).read_bytes() == manifest_bytes
    assert (root / advisory.OBSERVATIONS_FILE).read_bytes() == observations_bytes


def test_weekend_without_new_completed_session_preserves_fresh_close(
    inputs: dict[str, Path],
) -> None:
    _manifest, root = _build_at(
        inputs,
        retrieval_timestamp="2026-08-07T22:00:00Z",
        observation_timestamp="2026-08-07T20:00:00Z",
        run_id="friday-close",
    )
    context = consume_advisory_price_context(
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now="2026-08-09T12:00:00Z",
    )
    assert context["artifact_freshness_status"] == "fresh"
    assert context["effective_freshness_status"] == "fresh"
    assert context["effective_observation_age_completed_sessions"] == 0
    assert context["current_price"] == "123.4500"


def test_unavailable_effective_session_resolution_never_exposes_current_price(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest, root = _build(inputs)
    original_resolver = advisory.expected_completed_session

    def resolver(instrument, reference):
        if reference == advisory._timestamp(RETRIEVED_AT, "retrieval_timestamp"):
            return original_resolver(instrument, reference)
        return None, None

    monkeypatch.setattr(advisory, "expected_completed_session", resolver)
    context = consume_advisory_price_context(
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now=TRUSTED_NOW,
    )
    assert context["artifact_freshness_status"] == "fresh"
    assert context["effective_freshness_status"] == "invalid"
    assert context["price_context_status"] == "invalid"
    assert context["effective_observation_age_completed_sessions"] is None
    assert context["effective_freshness_error_code"] == "EXPECTED_SESSION_UNAVAILABLE"
    assert context["current_price"] is None


def test_consumer_rejects_trusted_time_before_artifact_generation(
    inputs: dict[str, Path],
) -> None:
    _manifest, root = _build(inputs)
    _assert_code(
        "FUTURE_RETRIEVAL_TIMESTAMP",
        consume_advisory_price_context,
        root,
        instrument_id="equity:aaa",
        canonical_ticker="AAA",
        universe_path=inputs["universe"],
        policy_path=inputs["policy"],
        trusted_now="2026-08-13T05:29:59Z",
    )


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
