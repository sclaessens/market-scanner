from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from inspect import signature
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from market_engine.data.local_market_data_universe import UNIVERSE_SNAPSHOT_SCHEMA_VERSION
from market_engine.data.scheduled_canonical_price_refresh import expected_completed_session
from market_engine.run import current_technical_screening as screening
from market_engine.run.current_technical_screening import (
    CurrentScreeningIssue,
    build_run33_grounded_handoff,
    load_validated_run33_handoff,
    run_current_technical_screening,
    validated_run33_handoff_payload,
)
from market_engine.source_refresh import advisory_ohlc_history as history
from market_engine.source_refresh import advisory_price_evidence as price


NOW = "2026-08-13T06:00:00Z"
SHA = "b" * 40


def _clock(value: str = NOW):
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return lambda: parsed


def _instrument(index: int) -> dict[str, object]:
    ticker = f"S{index}"
    return {"instrument_id": f"equity:{ticker.lower()}", "symbol": ticker, "source_symbol": ticker, "source_mapping_status": "mapped", "currency": "USD", "exchange": "US", "country": "US", "asset_type": "equity"}


def _sessions(end: date, count: int) -> list[str]:
    values = []
    while len(values) < count:
        if end.weekday() < 5: values.append(end.isoformat())
        end -= timedelta(days=1)
    return list(reversed(values))


def _bars(end: date, count: int = 252, slope: int = 1) -> list[dict[str, object]]:
    bars = []
    for index, session in enumerate(_sessions(end, count)):
        close = 100 + (index * slope / 20)
        bars.append({"session": session, "open": f"{close:.2f}", "high": f"{close + 2:.2f}", "low": f"{close - 2:.2f}", "close": f"{close:.2f}", "volume": "1000", "volume_status": "provider_reported"})
    return bars


@pytest.fixture
def route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(history, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(screening, "_repository_root", lambda: tmp_path)
    instruments = [_instrument(1), _instrument(2), _instrument(3)]
    monkeypatch.setattr(
        screening,
        "load_downstream_prestate",
        lambda: {
            "measurement_status": "measured",
            "authority_path": "test/downstream-prestate.json",
            "authority_sha256": "c" * 64,
            "by_ticker": {
                row["symbol"]: {
                    "instrument_id": row["instrument_id"],
                    "overall_fundamental_status": "partial",
                    "canonical_advice_input_ready": False,
                }
                for row in instruments
            },
        },
    )
    universe_path = tmp_path / "universe.json"; universe_path.write_text(json.dumps({"schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION, "universe_version": "screening-fixture", "instruments": instruments}))
    history_policy = json.loads(history.DEFAULT_POLICY_PATH.read_text()); history_policy_path = tmp_path / "history-policy.json"; history_policy_path.write_text(json.dumps(history_policy))
    price_policy = json.loads(price.DEFAULT_POLICY_PATH.read_text()); price_policy_path = tmp_path / "price-policy.json"; price_policy_path.write_text(json.dumps(price_policy))
    def provider(rows, at, policy_value):
        result = {}
        for index, instrument in enumerate(rows):
            _profile, expected = expected_completed_session(instrument, at)
            result[instrument["instrument_id"]] = {"instrument_id": instrument["instrument_id"], "canonical_ticker": instrument["symbol"], "source_symbol": instrument["source_symbol"], "currency": "USD", "price_basis": policy_value["price_basis"], "corporate_action_adjustment_policy": policy_value["corporate_action_adjustment_policy"], "bars": _bars(expected, slope=1 if index != 1 else -1)}
        return result
    _manifest, history_root = history._build_advisory_ohlc_history_impl(run_id="history", source_main_sha=SHA, universe_path=universe_path, policy_path=history_policy_path, clock=_clock(), provider=provider)
    return {"tmp": tmp_path, "instruments": instruments, "universe": universe_path, "history_policy": history_policy_path, "price_policy": price_policy_path, "history_root": history_root}


def _screen(route, run_id="screening"):
    return screening._run_current_technical_screening_impl(run_id=run_id, history_artifact_root=route["history_root"], universe_path=route["universe"], history_policy_path=route["history_policy"], now=_clock()())


def test_screening_recalculates_full_indicators_and_is_deterministic(route) -> None:
    manifest, root = _screen(route)
    index = json.loads((root / "universe_analysis_index.json").read_text())["records"]
    assert manifest["screened_count"] == 3
    assert all(row["setup_detection"]["ma200"] for row in index)
    assert all(row["traceability"]["end_date"] == "2026-08-12" for row in index)
    _manifest2, root2 = _screen(route, "screening-repeat")
    first = json.loads((root / "candidate_ranking.json").read_text())
    second = json.loads((root2 / "candidate_ranking.json").read_text())
    assert first["records"] == second["records"]
    keys = [(-row["candidate_score"], row["symbol"], row["instrument_id"]) for row in first["records"]]
    assert keys == sorted(keys)


def test_public_authority_apis_expose_no_freshness_time_override() -> None:
    forbidden = {"_clock", "clock", "trusted_now", "now", "as_of", "evaluation_time", "acquisition_timestamp"}
    for authority_api in (
        run_current_technical_screening,
        build_run33_grounded_handoff,
        load_validated_run33_handoff,
    ):
        assert forbidden.isdisjoint(signature(authority_api).parameters)


def test_screening_uses_only_loaded_history_and_isolates_one_bad_ticker(route, monkeypatch) -> None:
    context = history._load_advisory_ohlc_history_impl(route["history_root"], universe_path=route["universe"], policy_path=route["history_policy"], now=_clock()())
    effective = dict(context.effective_status); effective[route["instruments"][0]["instrument_id"]] = "invalid"
    policy = dict(context.policy); policy["minimum_fresh_screening_coverage_ratio"] = "0.50"
    altered = history._ValidatedHistoryContext(context.manifest, context.index, context.series, effective, context.universe, policy, context.root)
    monkeypatch.setattr(screening, "_load_advisory_ohlc_history_impl", lambda *args, **kwargs: altered)
    manifest, root = _screen(route, "isolated")
    rows = json.loads((root / "universe_analysis_index.json").read_text())["records"]
    assert manifest["screened_count"] == 2
    assert sum(row["screening_status"] == "blocked" for row in rows) == 1
    ranking = json.loads((root / "candidate_ranking.json").read_text())
    assert all(row["instrument_id"] != route["instruments"][0]["instrument_id"] for row in ranking["records"])


def test_less_than_25_candidates_is_honest_and_has_no_decision_semantics(route) -> None:
    _manifest, root = _screen(route)
    ranking = json.loads((root / "candidate_ranking.json").read_text())
    assert ranking["requested_top_limit"] == 25
    assert ranking["ranking_gap"] == 25 - len(ranking["records"])
    text = "\n".join(path.read_text() for path in root.iterdir() if path.suffix in {".json", ".md"}).lower()
    assert "price target" not in text and "position sizing" not in text and "broker order" not in text


def test_changed_history_or_forged_ranking_invalidates_screening(route) -> None:
    _manifest, root = _screen(route)
    ranking_path = root / "candidate_ranking.json"; ranking = json.loads(ranking_path.read_text()); ranking["records"] = []
    ranking_path.write_text(json.dumps(ranking, sort_keys=True, separators=(",", ":")) + "\n")
    checksums_path = root / "checksum_index.json"; checksums = json.loads(checksums_path.read_text()); checksums["files"]["candidate_ranking.json"] = history._sha256_file(ranking_path); checksums_path.write_bytes(history._canonical_json(checksums) + b"\n")
    loaded_history = history._load_advisory_ohlc_history_impl(route["history_root"], universe_path=route["universe"], policy_path=route["history_policy"], now=_clock()())
    with pytest.raises(CurrentScreeningIssue, match="SCREENING_BINDING_INVALID"):
        screening._load_screening(root, loaded_history)


def test_missing_volume_remains_nullable_and_never_becomes_zero() -> None:
    bars = _bars(date(2026, 8, 12), count=2)
    bars[0]["volume"] = None
    bars[0]["volume_status"] = "not_reported"
    frame = screening._frame(bars)
    assert str(frame["Volume"].dtype) == "Int64"
    assert frame["Volume"].isna().iloc[0]
    assert frame["Volume"].iloc[1] == 1000


def test_screening_policy_binding_blocks_rebound_policy_change(route, tmp_path) -> None:
    _manifest, root = _screen(route, "policy-binding")
    policy = json.loads(screening.DEFAULT_SCREENING_POLICY.read_text())
    policy["scoring_contract"] = "forged-scoring-v2"
    changed = tmp_path / "changed-screening-policy.json"
    changed.write_text(json.dumps(policy))
    loaded_history = history._load_advisory_ohlc_history_impl(
        route["history_root"], universe_path=route["universe"], policy_path=route["history_policy"], now=_clock()()
    )
    with pytest.raises(CurrentScreeningIssue, match="SCREENING_POLICY_INVALID"):
        screening._load_screening(root, loaded_history, screening_policy_path=changed)


def _price_artifact(route, overrides=None):
    def provider(instruments, at):
        result = {}
        loaded = history._load_advisory_ohlc_history_impl(route["history_root"], universe_path=route["universe"], policy_path=route["history_policy"], now=_clock()())
        for instrument in instruments:
            latest = loaded.series[instrument["instrument_id"]]["bars"][-1]
            result[instrument["instrument_id"]] = {"instrument_id": instrument["instrument_id"], "canonical_ticker": instrument["symbol"], "price": latest["close"], "currency": instrument["currency"], "observation_type": price.OBSERVATION_TYPE, "observation_timestamp": latest["session"] + "T20:00:00Z", "source_id": price.SOURCE_ID}
        for key, update in (overrides or {}).items():
            if update is None: result.pop(key)
            else: result[key].update(update)
        return result
    return price.build_advisory_price_artifact(run_id="price", source_main_sha=SHA, output_root=route["tmp"] / "price", universe_path=route["universe"], policy_path=route["price_policy"], retrieval_timestamp=NOW, provider=provider)[1]


def test_price_reconciliation_exact_match_and_pending_fundamentals_block_run33(route) -> None:
    _sm, screening_root = _screen(route)
    price_root = _price_artifact(route)
    manifest, root = screening._build_run33_grounded_handoff_impl(run_id="handoff", screening_root=screening_root, history_root=route["history_root"], price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"], price_policy_path=route["price_policy"], now=_clock()())
    reconciliation = json.loads((root / "technical_price_reconciliation.json").read_text())
    handoff = json.loads((root / "run33_candidate_input.json").read_text())
    assert reconciliation["counts"] == {"passed": 3}
    assert manifest["status"] == "conditional_blocked_pending_data11_approval"
    assert manifest["eligible_count"] == 0
    assert not any(row["eligible_for_run33"] for row in handoff["records"])
    assert manifest["downstream_execution"] == {"data07_calls": 0, "data06_calls": 0, "run31_calls": 0, "run33_calls": 0}
    schema = json.loads(Path("config/market_engine/run33_grounded_candidate_input_v1.schema.json").read_text())
    Draft202012Validator(schema).validate(handoff)
    manifest_schema = json.loads(Path("config/market_engine/run33_grounded_handoff_manifest_v1.schema.json").read_text())
    reconciliation_schema = json.loads(Path("config/market_engine/run33_technical_price_reconciliation_v1.schema.json").read_text())
    Draft202012Validator(manifest_schema).validate(manifest)
    Draft202012Validator(reconciliation_schema).validate(reconciliation)
    validated = screening._load_validated_run33_handoff_impl(
        root,
        screening_root=screening_root,
        history_root=route["history_root"],
        price_root=price_root,
        universe_path=route["universe"],
        history_policy_path=route["history_policy"],
        price_policy_path=route["price_policy"],
        now=_clock()(),
    )
    assert validated_run33_handoff_payload(validated) is None


def test_run33_loader_rejects_mapping_and_fully_rebound_forged_handoff(route) -> None:
    _sm, screening_root = _screen(route, "loader-screening")
    price_root = _price_artifact(route)
    _manifest, root = screening._build_run33_grounded_handoff_impl(
        run_id="loader-handoff", screening_root=screening_root, history_root=route["history_root"],
        price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"],
        price_policy_path=route["price_policy"], now=_clock()(),
    )
    kwargs = dict(
        screening_root=screening_root, history_root=route["history_root"], price_root=price_root,
        universe_path=route["universe"], history_policy_path=route["history_policy"],
        price_policy_path=route["price_policy"], now=_clock()(),
    )
    with pytest.raises(CurrentScreeningIssue, match="CALLER_CONTENT_FORBIDDEN"):
        screening._load_validated_run33_handoff_impl({"forged": True}, **kwargs)
    candidate_path = root / "run33_candidate_input.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["records"][0]["eligible_for_run33"] = True
    candidate_path.write_bytes(history._canonical_json(candidate) + b"\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_input_sha256"] = history._sha256(candidate_path.read_bytes())
    unsigned = dict(manifest); unsigned.pop("artifact_sha256")
    manifest["artifact_sha256"] = history._sha256(history._canonical_json(unsigned))
    manifest_path.write_bytes(history._canonical_json(manifest) + b"\n")
    checksums = json.loads((root / "checksum_index.json").read_text())
    checksums["files"]["manifest.json"] = history._sha256_file(manifest_path)
    checksums["files"]["run33_candidate_input.json"] = history._sha256_file(candidate_path)
    (root / "checksum_index.json").write_bytes(history._canonical_json(checksums) + b"\n")
    with pytest.raises(CurrentScreeningIssue, match="HANDOFF_SEMANTIC_REPLAY_INVALID"):
        screening._load_validated_run33_handoff_impl(root, **kwargs)


def test_public_run33_cannot_revive_stale_history_or_price(route) -> None:
    _screening_manifest, screening_root = _screen(route, "public-stale-screening")
    price_root = _price_artifact(route)
    _manifest, handoff_root = screening._build_run33_grounded_handoff_impl(
        run_id="public-stale-handoff", screening_root=screening_root, history_root=route["history_root"],
        price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"],
        price_policy_path=route["price_policy"], now=_clock()(),
    )
    kwargs = dict(
        screening_root=screening_root, history_root=route["history_root"], price_root=price_root,
        universe_path=route["universe"], history_policy_path=route["history_policy"],
        price_policy_path=route["price_policy"],
    )
    with pytest.raises(CurrentScreeningIssue, match="SCREENING_SEMANTIC_REPLAY_INVALID"):
        load_validated_run33_handoff(handoff_root, **kwargs)
    with pytest.raises(CurrentScreeningIssue, match="SCREENING_SEMANTIC_REPLAY_INVALID"):
        build_run33_grounded_handoff(run_id="public-stale-rebuild", **kwargs)


def test_current_history_with_missing_advisory_price_remains_blocked(route) -> None:
    _sm, screening_root = _screen(route)
    instrument_id = route["instruments"][0]["instrument_id"]
    price_root = _price_artifact(route, {instrument_id: None})
    _manifest, root = screening._build_run33_grounded_handoff_impl(run_id="missing-price", screening_root=screening_root, history_root=route["history_root"], price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"], price_policy_path=route["price_policy"], now=_clock()())
    row = next(value for value in json.loads((root / "run33_candidate_input.json").read_text())["records"] if value["instrument_id"] == instrument_id)
    assert row["conditions"]["current_technical_history"] is True
    assert row["conditions"]["fresh_advisory_price"] is False
    assert row["eligible_for_run33"] is False


def test_fresh_price_does_not_repair_insufficient_technical_history(route) -> None:
    instrument_id = route["instruments"][0]["instrument_id"]
    def provider(rows, at, policy_value):
        result = {}
        for instrument in rows:
            _profile, expected = expected_completed_session(instrument, at)
            count = 40 if instrument["instrument_id"] == instrument_id else 252
            result[instrument["instrument_id"]] = {"instrument_id": instrument["instrument_id"], "canonical_ticker": instrument["symbol"], "source_symbol": instrument["source_symbol"], "currency": "USD", "price_basis": policy_value["price_basis"], "corporate_action_adjustment_policy": policy_value["corporate_action_adjustment_policy"], "bars": _bars(expected, count=count)}
        return result
    _manifest, insufficient_root = history._build_advisory_ohlc_history_impl(run_id="insufficient", source_main_sha=SHA, universe_path=route["universe"], policy_path=route["history_policy"], clock=_clock(), provider=provider)
    route["history_root"] = insufficient_root
    _sm, screening_root = _screen(route, "insufficient-screening")
    price_root = _price_artifact(route)
    _manifest, root = screening._build_run33_grounded_handoff_impl(run_id="insufficient-handoff", screening_root=screening_root, history_root=insufficient_root, price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"], price_policy_path=route["price_policy"], now=_clock()())
    row = next(value for value in json.loads((root / "run33_candidate_input.json").read_text())["records"] if value["instrument_id"] == instrument_id)
    assert row["conditions"]["fresh_advisory_price"] is True
    assert row["conditions"]["current_technical_history"] is False
    assert row["conditions"]["current_technical_screening"] is False
    assert row["eligible_for_run33"] is False


@pytest.mark.parametrize("change,reason", [
    ({"observation_timestamp": "2026-08-11T20:00:00Z"}, "PRICE_SESSION_MISMATCH"),
    ({"canonical_ticker": "WRONG"}, "PRICE_IDENTITY_MISMATCH"),
    ({"currency": "EUR"}, "PRICE_CURRENCY_MISMATCH"),
    ({"price": "999.00"}, "PRICE_CLOSE_MISMATCH"),
])
def test_price_reconciliation_mismatches_are_explicit(route, change, reason) -> None:
    _sm, screening_root = _screen(route)
    instrument_id = route["instruments"][0]["instrument_id"]
    price_root = _price_artifact(route, {instrument_id: change})
    _manifest, root = screening._build_run33_grounded_handoff_impl(run_id="handoff-mismatch", screening_root=screening_root, history_root=route["history_root"], price_root=price_root, universe_path=route["universe"], history_policy_path=route["history_policy"], price_policy_path=route["price_policy"], now=_clock()())
    rows = json.loads((root / "technical_price_reconciliation.json").read_text())["records"]
    assert reason in next(row for row in rows if row["instrument_id"] == instrument_id)["reason_codes"]


def test_caller_projections_old_run30_and_tampered_portfolio_context_have_no_authority(route) -> None:
    with pytest.raises(CurrentScreeningIssue, match="CALLER_CONTENT_FORBIDDEN"):
        screening._load_screening({"records": []}, object())
    assert screening.DEFAULT_RUN30_RANKING.as_posix() not in json.loads((route["history_root"] / "manifest.json").read_text())
    with pytest.raises((CurrentScreeningIssue, TypeError)):
        screening._portfolio_binding({"forged": True})


def test_output_path_checks_precede_stage_execution(route, monkeypatch) -> None:
    calls = []; monkeypatch.setattr(screening, "_load_advisory_ohlc_history_impl", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(CurrentScreeningIssue, match="OUTPUT_PATH_INVALID"):
        screening._run_current_technical_screening_impl(run_id="unsafe", history_artifact_root=route["history_root"], output_root="../escape", universe_path=route["universe"], history_policy_path=route["history_policy"], now=_clock()())
    assert calls == []
