from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta, timezone
from inspect import signature
from io import StringIO
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from market_engine.data.local_market_data_universe import UNIVERSE_SNAPSHOT_SCHEMA_VERSION
from market_engine.data.scheduled_canonical_price_refresh import expected_completed_session
from market_engine.source_refresh import advisory_ohlc_history as history
from market_engine.source_refresh.advisory_ohlc_history import (
    AdvisoryHistoryIssue,
    build_advisory_ohlc_history,
    load_advisory_ohlc_history,
    validate_series_payload,
)


NOW = "2026-08-13T06:00:00Z"
SHA = "a" * 40


def _clock(value: str = NOW):
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return lambda: parsed


def _instrument(index: int) -> dict[str, object]:
    ticker = f"T{index:03d}"
    return {"instrument_id": f"equity:{ticker.lower()}", "symbol": ticker, "source_symbol": ticker, "source_mapping_status": "mapped", "currency": "USD", "exchange": "US", "country": "US", "asset_type": "equity"}


def _weekdays(end: str, count: int) -> list[str]:
    current = end if isinstance(end, date) else date.fromisoformat(end); result = []
    while len(result) < count:
        if current.weekday() < 5: result.append(current.isoformat())
        current -= timedelta(days=1)
    return list(reversed(result))


def _bars(end: str, count: int = 252) -> list[dict[str, object]]:
    return [{"session": session, "open": "100.00", "high": "125.00", "low": "99.00", "close": f"{100 + i % 20}.00", "volume": "1000", "volume_status": "provider_reported"} for i, session in enumerate(_weekdays(end, count))]


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(history, "_repository_root", lambda: tmp_path)
    instruments = [_instrument(1), _instrument(2), _instrument(3)]
    universe = {"schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION, "universe_version": "fixture-v1", "instruments": instruments}
    universe_path = tmp_path / "universe.json"; universe_path.write_text(json.dumps(universe), encoding="utf-8")
    policy = json.loads(history.DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    policy_path = tmp_path / "policy.json"; policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return {"root": tmp_path, "instruments": instruments, "universe": universe_path, "policy": policy_path}


def _provider(instruments, at, policy, overrides=None):
    result = {}
    for instrument in instruments:
        _profile, expected = expected_completed_session(instrument, at)
        result[instrument["instrument_id"]] = {"instrument_id": instrument["instrument_id"], "canonical_ticker": instrument["symbol"], "source_symbol": instrument["source_symbol"], "currency": instrument["currency"], "price_basis": policy["price_basis"], "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"], "bars": _bars(expected)}
    for key, value in (overrides or {}).items():
        if value is None: result.pop(key, None)
        else: result[key].update(value)
    return result


def _build(inputs, overrides=None, run_id="fixture"):
    return history._build_advisory_ohlc_history_impl(run_id=run_id, source_main_sha=SHA, universe_path=inputs["universe"], policy_path=inputs["policy"], provider=lambda rows, at, policy: _provider(rows, at, policy, overrides), clock=_clock())


def _fully_rebind(root: Path) -> None:
    manifest_path = root / "manifest.json"
    index = json.loads((root / "history_index.json").read_text())
    series = {}
    for row in index["records"]:
        if row["series_file"]:
            series[row["instrument_id"]] = json.loads((root / row["series_file"]).read_text())
    manifest = json.loads(manifest_path.read_text())
    manifest["observations_sha256"] = history._sha256(history._canonical_json({"index": index, "series": series}))
    unsigned = dict(manifest)
    unsigned.pop("artifact_sha256", None)
    manifest["artifact_sha256"] = history._sha256(history._canonical_json(unsigned))
    manifest_path.write_bytes(history._canonical_json(manifest) + b"\n")
    files = {
        path.relative_to(root).as_posix(): history._sha256_file(path)
        for path in root.rglob("*.json")
        if path.name != "checksum_index.json"
    }
    (root / "checksum_index.json").write_bytes(
        history._canonical_json({"schema_version": history.CHECKSUM_VERSION, "files": dict(sorted(files.items()))}) + b"\n"
    )


def test_roundtrip_binds_complete_identity_and_history(inputs) -> None:
    manifest, root = _build(inputs)
    loaded = history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())
    assert manifest["status_counts"] == {"fresh": 3, "stale": 0, "insufficient_history": 0, "missing": 0, "invalid": 0, "blocked_identity": 0, "blocked_adjustment_policy": 0, "attempted": 3}
    assert len(loaded.index) == len(loaded.series) == 3
    assert set(loaded.effective_status.values()) == {"fresh"}
    assert manifest["minimum_history_sessions"] == 252
    assert manifest["maximum_history_sessions"] == 420


def test_self_rebound_old_bars_cannot_claim_fresh_semantic_replay(inputs) -> None:
    _manifest, root = _build(inputs, run_id="self-rebound-old-bars")
    series_path = next((root / "series").glob("*.json"))
    payload = json.loads(series_path.read_text())
    for bar in payload["bars"]:
        bar["session"] = (date.fromisoformat(bar["session"]) - timedelta(days=7)).isoformat()
    series_path.write_bytes(history._canonical_json(payload) + b"\n")
    # Index freshness, global counts, lag claims, observations and every digest are
    # deliberately retained/re-signed as if the now-old series were still current.
    _fully_rebind(root)
    with pytest.raises(AdvisoryHistoryIssue, match="HISTORY_SEMANTIC_REPLAY_INVALID"):
        history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())


@pytest.mark.parametrize(
    "attack",
    ["source_id", "expected_session", "row_count", "first_session", "eligibility", "run_status", "bars"],
)
def test_fully_rebound_semantic_replay_attacks_fail_closed(inputs, attack) -> None:
    _manifest, root = _build(inputs, run_id=f"rebound-{attack}")
    index_path = root / "history_index.json"
    index = json.loads(index_path.read_text())
    series_path = root / index["records"][0]["series_file"]
    series = json.loads(series_path.read_text())
    if attack == "source_id":
        series["source_id"] = "forged-provider"
    elif attack == "expected_session":
        series["expected_session"] = "2026-08-11"
    elif attack == "row_count":
        index["records"][0]["row_count"] -= 1
    elif attack == "first_session":
        index["records"][0]["first_session"] = "2020-01-01"
    elif attack == "eligibility":
        eligibility_path = root / "screening_eligibility.json"
        eligibility = json.loads(eligibility_path.read_text())
        eligibility["records"][0]["eligible_for_current_screening"] = False
        eligibility_path.write_bytes(history._canonical_json(eligibility) + b"\n")
    elif attack == "run_status":
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["run_status"] = "blocked_provider_session_lag"
        manifest_path.write_bytes(history._canonical_json(manifest) + b"\n")
    else:
        series["bars"][-1]["session"] = "2026-08-11"
    index_path.write_bytes(history._canonical_json(index) + b"\n")
    series_path.write_bytes(history._canonical_json(series) + b"\n")
    _fully_rebind(root)
    with pytest.raises(AdvisoryHistoryIssue):
        history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())


def test_public_provider_and_time_authority_are_absent_and_cli_rejects_override() -> None:
    forbidden = {
        "provider", "_clock", "clock", "trusted_now", "now", "as_of", "evaluation_time",
        "acquisition_timestamp", "universe_path", "policy_path", "history_policy_path",
        "screening_policy_path", "price_policy_path", "source_main_sha",
    }
    assert forbidden.isdisjoint(signature(build_advisory_ohlc_history).parameters)
    assert forbidden.isdisjoint(signature(load_advisory_ohlc_history).parameters)
    with pytest.raises(TypeError, match="unexpected keyword argument 'provider'"):
        build_advisory_ohlc_history(run_id="forbidden-provider", provider=lambda *_: {})
    for function, args in (
        (build_advisory_ohlc_history, {"run_id": "forbidden-authority"}),
        (load_advisory_ohlc_history, {"artifact_root": "unused"}),
    ):
        for parameter in ("universe_path", "policy_path", "history_policy_path", "source_main_sha"):
            with pytest.raises(TypeError, match=f"unexpected keyword argument '{parameter}'"):
                function(**args, **{parameter: "forbidden"})
    with pytest.raises(TypeError, match="unexpected keyword argument '_clock'"):
        load_advisory_ohlc_history("unused", _clock=_clock())
    stderr = StringIO()
    with pytest.raises(SystemExit):
        history.run_command(["quality-gate", "--artifact-root", "unused", "--trusted-now", NOW], stderr=stderr)
    with pytest.raises(SystemExit):
        history.run_command(["build", "--run-id", "unused", "--source-main-sha", SHA], stderr=stderr)


def test_public_loader_cannot_backdate_a_stale_artifact(inputs, monkeypatch) -> None:
    _manifest, root = _build(inputs, run_id="public-no-backdating")
    monkeypatch.setattr(history, "DEFAULT_UNIVERSE_SNAPSHOT", inputs["universe"])
    monkeypatch.setattr(history, "DEFAULT_POLICY_PATH", inputs["policy"])
    monkeypatch.setattr(history, "_load_canonical_universe", lambda: {})
    loaded = load_advisory_ohlc_history(root)
    assert set(loaded.effective_status.values()) == {"stale"}
    assert history._effective_analytic_authority_status(loaded) == "unusable"


def test_monolithic_public_build_is_disabled_and_source_sha_resolution_remains_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_MAIN_SHA", "0" * 40)
    expected = history._current_repository_head_sha()
    assert expected == history.subprocess.run(
        ["git", "-C", str(history._repository_root()), "rev-parse", "--verify", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    with pytest.raises(AdvisoryHistoryIssue, match="BOUNDED_RUNTIME_REQUIRED"):
        build_advisory_ohlc_history(run_id="bounded-runtime-only")
    monkeypatch.setattr(history.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")))
    with pytest.raises(AdvisoryHistoryIssue, match="SOURCE_MAIN_SHA_UNRESOLVED"):
        history._current_repository_head_sha()


def test_canonical_production_universe_requires_exactly_952_identities(monkeypatch) -> None:
    assert len(history._load_canonical_universe()["instruments"]) == 952
    monkeypatch.setattr(history, "load_authoritative_universe", lambda _path: {"instruments": [{}] * 951})
    with pytest.raises(AdvisoryHistoryIssue, match="UNIVERSE_INVALID"):
        history._load_canonical_universe()


def test_build_measures_internal_acquisition_start_and_completion_separately(inputs) -> None:
    values = iter([
        datetime(2026, 8, 13, 6, tzinfo=UTC),
        datetime(2026, 8, 13, 6, 0, 5, tzinfo=UTC),
    ])
    seen = []
    manifest, _root = history._build_advisory_ohlc_history_impl(
        run_id="internal-clock", source_main_sha=SHA, universe_path=inputs["universe"], policy_path=inputs["policy"],
        provider=lambda rows, at, policy: seen.append(at) or _provider(rows, at, policy), clock=lambda: next(values),
    )
    assert seen == [datetime(2026, 8, 13, 6, tzinfo=UTC)]
    assert manifest["acquisition_started_at"] == "2026-08-13T06:00:00Z"
    assert manifest["acquisition_completed_at"] == "2026-08-13T06:00:05Z"


def test_representative_history_payloads_pass_real_json_schema(inputs) -> None:
    _manifest, root = _build(inputs)
    schema = json.loads(Path("config/market_engine/advisory_ohlc_history_v1.schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(json.loads((root / "manifest.json").read_text()))
    validator.validate(json.loads((root / "history_index.json").read_text()))
    validator.validate(json.loads(next((root / "series").glob("*.json")).read_text()))


def test_workflow_is_read_only_artifact_only_and_has_quality_gate() -> None:
    workflow = Path(".github/workflows/advisory-ohlc-history.yml").read_text()
    assert 'cron: "30 9 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "retention-days: 14" in workflow
    assert "Replay history semantics and enforce analytic usability" in workflow
    assert "if: always()" in workflow
    assert "publish" not in workflow.lower()
    assert "market-data" not in workflow
    assert "git push" not in workflow


@pytest.mark.parametrize("value", [1.2, True, "0", "0.0", "-1", "1e2", "NaN", "Infinity"])
def test_financial_price_domains_fail_closed(value) -> None:
    payload = {"schema_version": history.SERIES_VERSION, "instrument_id": "equity:a", "canonical_ticker": "A", "source_symbol": "A", "currency": "USD", "expected_session": "2026-08-12", "source_id": history.SOURCE_ID, "price_basis": "unadjusted_ohlc", "corporate_action_adjustment_policy": "provider_reported_unadjusted_with_adj_close_excluded", "bars": [{"session": "2026-08-12", "open": value, "high": "2", "low": "1", "close": "1.5", "volume": "1", "volume_status": "provider_reported"}]}
    with pytest.raises(AdvisoryHistoryIssue, match="PRICE_DOMAIN_INVALID"):
        validate_series_payload(payload)


@pytest.mark.parametrize("mutation,code", [
    (lambda bars: bars.append(deepcopy(bars[-1])), "SESSION_ORDER_INVALID"),
    (lambda bars: bars.reverse(), "SESSION_ORDER_INVALID"),
    (lambda bars: bars[-1].update(session="2999-01-01"), "FUTURE_SESSION"),
    (lambda bars: bars[-1].update(high="98"), "OHLC_RELATION_INVALID"),
    (lambda bars: bars[-1].update(low="126"), "OHLC_RELATION_INVALID"),
    (lambda bars: bars[-1].update(volume=0), "VOLUME_INVALID"),
])
def test_bar_semantics_fail_closed(mutation, code) -> None:
    bars = _bars("2026-08-12", 2); mutation(bars)
    payload = {"schema_version": history.SERIES_VERSION, "instrument_id": "equity:a", "canonical_ticker": "A", "source_symbol": "A", "currency": "USD", "expected_session": "2026-08-12", "source_id": history.SOURCE_ID, "price_basis": "unadjusted_ohlc", "corporate_action_adjustment_policy": "provider_reported_unadjusted_with_adj_close_excluded", "bars": bars}
    with pytest.raises(AdvisoryHistoryIssue) as caught: validate_series_payload(payload)
    assert caught.value.code == code


def test_missing_recent_listing_stale_identity_and_adjustment_are_distinct(inputs) -> None:
    instruments = inputs["instruments"]
    _profile, expected = expected_completed_session(instruments[1], history._timestamp(NOW, "now"))
    older = _weekdays(expected, 2)[0]
    overrides = {
        instruments[0]["instrument_id"]: None,
        instruments[1]["instrument_id"]: {"bars": _bars(older)},
        instruments[2]["instrument_id"]: {"bars": _bars(expected, 40)},
    }
    manifest, _root = _build(inputs, overrides)
    assert manifest["status_counts"]["missing"] == 1
    assert manifest["status_counts"]["stale"] == 1
    assert manifest["status_counts"]["insufficient_history"] == 1


def test_wrong_identity_currency_and_adjustment_are_blocked(inputs) -> None:
    ids = [row["instrument_id"] for row in inputs["instruments"]]
    manifest, _ = _build(inputs, {ids[0]: {"canonical_ticker": "WRONG"}, ids[1]: {"currency": "EUR"}, ids[2]: {"price_basis": "adjusted"}})
    assert manifest["status_counts"]["blocked_identity"] == 2
    assert manifest["status_counts"]["blocked_adjustment_policy"] == 1


def test_widespread_exact_one_session_lag_blocks_current_authority(inputs) -> None:
    overrides = {}
    for instrument in inputs["instruments"]:
        _profile, expected = expected_completed_session(instrument, history._timestamp(NOW, "now"))
        prior = _weekdays(expected, 2)[0]
        overrides[instrument["instrument_id"]] = {"bars": _bars(prior)}
    manifest, root = _build(inputs, overrides)
    assert manifest["run_status"] == "blocked_provider_session_lag"
    assert manifest["provider_session_lag"]["ratio"] == "1"
    eligibility = json.loads((root / "screening_eligibility.json").read_text())
    assert not any(row["eligible_for_current_screening"] for row in eligibility["records"])


def test_weekend_reuses_last_completed_session_without_staleness(inputs) -> None:
    weekend = "2026-08-16T06:00:00Z"
    manifest, _ = history._build_advisory_ohlc_history_impl(run_id="weekend", source_main_sha=SHA, universe_path=inputs["universe"], policy_path=inputs["policy"], provider=lambda rows, at, policy: _provider(rows, at, policy), clock=_clock(weekend))
    assert manifest["status_counts"]["fresh"] == 3


def test_provider_failure_is_distinct_and_fallback_is_bounded(inputs, monkeypatch) -> None:
    manifest, _ = history._build_advisory_ohlc_history_impl(run_id="failure", source_main_sha=SHA, universe_path=inputs["universe"], policy_path=inputs["policy"], provider=lambda *_: (_ for _ in ()).throw(RuntimeError()), clock=_clock())
    assert manifest["run_status"] == "blocked_provider_failure"
    calls = []
    monkeypatch.setattr(history, "download_yfinance_batch", lambda *args: {})
    monkeypatch.setattr(history, "_download_yfinance_history", lambda *args: calls.append(args) or None)
    policy = json.loads(Path(inputs["policy"]).read_text()); policy["max_individual_fallbacks"] = 2
    history._acquire_with_existing_adapter(inputs["instruments"], history._timestamp(NOW, "now"), policy)
    assert len(calls) == 2


def test_tampered_series_manifest_policy_and_checksum_fail_closed(inputs) -> None:
    _manifest, root = _build(inputs)
    series = next((root / "series").glob("*.json")); series.write_text(series.read_text().replace('"close":"119.00"', '"close":"118.00"'))
    with pytest.raises(AdvisoryHistoryIssue, match="ARTIFACT_INTEGRITY_INVALID"):
        history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())
    _manifest, root2 = _build(inputs, run_id="manifest-tamper"); data = json.loads((root2 / "manifest.json").read_text()); data["run_status"] = "completed"; (root2 / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(AdvisoryHistoryIssue): history._load_advisory_ohlc_history_impl(root2, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())
    _manifest, root3 = _build(inputs, run_id="policy-tamper"); policy = json.loads(Path(inputs["policy"]).read_text()); policy["max_individual_fallbacks"] = 1; Path(inputs["policy"]).write_text(json.dumps(policy))
    with pytest.raises(AdvisoryHistoryIssue, match="POLICY_BINDING_INVALID"):
        history._load_advisory_ohlc_history_impl(root3, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock()())


def test_load_time_freshness_and_untrusted_time_forms_fail_closed(inputs) -> None:
    _manifest, root = _build(inputs)
    loaded = history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock("2026-08-14T23:00:00Z")())
    assert set(loaded.effective_status.values()) == {"stale"}
    assert history._effective_analytic_authority_status(loaded) == "unusable"
    for value in (lambda: datetime(2026, 8, 13, 6), lambda: datetime(2026, 8, 13, 8, tzinfo=timezone(timedelta(hours=2))), lambda: datetime(2026, 8, 13, 6, 0, 0, 1, tzinfo=UTC)):
        with pytest.raises(AdvisoryHistoryIssue): history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=value())
    with pytest.raises(AdvisoryHistoryIssue, match="CLOCK_INVALID"):
        history._load_advisory_ohlc_history_impl(root, universe_path=inputs["universe"], policy_path=inputs["policy"], now=_clock("2026-08-12T06:00:00Z")())


def test_full_universe_fixture_reconciles_952_with_explicit_distribution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(history, "_repository_root", lambda: tmp_path)
    instruments = [_instrument(i) for i in range(952)]
    universe_path = tmp_path / "u.json"; universe_path.write_text(json.dumps({"schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION, "universe_version": "full-fixture", "instruments": instruments}))
    policy = json.loads(history.DEFAULT_POLICY_PATH.read_text()); policy.update(indicator_max_warmup_sessions=2, warmup_safety_margin_sessions=1, minimum_history_sessions=3, maximum_history_sessions=5)
    policy_path = tmp_path / "p.json"; policy_path.write_text(json.dumps(policy))
    def provider(rows, at, configured):
        result = _provider(rows, at, configured)
        result.pop(rows[-1]["instrument_id"])
        result[rows[-2]["instrument_id"]]["bars"] = result[rows[-2]["instrument_id"]]["bars"][:2]
        result[rows[-3]["instrument_id"]]["bars"][-1]["close"] = 1.5
        return result
    manifest, _ = history._build_advisory_ohlc_history_impl(run_id="full-952", source_main_sha=SHA, universe_path=universe_path, policy_path=policy_path, provider=provider, clock=_clock())
    assert manifest["status_counts"] == {"fresh": 949, "stale": 0, "insufficient_history": 1, "missing": 1, "invalid": 1, "blocked_identity": 0, "blocked_adjustment_policy": 0, "attempted": 952}
    assert manifest["run_status"] == "completed_with_blockers"
    assert manifest["analytic_authority_status"] == "usable"


@pytest.mark.parametrize("output", ["/tmp/absolute", "../escape", "artifacts/market_engine/advisory_ohlc_history_runs/extra"])
def test_output_path_authority_is_checked_before_provider(inputs, output) -> None:
    calls = []
    with pytest.raises(AdvisoryHistoryIssue, match="OUTPUT_PATH_INVALID"):
        history._build_advisory_ohlc_history_impl(run_id="unsafe", source_main_sha=SHA, output_root=output, universe_path=inputs["universe"], policy_path=inputs["policy"], provider=lambda *_: calls.append(True), clock=_clock())
    assert calls == []
