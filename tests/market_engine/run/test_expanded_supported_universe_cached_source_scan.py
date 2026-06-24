from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from market_engine.candidate_classification import (
    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION,
)
from market_engine.run import expanded_supported_universe_cached_source_scan as scan_module
from market_engine.run.expanded_supported_universe_cached_source_scan import (
    EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION,
    ExpandedSupportedUniverseCachedSourceScanError,
    build_expanded_supported_universe_cached_source_scan,
    to_plain_dict,
)
from market_engine.run.expanded_supported_universe_cached_source_scan_command import run_command
from market_engine.source_refresh.sec_companyfacts_snapshots import persist_sec_companyfacts_raw_snapshot
from market_engine.source_support import ProfessionalSwingSourceSupportStatus


HEADER = (
    "ticker,name,market,asset_type,active,universe_status,source_policy_hint,"
    "operator_priority,swing_profile,liquidity_profile,volatility_profile,"
    "market_cap_profile,theme,sector,notes"
)


def test_run23_processes_supported_cached_entries_only(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("AMD", name="AMD"), _candidate("VRT", name="Vertiv")],
    )
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")
    _persist_snapshot(source_root, "AMD", cik="0000002488")

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run23-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
    )

    assert result.format_version == EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION
    assert result.run_state == "completed"
    assert result.supported_cached_tickers == ("NVDA", "AMD")
    assert result.source_support_summary_counts["total_expanded_universe_entries"] == 3
    assert result.source_support_summary_counts["supported_cached"] == 2
    assert result.source_support_summary_counts["missing_snapshot"] == 1
    assert result.batch_payload is not None
    assert result.batch_payload["requested_tickers"] == ("NVDA", "AMD")
    assert result.portfolio_context["portfolio_context_source"] == "absent"
    assert [entry["ticker"] for entry in result.non_supported_entries] == ["VRT"]
    assert result.live_provider_call_made is False


def test_run23_blocks_without_supported_cached_entries(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("MISSING", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("VRT", name="Vertiv")])

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=_source_root(tmp_path),
        batch_id="me-run23-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
    )

    assert result.run_state == "blocked_no_supported_cached_entries"
    assert result.supported_cached_tickers == ()
    assert result.batch_payload is None
    assert result.source_support_summary_counts["missing_snapshot"] == 2
    assert result.blocked_reasons == (
        "No expanded Professional Swing Universe entries are supported_cached.",
    )


def test_run23_preserves_deterministic_supported_ticker_order(tmp_path: Path) -> None:
    universe_path = _write_universe(
        tmp_path,
        [_row("NVDA", priority=1), _row("MSFT", priority=2)],
    )
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("VRT", name="Vertiv"), _candidate("AMD", name="AMD")],
    )
    source_root = _source_root(tmp_path)
    for ticker, cik in (("NVDA", "0001045810"), ("MSFT", "0000789019"), ("AMD", "0000002488")):
        _persist_snapshot(source_root, ticker, cik=cik)

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run23-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
    )

    assert result.supported_cached_tickers == ("NVDA", "MSFT", "AMD")
    assert result.batch_payload is not None
    assert result.batch_payload["requested_tickers"] == ("NVDA", "MSFT", "AMD")


def test_run23_ticker_limit_applies_after_supported_cached_filter(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1), _row("MSFT", priority=2)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    for ticker, cik in (("NVDA", "0001045810"), ("MSFT", "0000789019"), ("AMD", "0000002488")):
        _persist_snapshot(source_root, ticker, cik=cik)

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run23-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
        ticker_limit=2,
    )

    assert result.supported_cached_tickers == ("NVDA", "MSFT")
    assert result.batch_payload is not None
    assert result.batch_payload["requested_tickers"] == ("NVDA", "MSFT")


def test_run24_default_behavior_passes_no_portfolio_context_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    for ticker, cik in (("NVDA", "0001045810"), ("AMD", "0000002488")):
        _persist_snapshot(source_root, ticker, cik=cik)
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(requested_tickers=kwargs["requested_tickers"])

    monkeypatch.setattr(scan_module, "build_cached_source_batch_dry_run", fake_build_cached_source_batch_dry_run)

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run24-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
    )

    assert captured["portfolio_contexts_by_ticker"] is None
    assert result.portfolio_context == {
        "enabled": False,
        "portfolio_context_source": "absent",
        "no_broker_or_live_portfolio_access": True,
        "no_portfolio_or_watchlist_mutation": True,
    }


def test_run24_accepts_non_production_portfolio_context_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    for ticker, cik in (("NVDA", "0001045810"), ("AMD", "0000002488")):
        _persist_snapshot(source_root, ticker, cik=cik)
    fixture_path = _write_portfolio_context_fixture(tmp_path)
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(requested_tickers=kwargs["requested_tickers"])

    monkeypatch.setattr(scan_module, "build_cached_source_batch_dry_run", fake_build_cached_source_batch_dry_run)

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run24-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
        non_production_portfolio_context_fixture_path=fixture_path,
    )

    contexts = captured["portfolio_contexts_by_ticker"]
    assert tuple(contexts) == ("NVDA", "AMD")
    assert contexts["NVDA"]["context_provenance"]["source"] == "non_production_fixture"
    assert contexts["NVDA"]["context_provenance"]["no_broker_or_live_portfolio_access"] is True
    assert contexts["AMD"]["position_state"] == "not_held"
    assert result.portfolio_context["portfolio_context_source"] == "non_production_fixture"
    assert result.portfolio_context["batch_contract_version"] == (
        "market-engine-local-portfolio-context-batch-v1"
    )
    assert result.portfolio_context["no_broker_or_live_portfolio_access"] is True
    assert result.portfolio_context["no_portfolio_or_watchlist_mutation"] is True


def test_run24_command_renders_non_production_fixture_provenance(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [])
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")
    fixture_path = _write_portfolio_context_fixture(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_command(
        [
            "--candidate-classification-summary",
            candidate_path.as_posix(),
            "--professional-swing-universe",
            universe_path.as_posix(),
            "--source-snapshot-root",
            source_root.as_posix(),
            "--batch-id",
            "me-run24-fixture",
            "--generated-at",
            "2026-06-24T12:00:00+00:00",
            "--non-production-portfolio-context-fixture",
            fixture_path.as_posix(),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "PORTFOLIO CONTEXT" in output
    assert "Portfolio context source: non_production_fixture" in output
    assert "No broker or live portfolio access: True" in output
    assert "No portfolio or watchlist mutation: True" in output
    assert "Non-production portfolio-context fixture only" in output
    for forbidden in ("BUY", "SELL", "HOLD", "target price", "conviction", "urgency", "tradeability"):
        assert forbidden not in output


def test_run24_nonexistent_portfolio_context_fixture_fails_closed(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [])
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")

    with pytest.raises(ExpandedSupportedUniverseCachedSourceScanError) as exc_info:
        build_expanded_supported_universe_cached_source_scan(
            candidate_classification_path=candidate_path,
            existing_universe_path=universe_path,
            source_snapshot_root=source_root,
            batch_id="me-run24-fixture",
            generated_at="2026-06-24T12:00:00+00:00",
            non_production_portfolio_context_fixture_path=tmp_path / "missing.json",
        )

    assert "failed closed" in str(exc_info.value)
    assert "Missing portfolio context file" in str(exc_info.value)


def test_run24_malformed_or_unsupported_portfolio_context_fixture_fails_closed(
    tmp_path: Path,
) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [])
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    unsupported_path = tmp_path / "unsupported.json"
    unsupported_path.write_text(
        json.dumps(
            {
                "portfolio_context_batch_format_version": "unsupported-v0",
                "non_production_local_context": True,
            }
        ),
        encoding="utf-8",
    )

    for fixture_path in (malformed_path, unsupported_path):
        with pytest.raises(ExpandedSupportedUniverseCachedSourceScanError):
            build_expanded_supported_universe_cached_source_scan(
                candidate_classification_path=candidate_path,
                existing_universe_path=universe_path,
                source_snapshot_root=source_root,
                batch_id="me-run24-fixture",
                generated_at="2026-06-24T12:00:00+00:00",
                non_production_portfolio_context_fixture_path=fixture_path,
            )


def test_run24_fixture_output_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1), _row("MSFT", priority=2)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    for ticker, cik in (("NVDA", "0001045810"), ("MSFT", "0000789019"), ("AMD", "0000002488")):
        _persist_snapshot(source_root, ticker, cik=cik)
    fixture_path = _write_portfolio_context_fixture(tmp_path)

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        return _batch_payload(requested_tickers=kwargs["requested_tickers"])

    monkeypatch.setattr(scan_module, "build_cached_source_batch_dry_run", fake_build_cached_source_batch_dry_run)

    first = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run24-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
        non_production_portfolio_context_fixture_path=fixture_path,
    ).to_payload()
    second = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run24-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
        non_production_portfolio_context_fixture_path=fixture_path,
    ).to_payload()

    assert first == second
    assert first["supported_cached_tickers"] == ("NVDA", "MSFT", "AMD")
    assert first["portfolio_context"]["context_tickers"] == ("AMD", "MSFT", "NVDA")


def test_run23_plain_payload_is_auditable(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")

    result = build_expanded_supported_universe_cached_source_scan(
        candidate_classification_path=candidate_path,
        existing_universe_path=universe_path,
        source_snapshot_root=source_root,
        batch_id="me-run23-fixture",
        generated_at="2026-06-24T12:00:00+00:00",
    )
    payload = to_plain_dict(result)

    assert payload["format_version"] == EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION
    assert payload["source_support_summary_counts"]["missing_snapshot"] == 1
    assert payload["non_supported_entries"][0]["status"] == ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT.value


def test_run23_command_renders_human_visible_summary(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("NVDA", priority=1)])
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    _persist_snapshot(source_root, "NVDA", cik="0001045810")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_command(
        [
            "--candidate-classification-summary",
            candidate_path.as_posix(),
            "--professional-swing-universe",
            universe_path.as_posix(),
            "--source-snapshot-root",
            source_root.as_posix(),
            "--batch-id",
            "me-run23-fixture",
            "--generated-at",
            "2026-06-24T12:00:00+00:00",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "EXPANDED SOURCE SUPPORT" in output
    assert "supported_cached: 1" in output
    assert "missing_snapshot: 1" in output
    assert "NVDA" in output
    assert "AMD | missing_snapshot" in output


def test_run23_module_has_no_provider_network_or_action_dependencies() -> None:
    import market_engine.run.expanded_supported_universe_cached_source_scan as scan
    import market_engine.run.local_portfolio_context_fixture as fixture

    module_names = set(scan.__dict__) | set(fixture.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names
    assert "broker" not in module_names


def _source_root(tmp_path: Path) -> Path:
    return tmp_path / "source_snapshots"


def _write_universe(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_candidate_summary(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "candidate_classification_summary.json"
    path.write_text(
        json.dumps(
            {
                "candidate_classification_format_version": (
                    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
                ),
                "candidate_classification_run_id": "candidate-fixture",
                "per_ticker_classifications": records,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _candidate(ticker: str, *, name: str | None = None) -> dict[str, object]:
    return {
        "ticker": ticker,
        "source_candidate_id": f"candidate:{ticker}",
        "candidate_bucket": "ready_for_manual_candidate_review",
        "candidate_rationale": "Local readable output has coherent review-only context.",
        "evidence_references": [
            {
                "reference_type": "operator_report",
                "reference": f"operator_report:{ticker}",
            }
        ],
        "blocking_reasons": [],
        "safety_flags": {
            "actionable_language_detected": False,
            "unsupported_input_detected": False,
            "malformed_input_detected": False,
            "stale_data_detected": False,
            "blocked_state_detected": False,
        },
        "proposed_universe_entry": {
            "ticker": ticker,
            "name": name or f"{ticker} Example",
            "market": "USA",
            "asset_type": "equity",
            "active": "true",
            "universe_status": "candidate",
            "source_policy_hint": "cached_source_candidate",
            "operator_priority": "",
            "swing_profile": "unknown",
            "liquidity_profile": "unknown",
            "volatility_profile": "unknown",
            "market_cap_profile": "unknown",
            "theme": "candidate_classification",
            "sector": "unknown",
            "notes": "non-actionable candidate classification source",
        },
    }


def _row(ticker: str, *, name: str | None = None, priority: int) -> str:
    return (
        f"{ticker},{name or ticker},USA,equity,true,candidate,cached_source_candidate,"
        f"{priority},trend_continuation,high,high,mega_cap,theme,technology,"
    )


def _persist_snapshot(source_root: Path, ticker: str, *, cik: str) -> None:
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-run23-fixtures",
        ticker=ticker,
        cik=cik,
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-24T12:00:00+00:00",
    )


def _write_portfolio_context_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "non_production_portfolio_context.json"
    path.write_text(
        json.dumps(
            {
                "portfolio_context_batch_format_version": (
                    "market-engine-local-portfolio-context-batch-v1"
                ),
                "non_production_local_context": True,
                "portfolio_write_authority": False,
                "portfolio_context_format_version": "market-engine-portfolio-context-v1",
                "portfolio_snapshot_timestamp": "2026-06-24T12:00:00+00:00",
                "portfolio_base_currency": "EUR",
                "default_position_state": "not_held",
                "portfolio_total_value": 0,
                "positions_by_ticker": {
                    "NVDA": {
                        "position_state": "not_held",
                        "current_quantity": 0,
                        "current_market_value": 0,
                        "current_ticker_exposure_pct": 0,
                    }
                },
                "exposure_buckets": {},
                "concentration_thresholds": {},
                "policy_constraints": {
                    "non_production_fixture_only": True,
                    "no_real_holdings_implied": True,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _batch_payload(*, requested_tickers: tuple[str, ...]) -> dict[str, Any]:
    return {
        "contract_version": "market-engine-cached-source-batch-dry-run-v1",
        "batch_id": "me-run24-fixture",
        "generated_at": "2026-06-24T12:00:00+00:00",
        "input_mode": "cached_source_batch",
        "source_mode": "cached_source_local_only",
        "source_snapshot_root": "source_snapshots",
        "operator_ticker_input_reference": "explicit_requested_tickers",
        "requested_tickers": requested_tickers,
        "ticker_universe_metadata": {},
        "batch_execution_state": "completed",
        "batch_counts": {
            "requested_count": len(requested_tickers),
            "completed_count": len(requested_tickers),
            "completed_with_limitations_count": 0,
            "blocked_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
        },
        "per_ticker_results": (),
        "batch_blocked_reasons": (),
        "batch_warnings": (),
        "artifact_manifest_reference": None,
        "forbidden_side_effect_confirmation": (
            "No provider, live market data, broker, message-delivery, scheduler, UI, "
            "portfolio, watchlist, production-report, or execution side effects are performed."
        ),
        "authority_boundary_confirmation": (
            "Decision Engine remains the only future action/allocation authority; "
            "the batch dry-run summarizes cached-source local execution state only."
        ),
        "provenance": {},
        "live_provider_call_made": False,
        "non_production_batch": True,
    }


def _companyfacts_payload() -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(100)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(20)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(30)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(5)]}
                },
            }
        }
    }


def _fact(value: int) -> dict[str, object]:
    return {
        "val": value,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "accn": "0000000000-25-000001",
        "frame": "CY2025",
    }
