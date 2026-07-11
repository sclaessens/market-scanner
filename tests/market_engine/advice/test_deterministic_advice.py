from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from market_engine.advice.deterministic_advice import build_advice_index


def test_invalid_artifact_becomes_unable_to_advise(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, [_status_row("BAD", status="invalid_artifact", artifact_path=None)])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "unable_to_advise"
    assert advice["tickers"][0]["confidence"] == "low"
    assert advice["tickers"][0]["advice_readiness"] == "not_ready"


def test_stale_with_missing_critical_data_becomes_unable_to_advise(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "STALE", fundamental_present=False)
    row = _status_row("STALE", artifact_path=artifact, context_stale=True)
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "unable_to_advise"
    assert "fundamental_context" in advice["tickers"][0]["missing_for_buy_candidate"]


def test_valid_non_stale_partial_setup_missing_becomes_watchlist(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "PARTIAL")
    row = _status_row("PARTIAL", artifact_path=artifact)
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    output = advice["tickers"][0]
    assert output["advice"] == "watchlist"
    assert output["confidence"] == "low"
    assert output["advice_readiness"] == "partial"
    assert "setup_price_market_context" in output["missing_for_buy_candidate"]


def test_blocked_with_known_explicit_blockers_becomes_watchlist(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "BLOCKED", setup_price_missing=False)
    row = _status_row(
        "BLOCKED",
        artifact_path=artifact,
        readiness_blocked_reasons=["known_manual_review_blocker"],
        evidence_families_missing=[],
        missing_data_summary=[],
    )
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "watchlist"
    assert "known_manual_review_blocker" in advice["tickers"][0]["blockers"]


def test_actionable_complete_setup_becomes_buy_candidate(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "BUY", setup_price_missing=False)
    row = _status_row(
        "BUY",
        status="review_ready",
        artifact_path=artifact,
        actionable_review_allowed=True,
        blocked_stage=None,
        blocked_reasons=[],
        readiness_blocked_reasons=[],
        evidence_families_missing=[],
        missing_data_summary=[],
    )
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "buy_candidate"
    assert advice["tickers"][0]["advice_readiness"] == "ready"


def test_actionable_setup_with_uncertainty_becomes_wait_for_price(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "WAIT", setup_price_missing=False)
    row = _status_row(
        "WAIT",
        status="review_ready",
        artifact_path=artifact,
        actionable_review_allowed=True,
        blocked_stage=None,
        blocked_reasons=["valuation_uncertainty"],
        readiness_blocked_reasons=[],
        evidence_families_missing=[],
        missing_data_summary=[],
    )
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "wait_for_price"
    assert "valuation_uncertainty" in advice["tickers"][0]["blockers"]


def test_unsupported_conflict_fixture_becomes_avoid_for_now(tmp_path: Path) -> None:
    artifact = _write_dry_run(tmp_path, "AVOID", advice_flags=["unsupported_state"])
    row = _status_row(
        "AVOID",
        artifact_path=artifact,
        readiness_blocked_reasons=["unsupported_state"],
    )
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "avoid_for_now"


def test_existing_healthy_position_becomes_hold_existing(tmp_path: Path) -> None:
    artifact = _write_dry_run(
        tmp_path,
        "HELD",
        portfolio_context={"existing_position": True, "risk_state": "normal"},
    )
    row = _status_row("HELD", artifact_path=artifact)
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "hold_existing"


def test_existing_loss_position_becomes_take_loss_review(tmp_path: Path) -> None:
    artifact = _write_dry_run(
        tmp_path,
        "LOSS",
        portfolio_context={
            "existing_position": True,
            "risk_state": "loss_review",
            "unrealized_return_pct": -18,
        },
    )
    row = _status_row("LOSS", artifact_path=artifact)
    index_path = _write_status_index(tmp_path, [row])

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert advice["tickers"][0]["advice"] == "take_loss_review"


def test_output_is_deterministically_sorted(tmp_path: Path) -> None:
    z_artifact = _write_dry_run(tmp_path, "ZZZ")
    a_artifact = _write_dry_run(tmp_path, "AAA")
    index_path = _write_status_index(
        tmp_path,
        [
            _status_row("ZZZ", artifact_path=z_artifact),
            _status_row("AAA", artifact_path=a_artifact),
        ],
    )

    advice = build_advice_index(index_path, run_id="run", generated_at="2026-07-11T00:00:00Z")

    assert [row["ticker"] for row in advice["tickers"]] == ["AAA", "ZZZ"]


def _write_status_index(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "ticker_status_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-ticker-status-index-v1",
                "artifact_type": "market-engine-ticker-status-index",
                "run_id": "status-run",
                "generated_at": "2026-07-11T00:00:00Z",
                "artifact_root": tmp_path.as_posix(),
                "summary": {"tickers_total": len(rows)},
                "tickers": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _status_row(
    ticker: str,
    *,
    artifact_path: Path | None,
    status: str = "blocked",
    readiness_level: str = "partial_analysis",
    context_stale: bool = False,
    actionable_review_allowed: bool = False,
    decision_engine_ready: bool = False,
    blocked_stage: str | None = "portfolio_review",
    blocked_reasons: list[str] | None = None,
    readiness_blocked_reasons: list[str] | None = None,
    missing_data_summary: list[str] | None = None,
    evidence_families_missing: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": status,
        "readiness_level": readiness_level,
        "context_stale": context_stale,
        "actionable_review_allowed": actionable_review_allowed,
        "decision_engine_ready": decision_engine_ready,
        "blocked_stage": blocked_stage,
        "blocked_reasons": blocked_reasons
        if blocked_reasons is not None
        else ["Stage preserves an upstream blocked state."],
        "readiness_blocked_reasons": readiness_blocked_reasons
        if readiness_blocked_reasons is not None
        else ["missing_setup_or_price_context"],
        "missing_data_summary": missing_data_summary
        if missing_data_summary is not None
        else ["portfolio_context"],
        "evidence_families_missing": evidence_families_missing
        if evidence_families_missing is not None
        else ["setup_price_market"],
        "artifact_path": artifact_path.as_posix() if artifact_path else None,
        "artifact_sha256": "sha",
    }


def _write_dry_run(
    tmp_path: Path,
    ticker: str,
    *,
    setup_price_missing: bool = True,
    fundamental_present: bool = True,
    advice_flags: list[str] | None = None,
    portfolio_context: dict[str, Any] | None = None,
) -> Path:
    path = tmp_path / ticker / "dry_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_results = []
    provenance: dict[str, Any] = {}
    if fundamental_present:
        stage_results.append(
            {"stage_name": "fundamental_observations", "status": "completed"}
        )
        provenance["fundamental_observations"] = {
            "fundamental_observations_run_id": f"{ticker.lower()}-fundamental"
        }
    payload: dict[str, Any] = {
        "payload": {
            "ticker": ticker,
            "stage_results": stage_results,
            "provenance_summary": provenance,
            "available_context_families": []
            if setup_price_missing
            else ["setup_price_context"],
        }
    }
    if advice_flags:
        payload["payload"]["advice_flags"] = advice_flags
    if portfolio_context:
        payload["payload"]["portfolio_context"] = portfolio_context
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
