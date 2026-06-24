from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
)


LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION = (
    "market-engine-local-portfolio-context-batch-v1"
)
DEFAULT_PORTFOLIO_CONTEXT_PATH = "data/market_engine/portfolio_contexts/local_portfolio_context.json"


class LocalPortfolioContextFixtureError(ValueError):
    pass


def load_local_portfolio_contexts_by_ticker(
    *,
    path: str | Path,
    requested_tickers: Sequence[str] | None,
    batch_id: str,
    generated_at: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    fixture_path = _validated_path(Path(path), field_name="portfolio_context_fixture_path")
    payload = _read_json_object(fixture_path, description="portfolio context")
    if payload.get("portfolio_context_batch_format_version") != LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION:
        raise LocalPortfolioContextFixtureError(
            "Portfolio context file must use "
            f"{LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION}."
        )
    if payload.get("non_production_local_context") is not True:
        raise LocalPortfolioContextFixtureError(
            "Portfolio context file must set non_production_local_context=true."
        )
    if payload.get("portfolio_write_authority") not in (None, False):
        raise LocalPortfolioContextFixtureError(
            "Portfolio context file must not grant portfolio write authority."
        )
    positions = _positions_by_ticker(payload.get("positions_by_ticker"))
    explicit_contexts = _explicit_contexts_by_ticker(payload.get("portfolio_contexts_by_ticker"))
    context_tickers = tuple(requested_tickers or sorted(set(positions) | set(explicit_contexts)))
    if not context_tickers:
        raise LocalPortfolioContextFixtureError(
            "Portfolio context file requires requested tickers or explicit per-ticker context."
        )
    default_position_state = str(payload.get("default_position_state") or "unknown")
    contexts = {
        ticker: _portfolio_context_for_ticker(
            ticker=str(ticker).upper(),
            payload=payload,
            position=positions.get(str(ticker).upper(), {}),
            explicit_context=explicit_contexts.get(str(ticker).upper()),
            path=fixture_path,
            batch_id=batch_id,
            generated_at=generated_at,
            default_position_state=default_position_state,
        )
        for ticker in context_tickers
    }
    metadata = {
        "enabled": True,
        "portfolio_context_source": "non_production_fixture",
        "source_path": fixture_path.as_posix(),
        "batch_contract_version": LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION,
        "portfolio_context_format_version": str(
            payload.get("portfolio_context_format_version")
            or MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION
        ),
        "portfolio_snapshot_timestamp": str(
            payload.get("portfolio_snapshot_timestamp") or generated_at
        ),
        "portfolio_base_currency": str(payload.get("portfolio_base_currency") or "EUR"),
        "portfolio_write_authority": False,
        "context_ticker_count": len(contexts),
        "default_position_state": default_position_state,
        "context_tickers": tuple(sorted(contexts)),
        "non_production_boundary": (
            "Non-production portfolio-context fixture only; no broker or live "
            "portfolio access and no portfolio or watchlist mutation."
        ),
        "no_broker_or_live_portfolio_access": True,
        "no_portfolio_or_watchlist_mutation": True,
    }
    return contexts, metadata


def absent_portfolio_context_metadata() -> dict[str, Any]:
    return {
        "enabled": False,
        "portfolio_context_source": "absent",
        "no_broker_or_live_portfolio_access": True,
        "no_portfolio_or_watchlist_mutation": True,
    }


def _portfolio_context_for_ticker(
    *,
    ticker: str,
    payload: Mapping[str, Any],
    position: Mapping[str, Any],
    explicit_context: Mapping[str, Any] | None,
    path: Path,
    batch_id: str,
    generated_at: str,
    default_position_state: str,
) -> Mapping[str, Any]:
    if explicit_context is not None:
        context = dict(explicit_context)
    else:
        context = {
            "portfolio_context_format_version": payload.get("portfolio_context_format_version")
            or MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
            "portfolio_context_run_id": f"{batch_id}-{ticker.lower()}-portfolio-context",
            "portfolio_snapshot_timestamp": payload.get("portfolio_snapshot_timestamp")
            or generated_at,
            "portfolio_base_currency": payload.get("portfolio_base_currency") or "EUR",
            "ticker": ticker,
            "position_state": position.get("position_state") or default_position_state,
            "current_quantity": position.get("current_quantity", 0),
            "current_market_value": position.get("current_market_value", 0),
            "portfolio_total_value": payload.get("portfolio_total_value"),
            "current_ticker_exposure_pct": position.get("current_ticker_exposure_pct", 0),
            "exposure_buckets": payload.get("exposure_buckets") or {},
            "concentration_thresholds": payload.get("concentration_thresholds") or {},
            "policy_constraints": payload.get("policy_constraints") or {},
            "missing_portfolio_context_fields": tuple(
                position.get("missing_portfolio_context_fields")
                or payload.get("missing_portfolio_context_fields")
                or ()
            ),
            "stale_portfolio_context_fields": tuple(
                position.get("stale_portfolio_context_fields")
                or payload.get("stale_portfolio_context_fields")
                or ()
            ),
            "context_provenance": {
                "source": "non_production_fixture",
                "source_path": path.as_posix(),
                "batch_id": batch_id,
                "generated_at": generated_at,
                "portfolio_write_authority": False,
                "no_broker_or_live_portfolio_access": True,
                "no_portfolio_or_watchlist_mutation": True,
            },
        }
    context["ticker"] = str(context.get("ticker") or ticker).upper()
    if context["ticker"] != ticker:
        raise LocalPortfolioContextFixtureError(
            f"Portfolio context ticker mismatch for {ticker}: {context['ticker']}"
        )
    return context


def _positions_by_ticker(value: Any) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LocalPortfolioContextFixtureError(
            "positions_by_ticker must be a JSON object keyed by ticker."
        )
    positions: dict[str, Mapping[str, Any]] = {}
    for ticker, payload in value.items():
        if not isinstance(payload, Mapping):
            raise LocalPortfolioContextFixtureError(
                "Each positions_by_ticker value must be a JSON object."
            )
        positions[str(ticker).upper()] = dict(payload)
    return positions


def _explicit_contexts_by_ticker(value: Any) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LocalPortfolioContextFixtureError(
            "portfolio_contexts_by_ticker must be a JSON object keyed by ticker."
        )
    contexts: dict[str, Mapping[str, Any]] = {}
    for ticker, payload in value.items():
        if not isinstance(payload, Mapping):
            raise LocalPortfolioContextFixtureError(
                "Each portfolio_contexts_by_ticker value must be a JSON object."
            )
        contexts[str(ticker).upper()] = dict(payload)
    return contexts


def _read_json_object(path: Path, *, description: str) -> Mapping[str, Any]:
    if not path.exists():
        raise LocalPortfolioContextFixtureError(f"Missing {description} file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalPortfolioContextFixtureError(
            f"Malformed {description} JSON: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LocalPortfolioContextFixtureError(
            f"Malformed {description} JSON: root must be an object."
        )
    return payload


def _validated_path(path: Path, *, field_name: str) -> Path:
    if ".." in path.parts:
        raise LocalPortfolioContextFixtureError(f"Unsafe {field_name}: {path}")
    return path
