from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

from market_engine.advice.setup_price_market_context import (
    extract_setup_price_market_context,
)


def test_extracts_partial_context_from_local_price_history(tmp_path: Path) -> None:
    _write_price_history(tmp_path, "AAA", close=105, ma20=100, ma50=95, ma200=90, atr=5)

    context = extract_setup_price_market_context(
        {"ticker": "AAA"},
        {"ticker": "AAA"},
        local_price_root=tmp_path,
    ).to_payload()

    assert context["schema_version"] == "market-engine-setup-price-market-context-v1"
    assert context["context_status"] == "partial"
    assert context["price_context_available"] is True
    assert context["setup_context_available"] is True
    assert context["market_context_available"] is False
    assert context["trend_state"] == "uptrend"
    assert context["setup_state"] == "pullback_watch"
    assert context["price_position"] == "near_entry_zone"
    assert context["risk_state"] == "normal"
    assert context["missing"] == ["market_context"]
    assert context["evidence"][0]["source_path"] == (tmp_path / "AAA.csv").as_posix()


def test_missing_local_price_history_fails_closed(tmp_path: Path) -> None:
    context = extract_setup_price_market_context(
        {"ticker": "AAA"},
        {"ticker": "AAA"},
        local_price_root=tmp_path,
    ).to_payload()

    assert context["context_status"] == "missing"
    assert context["trend_state"] == "unknown"
    assert "local_price_history" in context["missing"]
    assert context["blocked_reasons"] == ["local_price_history_not_found"]


def test_embedded_invalid_context_fails_closed(tmp_path: Path) -> None:
    context = extract_setup_price_market_context(
        {"ticker": "AAA"},
        {
            "ticker": "AAA",
            "setup_price_market_context": {
                "context_status": "not_allowed",
                "trend_state": "uptrend",
                "setup_state": "pullback_watch",
                "price_position": "near_entry_zone",
                "risk_state": "normal",
            },
        },
        local_price_root=tmp_path,
    ).to_payload()

    assert context["context_status"] == "invalid"
    assert context["missing"] == ["valid_setup_price_market_context"]
    assert context["blocked_reasons"] == ["setup_price_market_context_invalid"]


def test_setup_price_market_context_does_not_require_openai_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_" + "API_KEY", raising=False)
    monkeypatch.delenv("MARKET_ENGINE_" + "ADVISORY_MODEL", raising=False)
    _write_price_history(tmp_path, "AAA", close=105, ma20=100, ma50=95, ma200=90, atr=5)

    context = extract_setup_price_market_context(
        {"ticker": "AAA"},
        {"ticker": "AAA"},
        local_price_root=tmp_path,
    )

    assert context.context_status == "partial"


def test_setup_price_market_context_does_not_touch_network(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ME-DATA01 must not make provider/network calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    _write_price_history(tmp_path, "AAA", close=105, ma20=100, ma50=95, ma200=90, atr=5)

    context = extract_setup_price_market_context(
        {"ticker": "AAA"},
        {"ticker": "AAA"},
        local_price_root=tmp_path,
    )

    assert context.context_status == "partial"


def _write_price_history(
    root: Path,
    ticker: str,
    *,
    close: float,
    ma20: float,
    ma50: float,
    ma200: float,
    atr: float,
) -> None:
    path = root / f"{ticker}.csv"
    path.write_text(
        "\n".join(
            (
                "Date,Adj Close,Close,High,Low,Open,Volume,MA20,MA50,MA200,ATR14,20D_HIGH,20D_LOW,AVG_VOL_20",
                f"2026-04-30,{close},{close},{close + 1},{close - 1},{close},1000000,{ma20},{ma50},{ma200},{atr},{close + 2},{close - 10},1000000",
                "",
            )
        ),
        encoding="utf-8",
    )
