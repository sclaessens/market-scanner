from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "market-engine-setup-price-market-context-v1"

CONTEXT_STATUS_VALUES = ("available", "partial", "missing", "invalid")
TREND_STATE_VALUES = ("uptrend", "downtrend", "sideways", "unknown")
SETUP_STATE_VALUES = (
    "breakout_candidate",
    "pullback_watch",
    "extended_wait",
    "weak_setup",
    "no_clear_setup",
    "unknown",
)
PRICE_POSITION_VALUES = (
    "near_entry_zone",
    "above_preferred_entry",
    "below_support_or_breakdown",
    "fair_zone",
    "unknown",
)
RISK_STATE_VALUES = ("normal", "elevated", "high", "unknown")

DEFAULT_LOCAL_PRICE_ROOT = Path("data/processed")


@dataclass(frozen=True)
class SetupPriceMarketContext:
    schema_version: str
    ticker: str
    context_status: str
    price_context_available: bool
    setup_context_available: bool
    market_context_available: bool
    trend_state: str
    setup_state: str
    price_position: str
    risk_state: str
    evidence: tuple[Mapping[str, Any], ...]
    missing: tuple[str, ...]
    blocked_reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ticker": self.ticker,
            "context_status": self.context_status,
            "price_context_available": self.price_context_available,
            "setup_context_available": self.setup_context_available,
            "market_context_available": self.market_context_available,
            "trend_state": self.trend_state,
            "setup_state": self.setup_state,
            "price_position": self.price_position,
            "risk_state": self.risk_state,
            "evidence": [dict(item) for item in self.evidence],
            "missing": list(self.missing),
            "blocked_reasons": list(self.blocked_reasons),
        }


def extract_setup_price_market_context(
    ticker_status_row: Mapping[str, Any],
    dry_run_payload: Mapping[str, Any],
    *,
    local_price_root: str | Path = DEFAULT_LOCAL_PRICE_ROOT,
) -> SetupPriceMarketContext:
    ticker = _ticker(ticker_status_row, dry_run_payload)
    embedded = dry_run_payload.get("setup_price_market_context")
    if isinstance(embedded, dict):
        return _context_from_mapping(ticker, embedded)

    price_path = Path(local_price_root) / f"{ticker}.csv"
    if not price_path.exists():
        return _missing_context(
            ticker,
            missing=("local_price_history", "price_level_context", "setup_detection"),
            blocked_reasons=("local_price_history_not_found",),
        )

    try:
        row = _last_csv_row(price_path)
        values = _price_values(row)
    except (OSError, ValueError, KeyError):
        return _invalid_context(
            ticker,
            missing=("valid_local_price_history",),
            blocked_reasons=("local_price_history_invalid",),
        )

    trend_state = _trend_state(values)
    risk_state = _risk_state(values)
    setup_state = _setup_state(values, trend_state)
    price_position = _price_position(values, trend_state)
    missing = ("market_context",)
    return SetupPriceMarketContext(
        schema_version=SCHEMA_VERSION,
        ticker=ticker,
        context_status="partial",
        price_context_available=True,
        setup_context_available=True,
        market_context_available=False,
        trend_state=trend_state,
        setup_state=setup_state,
        price_position=price_position,
        risk_state=risk_state,
        evidence=(
            {
                "field": "local_price_history",
                "source_path": price_path.as_posix(),
                "source_family": "local_price_history",
                "as_of_date": row.get("Date"),
            },
            {
                "field": "derived_setup_price_market_context",
                "source_path": price_path.as_posix(),
                "source_family": "derived_from_local_price_history",
            },
        ),
        missing=missing,
        blocked_reasons=(),
    )


def _context_from_mapping(
    ticker: str,
    context: Mapping[str, Any],
) -> SetupPriceMarketContext:
    status = _allowed(context.get("context_status"), CONTEXT_STATUS_VALUES, "invalid")
    trend = _allowed(context.get("trend_state"), TREND_STATE_VALUES, "unknown")
    setup = _allowed(context.get("setup_state"), SETUP_STATE_VALUES, "unknown")
    price = _allowed(context.get("price_position"), PRICE_POSITION_VALUES, "unknown")
    risk = _allowed(context.get("risk_state"), RISK_STATE_VALUES, "unknown")
    if status == "invalid":
        missing = tuple(_strings(context.get("missing"))) or ("valid_setup_price_market_context",)
        blockers = tuple(_strings(context.get("blocked_reasons"))) or (
            "setup_price_market_context_invalid",
        )
    else:
        missing = tuple(_strings(context.get("missing")))
        blockers = tuple(_strings(context.get("blocked_reasons")))
    evidence = tuple(
        item
        for item in context.get("evidence", ())
        if isinstance(item, Mapping)
    )
    return SetupPriceMarketContext(
        schema_version=SCHEMA_VERSION,
        ticker=ticker,
        context_status=status,
        price_context_available=bool(context.get("price_context_available")),
        setup_context_available=bool(context.get("setup_context_available")),
        market_context_available=bool(context.get("market_context_available")),
        trend_state=trend,
        setup_state=setup,
        price_position=price,
        risk_state=risk,
        evidence=evidence,
        missing=missing,
        blocked_reasons=blockers,
    )


def _missing_context(
    ticker: str,
    *,
    missing: tuple[str, ...],
    blocked_reasons: tuple[str, ...],
) -> SetupPriceMarketContext:
    return SetupPriceMarketContext(
        schema_version=SCHEMA_VERSION,
        ticker=ticker,
        context_status="missing",
        price_context_available=False,
        setup_context_available=False,
        market_context_available=False,
        trend_state="unknown",
        setup_state="unknown",
        price_position="unknown",
        risk_state="unknown",
        evidence=(),
        missing=missing,
        blocked_reasons=blocked_reasons,
    )


def _invalid_context(
    ticker: str,
    *,
    missing: tuple[str, ...],
    blocked_reasons: tuple[str, ...],
) -> SetupPriceMarketContext:
    return SetupPriceMarketContext(
        schema_version=SCHEMA_VERSION,
        ticker=ticker,
        context_status="invalid",
        price_context_available=False,
        setup_context_available=False,
        market_context_available=False,
        trend_state="unknown",
        setup_state="unknown",
        price_position="unknown",
        risk_state="unknown",
        evidence=(),
        missing=missing,
        blocked_reasons=blocked_reasons,
    )


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("empty local price history")
    return rows[-1]


def _price_values(row: Mapping[str, str]) -> dict[str, float]:
    required = ("Close", "MA20", "MA50", "MA200", "ATR14", "20D_HIGH", "20D_LOW")
    values = {field: _number(row[field]) for field in required}
    if any(value is None for value in values.values()):
        raise ValueError("missing required price field")
    return {field: value for field, value in values.items() if value is not None}


def _trend_state(values: Mapping[str, float]) -> str:
    close = values["Close"]
    ma20 = values["MA20"]
    ma50 = values["MA50"]
    ma200 = values["MA200"]
    if close > ma20 and close > ma50 and (ma50 > ma200 or close > ma200):
        return "uptrend"
    if close > ma50 and ma50 > ma200:
        return "uptrend"
    if close < ma20 and close < ma50 and (ma50 < ma200 or close < ma200):
        return "downtrend"
    return "sideways"


def _setup_state(values: Mapping[str, float], trend_state: str) -> str:
    if trend_state == "downtrend":
        return "weak_setup"
    if trend_state != "uptrend":
        return "no_clear_setup"
    distance_atr = _distance_from_ma20_in_atr(values)
    if distance_atr > 1.5:
        return "breakout_candidate"
    if -0.75 <= distance_atr <= 1.5:
        return "pullback_watch"
    return "no_clear_setup"


def _price_position(values: Mapping[str, float], trend_state: str) -> str:
    close = values["Close"]
    low_20d = values["20D_LOW"]
    if trend_state == "downtrend" and close < values["MA50"]:
        return "below_support_or_breakdown"
    if close <= low_20d:
        return "below_support_or_breakdown"
    distance_atr = _distance_from_ma20_in_atr(values)
    if distance_atr > 1.5:
        return "above_preferred_entry"
    if -0.5 <= distance_atr <= 1.0:
        return "near_entry_zone"
    if 1.0 < distance_atr <= 1.5:
        return "fair_zone"
    return "unknown"


def _risk_state(values: Mapping[str, float]) -> str:
    atr_pct = values["ATR14"] / values["Close"]
    if atr_pct >= 0.10:
        return "high"
    if atr_pct >= 0.06:
        return "elevated"
    return "normal"


def _distance_from_ma20_in_atr(values: Mapping[str, float]) -> float:
    return (values["Close"] - values["MA20"]) / values["ATR14"]


def _ticker(
    ticker_status_row: Mapping[str, Any],
    dry_run_payload: Mapping[str, Any],
) -> str:
    value = ticker_status_row.get("ticker") or dry_run_payload.get("ticker") or "UNKNOWN"
    return str(value).strip().upper() or "UNKNOWN"


def _allowed(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _strings(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _number(value: str) -> float | None:
    if value == "":
        return None
    return float(value)
