from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from config.settings import DATA_DIR


WATCHLIST_ACTIVE_FILE = DATA_DIR / "watchlist" / "watchlist_active.csv"
WATCHLIST_STATUS_FILE = DATA_DIR / "watchlist" / "watchlist_status.csv"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
SCANNER_RANKED_FILE = DATA_DIR / "processed" / "scanner_ranked.csv"
PROCESSED_DIR = DATA_DIR / "processed"
THRESHOLDS_FILE = PROJECT_ROOT / "config" / "thresholds.yaml"


DEFAULT_THRESHOLDS = {
    "watchlist": {
        # Pullback
        "ready_distance_to_ma20_pct": 0.025,
        "neutral_ready_distance_to_ma20_pct": 0.015,
        "max_extended_from_ma20_pct": 0.06,

        # Breakout / VCP
        "ready_near_high_pct": 0.015,
        "breakout_break_above_high_pct": 0.002,
        "max_breakout_chase_pct": 0.03,
        "missed_breakout_pct": 0.08,
        "vcp_break_above_high_pct": 0.002,
        "vcp_max_range_pct": 0.08,
        "vcp_max_close_to_ma20_pct": 0.04,

        # Shared
        "reject_below_ma50": True,
        "expire_after_days": 20,
    }
}


def load_thresholds() -> dict:
    if yaml is None or not THRESHOLDS_FILE.exists():
        return DEFAULT_THRESHOLDS

    try:
        with THRESHOLDS_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_THRESHOLDS

    merged = DEFAULT_THRESHOLDS.copy()
    merged_watchlist = dict(DEFAULT_THRESHOLDS["watchlist"])
    merged_watchlist.update(data.get("watchlist", {}))
    merged["watchlist"] = merged_watchlist
    return merged


def load_market_regime() -> str:
    if not MARKET_REGIME_FILE.exists():
        return "NEUTRAL"

    try:
        df = pd.read_csv(MARKET_REGIME_FILE)
    except Exception:
        return "NEUTRAL"

    if df.empty:
        return "NEUTRAL"

    last_row = df.iloc[-1]
    for col in ["regime", "Regime", "market_regime"]:
        if col in df.columns:
            value = str(last_row[col]).strip().upper()
            if value:
                return value

    return "NEUTRAL"


def load_watchlist_active() -> pd.DataFrame:
    if not WATCHLIST_ACTIVE_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(WATCHLIST_ACTIVE_FILE)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper().str.strip()
    else:
        df["setup_type"] = "PULLBACK"

    if "is_active" not in df.columns:
        df["is_active"] = True

    return df


def load_indicator_file(ticker: str) -> pd.DataFrame:
    possible_paths = [
        PROCESSED_DIR / f"{ticker}_indicators.csv",
        PROCESSED_DIR / f"{ticker}.csv",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return pd.DataFrame()



def load_breakout_strength_map() -> dict[str, float]:
    """
    Haalt breakout_strength uit scanner_ranked.csv.
    De scanner blijft de bron van breakoutkwaliteit; de watchlist gebruikt dit
    alleen om zwakke breakouts niet als STOP BUY of BUY NOW te behandelen.
    """
    if not SCANNER_RANKED_FILE.exists():
        return {}

    try:
        df = pd.read_csv(SCANNER_RANKED_FILE)
    except Exception:
        return {}

    if df.empty or "ticker" not in df.columns or "breakout_strength" not in df.columns:
        return {}

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["breakout_strength"] = pd.to_numeric(df["breakout_strength"], errors="coerce")

    strength_map: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        strength = row.get("breakout_strength")
        if ticker and not pd.isna(strength):
            strength_map[ticker] = float(strength)

    return strength_map


def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_timestamp(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def round_or_none(value: float) -> Optional[float]:
    if pd.isna(value):
        return None
    return round(float(value), 2)


def build_metric_alias_map(columns: list[str]) -> dict[str, str]:
    normalized = {str(col).strip().upper(): col for col in columns}
    aliases = {
        "CLOSE": ["CLOSE", "close"],
        "HIGH": ["HIGH", "high"],
        "LOW": ["LOW", "low"],
        "VOLUME": ["VOLUME", "volume"],
        "MA20": ["MA20", "ma20"],
        "MA50": ["MA50", "ma50"],
        "MA200": ["MA200", "ma200"],
        "20D_HIGH": ["20D_HIGH", "HIGH_20D", "20d_high", "high_20d"],
        "20D_LOW": ["20D_LOW", "LOW_20D", "20d_low", "low_20d"],
        "AVG_VOL_20": ["AVG_VOL_20", "AVG_VOLUME_20D", "avg_vol_20", "avg_volume_20d"],
        "ATR14": ["ATR14", "ATR_14", "atr14", "atr_14"],
    }

    resolved = {}
    for canonical, options in aliases.items():
        resolved[canonical] = None
        for option in options:
            key = str(option).strip().upper()
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    return resolved


def get_latest_metrics(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None

    alias_map = build_metric_alias_map(list(df.columns))
    required = ["CLOSE", "MA20", "MA50", "MA200", "20D_HIGH"]
    if any(alias_map.get(name) is None for name in required):
        return None

    latest = df.iloc[-1]
    close = to_float(latest.get(alias_map["CLOSE"]))
    ma20 = to_float(latest.get(alias_map["MA20"]))
    ma50 = to_float(latest.get(alias_map["MA50"]))
    ma200 = to_float(latest.get(alias_map["MA200"]))
    high_20d = to_float(latest.get(alias_map["20D_HIGH"]))
    low_20d = to_float(latest.get(alias_map["20D_LOW"])) if alias_map.get("20D_LOW") else float("nan")
    avg_vol_20 = to_float(latest.get(alias_map["AVG_VOL_20"])) if alias_map.get("AVG_VOL_20") else float("nan")
    volume = to_float(latest.get(alias_map["VOLUME"])) if alias_map.get("VOLUME") else float("nan")
    high = to_float(latest.get(alias_map["HIGH"])) if alias_map.get("HIGH") else close
    low = to_float(latest.get(alias_map["LOW"])) if alias_map.get("LOW") else close
    atr14 = to_float(latest.get(alias_map["ATR14"])) if alias_map.get("ATR14") else float("nan")

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, high_20d]):
        return None

    candle_range = high - low if not pd.isna(high) and not pd.isna(low) else float("nan")
    range_pct = candle_range / close if close else float("nan")
    distance_to_ma20_pct = (close - ma20) / ma20 if ma20 else float("nan")
    distance_to_high_pct = (high_20d - close) / high_20d if high_20d else float("nan")
    breakout_above_high_pct = (close - high_20d) / high_20d if high_20d else float("nan")
    price_vs_ma50_pct = (close - ma50) / ma50 if ma50 else float("nan")
    atr_pct = atr14 / close if close and not pd.isna(atr14) else float("nan")

    extension_atr = float("nan")
    if not pd.isna(atr14) and atr14 > 0:
        extension_atr = (close - ma20) / atr14

    compression_ready = False
    if not pd.isna(range_pct):
        compression_ready = range_pct <= 0.08
    if not pd.isna(atr_pct):
        compression_ready = compression_ready and atr_pct <= 0.05

    return {
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "high_20d": round(high_20d, 2),
        "low_20d": round_or_none(low_20d),
        "avg_vol_20": round_or_none(avg_vol_20),
        "volume": round_or_none(volume),
        "atr14": round_or_none(atr14),
        "range_pct": range_pct,
        "atr_pct": atr_pct,
        "extension_atr": extension_atr,
        "distance_to_ma20_pct": distance_to_ma20_pct,
        "distance_to_high_pct": distance_to_high_pct,
        "breakout_above_high_pct": breakout_above_high_pct,
        "price_vs_ma50_pct": price_vs_ma50_pct,
        "compression_ready": compression_ready,
    }


def resolve_regime_bias(regime: str) -> str:
    regime = str(regime).upper().strip()
    if regime in {"BULLISH", "NEUTRAL", "BEARISH"}:
        return regime
    return "NEUTRAL"


def evaluate_pullback(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_ma20 = metrics["distance_to_ma20_pct"]

    ready_distance = cfg["ready_distance_to_ma20_pct"]
    neutral_ready_distance = cfg["neutral_ready_distance_to_ma20_pct"]
    max_extended = cfg["max_extended_from_ma20_pct"]
    reject_below_ma50 = cfg["reject_below_ma50"]

    if reject_below_ma50 and close < ma50:
        return "REJECTED", "none", "below_ma50", "Trend is gebroken onder MA50."

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish_regime", "Pullback is niet koopbaar in bearish regime."

    if pd.isna(dist_ma20):
        return "WAIT", "wait", "missing_ma20_distance", "Afstand tot MA20 kon niet berekend worden."

    if close < ma20:
        return "WAIT", "wait", "pullback_not_confirmed", "Prijs zit nog onder MA20. Eerst reclaim nodig."

    if dist_ma20 > max_extended:
        return "WAIT", "wait", "too_extended_above_ma20", "Prijs staat te ver boven MA20 voor een nette pullback."

    current_ready_band = ready_distance if regime == "BULLISH" else neutral_ready_distance
    if dist_ma20 <= current_ready_band and close > ma50:
        if regime == "BULLISH":
            return "READY", "buy", "pullback_ready_near_ma20", "Pullback is bevestigd dicht bij MA20 in een gezonde trend."
        return "READY", "buy", "pullback_ready_near_ma20_neutral", "Pullback is scherp genoeg ondanks neutraal regime."

    return "WAIT", "wait", "pullback_wait_better_entry", "Trend is ok, maar de instap ligt nog niet mooi bij MA20."


def evaluate_breakout(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_high = metrics["distance_to_high_pct"]
    breakout_above_high_pct = metrics["breakout_above_high_pct"]
    extension_atr = metrics.get("extension_atr")
    breakout_strength = metrics.get("breakout_strength")

    ready_near_high = cfg["ready_near_high_pct"]
    breakout_break_above_high_pct = cfg.get("breakout_break_above_high_pct", 0.002)
    max_breakout_chase_pct = cfg.get("max_breakout_chase_pct", 0.03)
    missed_breakout_pct = cfg.get("missed_breakout_pct", 0.08)
    reject_below_ma50 = cfg["reject_below_ma50"]

    if reject_below_ma50 and close < ma50:
        return (
            "REJECTED",
            "none",
            "below_ma50",
            "Trend is gebroken onder MA50.",
        )

    if regime == "BEARISH":
        return (
            "WAIT",
            "wait",
            "bearish_regime",
            "Breakout krijgt geen groen licht in bearish regime.",
        )

    if pd.isna(dist_high) or pd.isna(breakout_above_high_pct):
        return (
            "WAIT",
            "wait",
            "missing_high_distance",
            "Afstand tot de breakout-trigger kon niet berekend worden.",
        )

    if close < ma20:
        return (
            "WAIT",
            "wait",
            "below_ma20",
            "Prijs zit nog onder MA20. Breakout mist korte-trend steun.",
        )

    # 1. Extension krijgt voorrang.
    # Een aandeel kan dicht bij de trigger zitten, maar toch al te ver boven MA20 staan.
    if extension_atr is not None and not pd.isna(extension_atr):
        if extension_atr >= 3.5:
            return (
                "MISSED",
                "none",
                "too_extended_breakout",
                "Breakout is veel te ver opgelopen tegenover MA20. Niet najagen.",
            )

        if extension_atr >= 2.0:
            return (
                "WAIT",
                "wait",
                "extended_breakout",
                "Breakout is sterk, maar de koers staat te ver boven MA20. Wacht op pullback.",
            )

    # 2. Mini-step: breakout quality filter.
    # Zwakke breakouts mogen geen READY / STOP BUY worden.
    if breakout_strength is not None and not pd.isna(breakout_strength):
        if breakout_strength < 2.5:
            return (
                "WAIT",
                "wait",
                "weak_breakout",
                "Breakout mist kracht. Wacht op een betere setup.",
            )

    # 3. Te ver boven de breakout-trigger = gemiste trade.
    if breakout_above_high_pct >= missed_breakout_pct:
        return (
            "MISSED",
            "none",
            "missed_breakout",
            "Breakout is al te ver gevorderd. Niet najagen; wacht op een nieuwe setup of pullback.",
        )

    # 4. Net boven trigger = BUY NOW, maar alleen als hij niet te ver doorgeschoten is.
    if breakout_above_high_pct >= breakout_break_above_high_pct:
        if breakout_above_high_pct <= max_breakout_chase_pct:
            return (
                "READY",
                "buy",
                "breakout_triggered",
                "Prijs breekt gecontroleerd door de 20D high.",
            )

        return (
            "WAIT",
            "wait",
            "late_breakout",
            "Breakout is gebeurd, maar niet meer mooi instapbaar.",
        )

    # 5. Net onder trigger = SET STOP BUY, maar alleen als hij sterk genoeg en niet extended is.
    if dist_high <= ready_near_high:
        if regime == "NEUTRAL":
            return (
                "WAIT",
                "wait",
                "breakout_near_trigger_neutral",
                "Prijs zit dicht bij trigger, maar neutraal regime vraagt bevestiging.",
            )

        return (
            "READY",
            "buy",
            "breakout_near_trigger",
            "Prijs zit vlak onder de breakout-trigger.",
        )

    return (
        "WAIT",
        "wait",
        "breakout_below_trigger",
        "Prijs zit nog onder de breakout-trigger.",
    )

def evaluate_vcp(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_high = metrics["distance_to_high_pct"]
    breakout_above_high_pct = metrics["breakout_above_high_pct"]
    range_pct = metrics["range_pct"]
    compression_ready = metrics["compression_ready"]

    ready_near_high = cfg["ready_near_high_pct"]
    vcp_break_above_high_pct = cfg.get("vcp_break_above_high_pct", 0.002)
    vcp_max_range_pct = cfg.get("vcp_max_range_pct", 0.08)
    vcp_max_close_to_ma20_pct = cfg.get("vcp_max_close_to_ma20_pct", 0.04)
    reject_below_ma50 = cfg["reject_below_ma50"]

    if reject_below_ma50 and close < ma50:
        return "REJECTED", "none", "below_ma50", "VCP-structuur is ongeldig onder MA50."

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish_regime", "VCP wordt niet geactiveerd in bearish regime."

    if pd.isna(dist_high):
        return "WAIT", "wait", "missing_high_distance", "Afstand tot pivot kon niet berekend worden."

    if close < ma20:
        return "WAIT", "wait", "below_ma20", "Prijs zit nog onder MA20. Structuur is nog niet rijp."

    if abs(metrics["distance_to_ma20_pct"]) > vcp_max_close_to_ma20_pct:
        return "WAIT", "wait", "vcp_too_far_from_ma20", "Prijs zit te ver van MA20 voor een strakke VCP."

    if pd.isna(range_pct) or range_pct > vcp_max_range_pct or not compression_ready:
        return "WAIT", "wait", "vcp_pattern_not_ready", "Compressie is nog niet strak genoeg."

    if breakout_above_high_pct >= vcp_break_above_high_pct:
        return "READY", "buy", "vcp_breakout_triggered", "Compressie is klaar en prijs breekt al door de pivot."

    if dist_high <= ready_near_high:
        if regime == "NEUTRAL":
            return "WAIT", "wait", "vcp_near_trigger_neutral", "VCP zit dicht bij pivot, maar neutraal regime vraagt extra bevestiging."
        return "READY", "buy", "vcp_near_trigger", "Compressie is aanwezig en prijs zit vlak bij de pivot."

    return "WAIT", "wait", "vcp_wait_for_trigger", "Patroon oogt goed, maar de breakout-trigger is nog niet geraakt."


def apply_expiry(
    status: str,
    added_at: pd.Timestamp,
    expire_after_days: int,
    reason_code: str,
) -> tuple[str, Optional[str], Optional[str]]:
    if pd.isna(added_at):
        return status, None, None

    # Deze statussen mogen nooit door expiry overschreven worden
    if status in {"READY", "REJECTED", "MISSED"}:
        return status, None, None

    # Sterke aandelen die gewoon te extended zijn, niet verwijderen.
    # Dit zijn geen dode setups, maar "wacht op betere entry".
    keep_waiting_reasons = {
        "too_extended_above_ma20",
        "pullback_wait_better_entry",
        "breakout_below_trigger",
        "breakout_near_trigger_neutral",
        "vcp_wait_for_trigger",
        "vcp_near_trigger_neutral",
        "vcp_pattern_not_ready",
        "vcp_too_far_from_ma20",
    }

    if reason_code in keep_waiting_reasons:
        return status, None, None

    days_active = (pd.Timestamp.now().normalize() - added_at.normalize()).days

    if days_active > expire_after_days:
        return (
            "EXPIRED",
            f"expired_after_{days_active}_days",
            f"Setup staat al {days_active} dagen open zonder geldige trigger.",
        )

    return status, None, None


def get_setup_label(setup_type: str) -> str:
    mapping = {
        "PULLBACK": "Pullback",
        "BREAKOUT": "Breakout",
        "VCP": "Rustige opbouw",
    }
    return mapping.get(setup_type, setup_type.title())


def build_watchlist_plan(
    setup_type: str,
    status: str,
    reason_code: str,
    regime: str,
    metrics: dict,
) -> dict:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    high_20d = metrics["high_20d"]

    plan = {
        "setup_label": get_setup_label(setup_type),
        "action_now": "WAIT",
        "trigger_type": "none",
        "trigger_price": None,
        "entry_plan": "wait",
        "why_now": "Setup is nog niet klaar.",
        "urgency": "low",
    }

    if status == "READY":
        if setup_type == "PULLBACK":
            plan.update(
                action_now="BUY NOW",
                trigger_type="buy_now",
                trigger_price=close,
                entry_plan="market_or_limit",
                why_now="Pullback is bevestigd en de instapzone is actief.",
                urgency="high",
            )
        elif setup_type == "BREAKOUT":
            if reason_code == "breakout_near_trigger":
                plan.update(
                    action_now="SET STOP BUY",
                    trigger_type="buy_above",
                    trigger_price=high_20d,
                    entry_plan="stop_order",
                    why_now="Prijs zit vlak onder de breakout-trigger. Koop pas bij bevestiging boven de trigger.",
                    urgency="medium",
                )
            else:
                plan.update(
                    action_now="BUY NOW",
                    trigger_type="buy_now",
                    trigger_price=close,
                    entry_plan="market_or_stop",
                    why_now="Prijs breekt gecontroleerd door de breakout-trigger.",
                    urgency="high",
                )
        elif setup_type == "VCP":
            if reason_code == "vcp_near_trigger":
                plan.update(
                    action_now="SET STOP BUY",
                    trigger_type="buy_above",
                    trigger_price=high_20d,
                    entry_plan="stop_order",
                    why_now="VCP zit vlak onder pivot. Koop pas bij bevestiging boven de trigger.",
                    urgency="medium",
                )
            else:
                plan.update(
                    action_now="BUY NOW",
                    trigger_type="buy_now",
                    trigger_price=close,
                    entry_plan="market_or_stop",
                    why_now="VCP breekt gecontroleerd door de pivot.",
                    urgency="high",
                )
        return plan

    if reason_code in {"extended_breakout", "too_extended_breakout", "late_breakout", "missed_breakout"}:
        plan.update(
            action_now="SET LIMIT BUY",
            trigger_type="limit_buy",
            trigger_price=ma20,
            entry_plan="wait_for_pullback",
            why_now="Breakout is sterk, maar te ver opgelopen. Wacht op pullback richting MA20.",
            urgency="low",
        )
        return plan

    if status == "MISSED":
        plan.update(
            action_now="SET LIMIT BUY",
            trigger_type="limit_buy",
            trigger_price=ma20,
            entry_plan="wait_for_pullback",
            why_now="Breakout is al gebeurd. Niet najagen; wacht op pullback richting MA20.",
            urgency="low",
        )
        return plan

    if status == "REJECTED":
        plan.update(
            action_now="REMOVE",
            trigger_type="none",
            trigger_price=None,
            entry_plan="remove_from_watchlist",
            why_now="De setup is niet meer geldig.",
            urgency="medium",
        )
        return plan

    if status == "EXPIRED":
        plan.update(
            action_now="REMOVE",
            trigger_type="none",
            trigger_price=None,
            entry_plan="expire",
            why_now="De setup staat te lang open zonder trigger.",
            urgency="low",
        )
        return plan

    if reason_code in {"pullback_not_confirmed", "below_ma20"}:
        plan.update(
            action_now="WAIT",
            trigger_type="buy_above",
            trigger_price=ma20,
            entry_plan="stop_order",
            why_now="Eerst een reclaim boven MA20 nodig.",
            urgency="low",
        )
        return plan

    if reason_code in {"too_extended_above_ma20", "pullback_wait_better_entry"}:
        plan.update(
            action_now="SET LIMIT BUY",
            trigger_type="limit_buy",
            trigger_price=ma20,
            entry_plan="limit_order",
            why_now="Wacht op een rustigere pullback richting MA20.",
            urgency="low",
        )
        return plan

    if reason_code in {
        "breakout_below_trigger",
        "breakout_near_trigger_neutral",
        "vcp_wait_for_trigger",
        "vcp_near_trigger_neutral",
    }:
        plan.update(
            action_now="SET STOP BUY",
            trigger_type="buy_above",
            trigger_price=high_20d,
            entry_plan="stop_order",
            why_now="De setup leeft nog, maar koop pas bij bevestiging boven de trigger.",
            urgency="low" if regime == "NEUTRAL" else "medium",
        )
        return plan

    if reason_code in {"vcp_pattern_not_ready", "vcp_too_far_from_ma20"}:
        plan.update(
            action_now="WAIT",
            trigger_type="buy_above",
            trigger_price=high_20d,
            entry_plan="stop_order",
            why_now="De VCP-structuur is nog niet strak genoeg.",
            urgency="low",
        )
        return plan

    if reason_code == "bearish_regime":
        plan.update(
            action_now="WAIT",
            trigger_type="none",
            trigger_price=None,
            entry_plan="wait",
            why_now="De markt geeft tegenwind. Geen agressieve entries.",
            urgency="low",
        )
        return plan

    if reason_code in {"missing_ma20_distance", "missing_high_distance", "missing_indicator_data"}:
        plan.update(
            action_now="REMOVE",
            trigger_type="none",
            trigger_price=None,
            entry_plan="data_check",
            why_now="Er ontbreekt data om de setup veilig te beoordelen.",
            urgency="medium",
        )
        return plan

    plan.update(
        action_now="WAIT",
        trigger_type="none",
        trigger_price=None,
        entry_plan="wait",
        why_now="Wacht op betere bevestiging.",
        urgency="low",
    )
    return plan


def evaluate_row(
    row: pd.Series,
    regime: str,
    cfg: dict,
    breakout_strength_map: dict[str, float],
) -> dict:
    ticker = str(row.get("ticker", "")).upper().strip()
    setup_type = str(row.get("setup_type", "PULLBACK")).upper().strip()
    is_active = bool(row.get("is_active", True))
    added_at = parse_timestamp(row.get("added_at"))
    regime = resolve_regime_bias(regime)

    base_result = {
        "ticker": ticker,
        "is_active": is_active,
        "setup_type": setup_type,
        "setup_label": get_setup_label(setup_type),
        "status": "WAIT",
        "entry_bias": "wait",
        "action_now": "WAIT",
        "trigger_type": "none",
        "trigger_price": None,
        "entry_plan": "wait",
        "why_now": "",
        "urgency": "low",
        "added_at": row.get("added_at"),
        "last_reviewed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "",
        "reason_text": "",
        "reason_code": "",
        "regime": regime,
        "close": None,
        "ma20": None,
        "ma50": None,
        "ma200": None,
        "high_20d": None,
    }

    if not is_active:
        base_result["status"] = "INACTIVE"
        base_result["entry_bias"] = "none"
        base_result["reason"] = "Aandeel staat niet meer actief op de watchlist."
        base_result["reason_text"] = base_result["reason"]
        base_result["reason_code"] = "not_active"
        base_result["action_now"] = "WAIT"
        base_result["why_now"] = "Aandeel staat niet meer actief op de watchlist."
        return base_result

    df_ind = load_indicator_file(ticker)
    metrics = get_latest_metrics(df_ind)

    if metrics is None:
        base_result["status"] = "REJECTED"
        base_result["entry_bias"] = "none"
        base_result["reason"] = "Er ontbreekt indicatorendata om deze setup te evalueren."
        base_result["reason_text"] = base_result["reason"]
        base_result["reason_code"] = "missing_indicator_data"
        base_result["action_now"] = "REMOVE"
        base_result["entry_plan"] = "data_check"
        base_result["why_now"] = "Er ontbreekt indicatorendata om dit aandeel correct te evalueren."
        base_result["urgency"] = "medium"
        return base_result

    # Mini-step: breakout_strength toevoegen uit scanner_ranked.csv.
    # Dit verandert de CSV-structuur niet, maar maakt breakout-evaluatie slimmer.
    metrics["breakout_strength"] = breakout_strength_map.get(ticker)

    base_result["close"] = metrics["close"]
    base_result["ma20"] = metrics["ma20"]
    base_result["ma50"] = metrics["ma50"]
    base_result["ma200"] = metrics["ma200"]
    base_result["high_20d"] = metrics["high_20d"]

    if setup_type == "BREAKOUT":
        status, entry_bias, reason_code, reason_text = evaluate_breakout(
            metrics,
            regime,
            cfg,
        )
    elif setup_type == "VCP":
        status, entry_bias, reason_code, reason_text = evaluate_vcp(
            metrics,
            regime,
            cfg,
        )
    else:
        status, entry_bias, reason_code, reason_text = evaluate_pullback(
            metrics,
            regime,
            cfg,
        )

    status, expiry_reason_code, expiry_reason_text = apply_expiry(
        status,
        added_at,
        cfg["expire_after_days"],
        reason_code,
    )

    if expiry_reason_code:
        reason_code = expiry_reason_code
        reason_text = expiry_reason_text or reason_text
        entry_bias = "none"

    if status == "MISSED":
        entry_bias = "none"

    plan = build_watchlist_plan(
        setup_type,
        status,
        reason_code,
        regime,
        metrics,
    )

    base_result["status"] = status
    base_result["entry_bias"] = entry_bias
    base_result["reason"] = reason_text
    base_result["reason_text"] = reason_text
    base_result["reason_code"] = reason_code
    base_result["action_now"] = plan["action_now"]
    base_result["trigger_type"] = plan["trigger_type"]
    base_result["trigger_price"] = plan["trigger_price"]
    base_result["entry_plan"] = plan["entry_plan"]
    base_result["why_now"] = plan["why_now"]
    base_result["urgency"] = plan["urgency"]

    return base_result


def main() -> None:
    thresholds = load_thresholds()
    cfg = thresholds["watchlist"]
    regime = load_market_regime()

    active_df = load_watchlist_active()
    breakout_strength_map = load_breakout_strength_map()
    WATCHLIST_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    empty_cols = [
        "ticker",
        "is_active",
        "setup_type",
        "setup_label",
        "status",
        "entry_bias",
        "action_now",
        "trigger_type",
        "trigger_price",
        "entry_plan",
        "why_now",
        "urgency",
        "added_at",
        "last_reviewed_at",
        "reason",
        "reason_text",
        "reason_code",
        "regime",
        "close",
        "ma20",
        "ma50",
        "ma200",
        "high_20d",
    ]

    if active_df.empty:
        pd.DataFrame(columns=empty_cols).to_csv(WATCHLIST_STATUS_FILE, index=False)
        print(f"No active watchlist found. Empty status file written to: {WATCHLIST_STATUS_FILE}")
        return

    rows = [
        evaluate_row(row, regime, cfg, breakout_strength_map)
        for _, row in active_df.iterrows()
    ]
    result_df = pd.DataFrame(rows)

    for col in empty_cols:
        if col not in result_df.columns:
            result_df[col] = None

    result_df = result_df[empty_cols]

    sort_order = {
        "READY": 0,
        "MISSED": 1,
        "WAIT": 2,
        "REJECTED": 3,
        "EXPIRED": 4,
        "INACTIVE": 5,
    }
    result_df["sort_key"] = result_df["status"].map(sort_order).fillna(99)
    result_df = result_df.sort_values(["sort_key", "ticker"]).drop(columns=["sort_key"])

    result_df.to_csv(WATCHLIST_STATUS_FILE, index=False)
    print(f"Watchlist status written to: {WATCHLIST_STATUS_FILE}")


if __name__ == "__main__":
    main()