from datetime import datetime
from config.settings import REPORT_EMPTY_TEXT


def format_section(items):
    if not items:
        return f"- {REPORT_EMPTY_TEXT}"
    return "\n".join(f"- {item}" for item in items)


def build_report(universe_size, liquid_universe_size, regime, vcp, pullbacks, breakouts, weakening):
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""# Legacy Market Scan - {today}

Universe size (raw): **{universe_size}**
Universe size (liquidity-classified): **{liquid_universe_size}**

## Market Regime (QQQ)
{regime}

## VCP setups
{format_section(vcp)}

## Pullback setups
{format_section(pullbacks)}

## Breakouts
{format_section(breakouts)}

## Trend failures / weakening
{format_section(weakening)}
"""
