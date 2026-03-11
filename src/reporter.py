from datetime import datetime
from config.settings import REPORT_EMPTY_TEXT


def format_section(items):

    if not items:
        return f"- {REPORT_EMPTY_TEXT}"

    return "\n".join(f"- {i}" for i in items)


def build_report(
    universe_raw,
    universe_filtered,
    regime,
    vcp,
    pullbacks,
    breakouts,
    weakening
):

    today = datetime.now().strftime("%Y-%m-%d")

    report = f"""# Market Scan — {today}

Universe size (raw): **{universe_raw}**
Universe size (liquid-filtered): **{universe_filtered}**

## Market Regime (QQQ)
{regime}

## VCP setups (compression → potential breakout)
{format_section(vcp)}

## Pullback setups (actionable)
{format_section(pullbacks)}

## Breakouts (watch for pullback/retest)
{format_section(breakouts)}

## Trend failures / weakening (review exits)
{format_section(weakening)}
"""

    return report
