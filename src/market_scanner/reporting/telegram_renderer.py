"""Pure in-memory renderer for the v2 Telegram UX scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from market_scanner.reporting.telegram_contracts import (
    CandidateDisplayGroup,
    TelegramSection,
    ThresholdDirection,
    empty_section_text,
    telegram_section_order,
)


@dataclass(frozen=True)
class PortfolioDisplayRecord:
    """Display-ready portfolio row supplied by upstream reporting inputs."""

    ticker: str
    profit_loss_percent_display: str
    current_price_display: str
    target_price_display: str
    action_status: str
    instrument_group: str | None = None
    compact_note: str | None = None


@dataclass(frozen=True)
class CandidateDisplayRecord:
    """Display-ready candidate row supplied by upstream reporting inputs."""

    ticker: str
    candidate_group: CandidateDisplayGroup
    threshold_price_display: str
    threshold_direction: ThresholdDirection
    action_status: str | None = None


@dataclass(frozen=True)
class DataStatusRecord:
    """Display-ready source-data status text supplied by upstream inputs."""

    data_status: str
    review_reason: str


@dataclass(frozen=True)
class TelegramSummaryInput:
    """Explicit in-memory input for the compact Telegram renderer."""

    header: str
    portfolio_rows: Sequence[PortfolioDisplayRecord]
    buy_now_candidates: Sequence[CandidateDisplayRecord]
    pullback_candidates: Sequence[CandidateDisplayRecord]
    breakout_candidates: Sequence[CandidateDisplayRecord]
    data_status: DataStatusRecord


def render_telegram_summary(summary: TelegramSummaryInput) -> str:
    """Render the approved portfolio-first Telegram message shape."""

    sections: list[str] = []

    for section in telegram_section_order():
        if section == TelegramSection.HEADER:
            sections.append(summary.header)
        elif section == TelegramSection.PORTFOLIO:
            sections.append(_render_portfolio_section(summary.portfolio_rows))
        elif section == TelegramSection.BUY_NOW:
            sections.append(
                _render_candidate_section(
                    title="Buy now",
                    group=CandidateDisplayGroup.BUY_NOW,
                    rows=summary.buy_now_candidates,
                )
            )
        elif section == TelegramSection.BUY_ON_PULLBACK:
            sections.append(
                _render_candidate_section(
                    title="Buy on pullback",
                    group=CandidateDisplayGroup.BUY_ON_PULLBACK,
                    rows=summary.pullback_candidates,
                )
            )
        elif section == TelegramSection.BUY_ON_BREAKOUT:
            sections.append(
                _render_candidate_section(
                    title="Buy on breakout",
                    group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
                    rows=summary.breakout_candidates,
                )
            )
        elif section == TelegramSection.DATA_STATUS:
            sections.append(_render_data_status_section(summary.data_status))

    return "\n\n".join(sections)


def render_portfolio_row(row: PortfolioDisplayRecord) -> str:
    """Render a display-ready portfolio row without calculating values."""

    if row.compact_note:
        return f"{row.ticker}: {row.profit_loss_percent_display} | {row.compact_note}"

    return (
        f"{row.ticker}: {row.profit_loss_percent_display} | "
        f"price {row.current_price_display} | "
        f"target {row.target_price_display} | "
        f"{row.action_status}"
    )


def render_candidate_row(row: CandidateDisplayRecord) -> str:
    """Render a display-ready candidate row without deciding its group."""

    if row.threshold_direction == ThresholdDirection.NOT_APPLICABLE:
        if row.action_status:
            return f"{row.ticker} | {row.action_status}"
        return row.ticker

    return (
        f"{row.ticker} {row.threshold_direction.value} "
        f"{row.threshold_price_display}"
    )


def render_data_status(row: DataStatusRecord) -> str:
    """Render compact data-status text without quality inference."""

    return f"{row.data_status} -> {row.review_reason}"


def _render_portfolio_section(rows: Sequence[PortfolioDisplayRecord]) -> str:
    rendered_rows = [render_portfolio_row(row) for row in rows]
    return "\n".join(("Portfolio", *rendered_rows))


def _render_data_status_section(row: DataStatusRecord) -> str:
    rendered_rows = [render_data_status(row)]
    return "\n".join(("Data status", *rendered_rows))


def _render_candidate_section(
    *,
    title: str,
    group: CandidateDisplayGroup,
    rows: Sequence[CandidateDisplayRecord],
) -> str:
    rendered_rows = [render_candidate_row(row) for row in rows]
    if not rendered_rows:
        rendered_rows = [empty_section_text()[group]]

    return "\n".join((title, *rendered_rows))
