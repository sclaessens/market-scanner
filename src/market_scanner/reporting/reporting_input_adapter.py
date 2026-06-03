"""Synthetic reporting input adapter for the v2 Telegram renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from market_scanner.reporting.reporting_input_contracts import (
    AGGREGATION_CONTRACT_VERSION,
    ReportingInputContractIssue,
    ReportingInputRole,
    validate_candidate_display_input_shape,
    validate_portfolio_display_input_shape,
    validate_reporting_input_trace_shape,
    validate_source_data_status_input_shape,
)
from market_scanner.reporting.telegram_contracts import (
    CandidateDisplayGroup,
    ThresholdDirection,
)
from market_scanner.reporting.telegram_renderer import (
    CandidateDisplayRecord,
    DataStatusRecord,
    PortfolioDisplayRecord,
    TelegramSummaryInput,
)


@dataclass(frozen=True)
class ReportingPortfolioDisplayInput:
    """Synthetic display-ready portfolio input with traceability."""

    ticker: str
    profit_loss_percent_display: str
    current_price_display: str
    target_price_display: str
    action_status: str
    currency: str
    source_reference: str
    instrument_group: str | None = None
    compact_note: str | None = None
    source_role: ReportingInputRole = ReportingInputRole.PORTFOLIO_DISPLAY_INPUT
    aggregation_contract_version: str = AGGREGATION_CONTRACT_VERSION


@dataclass(frozen=True)
class ReportingCandidateDisplayInput:
    """Synthetic display-ready candidate input with traceability."""

    ticker: str
    candidate_group: CandidateDisplayGroup
    threshold_price_display: str
    threshold_direction: ThresholdDirection
    action_status: str
    currency: str
    source_reference: str
    source_role: ReportingInputRole = ReportingInputRole.CANDIDATE_DISPLAY_INPUT
    aggregation_contract_version: str = AGGREGATION_CONTRACT_VERSION


@dataclass(frozen=True)
class ReportingDataStatusInput:
    """Synthetic display-ready source-data status input with traceability."""

    data_status: str
    review_reason: str
    source_reference: str
    source_role: ReportingInputRole = ReportingInputRole.SOURCE_DATA_STATUS_INPUT
    aggregation_contract_version: str = AGGREGATION_CONTRACT_VERSION


def build_telegram_summary_input(
    *,
    portfolio_rows: Sequence[ReportingPortfolioDisplayInput],
    candidate_rows: Sequence[ReportingCandidateDisplayInput],
    data_status: ReportingDataStatusInput,
    header: str = "Market Scanner",
) -> TelegramSummaryInput:
    """Build renderer input from explicit synthetic display records."""

    for row in portfolio_rows:
        _raise_if_issues(_validate_portfolio_row(row))

    for row in candidate_rows:
        _raise_if_issues(_validate_candidate_row(row))

    _raise_if_issues(_validate_data_status(data_status))

    buy_now_candidates: list[CandidateDisplayRecord] = []
    pullback_candidates: list[CandidateDisplayRecord] = []
    breakout_candidates: list[CandidateDisplayRecord] = []

    for row in candidate_rows:
        rendered_row = CandidateDisplayRecord(
            ticker=row.ticker,
            candidate_group=row.candidate_group,
            threshold_price_display=row.threshold_price_display,
            threshold_direction=row.threshold_direction,
            action_status=row.action_status,
        )
        if row.candidate_group == CandidateDisplayGroup.BUY_NOW:
            buy_now_candidates.append(rendered_row)
        elif row.candidate_group == CandidateDisplayGroup.BUY_ON_PULLBACK:
            pullback_candidates.append(rendered_row)
        elif row.candidate_group == CandidateDisplayGroup.BUY_ON_BREAKOUT:
            breakout_candidates.append(rendered_row)

    return TelegramSummaryInput(
        header=header,
        portfolio_rows=tuple(
            PortfolioDisplayRecord(
                ticker=row.ticker,
                profit_loss_percent_display=row.profit_loss_percent_display,
                current_price_display=row.current_price_display,
                target_price_display=row.target_price_display,
                action_status=row.action_status,
                instrument_group=row.instrument_group,
                compact_note=row.compact_note,
            )
            for row in portfolio_rows
        ),
        buy_now_candidates=tuple(buy_now_candidates),
        pullback_candidates=tuple(pullback_candidates),
        breakout_candidates=tuple(breakout_candidates),
        data_status=DataStatusRecord(
            data_status=data_status.data_status,
            review_reason=data_status.review_reason,
        ),
    )


def _validate_portfolio_row(
    row: ReportingPortfolioDisplayInput,
) -> tuple[ReportingInputContractIssue, ...]:
    record = {
        "ticker": row.ticker,
        "profit_loss_percent_display": row.profit_loss_percent_display,
        "current_price_display": row.current_price_display,
        "target_price_display": row.target_price_display,
        "action_status": row.action_status,
        "currency": row.currency,
        "source_reference": row.source_reference,
    }
    return (
        validate_portfolio_display_input_shape(record)
        + validate_reporting_input_trace_shape(_trace_record(row))
    )


def _validate_candidate_row(
    row: ReportingCandidateDisplayInput,
) -> tuple[ReportingInputContractIssue, ...]:
    record = {
        "ticker": row.ticker,
        "candidate_group": row.candidate_group.value,
        "threshold_price_display": row.threshold_price_display,
        "threshold_direction": row.threshold_direction.value,
        "action_status": row.action_status,
        "currency": row.currency,
        "source_reference": row.source_reference,
    }
    return (
        validate_candidate_display_input_shape(record)
        + validate_reporting_input_trace_shape(_trace_record(row))
    )


def _validate_data_status(
    row: ReportingDataStatusInput,
) -> tuple[ReportingInputContractIssue, ...]:
    record = {
        "data_status": row.data_status,
        "review_reason": row.review_reason,
        "source_reference": row.source_reference,
    }
    return (
        validate_source_data_status_input_shape(record)
        + validate_reporting_input_trace_shape(_trace_record(row))
    )


def _trace_record(
    row: ReportingPortfolioDisplayInput
    | ReportingCandidateDisplayInput
    | ReportingDataStatusInput,
) -> dict[str, str]:
    return {
        "source_role": row.source_role.value,
        "source_reference": row.source_reference,
        "aggregation_contract_version": row.aggregation_contract_version,
    }


def _raise_if_issues(issues: tuple[ReportingInputContractIssue, ...]) -> None:
    if issues:
        issue_text = ", ".join(
            f"{issue.field_name}:{issue.issue_code.value}" for issue in issues
        )
        raise ValueError(f"Invalid reporting input record: {issue_text}")
