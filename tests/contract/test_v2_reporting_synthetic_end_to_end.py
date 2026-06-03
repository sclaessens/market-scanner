from pathlib import Path

from market_scanner.reporting.reporting_input_adapter import (
    ReportingCandidateDisplayInput,
    ReportingDataStatusInput,
    ReportingPortfolioDisplayInput,
    build_telegram_summary_input,
)
from market_scanner.reporting.telegram_contracts import (
    CandidateDisplayGroup,
    ThresholdDirection,
)
from market_scanner.reporting.telegram_renderer import render_telegram_summary


EXPECTED_COMPACT_MESSAGE = """Market Scanner

Portfolio
ASML: +8.4% | price €XXX | target €XXX | REVIEW
Thales: +14.2% | price €XXX | target €XXX | HOLD
Costco: -3.1% | price $XXX | target $XXX | REVIEW
ETFs: +2-8% | keep accumulating

Buy now
No candidates today.

Buy on pullback
AMD below $XXX
ASML below €XXX

Buy on breakout
NVIDIA above $XXX
Meta above $XXX

Data status
Fundamental data incomplete -> many REVIEW."""


def _synthetic_portfolio_display_input():
    return (
        ReportingPortfolioDisplayInput(
            ticker="ASML",
            profit_loss_percent_display="+8.4%",
            current_price_display="€XXX",
            target_price_display="€XXX",
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-portfolio-display-asml",
        ),
        ReportingPortfolioDisplayInput(
            ticker="Thales",
            profit_loss_percent_display="+14.2%",
            current_price_display="€XXX",
            target_price_display="€XXX",
            action_status="HOLD",
            currency="EUR",
            source_reference="synthetic-portfolio-display-thales",
        ),
        ReportingPortfolioDisplayInput(
            ticker="Costco",
            profit_loss_percent_display="-3.1%",
            current_price_display="$XXX",
            target_price_display="$XXX",
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-portfolio-display-costco",
        ),
        ReportingPortfolioDisplayInput(
            ticker="ETFs",
            profit_loss_percent_display="+2-8%",
            current_price_display="not_applicable",
            target_price_display="not_applicable",
            action_status="not_applicable",
            currency="mixed",
            source_reference="synthetic-portfolio-display-etfs",
            instrument_group="ETF",
            compact_note="keep accumulating",
        ),
    )


def _synthetic_candidate_display_input():
    return (
        ReportingCandidateDisplayInput(
            ticker="AMD",
            candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.BELOW,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-display-amd",
        ),
        ReportingCandidateDisplayInput(
            ticker="ASML",
            candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
            threshold_price_display="€XXX",
            threshold_direction=ThresholdDirection.BELOW,
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-candidate-display-asml",
        ),
        ReportingCandidateDisplayInput(
            ticker="NVIDIA",
            candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.ABOVE,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-display-nvidia",
        ),
        ReportingCandidateDisplayInput(
            ticker="Meta",
            candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.ABOVE,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-display-meta",
        ),
    )


def _synthetic_source_data_readiness_input():
    return ReportingDataStatusInput(
        data_status="Fundamental data incomplete",
        review_reason="many REVIEW.",
        source_reference="synthetic-source-data-readiness",
    )


def test_synthetic_reporting_chain_renders_approved_compact_message():
    summary_input = build_telegram_summary_input(
        portfolio_rows=_synthetic_portfolio_display_input(),
        candidate_rows=_synthetic_candidate_display_input(),
        data_status=_synthetic_source_data_readiness_input(),
    )

    message = render_telegram_summary(summary_input)

    assert message == EXPECTED_COMPACT_MESSAGE


def test_synthetic_reporting_chain_preserves_contract_section_routing():
    summary_input = build_telegram_summary_input(
        portfolio_rows=_synthetic_portfolio_display_input(),
        candidate_rows=_synthetic_candidate_display_input(),
        data_status=_synthetic_source_data_readiness_input(),
    )

    assert tuple(row.ticker for row in summary_input.portfolio_rows) == (
        "ASML",
        "Thales",
        "Costco",
        "ETFs",
    )
    assert summary_input.buy_now_candidates == ()
    assert tuple(row.ticker for row in summary_input.pullback_candidates) == (
        "AMD",
        "ASML",
    )
    assert tuple(row.ticker for row in summary_input.breakout_candidates) == (
        "NVIDIA",
        "Meta",
    )
    assert summary_input.data_status.data_status == "Fundamental data incomplete"
    assert summary_input.data_status.review_reason == "many REVIEW."


def test_synthetic_reporting_chain_is_in_memory_and_creates_no_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    summary_input = build_telegram_summary_input(
        portfolio_rows=_synthetic_portfolio_display_input(),
        candidate_rows=_synthetic_candidate_display_input(),
        data_status=_synthetic_source_data_readiness_input(),
    )
    render_telegram_summary(summary_input)

    assert list(tmp_path.iterdir()) == []
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_synthetic_reporting_chain_does_not_emit_forbidden_authority_language():
    summary_input = build_telegram_summary_input(
        portfolio_rows=_synthetic_portfolio_display_input(),
        candidate_rows=_synthetic_candidate_display_input(),
        data_status=_synthetic_source_data_readiness_input(),
    )
    message = render_telegram_summary(summary_input).lower()

    for forbidden in (
        "allocation",
        "execution",
        "urgency",
        "conviction",
        "tradeability",
        "ranking",
        "score",
        "recommendation",
    ):
        assert forbidden not in message
