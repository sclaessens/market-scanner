from importlib import reload
from pathlib import Path

import pytest

from market_scanner.reporting import reporting_input_adapter
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
from market_scanner.reporting.telegram_renderer import (
    render_telegram_summary,
)


def _portfolio_rows():
    return (
        ReportingPortfolioDisplayInput(
            ticker="ASML",
            profit_loss_percent_display="+8.4%",
            current_price_display="€XXX",
            target_price_display="€XXX",
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-portfolio-asml",
        ),
        ReportingPortfolioDisplayInput(
            ticker="Thales",
            profit_loss_percent_display="+14.2%",
            current_price_display="€XXX",
            target_price_display="€XXX",
            action_status="HOLD",
            currency="EUR",
            source_reference="synthetic-portfolio-thales",
        ),
        ReportingPortfolioDisplayInput(
            ticker="Costco",
            profit_loss_percent_display="-3.1%",
            current_price_display="$XXX",
            target_price_display="$XXX",
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-portfolio-costco",
        ),
        ReportingPortfolioDisplayInput(
            ticker="ETFs",
            profit_loss_percent_display="+2-8%",
            current_price_display="not_applicable",
            target_price_display="not_applicable",
            action_status="not_applicable",
            currency="mixed",
            source_reference="synthetic-portfolio-etfs",
            instrument_group="ETF",
            compact_note="keep accumulating",
        ),
    )


def _candidate_rows():
    return (
        ReportingCandidateDisplayInput(
            ticker="AMD",
            candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.BELOW,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-amd",
        ),
        ReportingCandidateDisplayInput(
            ticker="ASML",
            candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
            threshold_price_display="€XXX",
            threshold_direction=ThresholdDirection.BELOW,
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-candidate-asml",
        ),
        ReportingCandidateDisplayInput(
            ticker="NVIDIA",
            candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.ABOVE,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-nvidia",
        ),
        ReportingCandidateDisplayInput(
            ticker="Meta",
            candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
            threshold_price_display="$XXX",
            threshold_direction=ThresholdDirection.ABOVE,
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-candidate-meta",
        ),
    )


def _data_status():
    return ReportingDataStatusInput(
        data_status="Fundamental data incomplete",
        review_reason="many REVIEW.",
        source_reference="synthetic-source-data-status",
    )


def _summary_input():
    return build_telegram_summary_input(
        portfolio_rows=_portfolio_rows(),
        candidate_rows=_candidate_rows(),
        data_status=_data_status(),
    )


def test_synthetic_adapter_builds_renderer_input_for_approved_message():
    assert (
        render_telegram_summary(_summary_input())
        == """Market Scanner

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
    )


def test_synthetic_portfolio_rows_are_routed_to_portfolio_section():
    summary = _summary_input()

    assert tuple(row.ticker for row in summary.portfolio_rows) == (
        "ASML",
        "Thales",
        "Costco",
        "ETFs",
    )


def test_candidate_groups_are_routed_to_expected_sections():
    summary = _summary_input()

    assert summary.buy_now_candidates == ()
    assert tuple(row.ticker for row in summary.pullback_candidates) == ("AMD", "ASML")
    assert tuple(row.ticker for row in summary.breakout_candidates) == (
        "NVIDIA",
        "Meta",
    )


def test_data_status_input_is_routed_to_data_status_section():
    summary = _summary_input()

    assert summary.data_status.data_status == "Fundamental data incomplete"
    assert summary.data_status.review_reason == "many REVIEW."


def test_values_are_preserved_exactly_as_supplied():
    portfolio_row = _summary_input().portfolio_rows[0]
    pullback_row = _summary_input().pullback_candidates[0]

    assert portfolio_row.profit_loss_percent_display == "+8.4%"
    assert portfolio_row.current_price_display == "€XXX"
    assert portfolio_row.target_price_display == "€XXX"
    assert pullback_row.threshold_price_display == "$XXX"
    assert pullback_row.threshold_direction == ThresholdDirection.BELOW


def test_missing_target_price_remains_explicit_not_zero():
    rows = (
        ReportingPortfolioDisplayInput(
            ticker="ASML",
            profit_loss_percent_display="+8.4%",
            current_price_display="€XXX",
            target_price_display="target unavailable",
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-missing-target",
        ),
    )

    rendered = render_telegram_summary(
        build_telegram_summary_input(
            portfolio_rows=rows,
            candidate_rows=(),
            data_status=_data_status(),
        )
    )

    assert "target unavailable" in rendered
    assert "target 0" not in rendered


def test_missing_profit_loss_remains_explicit_not_zero():
    rows = (
        ReportingPortfolioDisplayInput(
            ticker="Costco",
            profit_loss_percent_display="P/L unavailable",
            current_price_display="$XXX",
            target_price_display="$XXX",
            action_status="REVIEW",
            currency="USD",
            source_reference="synthetic-missing-profit-loss",
        ),
    )

    rendered = render_telegram_summary(
        build_telegram_summary_input(
            portfolio_rows=rows,
            candidate_rows=(),
            data_status=_data_status(),
        )
    )

    assert "P/L unavailable" in rendered
    assert "0%" not in rendered


def test_adapter_rejects_missing_required_display_values():
    rows = (
        ReportingPortfolioDisplayInput(
            ticker="ASML",
            profit_loss_percent_display="",
            current_price_display="€XXX",
            target_price_display="€XXX",
            action_status="REVIEW",
            currency="EUR",
            source_reference="synthetic-invalid-row",
        ),
    )

    with pytest.raises(ValueError, match="profit_loss_percent_display"):
        build_telegram_summary_input(
            portfolio_rows=rows,
            candidate_rows=(),
            data_status=_data_status(),
        )


def test_adapter_does_not_calculate_target_threshold_or_profit_loss_fields():
    source = Path(reporting_input_adapter.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "target_price_calculation",
        "buy_threshold_calculation",
        "breakout_threshold_calculation",
        "profit_loss_calculation",
        "current_price_fetch",
    ):
        assert forbidden not in source


def test_adapter_does_not_rank_score_or_create_action_status_values():
    source = Path(reporting_input_adapter.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "ranking",
        "score",
        "recommendation",
        "decision_override",
        "final_action",
    ):
        assert forbidden not in source


def test_adapter_output_avoids_forbidden_authority_language():
    rendered = render_telegram_summary(_summary_input()).lower()

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
        assert forbidden not in rendered


def test_importing_adapter_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(reporting_input_adapter)

    assert list(tmp_path.iterdir()) == []


def test_adapter_creates_no_report_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    build_telegram_summary_input(
        portfolio_rows=_portfolio_rows(),
        candidate_rows=_candidate_rows(),
        data_status=_data_status(),
    )

    assert list(tmp_path.iterdir()) == []
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_adapter_does_not_import_legacy_scripts():
    source = Path(reporting_input_adapter.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source
