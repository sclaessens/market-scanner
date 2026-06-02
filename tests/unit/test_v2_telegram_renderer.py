from importlib import reload
from pathlib import Path

from market_scanner.reporting import telegram_renderer
from market_scanner.reporting.telegram_contracts import (
    CandidateDisplayGroup,
    ThresholdDirection,
)
from market_scanner.reporting.telegram_renderer import (
    CandidateDisplayRecord,
    DataStatusRecord,
    PortfolioDisplayRecord,
    TelegramSummaryInput,
    render_candidate_row,
    render_portfolio_row,
    render_telegram_summary,
)


def _synthetic_summary() -> TelegramSummaryInput:
    return TelegramSummaryInput(
        header="Market Scanner",
        portfolio_rows=(
            PortfolioDisplayRecord(
                ticker="ASML",
                profit_loss_percent_display="+8.4%",
                current_price_display="€XXX",
                target_price_display="€XXX",
                action_status="REVIEW",
            ),
            PortfolioDisplayRecord(
                ticker="Thales",
                profit_loss_percent_display="+14.2%",
                current_price_display="€XXX",
                target_price_display="€XXX",
                action_status="HOLD",
            ),
            PortfolioDisplayRecord(
                ticker="Costco",
                profit_loss_percent_display="-3.1%",
                current_price_display="$XXX",
                target_price_display="$XXX",
                action_status="REVIEW",
            ),
            PortfolioDisplayRecord(
                ticker="ETFs",
                profit_loss_percent_display="+2-8%",
                current_price_display="not_applicable",
                target_price_display="not_applicable",
                action_status="not_applicable",
                instrument_group="ETF",
                compact_note="keep accumulating",
            ),
        ),
        buy_now_candidates=(),
        pullback_candidates=(
            CandidateDisplayRecord(
                ticker="AMD",
                candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
                threshold_price_display="$XXX",
                threshold_direction=ThresholdDirection.BELOW,
            ),
            CandidateDisplayRecord(
                ticker="ASML",
                candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
                threshold_price_display="€XXX",
                threshold_direction=ThresholdDirection.BELOW,
            ),
        ),
        breakout_candidates=(
            CandidateDisplayRecord(
                ticker="NVIDIA",
                candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
                threshold_price_display="$XXX",
                threshold_direction=ThresholdDirection.ABOVE,
            ),
            CandidateDisplayRecord(
                ticker="Meta",
                candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
                threshold_price_display="$XXX",
                threshold_direction=ThresholdDirection.ABOVE,
            ),
        ),
        data_status=DataStatusRecord(
            data_status="Fundamental data incomplete",
            review_reason="many REVIEW.",
        ),
    )


def test_renderer_matches_approved_synthetic_message_shape():
    assert (
        render_telegram_summary(_synthetic_summary())
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


def test_portfolio_section_appears_before_candidate_sections():
    message = render_telegram_summary(_synthetic_summary())

    assert message.index("Portfolio") < message.index("Buy now")
    assert message.index("Buy now") < message.index("Buy on pullback")
    assert message.index("Buy on pullback") < message.index("Buy on breakout")
    assert message.index("Buy on breakout") < message.index("Data status")


def test_portfolio_rows_render_display_ready_values_only():
    row = PortfolioDisplayRecord(
        ticker="ASML",
        profit_loss_percent_display="+8.4%",
        current_price_display="€XXX",
        target_price_display="€XXX",
        action_status="REVIEW",
    )

    assert render_portfolio_row(row) == (
        "ASML: +8.4% | price €XXX | target €XXX | REVIEW"
    )


def test_etf_aggregate_row_can_render_compactly():
    row = PortfolioDisplayRecord(
        ticker="ETFs",
        profit_loss_percent_display="+2-8%",
        current_price_display="not_applicable",
        target_price_display="not_applicable",
        action_status="not_applicable",
        instrument_group="ETF",
        compact_note="keep accumulating",
    )

    assert render_portfolio_row(row) == "ETFs: +2-8% | keep accumulating"


def test_empty_buy_now_section_stays_explicit_and_compact():
    message = render_telegram_summary(_synthetic_summary())

    assert "Buy now\nNo candidates today." in message


def test_pullback_and_breakout_rows_use_upstream_threshold_direction():
    pullback = CandidateDisplayRecord(
        ticker="AMD",
        candidate_group=CandidateDisplayGroup.BUY_ON_PULLBACK,
        threshold_price_display="$XXX",
        threshold_direction=ThresholdDirection.BELOW,
    )
    breakout = CandidateDisplayRecord(
        ticker="NVIDIA",
        candidate_group=CandidateDisplayGroup.BUY_ON_BREAKOUT,
        threshold_price_display="$XXX",
        threshold_direction=ThresholdDirection.ABOVE,
    )

    assert render_candidate_row(pullback) == "AMD below $XXX"
    assert render_candidate_row(breakout) == "NVIDIA above $XXX"


def test_data_status_line_renders_compactly():
    message = render_telegram_summary(_synthetic_summary())

    assert "Data status\nFundamental data incomplete -> many REVIEW." in message


def test_missing_profit_loss_and_target_remain_explicit_not_zero():
    row = PortfolioDisplayRecord(
        ticker="ASML",
        profit_loss_percent_display="P/L unavailable",
        current_price_display="€XXX",
        target_price_display="target unavailable",
        action_status="REVIEW",
    )

    rendered = render_portfolio_row(row)

    assert "P/L unavailable" in rendered
    assert "target unavailable" in rendered
    assert "0%" not in rendered
    assert "target 0" not in rendered


def test_renderer_output_avoids_forbidden_authority_language():
    message = render_telegram_summary(_synthetic_summary()).lower()

    for forbidden in (
        "urgency",
        "conviction",
        "tradeability",
        "ranking",
        "score",
        "recommended",
        "priority",
    ):
        assert forbidden not in message


def test_renderer_output_avoids_legacy_debug_labels_near_top():
    message = render_telegram_summary(_synthetic_summary())
    first_lines = "\n".join(message.splitlines()[:12])

    for forbidden in (
        "Reporting contract",
        "Source artifact",
        "Dashboard artifact",
        "omitted_row_count",
        "Grouping rule",
        "Truncation rule",
    ):
        assert forbidden not in first_lines


def test_importing_renderer_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(telegram_renderer)

    assert list(tmp_path.iterdir()) == []


def test_renderer_does_not_create_report_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    render_telegram_summary(_synthetic_summary())

    assert not Path("reports/daily/telegram_message.txt").exists()
    assert list(tmp_path.iterdir()) == []


def test_renderer_does_not_import_legacy_scripts():
    source = Path(telegram_renderer.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source
