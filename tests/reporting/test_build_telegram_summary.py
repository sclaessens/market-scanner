import pandas as pd

from scripts.reporting.build_telegram_summary import append_all_decision_sections


def test_low_information_scanner_observations_are_summarized():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "final_action": "REVIEW",
                "source_layer": "SCANNER",
                "setup_type": None,
                "tradeability": "NOT_TRADEABLE",
                "conviction": "LOW",
                "allocation_priority": 25,
                "validation_state": "INCOMPLETE",
                "context_strength": "WEAK",
                "timing_state": "UNKNOWN",
                "portfolio_state": "NONE",
                "execution_style": "PASSIVE",
                "trigger_price": None,
                "close": 10,
                "decision_reason": "structure_not_coherent:no_setup",
            },
            {
                "ticker": "BBB",
                "final_action": "WAIT",
                "source_layer": "SCANNER",
                "setup_type": "BREAKOUT",
                "tradeability": "WATCH",
                "conviction": "HIGH",
                "allocation_priority": 75,
                "validation_state": "COHERENT",
                "context_strength": "STRONG",
                "timing_state": "UNKNOWN",
                "portfolio_state": "NONE",
                "execution_style": "PASSIVE",
                "trigger_price": None,
                "close": 20,
                "decision_reason": "classification_recognized_allocation_not_ready:coherent_breakout",
            },
        ]
    )

    lines = []
    append_all_decision_sections(lines, df)
    message = "\n".join(lines)

    assert "Watch / Observation Candidates" in message
    assert "- BREAKOUT / STRONG: 1 | examples: BBB" in message
    assert "- AAA |" not in message
    assert "Low-information scanner observations omitted: 1" in message
    assert "final_decisions.csv" in message
    assert "tradeability" not in message
    assert "DE order" not in message
    assert "validation" not in message


def test_watchlist_review_rows_are_displayed():
    df = pd.DataFrame(
        [
            {
                "ticker": "CCC",
                "final_action": "REVIEW",
                "source_layer": "WATCHLIST",
                "setup_type": None,
                "tradeability": "NOT_TRADEABLE",
                "conviction": "LOW",
                "allocation_priority": 25,
                "validation_state": "INCOMPLETE",
                "context_strength": "WEAK",
                "timing_state": "WAIT",
                "portfolio_state": "NONE",
                "execution_style": "PASSIVE",
                "trigger_price": 12,
                "close": 11,
                "decision_reason": "structure_not_coherent:no_setup",
            }
        ]
    )

    lines = []
    append_all_decision_sections(lines, df)
    message = "\n".join(lines)

    assert "Portfolio / Active Decisions" in message
    assert "REVIEW" in message
    assert "- CCC — REVIEW | timing WAIT | context WEAK | trigger 12.00 | close 11.00" in message
    assert "Low-information scanner observations omitted" not in message


def test_compact_rows_do_not_dump_internal_fields():
    df = pd.DataFrame(
        [
            {
                "ticker": "DDD",
                "final_action": "HOLD",
                "source_layer": "PORTFOLIO",
                "setup_type": None,
                "tradeability": "HELD",
                "conviction": "MEDIUM",
                "allocation_priority": 40,
                "validation_state": "UNKNOWN",
                "context_strength": "UNKNOWN",
                "timing_state": "UNKNOWN",
                "portfolio_state": "NORMAL",
                "execution_style": "NONE",
                "trigger_price": None,
                "close": 42,
                "decision_reason": "portfolio_state_normal",
            }
        ]
    )

    lines = []
    append_all_decision_sections(lines, df)
    message = "\n".join(lines)

    assert "- DDD — HOLD" in message
    assert "portfolio NORMAL" in message
    assert "close 42.00" in message
    assert "source PORTFOLIO" not in message
    assert "tradeability HELD" not in message
    assert "confidence MEDIUM" not in message
    assert "DE order 40" not in message
    assert "validation UNKNOWN" not in message
    assert "style" not in message
