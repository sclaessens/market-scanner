from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reporting import build_reporting_layer as reporting
from scripts.reporting import build_telegram_summary


def test_telegram_summary_wrapper_delegates_to_reporting_layer(
    monkeypatch,
    tmp_path: Path,
):
    final_decisions = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-05-10",
                "final_action": "HOLD",
                "allocation_decision": "SOURCE_HOLD",
                "execution_decision": "SOURCE_NONE",
                "portfolio_decision_state": "SOURCE_PORTFOLIO",
                "opportunity_decision_state": "SOURCE_OPPORTUNITY",
                "arbitration_state": "SOURCE_CLEAR",
                "allocation_rationale": "source allocation rationale",
                "execution_rationale": "source execution rationale",
                "arbitration_reason": "source arbitration reason",
                "conflict_resolution_reason": "source conflict reason",
                "source_provenance": "DECISION_ENGINE",
                "decision_contract_version": "DECISION_CONTRACT_V1",
                "input_row_hash": "hash-aaa",
            }
        ]
    )
    final_path = tmp_path / "data/processed/final_decisions.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_decisions.to_csv(final_path, index=False)

    monkeypatch.setattr(reporting, "FINAL_DECISIONS_FILE", final_path)
    monkeypatch.setattr(reporting, "STABILITY_STATE_FILE", tmp_path / "data/processed/stability_state.csv")
    monkeypatch.setattr(
        reporting,
        "REPORTING_DASHBOARD_FILE",
        tmp_path / "data/processed/reporting_dashboard_data.csv",
    )
    monkeypatch.setattr(reporting, "REPORTING_LOG_FILE", tmp_path / "data/logs/reporting_layer_log.csv")
    monkeypatch.setattr(reporting, "TELEGRAM_MESSAGE_FILE", tmp_path / "reports/daily/telegram_message.txt")
    monkeypatch.setattr(build_telegram_summary, "TELEGRAM_MESSAGE_FILE", reporting.TELEGRAM_MESSAGE_FILE)

    wrapper_text = build_telegram_summary.build_telegram_summary_text()
    _, _, authoritative_text = reporting.build_reporting_layer()

    assert wrapper_text == authoritative_text
    assert "Decision output: HOLD" in wrapper_text
    assert "Low-information scanner observations omitted" not in wrapper_text
