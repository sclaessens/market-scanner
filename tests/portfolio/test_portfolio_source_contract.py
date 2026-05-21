from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.portfolio import build_portfolio
from scripts.core import build_portfolio_intelligence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_TRANSACTIONS = PROJECT_ROOT / "data" / "portfolio" / "portfolio_transactions.csv"
ACTIVE_POSITIONS = PROJECT_ROOT / "data" / "portfolio" / "portfolio_positions.csv"

TRANSACTION_DERIVED_COLUMNS = [
    "ticker",
    "quantity",
    "avg_cost",
    "status",
    "last_action",
    "last_action_at",
]

DESCRIPTIVE_ENRICHMENT_COLUMNS = [
    "last_price",
    "market_value",
    "unrealized_pnl",
    "pnl_pct",
]

PORTFOLIO_DECISION_AUTHORITY_COLUMNS = [
    "portfolio_action",
    "allocation_decision",
    "buy_decision",
    "sell_decision",
    "trim_recommendation",
    "add_recommendation",
]


def test_active_positions_are_rebuildable_from_transaction_source(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "portfolio_positions.csv"
    monkeypatch.setattr(build_portfolio, "TRANSACTIONS_FILE", str(ACTIVE_TRANSACTIONS))
    monkeypatch.setattr(build_portfolio, "POSITIONS_FILE", str(output_path))
    monkeypatch.setattr(build_portfolio, "PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))

    rebuilt = build_portfolio.build_portfolio()
    active = pd.read_csv(ACTIVE_POSITIONS)

    pd.testing.assert_frame_equal(
        active[TRANSACTION_DERIVED_COLUMNS].reset_index(drop=True),
        rebuilt[TRANSACTION_DERIVED_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
    assert active["ticker"].tolist() == ["COST", "MRVL", "ON", "TECK"]
    assert set(active["status"]) == {"OPEN"}
    assert active["ticker"].is_unique
    assert rebuilt["ticker"].is_unique
    assert len(active) == len(rebuilt)


def test_last_price_is_optional_descriptive_enrichment(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "portfolio_positions.csv"
    monkeypatch.setattr(build_portfolio, "TRANSACTIONS_FILE", str(ACTIVE_TRANSACTIONS))
    monkeypatch.setattr(build_portfolio, "POSITIONS_FILE", str(output_path))
    monkeypatch.setattr(build_portfolio, "PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))

    rebuilt = build_portfolio.build_portfolio()
    active = pd.read_csv(ACTIVE_POSITIONS)

    for column in DESCRIPTIVE_ENRICHMENT_COLUMNS:
        assert column in active.columns
        assert column in rebuilt.columns

    assert set(DESCRIPTIVE_ENRICHMENT_COLUMNS).isdisjoint(TRANSACTION_DERIVED_COLUMNS)
    assert active["last_price"].isna().all()

    for column in PORTFOLIO_DECISION_AUTHORITY_COLUMNS:
        assert column not in active.columns
        assert column not in rebuilt.columns


def test_last_action_is_not_portfolio_intelligence_authority(monkeypatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    portfolio_dir = tmp_path / "data" / "portfolio"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True)
    portfolio_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    timing_path = processed_dir / "timing_state_layer.csv"
    positions_path = portfolio_dir / "portfolio_positions.csv"
    output_path = processed_dir / "portfolio_intelligence.csv"
    log_path = logs_dir / "portfolio_intelligence_log.csv"

    pd.DataFrame(
        [
            {
                "ticker": "COST",
                "date": "2026-05-20",
                "quality_state": "INSUFFICIENT_DATA",
                "quality_reason": "source unavailable",
                "timing_state": "UNCLASSIFIED",
                "timing_reason": "source unavailable",
            },
            {
                "ticker": "ASML",
                "date": "2026-05-20",
                "quality_state": "INSUFFICIENT_DATA",
                "quality_reason": "source unavailable",
                "timing_state": "UNCLASSIFIED",
                "timing_reason": "source unavailable",
            },
        ]
    ).to_csv(timing_path, index=False)
    pd.read_csv(ACTIVE_POSITIONS).to_csv(positions_path, index=False)

    monkeypatch.setattr(build_portfolio_intelligence, "INPUT_PATH", timing_path)
    monkeypatch.setattr(build_portfolio_intelligence, "PORTFOLIO_PATH", positions_path)
    monkeypatch.setattr(build_portfolio_intelligence, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(build_portfolio_intelligence, "LOG_PATH", log_path)

    output = build_portfolio_intelligence.build_portfolio_intelligence()

    assert output["in_portfolio"].tolist() == ["PRESENT", "ABSENT"]
    assert "last_action" not in output.columns
    assert "last_action_at" not in output.columns
