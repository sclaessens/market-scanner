from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_full_pipeline, run_scan


def test_run_full_pipeline_step_prints_neutral_success(monkeypatch, capsys):
    calls = []

    def fake_run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_full_pipeline.subprocess, "run", fake_run)

    run_full_pipeline.run_step("Example step", ["python", "example.py"])

    output = capsys.readouterr().out
    assert calls == [["python", "example.py"]]
    assert "Pipeline step started: Example step" in output
    assert "Command: python example.py" in output
    assert "Pipeline step completed: Example step" in output


def test_run_full_pipeline_step_prints_failure_context(monkeypatch, capsys):
    def fake_run(command):
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_full_pipeline.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_full_pipeline.run_step("Example step", ["python", "example.py"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 7
    assert "Pipeline step failed: Example step" in output
    assert "Return code: 7" in output


def test_scan_artifact_message_includes_optional_row_count():
    assert (
        run_scan.format_artifact_message(run_scan.SCANNER_RANKED_FILE, row_count=12)
        == f"Artifact written: {run_scan.SCANNER_RANKED_FILE} rows=12"
    )
    assert (
        run_scan.format_artifact_message(run_scan.TELEGRAM_MESSAGE_FILE)
        == f"Artifact written: {run_scan.TELEGRAM_MESSAGE_FILE}"
    )


def test_scan_progress_uses_neutral_operational_language(capsys):
    run_scan.print_scan_progress(
        processed_count=25,
        total_count=100,
        setup_count=4,
        failed_count=1,
    )

    output = capsys.readouterr().out
    assert "Scanner progress: processed=25/100" in output
    assert "setup_rows_collected=4" in output
    assert "failed_rows=1" in output
    assert "tradeable" not in output.lower()
    assert "conviction" not in output.lower()
    assert "urgency" not in output.lower()


def test_run_scan_rebuilds_required_layers_before_final_decisions(monkeypatch, tmp_path: Path):
    order = []

    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"
    portfolio_dir = tmp_path / "data" / "portfolio"
    reports_dir = tmp_path / "reports" / "daily"
    for directory in [processed_dir, logs_dir, portfolio_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_scan, "MIN_HISTORY_ROWS", 1)
    monkeypatch.setattr(run_scan, "SCAN_PROGRESS_INTERVAL", 1)
    monkeypatch.setattr(run_scan, "FAILED_TICKERS_FILE", logs_dir / "failed_tickers.csv")
    monkeypatch.setattr(run_scan, "TELEGRAM_MESSAGE_FILE", reports_dir / "telegram_message.txt")
    monkeypatch.setattr(run_scan, "SCANNER_RANKED_FILE", processed_dir / "scanner_ranked.csv")
    monkeypatch.setattr(run_scan, "VALIDATION_LAYER_FILE", processed_dir / "validation_layer.csv")
    monkeypatch.setattr(run_scan, "CONTEXT_LAYER_FILE", processed_dir / "context_strength.csv")
    monkeypatch.setattr(run_scan, "FUNDAMENTAL_QUALITY_FILE", processed_dir / "fundamental_quality.csv")
    monkeypatch.setattr(run_scan, "TIMING_STATE_LAYER_FILE", processed_dir / "timing_state_layer.csv")
    monkeypatch.setattr(run_scan, "PORTFOLIO_POSITIONS_FILE", portfolio_dir / "portfolio_positions.csv")
    monkeypatch.setattr(run_scan, "PORTFOLIO_REVIEW_FILE", portfolio_dir / "portfolio_review.csv")
    monkeypatch.setattr(run_scan, "PORTFOLIO_INTELLIGENCE_FILE", processed_dir / "portfolio_intelligence.csv")
    monkeypatch.setattr(run_scan, "FINAL_DECISIONS_FILE", processed_dir / "final_decisions.csv")
    monkeypatch.setattr(run_scan, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_scan, "load_tickers", lambda: ["AAA"])
    monkeypatch.setattr(run_scan, "classify_market_regime", lambda **_: "BULLISH")
    monkeypatch.setattr(
        run_scan,
        "fetch_ohlcv_data",
        lambda ticker: pd.DataFrame({"Close": [100.0]}),
    )
    monkeypatch.setattr(
        run_scan,
        "add_indicators",
        lambda df: df.assign(MA50=95.0, MA200=90.0),
    )
    monkeypatch.setattr(
        run_scan,
        "scan_ticker",
        lambda **_: {
            "ticker": "AAA",
            "date": "2026-05-19",
            "sector": "TECHNOLOGY",
            "rs_20d_pct": 1.0,
        },
    )
    monkeypatch.setattr(run_scan, "rank_setups", lambda setups, top_n: setups)

    def record(name: str, rows: list[dict]) -> pd.DataFrame:
        order.append(name)
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        run_scan,
        "build_validation_layer",
        lambda: record("validation", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_context_layer",
        lambda: record("context", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_layer",
        lambda: record("fundamental", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_timing_state_layer",
        lambda: record("timing", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_portfolio",
        lambda: record("portfolio_state", [{"ticker": "AAA", "status": "OPEN"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "evaluate_positions",
        lambda: record("portfolio_review", [{"ticker": "AAA", "risk_state": "NORMAL"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_portfolio_intelligence",
        lambda: record("portfolio_intelligence", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_final_decisions",
        lambda: record("final_decisions", [{"ticker": "AAA", "date": "2026-05-19"}]),
    )
    monkeypatch.setattr(run_scan, "build_telegram_summary_text", lambda: "summary")
    monkeypatch.setattr(run_scan, "save_summary", lambda text: order.append("reporting"))
    monkeypatch.setattr(run_scan, "send_daily_summary", lambda: order.append("telegram_delivery"))

    run_scan.main()

    assert order == [
        "validation",
        "context",
        "fundamental",
        "timing",
        "portfolio_state",
        "portfolio_review",
        "portfolio_intelligence",
        "final_decisions",
        "reporting",
        "telegram_delivery",
    ]
    assert order.index("portfolio_intelligence") < order.index("final_decisions")
