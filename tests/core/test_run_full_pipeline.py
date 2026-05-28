from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_full_pipeline, run_scan


def test_run_full_pipeline_default_preserves_existing_run_scan_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, command: list[str]) -> None:
        calls.append((name, command))

    monkeypatch.setattr(run_full_pipeline, "run_step", fake_run_step)

    run_full_pipeline.main([])

    assert calls == [("1. Core end-to-end pipeline", [run_full_pipeline.sys.executable, "scripts/run_scan.py"])]


def test_run_full_pipeline_passes_optional_fundamentals_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    history_path = tmp_path / "fundamentals_history.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    analysis_path = tmp_path / "fundamental_analysis.csv"

    def fake_run_step(name: str, command: list[str]) -> None:
        calls.append(command)

    monkeypatch.setattr(run_full_pipeline, "run_step", fake_run_step)

    run_full_pipeline.main(
        [
            "--fundamentals-history-path",
            str(history_path),
            "--fundamental-metrics-output-path",
            str(metrics_path),
            "--fundamental-analysis-output-path",
            str(analysis_path),
        ]
    )

    assert calls == [
        [
            run_full_pipeline.sys.executable,
            "scripts/run_scan.py",
            "--fundamentals-history-path",
            str(history_path),
            "--fundamental-metrics-output-path",
            str(metrics_path),
            "--fundamental-analysis-output-path",
            str(analysis_path),
        ]
    ]


def test_build_fundamentals_pipeline_default_uses_existing_fundamental_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    expected_quality = pd.DataFrame([{"ticker": "AAA", "date": "2026-05-28"}])

    monkeypatch.setattr(run_scan, "build_fundamental_layer", lambda: calls.append("quality") or expected_quality)
    monkeypatch.setattr(
        run_scan,
        "validate_fundamentals_history",
        lambda path: pytest.fail("raw history validation should not run in default mode"),
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_metrics",
        lambda input_path, output_path: pytest.fail("metrics should not run in default mode"),
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_analysis",
        lambda quality_path, metrics_path, output_path: pytest.fail("analysis should not run in default mode"),
    )

    quality_df, metrics_df, analysis_df = run_scan.build_fundamentals_pipeline()

    assert calls == ["quality"]
    assert quality_df.equals(expected_quality)
    assert metrics_df is None
    assert analysis_df is None


def test_build_fundamentals_pipeline_optional_raw_history_runs_ordered_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    history_path = tmp_path / "fundamentals_history.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    quality_path = tmp_path / "fundamental_quality.csv"
    analysis_path = tmp_path / "fundamental_analysis.csv"
    expected_quality = pd.DataFrame([{"ticker": "AAA", "date": "2026-05-28"}])
    expected_metrics = pd.DataFrame([{"ticker": "AAA"}])
    expected_analysis = pd.DataFrame([{"ticker": "AAA"}])
    monkeypatch.setattr(run_scan, "FUNDAMENTAL_QUALITY_FILE", quality_path)

    def fake_validate(path: Path) -> dict:
        order.append("validate")
        assert path == history_path
        return {"status": "VALID"}

    def fake_metrics(input_path: Path, output_path: Path) -> pd.DataFrame:
        order.append("metrics")
        assert input_path == history_path
        assert output_path == metrics_path
        return expected_metrics

    def fake_quality(**kwargs) -> pd.DataFrame:
        order.append("quality")
        assert kwargs == {
            "fundamentals_history_path": history_path,
            "fundamental_metrics_path": metrics_path,
        }
        return expected_quality

    def fake_analysis(quality_input_path: Path, metrics_input_path: Path, output_path: Path) -> pd.DataFrame:
        order.append("analysis")
        assert quality_input_path == quality_path
        assert metrics_input_path == metrics_path
        assert output_path == analysis_path
        return expected_analysis

    monkeypatch.setattr(run_scan, "validate_fundamentals_history", fake_validate)
    monkeypatch.setattr(run_scan, "build_fundamental_metrics", fake_metrics)
    monkeypatch.setattr(run_scan, "build_fundamental_layer", fake_quality)
    monkeypatch.setattr(run_scan, "build_fundamental_analysis", fake_analysis)

    quality_df, metrics_df, analysis_df = run_scan.build_fundamentals_pipeline(
        fundamentals_history_path=history_path,
        fundamental_metrics_output_path=metrics_path,
        fundamental_analysis_output_path=analysis_path,
    )

    assert order == ["validate", "metrics", "quality", "analysis"]
    assert quality_df.equals(expected_quality)
    assert metrics_df.equals(expected_metrics)
    assert analysis_df.equals(expected_analysis)


def test_invalid_raw_history_fails_before_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    history_path = tmp_path / "fundamentals_history.csv"

    monkeypatch.setattr(
        run_scan,
        "validate_fundamentals_history",
        lambda path: calls.append("validate") or {"status": "INVALID", "missing_required_columns": ["ticker"]},
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_metrics",
        lambda input_path, output_path: pytest.fail("metrics should not run after invalid validation"),
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_layer",
        lambda **kwargs: pytest.fail("quality should not run after invalid validation"),
    )
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_analysis",
        lambda quality_path, metrics_path, output_path: pytest.fail("analysis should not run after invalid validation"),
    )

    with pytest.raises(ValueError, match="fundamentals history validation failed"):
        run_scan.build_fundamentals_pipeline(fundamentals_history_path=history_path)

    assert calls == ["validate"]


def test_optional_analysis_failure_does_not_block_quality_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history_path = tmp_path / "fundamentals_history.csv"
    expected_quality = pd.DataFrame([{"ticker": "AAA", "date": "2026-05-28"}])
    expected_metrics = pd.DataFrame([{"ticker": "AAA"}])

    monkeypatch.setattr(run_scan, "validate_fundamentals_history", lambda path: {"status": "VALID"})
    monkeypatch.setattr(run_scan, "build_fundamental_metrics", lambda input_path, output_path: expected_metrics)
    monkeypatch.setattr(
        run_scan,
        "build_fundamental_layer",
        lambda **kwargs: expected_quality,
    )

    def fake_analysis(quality_path: Path, metrics_path: Path, output_path: Path) -> pd.DataFrame:
        raise ValueError("analysis input unavailable")

    monkeypatch.setattr(run_scan, "build_fundamental_analysis", fake_analysis)

    quality_df, metrics_df, analysis_df = run_scan.build_fundamentals_pipeline(fundamentals_history_path=history_path)

    assert quality_df.equals(expected_quality)
    assert metrics_df.equals(expected_metrics)
    assert analysis_df is None
    assert "Optional fundamental analysis skipped" in capsys.readouterr().out


def test_run_scan_continues_downstream_with_fundamental_quality_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    history_path = tmp_path / "fundamentals_history.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    analysis_path = tmp_path / "fundamental_analysis.csv"

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
    monkeypatch.setattr(run_scan, "fetch_ohlcv_data", lambda ticker: pd.DataFrame({"Close": [100.0]}))
    monkeypatch.setattr(run_scan, "add_indicators", lambda df: df.assign(MA50=95.0, MA200=90.0))
    monkeypatch.setattr(
        run_scan,
        "scan_ticker",
        lambda **_: {"ticker": "AAA", "date": "2026-05-28", "sector": "TECHNOLOGY", "rs_20d_pct": 1.0},
    )
    monkeypatch.setattr(run_scan, "rank_setups", lambda setups, top_n: setups)

    def record(name: str, rows: list[dict]) -> pd.DataFrame:
        order.append(name)
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        run_scan,
        "build_validation_layer",
        lambda: record("validation", [{"ticker": "AAA", "date": "2026-05-28"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_context_layer",
        lambda: record("context", [{"ticker": "AAA", "date": "2026-05-28"}]),
    )

    def fake_fundamentals_pipeline(**kwargs):
        order.append("fundamental")
        assert kwargs == {
            "fundamentals_history_path": str(history_path),
            "fundamental_metrics_output_path": str(metrics_path),
            "fundamental_analysis_output_path": str(analysis_path),
        }
        return (
            pd.DataFrame([{"ticker": "AAA", "date": "2026-05-28"}]),
            pd.DataFrame([{"ticker": "AAA"}]),
            pd.DataFrame([{"ticker": "AAA"}]),
        )

    monkeypatch.setattr(run_scan, "build_fundamentals_pipeline", fake_fundamentals_pipeline)
    monkeypatch.setattr(
        run_scan,
        "build_timing_state_layer",
        lambda: record("timing", [{"ticker": "AAA", "date": "2026-05-28"}]),
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
        lambda: record("portfolio_intelligence", [{"ticker": "AAA", "date": "2026-05-28"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_final_decisions",
        lambda: record("final_decisions", [{"ticker": "AAA", "date": "2026-05-28"}]),
    )
    monkeypatch.setattr(
        run_scan,
        "build_reporting_layer",
        lambda: (
            record("reporting", [{"ticker": "AAA", "date": "2026-05-28"}]),
            {"input_status": "SOURCE_AVAILABLE"},
            "summary",
        ),
    )
    monkeypatch.setattr(run_scan, "write_reporting_outputs", lambda dashboard, log_row, text: order.append("reporting_outputs"))
    monkeypatch.setattr(run_scan, "send_daily_summary", lambda: order.append("telegram_delivery"))

    run_scan.main(
        [
            "--fundamentals-history-path",
            str(history_path),
            "--fundamental-metrics-output-path",
            str(metrics_path),
            "--fundamental-analysis-output-path",
            str(analysis_path),
        ]
    )

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
        "reporting_outputs",
        "telegram_delivery",
    ]
    assert order.index("fundamental") < order.index("timing")
