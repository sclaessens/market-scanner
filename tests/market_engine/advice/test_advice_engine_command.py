from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from typing import Any

from market_engine.advice.advice_engine_command import run_command


def test_command_writes_all_output_files(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, ["BBB", "AAA"])
    output_root = tmp_path / "out"
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run-id",
            "advice-run",
            "--generated-at",
            "2026-07-11T00:00:00Z",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    output_dir = output_root / "advice-run"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "advice_index.json",
        "advice_index.md",
        "advice_summary.json",
        "manifest.json",
        "unable_to_advise.json",
    ]
    advice = json.loads((output_dir / "advice_index.json").read_text())
    assert [row["ticker"] for row in advice["tickers"]] == ["AAA", "BBB"]
    unable = json.loads((output_dir / "unable_to_advise.json").read_text())
    assert unable["tickers"] == []
    markdown = (output_dir / "advice_index.md").read_text()
    assert "# Market Engine Advice Index" in markdown
    assert "| AAA | watchlist | low | partial |" in markdown


def test_advice_engine_does_not_require_openai_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_" + "API_KEY", raising=False)
    monkeypatch.delenv("MARKET_ENGINE_" + "ADVISORY_MODEL", raising=False)
    index_path = _write_status_index(tmp_path, ["AAA"])

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "advice-run",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0


def test_advice_engine_does_not_touch_network(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ME-ADV01 must not make provider/network calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    index_path = _write_status_index(tmp_path, ["AAA"])

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "advice-run",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0


def test_unable_to_advise_file_contains_only_unable_tickers(tmp_path: Path) -> None:
    valid = _write_dry_run(tmp_path, "OK")
    index_path = tmp_path / "ticker_status_index.json"
    rows = [
        _status_row("OK", valid),
        {
            **_status_row("BAD", None),
            "status": "invalid_artifact",
            "artifact_path": None,
        },
    ]
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-ticker-status-index-v1",
                "artifact_type": "market-engine-ticker-status-index",
                "run_id": "status-run",
                "generated_at": "2026-07-11T00:00:00Z",
                "summary": {"tickers_total": 2},
                "tickers": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "advice-run",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    unable = json.loads(
        (tmp_path / "out" / "advice-run" / "unable_to_advise.json").read_text()
    )
    assert [row["ticker"] for row in unable["tickers"]] == ["BAD"]


def _write_status_index(tmp_path: Path, tickers: list[str]) -> Path:
    rows = [_status_row(ticker, _write_dry_run(tmp_path, ticker)) for ticker in tickers]
    path = tmp_path / "ticker_status_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-ticker-status-index-v1",
                "artifact_type": "market-engine-ticker-status-index",
                "run_id": "status-run",
                "generated_at": "2026-07-11T00:00:00Z",
                "summary": {"tickers_total": len(rows)},
                "tickers": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _status_row(ticker: str, artifact_path: Path | None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": "blocked",
        "readiness_level": "partial_analysis",
        "context_stale": False,
        "actionable_review_allowed": False,
        "decision_engine_ready": False,
        "blocked_stage": "portfolio_review",
        "blocked_reasons": ["Stage preserves an upstream blocked state."],
        "readiness_blocked_reasons": ["missing_setup_or_price_context"],
        "missing_data_summary": ["portfolio_context"],
        "evidence_families_missing": ["setup_price_market"],
        "artifact_path": artifact_path.as_posix() if artifact_path else None,
        "artifact_sha256": "sha",
    }


def _write_dry_run(tmp_path: Path, ticker: str) -> Path:
    path = tmp_path / ticker / "dry_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "payload": {
                    "ticker": ticker,
                    "stage_results": [
                        {
                            "stage_name": "fundamental_observations",
                            "status": "completed",
                        }
                    ],
                    "provenance_summary": {
                        "fundamental_observations": {
                            "fundamental_observations_run_id": f"{ticker.lower()}-fundamental"
                        }
                    },
                    "available_context_families": [],
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
