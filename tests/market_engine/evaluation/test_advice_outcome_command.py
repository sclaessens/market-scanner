from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

from market_engine.evaluation.advice_outcome_command import run_command


def test_command_writes_outcome_outputs(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, "AAA")
    _write_price_csv(tmp_path / "prices" / "AAA.csv")
    stdout = io.StringIO()

    exit_code = run_command(
        [
            "--advice-index",
            advice_index.as_posix(),
            "--price-data-root",
            (tmp_path / "prices").as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "me-eval01-advice-outcomes-20260711T150000Z",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["summary"]["resolved_outcomes"] == 1
    assert (tmp_path / "out" / "me-eval01-advice-outcomes-20260711T150000Z" / "advice_outcome_index.json").exists()


def test_command_rejects_existing_output_without_allow_overwrite(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, "AAA")
    _write_price_csv(tmp_path / "prices" / "AAA.csv")
    output_dir = tmp_path / "out" / "existing"
    output_dir.mkdir(parents=True)
    stderr = io.StringIO()

    exit_code = run_command(
        [
            "--advice-index",
            advice_index.as_posix(),
            "--price-data-root",
            (tmp_path / "prices").as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "existing",
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "output directory already exists" in stderr.getvalue()


def test_command_allow_overwrite_reuses_output_directory(tmp_path: Path) -> None:
    advice_index = _write_advice_index(tmp_path, "AAA")
    _write_price_csv(tmp_path / "prices" / "AAA.csv")
    output_dir = tmp_path / "out" / "existing"
    output_dir.mkdir(parents=True)

    exit_code = run_command(
        [
            "--advice-index",
            advice_index.as_posix(),
            "--price-data-root",
            (tmp_path / "prices").as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "existing",
            "--allow-overwrite",
            "--horizons",
            "5,21,63",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert (output_dir / "manifest.json").exists()


def _write_advice_index(root: Path, ticker: str) -> Path:
    path = root / "advice_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-advice-batch-index-v1",
                "artifact_type": "market-engine-deterministic-advice-batch-index",
                "run_id": "advice-run-20260711T140000Z",
                "generated_at": "2026-07-11T14:00:00Z",
                "summary": {"tickers_total": 1},
                "tickers": [
                    {
                        "ticker": ticker,
                        "advice": "buy_candidate",
                        "confidence": "medium",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_price_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Date,Adj Close,Close"]
    start = date(2026, 7, 11)
    for index in range(70):
        row_date = start + timedelta(days=index)
        lines.append(f"{row_date.isoformat()},{100 + index},{100 + index}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
