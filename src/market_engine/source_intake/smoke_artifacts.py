from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from market_engine.source_intake.coverage_review import build_source_coverage_review
from market_engine.source_intake.models import BatchSourceIntakeSummary, TickerSourceResult


SEC_COMPANYFACTS_SMOKE_ROOT = Path("data/market_engine/smokes/source_intake/sec_companyfacts")
SMOKE_DISCLAIMER = "source coverage evidence only; not analysis; not source truth"


def make_smoke_run_id(now: datetime | None = None) -> str:
    timestamp = now or datetime.now(UTC)
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def default_sec_companyfacts_artifact_dir(run_id: str) -> Path:
    return SEC_COMPANYFACTS_SMOKE_ROOT / run_id


def write_sec_companyfacts_smoke_artifacts(
    *,
    summary: BatchSourceIntakeSummary,
    tickers: Iterable[str],
    max_tickers: int,
    run_id: str,
    artifact_dir: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    output_dir = artifact_dir or default_sec_companyfacts_artifact_dir(run_id)
    _validate_artifact_dir(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke artifact directory: {output_dir}")

    output_dir.mkdir(parents=True)
    review = build_source_coverage_review(summary)
    tickers_tuple = tuple(tickers)
    run_timestamp = (timestamp or datetime.now(UTC)).isoformat()

    _write_csv(
        output_dir / "coverage_summary.csv",
        ("provider_name", "ticker_count", "readiness_status", "count", "disclaimer"),
        [
            (
                summary.provider_name,
                summary.total_tickers,
                readiness_status,
                count,
                SMOKE_DISCLAIMER,
            )
            for readiness_status, count in review.readiness_counts.items()
        ],
    )
    _write_csv(
        output_dir / "ticker_results.csv",
        (
            "ticker",
            "provider_name",
            "readiness_status",
            "available_fields",
            "missing_fields",
            "raw_evidence_present",
            "raw_evidence_summary",
            "provider_error_type",
            "provider_error_message",
            "intake_success",
        ),
        [_ticker_row(result) for result in summary.results],
    )
    _write_csv(
        output_dir / "missing_fields.csv",
        ("field_name", "missing_count", "disclaimer"),
        [
            (field_name, count, SMOKE_DISCLAIMER)
            for field_name, count in review.missing_field_frequency.items()
        ],
    )
    _write_csv(
        output_dir / "provider_errors.csv",
        ("ticker", "provider_error_type", "provider_error_message", "disclaimer"),
        [
            (
                result.ticker,
                result.error.error_type if result.error is not None else "",
                result.error.message if result.error is not None else "",
                SMOKE_DISCLAIMER,
            )
            for result in summary.results
            if result.error is not None
        ],
    )
    metadata = {
        "provider_name": summary.provider_name,
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "tickers": list(tickers_tuple),
        "ticker_count": len(tickers_tuple),
        "max_tickers": max_tickers,
        "required_fields": list(summary.required_fields),
        "readiness_counts": review.readiness_counts,
        "missing_field_frequency": review.missing_field_frequency,
        "provider_error_categories": review.provider_error_categories,
        "artifact_files": [
            "coverage_summary.csv",
            "ticker_results.csv",
            "missing_fields.csv",
            "provider_errors.csv",
            "smoke_metadata.json",
        ],
        "disclaimer": SMOKE_DISCLAIMER,
    }
    (output_dir / "smoke_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def _validate_artifact_dir(path: Path) -> None:
    expected = SEC_COMPANYFACTS_SMOKE_ROOT.parts
    parts = path.parts
    for index in range(0, len(parts) - len(expected) + 1):
        if parts[index : index + len(expected)] == expected:
            return
    raise ValueError(f"smoke artifacts must be written under {SEC_COMPANYFACTS_SMOKE_ROOT}")


def _ticker_row(result: TickerSourceResult) -> tuple[object, ...]:
    payload = asdict(result)
    return (
        result.ticker,
        result.provider_name,
        result.readiness_status.value,
        "|".join(result.available_fields),
        "|".join(result.missing_fields),
        result.raw_evidence_present,
        result.raw_evidence_summary or "",
        payload["error"]["error_type"] if payload["error"] is not None else "",
        payload["error"]["message"] if payload["error"] is not None else "",
        result.intake_success,
    )


def _write_csv(path: Path, headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
