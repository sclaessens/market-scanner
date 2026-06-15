from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from market_engine.source_intake.fake_provider import FakeProviderScenario, FakeSourceProvider
from market_engine.source_intake.models import BatchSourceIntakeSummary
from market_engine.source_intake.runner import run_source_intake


DEFAULT_REQUIRED_FIELDS = ("revenue", "operating_cash_flow", "capital_expenditures")
DEFAULT_TICKERS = ("AAPL", "MSFT", "MISSING", "PARTIAL", "UNSUPPORTED", "INVALID", "ERROR")


def build_fake_provider() -> FakeSourceProvider:
    return FakeSourceProvider(
        scenarios={
            "AAPL": FakeProviderScenario(
                fields={
                    "revenue": 100,
                    "operating_cash_flow": 25,
                    "capital_expenditures": 5,
                },
                raw_evidence={"fixture": "AAPL"},
                raw_evidence_summary="complete fake source",
            ),
            "MSFT": FakeProviderScenario(
                fields={
                    "revenue": 200,
                    "operating_cash_flow": 50,
                    "capital_expenditures": 10,
                },
                raw_evidence={"fixture": "MSFT"},
                raw_evidence_summary="complete fake source",
            ),
            "PARTIAL": FakeProviderScenario(
                fields={
                    "revenue": 10,
                    "operating_cash_flow": None,
                },
                raw_evidence={"fixture": "PARTIAL"},
                raw_evidence_summary="partial fake source",
            ),
            "MISSING": FakeProviderScenario(missing_source=True),
            "UNSUPPORTED": FakeProviderScenario(unsupported=True),
            "INVALID": FakeProviderScenario(invalid=True),
            "ERROR": FakeProviderScenario(provider_error=True),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Market Engine source intake smoke with a fake provider.")
    parser.add_argument("--ticker-file", type=Path, help="Optional local ticker file with one ticker per line.")
    parser.add_argument(
        "--write-artifact",
        type=Path,
        help="Optional non-production JSON artifact path for local smoke output.",
    )
    args = parser.parse_args(argv)

    tickers = _read_tickers(args.ticker_file) if args.ticker_file else list(DEFAULT_TICKERS)
    summary = run_source_intake(
        tickers=tickers,
        provider=build_fake_provider(),
        required_fields=DEFAULT_REQUIRED_FIELDS,
    )
    print(_format_summary(summary))

    if args.write_artifact is not None:
        args.write_artifact.parent.mkdir(parents=True, exist_ok=True)
        args.write_artifact.write_text(
            json.dumps(_summary_to_jsonable(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0


def _read_tickers(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_summary(summary: BatchSourceIntakeSummary) -> str:
    status_parts = ", ".join(
        f"{status.value}={count}"
        for status, count in sorted(summary.status_counts.items(), key=lambda item: item[0].value)
    )
    missing_parts = ", ".join(
        f"{field}={count}"
        for field, count in sorted(summary.missing_field_frequency.items())
    )
    return (
        f"provider={summary.provider_name}\n"
        f"tickers={summary.total_tickers}\n"
        f"intake_success={summary.intake_success_count}\n"
        f"intake_failure={summary.intake_failure_count}\n"
        f"statuses={status_parts or 'none'}\n"
        f"missing_fields={missing_parts or 'none'}"
    )


def _summary_to_jsonable(summary: BatchSourceIntakeSummary) -> dict[str, object]:
    payload = asdict(summary)
    payload["status_counts"] = {
        status.value if hasattr(status, "value") else str(status): count
        for status, count in summary.status_counts.items()
    }
    for result in payload["results"]:
        result["readiness_status"] = result["readiness_status"].value
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
