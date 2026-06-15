from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from market_engine.source_intake.coverage_review import (
    build_source_coverage_review,
    format_source_coverage_review,
)
from market_engine.source_intake.fake_provider import FakeProviderScenario, FakeSourceProvider
from market_engine.source_intake.models import BatchSourceIntakeSummary
from market_engine.source_intake.runner import run_source_intake
from market_engine.source_intake.sec_companyfacts_provider import (
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    SMOKE_TICKER_CIKS,
    SecCompanyFactsProvider,
)


DEFAULT_REQUIRED_FIELDS = ("revenue", "operating_cash_flow", "capital_expenditures")
DEFAULT_TICKERS = ("AAPL", "MSFT", "MISSING", "PARTIAL", "UNSUPPORTED", "INVALID", "ERROR")
DEFAULT_MAX_TICKERS = 5


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
    parser = argparse.ArgumentParser(description="Run bounded Market Engine source intake smoke.")
    parser.add_argument(
        "--provider",
        choices=("fake", "sec-companyfacts"),
        default="fake",
        help="Provider to use. Default is fake provider only.",
    )
    parser.add_argument("--tickers", nargs="*", help="Explicit ticker list for the smoke run.")
    parser.add_argument(
        "--use-sec-sample",
        action="store_true",
        help="Use the bounded built-in SEC sample ticker list.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=DEFAULT_MAX_TICKERS,
        help="Maximum ticker count allowed for a bounded real-provider smoke.",
    )
    parser.add_argument("--ticker-file", type=Path, help="Optional local ticker file with one ticker per line.")
    parser.add_argument(
        "--write-artifact",
        type=Path,
        help="Deprecated alias for --write-smoke-artifact.",
    )
    parser.add_argument(
        "--write-smoke-artifact",
        type=Path,
        help="Optional non-production JSON artifact path for local smoke output.",
    )
    args = parser.parse_args(argv)

    tickers = _resolve_tickers(args)
    provider = build_fake_provider()
    required_fields = DEFAULT_REQUIRED_FIELDS
    if args.provider == "sec-companyfacts":
        _validate_real_provider_args(tickers, args.max_tickers)
        print("bounded_real_provider_smoke=true")
        print("provider_warning=manual SEC CompanyFacts smoke; source coverage evidence only")
        provider = SecCompanyFactsProvider()
        required_fields = SEC_COMPANYFACTS_REQUIRED_FIELDS

    summary = run_source_intake(
        tickers=tickers,
        provider=provider,
        required_fields=required_fields,
    )
    review = build_source_coverage_review(summary)
    print(format_source_coverage_review(review))

    artifact_path = args.write_smoke_artifact or args.write_artifact
    if artifact_path is not None:
        _write_artifact(path=artifact_path, summary=summary)

    return 0


def _read_tickers(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return list(args.tickers)
    if args.ticker_file:
        return _read_tickers(args.ticker_file)
    if args.provider == "sec-companyfacts" and args.use_sec_sample:
        return list(SMOKE_TICKER_CIKS)
    if args.provider == "sec-companyfacts":
        raise SystemExit("sec-companyfacts provider requires --tickers, --ticker-file, or --use-sec-sample")
    return list(DEFAULT_TICKERS)


def _validate_real_provider_args(tickers: list[str], max_tickers: int) -> None:
    if max_tickers < 1:
        raise SystemExit("--max-tickers must be at least 1")
    if len(tickers) > max_tickers:
        raise SystemExit(f"refusing to run {len(tickers)} tickers with max_tickers={max_tickers}")


def _write_artifact(path: Path, summary: BatchSourceIntakeSummary) -> None:
    allowed_root = Path("data/market_engine/smokes/source_intake")
    if allowed_root not in path.parents and path != allowed_root:
        raise SystemExit(f"smoke artifacts must be written under {allowed_root}")
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing smoke artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_summary_to_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
