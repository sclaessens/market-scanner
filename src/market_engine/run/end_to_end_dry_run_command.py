from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.run.end_to_end_dry_run import (
    APPROVED_DRY_RUN_INPUT_MODES,
    build_market_engine_end_to_end_dry_run,
)


DEFAULT_LOCAL_DRY_RUN_ID = "market-engine-local-dry-run"


class _DryRunCommandError(ValueError):
    pass


def run_market_engine_end_to_end_dry_run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    parser = _argument_parser()
    args = parser.parse_args(argv)

    try:
        stage_payloads = _stage_payloads_for_command(args)
    except _DryRunCommandError as exc:
        print(str(exc), file=error_stream)
        return 2

    dry_run = build_market_engine_end_to_end_dry_run(
        stage_payloads,
        dry_run_id=args.dry_run_id,
        input_mode=args.input_mode,
        generated_at=args.generated_at or _generated_at_utc(),
    )

    json.dump(
        dry_run.to_payload(),
        output_stream,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    output_stream.write("\n")
    return 0


def main() -> None:
    raise SystemExit(run_market_engine_end_to_end_dry_run_command())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-dry-run",
        description=(
            "Run the deterministic local Market Engine end-to-end dry-run harness "
            "and print the market-engine-end-to-end-dry-run-v1 payload as JSON."
        ),
    )
    parser.add_argument(
        "--dry-run-id",
        default=DEFAULT_LOCAL_DRY_RUN_ID,
        help="Dry-run identifier to include in the emitted payload.",
    )
    parser.add_argument(
        "--input-mode",
        choices=APPROVED_DRY_RUN_INPUT_MODES,
        default="synthetic_contract_fixture",
        help="Approved dry-run input mode. Defaults to the embedded synthetic fixture.",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional generated timestamp to include in the emitted payload.",
    )
    parser.add_argument(
        "--stage-payloads-json",
        default=None,
        help=(
            "Optional local JSON file containing a mapping of approved stage payloads. "
            "Required for non-synthetic input modes."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON instead of pretty printed JSON.",
    )
    return parser


def _stage_payloads_for_command(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.stage_payloads_json:
        return _load_stage_payloads(Path(args.stage_payloads_json))
    if args.input_mode == "synthetic_contract_fixture":
        return build_synthetic_dry_run_stage_payloads()
    raise _DryRunCommandError(
        "--stage-payloads-json is required when --input-mode is not "
        "synthetic_contract_fixture."
    )


def _load_stage_payloads(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as payload_file:
            payload = json.load(payload_file)
    except OSError as exc:
        raise _DryRunCommandError(f"Unable to read stage payload JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _DryRunCommandError(f"Stage payload JSON is invalid: {path}") from exc

    if not isinstance(payload, Mapping):
        raise _DryRunCommandError("Stage payload JSON must contain an object at the top level.")
    return payload


def _generated_at_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_synthetic_dry_run_stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "source_context": {
            "source_context_format_version": "sec-companyfacts-source-context-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "source_context_state": "AVAILABLE",
            "source_refresh_snapshot_id": "source-run-001",
            "fixture_backed": True,
        },
        "fundamental_observations": {
            "fundamental_observations_format_version": (
                "sec-companyfacts-fundamental-observations-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "observation_run_id": "fundamental-run-001",
            "source_context_reference": {"source_refresh_snapshot_id": "source-run-001"},
        },
        "derived_observations": {
            "derived_observations_format_version": (
                "sec-companyfacts-derived-cash-generation-observations-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "derived_observation_run_id": "derived-run-001",
            "fundamental_observations_reference": {
                "observation_run_id": "fundamental-run-001"
            },
        },
        "setup_detection": {
            "setup_detection_format_version": "sec-companyfacts-setup-detection-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "setup_detection_run_id": "setup-run-001",
            "derived_observations_reference": {
                "derived_observation_run_id": "derived-run-001"
            },
        },
        "analysis_review": {
            "analysis_review_format_version": "sec-companyfacts-analysis-review-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "analysis_review_run_id": "analysis-review-run-001",
            "setup_detection_reference": {"setup_detection_run_id": "setup-run-001"},
        },
        "recommendation_review": {
            "recommendation_review_format_version": (
                "sec-companyfacts-recommendation-review-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "recommendation_review_run_id": "rr-run-001",
            "input_provenance": {
                "analysis_review_run_id": "analysis-review-run-001",
                "setup_detection_run_id": "setup-run-001",
            },
        },
        "portfolio_review": {
            "portfolio_review_format_version": "sec-companyfacts-portfolio-review-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "portfolio_review_run_id": "portfolio-review-run-001",
            "portfolio_context_reference": {
                "portfolio_context_format_version": "market-engine-portfolio-context-v1",
                "portfolio_context_run_id": "portfolio-context-run-001",
                "current_quantity": 0,
                "current_market_value": 0.0,
            },
            "recommendation_review_reference": {
                "recommendation_review_run_id": "rr-run-001"
            },
        },
        "decision_engine_handoff": {
            "handoff_format_version": "market-engine-decision-engine-handoff-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "handoff_run_id": "handoff-run-001",
            "portfolio_review_reference": {
                "portfolio_review_run_id": "portfolio-review-run-001"
            },
            "portfolio_context_reference": {
                "portfolio_context_run_id": "portfolio-context-run-001"
            },
            "handoff_readiness_state": "ready_for_decision_engine_review",
            "audit_provenance": {"portfolio_review_run_id": "portfolio-review-run-001"},
        },
        "delivery_reporting": {
            "report_format_version": "market-engine-delivery-report-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "report_id": "delivery-report-001",
            "source_handoff_run_id": "handoff-run-001",
            "delivery_state": "ready_for_user_review",
            "upstream_provenance_summary": {
                "decision_engine_handoff": {"handoff_run_id": "handoff-run-001"}
            },
            "forbidden_language_guardrails": ("buy", "sell", "hold"),
        },
    }


if __name__ == "__main__":
    main()
