from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence, TextIO

from market_engine.run.end_to_end_dry_run import (
    APPROVED_DRY_RUN_INPUT_MODES,
    build_market_engine_end_to_end_dry_run,
)
from market_engine.run.cached_source_execution import (
    CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
    CachedSourceLocalExecutionError,
    build_cached_source_local_execution_stage_payloads,
    load_cached_source_local_execution_stage_payloads,
    load_portfolio_context_payload,
)
from market_engine.run.local_dry_run_artifacts import (
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    LocalDryRunArtifactError,
    persist_market_engine_local_dry_run_artifact,
)
from market_engine.run.local_dry_run_inputs import (
    LocalDryRunInputError,
    load_market_engine_local_dry_run_input,
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
    dry_run_payload = dry_run.to_payload()

    if args.write_local_artifact:
        try:
            persist_market_engine_local_dry_run_artifact(
                dry_run_payload,
                output_root=args.artifact_output_root,
                artifact_created_at=args.artifact_created_at or _generated_at_utc(),
            )
        except LocalDryRunArtifactError as exc:
            print(str(exc), file=error_stream)
            return 2

    json.dump(
        dry_run_payload,
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
            "Optional local JSON file containing approved stage payloads. "
            "For local_snapshot_fixture, the file must be a non-production "
            "market-engine-local-dry-run-input-fixture-v1 wrapper. "
            "Required for non-synthetic input modes."
        ),
    )
    parser.add_argument(
        "--source-snapshot-json",
        default=None,
        help=(
            "Local cached SEC CompanyFacts source snapshot JSON. Required for "
            "cached_source_snapshot unless --stage-payloads-json provides the "
            "cached-source local execution wrapper."
        ),
    )
    parser.add_argument(
        "--source-snapshot-root",
        default="data/market_engine/source_snapshots",
        help=(
            "Approved local cached-source root used to validate "
            "--source-snapshot-json containment."
        ),
    )
    parser.add_argument(
        "--portfolio-context-json",
        default=None,
        help=(
            "Optional local portfolio-context JSON used only with "
            "cached_source_snapshot."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON instead of pretty printed JSON.",
    )
    parser.add_argument(
        "--write-local-artifact",
        action="store_true",
        help=(
            "Persist the emitted dry-run payload as a local non-production JSON "
            "artifact. Disabled by default."
        ),
    )
    parser.add_argument(
        "--artifact-output-root",
        default=MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
        help=(
            "Local dry-run artifact root. Used only with --write-local-artifact. "
            "Defaults to artifacts/market_engine/dry_runs."
        ),
    )
    parser.add_argument(
        "--artifact-created-at",
        default=None,
        help=(
            "Optional artifact creation timestamp for deterministic local artifact "
            "metadata. Used only with --write-local-artifact."
        ),
    )
    return parser


def _stage_payloads_for_command(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.input_mode == CACHED_SOURCE_SNAPSHOT_INPUT_MODE:
        try:
            if args.stage_payloads_json:
                return load_cached_source_local_execution_stage_payloads(
                    args.stage_payloads_json,
                    dry_run_id=args.dry_run_id,
                    generated_at=args.generated_at,
                )
            if not args.source_snapshot_json:
                raise CachedSourceLocalExecutionError(
                    "--source-snapshot-json is required when --input-mode is "
                    "cached_source_snapshot unless --stage-payloads-json is supplied."
                )
            portfolio_context_payload = (
                load_portfolio_context_payload(args.portfolio_context_json)
                if args.portfolio_context_json
                else None
            )
            return build_cached_source_local_execution_stage_payloads(
                source_snapshot_path=args.source_snapshot_json,
                source_snapshot_root=args.source_snapshot_root,
                dry_run_id=args.dry_run_id,
                generated_at=args.generated_at,
                portfolio_context_payload=portfolio_context_payload,
            )
        except CachedSourceLocalExecutionError as exc:
            raise _DryRunCommandError(str(exc)) from exc

    if args.stage_payloads_json:
        try:
            return load_market_engine_local_dry_run_input(
                args.stage_payloads_json,
                input_mode=args.input_mode,
            )
        except LocalDryRunInputError as exc:
            raise _DryRunCommandError(str(exc)) from exc
    if args.input_mode == "synthetic_contract_fixture":
        return build_synthetic_dry_run_stage_payloads()
    raise _DryRunCommandError(
        "--stage-payloads-json is required when --input-mode is not "
        "synthetic_contract_fixture."
    )


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
