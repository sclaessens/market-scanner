from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.run.cached_source_execution import (
    CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
    build_cached_source_local_execution_stage_payloads,
    validate_cached_source_snapshot_consumption_compatibility,
)
from market_engine.run.end_to_end_dry_run import (
    build_market_engine_end_to_end_dry_run,
)
from market_engine.run.local_dry_run_artifacts import (
    persist_market_engine_local_dry_run_artifact,
)
from market_engine.source_acquisition.automated_cached_source_acquisition import (
    REQUEST_FORMAT,
    run_automated_cached_source_acquisition,
)
from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    build_cached_source_snapshot_staging_validation,
)


SUMMARY_FORMAT = "market-engine-me-run27-cross-ticker-summary-v1"
BOUNDED_VALIDATION_TICKERS = ("NVDA", "AMD", "ASML")
SOURCE_FAMILY = "company_profile"


def run_company_profile_cross_ticker_dry_run(
    *,
    run_id: str,
    generated_at: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-RUN27 artifact root already exists: {root}")

    acquisition_root = root / "acquisition"
    dry_run_root = root / "dry_runs"
    root.mkdir(parents=True)

    acquisition = run_automated_cached_source_acquisition(
        _acquisition_request(
            run_id=run_id,
            generated_at=generated_at,
            destination_root=acquisition_root,
        )
    )
    staging = build_cached_source_snapshot_staging_validation(
        staging_root=acquisition_root,
        validated_at=generated_at,
        tickers=BOUNDED_VALIDATION_TICKERS,
    )
    _write_json(root / "staging_validation.json", staging)

    acquisition_by_ticker = {
        str(entry["ticker"]): entry for entry in acquisition["entries"]
    }
    staging_by_ticker = {
        str(entry["ticker"]): entry for entry in staging["entries"]
    }
    ticker_results = []
    for ticker in BOUNDED_VALIDATION_TICKERS:
        acquisition_entry = acquisition_by_ticker[ticker]
        staging_entry = staging_by_ticker[ticker]
        snapshot_path = (
            acquisition_root / ticker / SOURCE_FAMILY / "company_profile.json"
        )
        gate = validate_cached_source_snapshot_consumption_compatibility(snapshot_path)
        dry_run_id = f"{run_id}-{ticker.lower()}"
        stage_payloads = build_cached_source_local_execution_stage_payloads(
            source_snapshot_path=snapshot_path,
            source_snapshot_root=acquisition_root,
            dry_run_id=dry_run_id,
            generated_at=generated_at,
        )
        dry_run = build_market_engine_end_to_end_dry_run(
            stage_payloads,
            dry_run_id=dry_run_id,
            input_mode=CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            generated_at=generated_at,
        ).to_payload()
        persisted = persist_market_engine_local_dry_run_artifact(
            dry_run,
            output_root=dry_run_root,
            artifact_created_at=generated_at,
        )
        ticker_results.append(
            _ticker_result(
                ticker=ticker,
                acquisition_entry=acquisition_entry,
                staging_entry=staging_entry,
                gate=gate,
                dry_run=dry_run,
                dry_run_artifact_path=persisted.artifact_path,
                snapshot_path=snapshot_path,
            )
        )

    expected_controlled_stop = all(
        result["acquisition_state"] == "completed"
        and result["staging_validation_state"] == "accepted"
        and result["compatibility_gate_state"] == "allowed"
        and result["source_context_state"] == "consumed"
        and result["fundamental_observations_state"] == "completed"
        and result["analysis_review_state"] == "completed"
        and result["analysis_context_available"] is True
        and result["stop_stage"] == "recommendation_review"
        and result["company_profile_observations_produced"] is True
        for result in ticker_results
    )
    summary = {
        "summary_format": SUMMARY_FORMAT,
        "run_id": run_id,
        "generated_at": generated_at,
        "artifact_root": _display_path(root),
        "bounded_validation_tickers": BOUNDED_VALIDATION_TICKERS,
        "source_family": SOURCE_FAMILY,
        "acquisition_summary": acquisition["summary"],
        "acquisition_safety": acquisition["safety"],
        "staging_validation_counts": staging["counts"],
        "ticker_results": ticker_results,
        "ticker_agnostic_execution": True,
        "overall_result": (
            "completed_with_controlled_stop"
            if expected_controlled_stop
            else "blocked_with_evidence"
        ),
        "generated_artifacts_committed": False,
    }
    _write_json(root / "me_run27_summary.json", summary)
    (root / "me_run27_summary.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary


def _acquisition_request(
    *,
    run_id: str,
    generated_at: str,
    destination_root: Path,
) -> dict[str, Any]:
    return {
        "request_format": REQUEST_FORMAT,
        "request_id": run_id,
        "requested_at": generated_at,
        "generated_at": generated_at,
        "run_mode": "dry_run",
        "ticker_source": {
            "mode": "explicit_list",
            "source_id": "me_run27_bounded_validation_set",
        },
        "tickers": BOUNDED_VALIDATION_TICKERS,
        "source_families": (SOURCE_FAMILY,),
        "destination_root": destination_root.as_posix(),
        "freshness_policy": {
            "default_max_age_days": 7,
            "per_source_family": {
                SOURCE_FAMILY: {
                    "max_age_days": 30,
                    "source_timestamp_required": False,
                }
            },
        },
        "provider_policy": {
            "approved_adapters": (
                {
                    "adapter_id": "fake_company_profile_adapter",
                    "adapter_version": "test-v1",
                    "source_families": (SOURCE_FAMILY,),
                    "allowed_run_modes": ("dry_run", "local_non_production"),
                    "provider_name": "deterministic_fake_provider",
                    "canonical_source_identity": "fake://company_profile",
                    "network_required": False,
                    "rate_limit_policy": "not_applicable",
                    "error_policy": "fail_closed",
                },
            ),
            "allow_hidden_fallback": False,
            "allow_silent_substitution": False,
            "allow_fabricated_data": False,
        },
        "safety_flags": {
            "allow_provider_calls": False,
            "allow_network": False,
            "allow_production_writes": False,
            "allow_telegram_send": False,
            "allow_portfolio_writes": False,
            "allow_watchlist_writes": False,
            "allow_broker_actions": False,
        },
        "operator_context": {
            "requested_by": "operator",
            "purpose": "ME-RUN27 bounded cross-ticker validation",
            "notes": (),
        },
    }


def _ticker_result(
    *,
    ticker: str,
    acquisition_entry: Mapping[str, Any],
    staging_entry: Mapping[str, Any],
    gate: Mapping[str, Any],
    dry_run: Mapping[str, Any],
    dry_run_artifact_path: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    stage_results = {
        str(stage["stage_name"]): stage for stage in dry_run["stage_results"]
    }
    source_profile = dry_run["provenance_summary"]["source_context"][
        "company_profile"
    ]
    observation_profile = dry_run["provenance_summary"][
        "fundamental_observations"
    ]["company_profile"]
    analysis_profile = dry_run["provenance_summary"]["analysis_review"][
        "company_profile"
    ]
    observations = observation_profile["observations"]
    return {
        "ticker": ticker,
        "acquisition_state": acquisition_entry["status"],
        "acquisition_package_path": _display_path(snapshot_path.parent),
        "staging_validation_state": staging_entry["staging_validation_status"],
        "staging_validation_reasons": staging_entry["issues"],
        "compatibility_gate_state": "allowed" if gate["allowed"] else "blocked",
        "compatibility_gate_reason_codes": gate["reason_codes"],
        "source_context_state": source_profile["consumption_state"],
        "fundamental_observations_state": stage_results[
            "fundamental_observations"
        ]["status"],
        "analysis_review_state": stage_results["analysis_review"]["status"],
        "analysis_context_available": (
            analysis_profile["context_state"] == "descriptive_context_available"
        ),
        "completed_stages": tuple(
            stage["stage_name"]
            for stage in dry_run["stage_results"]
            if stage["status"] in {"completed", "completed_with_limitations"}
        ),
        "stop_stage": dry_run["blocked_stage"],
        "blocker_reasons": dry_run["blocked_reasons"],
        "dry_run_artifact_path": _display_path(dry_run_artifact_path),
        "company_profile_observations_produced": bool(observations),
        "company_profile_observation_codes": tuple(
            observation["observation_code"] for observation in observations
        ),
    }


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ME-RUN27 company_profile cross-ticker summary",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        f"Overall result: `{summary['overall_result']}`",
        "",
        "| Ticker | Acquisition | Staging | Gate | Source Context | Fundamental Observations | Analysis Review | Stop stage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in summary["ticker_results"]:
        lines.append(
            f"| {result['ticker']} | {result['acquisition_state']} | "
            f"{result['staging_validation_state']} | "
            f"{result['compatibility_gate_state']} | "
            f"{result['source_context_state']} | "
            f"{result['fundamental_observations_state']} | "
            f"{result['analysis_review_state']} | "
            f"{result['stop_stage']} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic ME-RUN27 company_profile validation."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    summary = run_company_profile_cross_ticker_dry_run(
        run_id=args.run_id,
        generated_at=args.generated_at,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
