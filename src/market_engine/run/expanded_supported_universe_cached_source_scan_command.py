from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence, TextIO

from market_engine.run.expanded_supported_universe_cached_source_scan import (
    DEFAULT_ARTIFACT_OUTPUT_ROOT,
    DEFAULT_SOURCE_SNAPSHOT_ROOT,
    ExpandedSupportedUniverseCachedSourceScanError,
    build_expanded_supported_universe_cached_source_scan,
)
from market_engine.ticker_universe.professional_swing import PROFESSIONAL_SWING_UNIVERSE_PATH


DEFAULT_BATCH_ID_PREFIX = "me-run23-expanded-supported-universe"


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        generated_at = args.generated_at or _generated_at_utc()
        batch_id = args.batch_id or _default_batch_id(generated_at)
        result = build_expanded_supported_universe_cached_source_scan(
            candidate_classification_path=args.candidate_classification_summary,
            existing_universe_path=args.professional_swing_universe,
            source_snapshot_root=args.source_snapshot_root,
            batch_id=batch_id,
            generated_at=generated_at,
            ticker_limit=args.ticker_limit,
            write_local_artifacts=bool(args.write_local_artifacts),
            artifact_output_root=args.artifact_output_root,
        )
    except ExpandedSupportedUniverseCachedSourceScanError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2

    render_human_visible_output(result.to_payload(), stdout=stdout)
    if args.emit_json:
        print("\nJSON PAYLOAD", file=stdout)
        print(json.dumps(result.to_payload(), indent=2, sort_keys=True), file=stdout)
    return 0


def render_human_visible_output(payload: Mapping[str, Any], *, stdout: TextIO) -> None:
    counts = payload["source_support_summary_counts"]
    batch = payload.get("batch_payload") or {}

    _section("RUN CONTEXT", stdout)
    _line("Contract", payload["format_version"], stdout)
    _line("Batch id", payload["batch_id"], stdout)
    _line("Generated at", payload["generated_at"], stdout)
    _line("Universe path", payload["input_universe_path"], stdout)
    _line("Candidate summary", payload["input_candidate_classification_path"], stdout)
    _line("Cached-source root", payload["source_snapshot_root"], stdout)
    _line("Run state", payload["run_state"], stdout)

    _section("EXPANDED SOURCE SUPPORT", stdout)
    _line("Expanded entries", payload["expanded_universe_count"], stdout)
    for key in (
        "supported_cached",
        "missing_snapshot",
        "unsupported_sec_companyfacts",
        "missing_required_source_field",
        "malformed_or_unreadable_source_artifact",
        "ambiguous_identity",
        "manual_review_only",
        "excluded",
        "blocked_unsupported_or_manual_review_total",
    ):
        _line(key, counts.get(key, 0), stdout)

    _section("SUPPORTED CACHED TICKERS PROCESSED", stdout)
    supported = tuple(payload.get("supported_cached_tickers") or ())
    print(", ".join(supported) if supported else "none", file=stdout)

    _section("NON-SUPPORTED ENTRIES", stdout)
    non_supported = tuple(payload.get("non_supported_entries") or ())
    if not non_supported:
        print("none", file=stdout)
    for item in non_supported:
        print(
            f"{item['ticker']} | {item['status']} | origin={item['universe_entry_origin']} | reason={item['reason']}",
            file=stdout,
        )

    _section("CACHED-SOURCE BATCH", stdout)
    if not batch:
        print("not processed", file=stdout)
        for reason in payload.get("blocked_reasons") or ():
            print(f"blocked reason: {reason}", file=stdout)
    else:
        batch_counts = batch["batch_counts"]
        _line("Batch contract", batch["contract_version"], stdout)
        _line("Batch state", batch["batch_execution_state"], stdout)
        _line("Requested", batch_counts["requested_count"], stdout)
        _line("Completed", batch_counts["completed_count"], stdout)
        _line("Completed with limitations", batch_counts["completed_with_limitations_count"], stdout)
        _line("Blocked", batch_counts["blocked_count"], stdout)
        _line("Failed", batch_counts["failed_count"], stdout)
        _line("Skipped", batch_counts["skipped_count"], stdout)
        _line("Artifact manifest", batch.get("artifact_manifest_reference") or "not written", stdout)

    _section("SAFETY BOUNDARY", stdout)
    print(payload["safety_boundary"], file=stdout)
    _line("Live provider call made", payload["live_provider_call_made"], stdout)
    _line("Non-production batch", payload["non_production_batch"], stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ME-RUN23 expanded supported-universe cached-source scan."
    )
    parser.add_argument("--candidate-classification-summary", required=True)
    parser.add_argument("--professional-swing-universe", default=str(PROFESSIONAL_SWING_UNIVERSE_PATH))
    parser.add_argument("--source-snapshot-root", default=str(DEFAULT_SOURCE_SNAPSHOT_ROOT))
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--ticker-limit", type=int, default=None)
    parser.add_argument("--write-local-artifacts", action="store_true")
    parser.add_argument("--artifact-output-root", default=str(DEFAULT_ARTIFACT_OUTPUT_ROOT))
    parser.add_argument("--emit-json", action="store_true")
    return parser


def _generated_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _default_batch_id(generated_at: str) -> str:
    safe_timestamp = generated_at.replace(":", "").replace("+", "").replace("-", "")
    return f"{DEFAULT_BATCH_ID_PREFIX}-{safe_timestamp}"


def _section(title: str, stdout: TextIO) -> None:
    print(f"\n## {title}", file=stdout)


def _line(label: str, value: Any, stdout: TextIO) -> None:
    print(f"{label}: {value}", file=stdout)


if __name__ == "__main__":
    raise SystemExit(main())
