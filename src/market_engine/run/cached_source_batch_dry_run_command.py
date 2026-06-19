from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.run.cached_source_batch_execution import (
    CachedSourceBatchDryRunError,
    build_cached_source_batch_dry_run,
)
from market_engine.ticker_universe import (
    CANONICAL_TICKER_UNIVERSE_PATH,
    CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION,
    CanonicalTickerUniverseValidationError,
    load_canonical_ticker_universe,
)


MARKET_ENGINE_REAL_CACHED_SOURCE_BATCH_DRY_RUN_VISIBILITY_FORMAT_VERSION = (
    "market-engine-real-cached-source-batch-dry-run-visibility-v1"
)
DEFAULT_CACHED_SOURCE_SNAPSHOT_ROOT = "data/market_engine/source_snapshots"
DEFAULT_BATCH_ID_PREFIX = "real-cached-source-batch-dry-run"
DEFAULT_ARTIFACT_OUTPUT_ROOT = "artifacts/market_engine"


class CachedSourceBatchDryRunCommandError(ValueError):
    pass


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
        command_result = build_command_result(args)
    except (CachedSourceBatchDryRunCommandError, CachedSourceBatchDryRunError) as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2

    render_human_visible_output(command_result, stdout=stdout)
    if args.emit_json:
        print("\nJSON PAYLOAD", file=stdout)
        print(json.dumps(command_result, indent=2, sort_keys=True), file=stdout)
    return 0


def build_command_result(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = args.generated_at or _generated_at_utc()
    batch_id = args.batch_id or _default_batch_id(generated_at)
    requested_tickers, canonical_universe_metadata = _requested_tickers_from_args(args)
    artifact_created_at = generated_at if args.write_local_artifacts else None

    batch_payload = build_cached_source_batch_dry_run(
        source_snapshot_root=args.source_snapshot_root,
        batch_id=batch_id,
        generated_at=generated_at,
        requested_tickers=requested_tickers,
        discover_cached_tickers=bool(args.discover_cached_tickers),
        ticker_limit=args.ticker_limit,
        write_local_artifacts=bool(args.write_local_artifacts),
        artifact_output_root=args.artifact_output_root,
        artifact_created_at=artifact_created_at,
    )

    return {
        "visibility_contract_version": (
            MARKET_ENGINE_REAL_CACHED_SOURCE_BATCH_DRY_RUN_VISIBILITY_FORMAT_VERSION
        ),
        "command": _command_line_from_args(args),
        "run_context": {
            "batch_id": batch_payload["batch_id"],
            "generated_at": generated_at,
            "source_snapshot_root": batch_payload["source_snapshot_root"],
            "artifact_writing_enabled": bool(args.write_local_artifacts),
            "artifact_output_root": args.artifact_output_root,
            "ticker_limit": args.ticker_limit,
            "operator_ticker_input_reference": batch_payload[
                "operator_ticker_input_reference"
            ],
            "canonical_ticker_universe": canonical_universe_metadata,
            "overwrite_protection": "enabled",
        },
        "batch_payload": batch_payload,
        "next_review_actions": _next_review_actions(batch_payload, args),
    }


def render_human_visible_output(
    command_result: Mapping[str, Any],
    *,
    stdout: TextIO,
) -> None:
    batch = command_result["batch_payload"]
    context = command_result["run_context"]
    counts = batch["batch_counts"]
    metadata = batch["ticker_universe_metadata"]
    results = batch["per_ticker_results"]

    _section("RUN CONTEXT", stdout)
    _line("Visibility contract", command_result["visibility_contract_version"], stdout)
    _line("Command", command_result["command"], stdout)
    _line("Batch id", context["batch_id"], stdout)
    _line("Generated at", context["generated_at"], stdout)
    _line("Cached-source root", context["source_snapshot_root"], stdout)
    _line("Artifact writing", _yes_no(context["artifact_writing_enabled"]), stdout)
    _line("Artifact output root", context["artifact_output_root"], stdout)
    _line("Overwrite protection", context["overwrite_protection"], stdout)

    _section("INPUT DISCOVERY", stdout)
    _line("Input mode", batch["input_mode"], stdout)
    _line("Source mode", batch["source_mode"], stdout)
    _line("Ticker input", context["operator_ticker_input_reference"], stdout)
    canonical_universe = context.get("canonical_ticker_universe")
    if canonical_universe:
        _line("Canonical universe path", canonical_universe["source_path"], stdout)
        _line("Canonical universe contract", canonical_universe["contract_version"], stdout)
        _line("Canonical loaded rows", canonical_universe["loaded_row_count"], stdout)
        _line("Canonical selected rows", canonical_universe["selected_row_count"], stdout)
        _line("Excluded manual-review-only", ", ".join(canonical_universe["excluded_manual_review_only_tickers"]) or "none", stdout)
    _line("Discovered cached-source tickers", counts["discovered_cached_source_count"], stdout)
    _line("Selected ticker count", counts["requested_count"], stdout)
    _line("Ticker limit", context["ticker_limit"] or "none", stdout)

    _section("SELECTED TICKERS", stdout)
    selected = batch.get("requested_tickers") or ()
    print(", ".join(selected) if selected else "none", file=stdout)
    discovered = metadata.get("discovered_cached_source_tickers") or ()
    if discovered:
        _line("Discovered", ", ".join(discovered), stdout)

    _section("EXECUTION PROGRESS", stdout)
    for index, result in enumerate(results, start=1):
        reason = _first_reason(result)
        artifact = result.get("artifact_reference") or "not written"
        snapshot = result.get("source_snapshot_reference") or "none"
        print(
            f"{index:03d}. {result['ticker']} | {result['execution_state']} | "
            f"snapshot={snapshot} | artifact={artifact}",
            file=stdout,
        )
        if reason:
            print(f"     reason: {reason}", file=stdout)

    _section("BATCH SUMMARY", stdout)
    _line("Batch contract", batch["contract_version"], stdout)
    _line("Batch state", batch["batch_execution_state"], stdout)
    _line("Requested", counts["requested_count"], stdout)
    _line("Completed", counts["completed_count"], stdout)
    _line("Completed with limitations", counts["completed_with_limitations_count"], stdout)
    _line("Blocked", counts["blocked_count"], stdout)
    _line("Failed", counts["failed_count"], stdout)
    _line("Skipped", counts["skipped_count"], stdout)
    _line("Executed", counts["executed_count"], stdout)

    _section("BLOCKED / FAILED TICKERS", stdout)
    blocked_or_failed = [
        result
        for result in results
        if str(result["execution_state"]).startswith(("blocked", "failed"))
    ]
    if not blocked_or_failed:
        print("none", file=stdout)
    for result in blocked_or_failed:
        print(
            f"{result['ticker']} | {result['execution_state']} | "
            f"reason={_first_reason(result) or 'unspecified'}",
            file=stdout,
        )

    _section("ARTIFACTS", stdout)
    _line("Artifact writing", _yes_no(context["artifact_writing_enabled"]), stdout)
    _line("Artifact output root", context["artifact_output_root"], stdout)
    _line("Batch manifest", batch.get("artifact_manifest_reference") or "not written", stdout)
    _line("Artifact write count", _artifact_write_count(results), stdout)
    print("Generated artifacts are not committed by default.", file=stdout)

    _section("FORBIDDEN SIDE-EFFECT CONFIRMATION", stdout)
    print(batch["forbidden_side_effect_confirmation"], file=stdout)
    print(batch["authority_boundary_confirmation"], file=stdout)

    _section("NEXT REVIEW ACTIONS", stdout)
    for action in command_result["next_review_actions"]:
        print(f"- {action}", file=stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a human-visible cached-source batch dry-run."
    )
    parser.add_argument("--source-snapshot-root", default=DEFAULT_CACHED_SOURCE_SNAPSHOT_ROOT)
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--tickers", default=None)
    ticker_group.add_argument("--ticker-file", default=None)
    ticker_group.add_argument(
        "--canonical-ticker-universe",
        nargs="?",
        const=str(CANONICAL_TICKER_UNIVERSE_PATH),
        default=None,
    )
    ticker_group.add_argument("--discover-cached-tickers", action="store_true")
    parser.add_argument("--ticker-limit", type=int, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--write-local-artifacts", action="store_true")
    parser.add_argument("--artifact-output-root", default=DEFAULT_ARTIFACT_OUTPUT_ROOT)
    parser.add_argument("--emit-json", action="store_true")
    return parser


def _requested_tickers_from_args(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...] | None, dict[str, Any] | None]:
    if args.discover_cached_tickers:
        return None, None
    if args.tickers:
        return _parse_ticker_tokens(args.tickers.split(",")), None
    if args.ticker_file:
        return _parse_ticker_file(Path(args.ticker_file)), None
    if args.canonical_ticker_universe:
        return _parse_canonical_ticker_universe(Path(args.canonical_ticker_universe))
    raise CachedSourceBatchDryRunCommandError(
        "Provide --tickers, --ticker-file, --canonical-ticker-universe, "
        "or --discover-cached-tickers."
    )


def _parse_canonical_ticker_universe(
    path: Path,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    try:
        universe = load_canonical_ticker_universe(path, include_inactive=True)
    except CanonicalTickerUniverseValidationError as exc:
        raise CachedSourceBatchDryRunCommandError(
            f"Unable to load canonical ticker universe: {exc}"
        ) from exc

    selected_entries = tuple(
        entry
        for entry in universe.entries
        if entry.active and entry.source_policy == "cached_source_only"
    )
    selected_tickers = tuple(entry.ticker for entry in selected_entries)
    if not selected_tickers:
        raise CachedSourceBatchDryRunCommandError(
            "Canonical ticker universe produced no active cached_source_only tickers."
        )

    excluded_inactive = tuple(entry.ticker for entry in universe.entries if not entry.active)
    excluded_manual_review_only = tuple(
        entry.ticker
        for entry in universe.entries
        if entry.source_policy == "manual_review_only"
    )
    excluded_blocked = tuple(
        entry.ticker for entry in universe.entries if entry.source_policy == "blocked"
    )
    excluded_other_source_policy = tuple(
        entry.ticker
        for entry in universe.entries
        if entry.active
        and entry.source_policy
        not in {"cached_source_only", "manual_review_only", "blocked"}
    )
    metadata = {
        "contract_version": CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION,
        "source_path": universe.source_path,
        "loaded_row_count": universe.loaded_row_count,
        "selected_row_count": len(selected_tickers),
        "selection_policy": "active_true_and_source_policy_cached_source_only",
        "selected_tickers": selected_tickers,
        "excluded_inactive_tickers": excluded_inactive,
        "excluded_manual_review_only_tickers": excluded_manual_review_only,
        "excluded_blocked_tickers": excluded_blocked,
        "excluded_other_source_policy_tickers": excluded_other_source_policy,
    }
    return selected_tickers, metadata


def _parse_ticker_file(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CachedSourceBatchDryRunCommandError(
            f"Unable to read ticker file: {path}"
        ) from exc
    tokens: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens.extend(stripped.split(","))
    return _parse_ticker_tokens(tokens)


def _parse_ticker_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    tickers = tuple(token.strip().upper() for token in tokens if token.strip())
    if not tickers:
        raise CachedSourceBatchDryRunCommandError("Ticker input cannot be empty.")
    if len(set(tickers)) != len(tickers):
        raise CachedSourceBatchDryRunCommandError("Ticker input must be unique.")
    return tickers


def _generated_at_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_batch_id(generated_at: str) -> str:
    safe_timestamp = generated_at.replace(":", "").replace("-", "").replace("Z", "z")
    return f"{DEFAULT_BATCH_ID_PREFIX}-{safe_timestamp}"


def _command_line_from_args(args: argparse.Namespace) -> str:
    parts = ["market-engine-cached-source-batch-dry-run"]
    parts.extend(["--source-snapshot-root", str(args.source_snapshot_root)])
    if args.tickers:
        parts.extend(["--tickers", args.tickers])
    elif args.ticker_file:
        parts.extend(["--ticker-file", args.ticker_file])
    elif args.canonical_ticker_universe:
        parts.extend(["--canonical-ticker-universe", str(args.canonical_ticker_universe)])
    elif args.discover_cached_tickers:
        parts.append("--discover-cached-tickers")
    if args.ticker_limit is not None:
        parts.extend(["--ticker-limit", str(args.ticker_limit)])
    if args.batch_id:
        parts.extend(["--batch-id", args.batch_id])
    if args.generated_at:
        parts.extend(["--generated-at", args.generated_at])
    if args.write_local_artifacts:
        parts.append("--write-local-artifacts")
        parts.extend(["--artifact-output-root", str(args.artifact_output_root)])
    if args.emit_json:
        parts.append("--emit-json")
    return " ".join(parts)


def _next_review_actions(batch_payload: Mapping[str, Any], args: argparse.Namespace) -> tuple[str, ...]:
    actions = [
        "Capture this terminal output with `| tee /dev/tty | pbcopy`.",
        "Review completed, blocked, and failed ticker counts before broadening input.",
    ]
    if args.write_local_artifacts:
        actions.append("Inspect the batch manifest and at least one per-ticker artifact.")
        actions.append("Do not commit generated artifacts by default.")
    if batch_payload["batch_counts"].get("blocked_count") or batch_payload["batch_counts"].get("failed_count"):
        actions.append("Triage blocked or failed tickers from the visible reasons above.")
    return tuple(actions)


def _artifact_write_count(results: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for result in results if result.get("artifact_reference"))


def _first_reason(result: Mapping[str, Any]) -> str:
    reasons = result.get("blocked_reasons") or ()
    return str(reasons[0]) if reasons else ""


def _section(title: str, stdout: TextIO) -> None:
    print(f"\n{title}", file=stdout)
    print("-" * len(title), file=stdout)


def _line(label: str, value: object, stdout: TextIO) -> None:
    print(f"{label}: {value}", file=stdout)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
