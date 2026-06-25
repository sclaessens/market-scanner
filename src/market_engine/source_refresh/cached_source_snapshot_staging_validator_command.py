from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    build_cached_source_snapshot_staging_validation,
)


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
    report = build_cached_source_snapshot_staging_validation(
        staging_root=args.staging_root,
        validated_at=args.validated_at,
        tickers=args.ticker,
    )
    if args.output_json:
        try:
            _write_json_report(report, Path(args.output_json))
        except OSError as exc:
            print(f"ERROR: unable to write staging validation report: {exc}", file=stderr)
            return 2

    if args.human:
        render_human_readable_staging_validation(report, stdout=stdout)
    else:
        json.dump(report, stdout, indent=2, sort_keys=True)
        stdout.write("\n")
    return 0


def render_human_readable_staging_validation(
    report: Mapping[str, Any],
    *,
    stdout: TextIO,
) -> None:
    counts = report["counts"]
    print("CACHED-SOURCE SNAPSHOT STAGING VALIDATION", file=stdout)
    print(f"Report format: {report['report_format_version']}", file=stdout)
    print(f"Staging root: {report['staging_root']}", file=stdout)
    print(f"Validated at: {report['validated_at']}", file=stdout)
    print(
        "Counts: "
        f"total={counts['total_inspected_entries']} "
        f"accepted={counts['accepted_entries']} "
        f"rejected={counts['rejected_entries']} "
        f"missing_manifest={counts['missing_manifest_count']} "
        f"malformed_manifest={counts['malformed_manifest_count']} "
        f"unknown_format={counts['unknown_format_count']} "
        f"missing_referenced_file={counts['missing_referenced_file_count']} "
        f"hash_mismatch={counts['hash_mismatch_count']} "
        f"size_mismatch={counts['size_mismatch_count']} "
        f"stale={counts['stale_count']} "
        f"fixture_or_test={counts['fixture_or_test_material_count']} "
        f"validation_status_blocked={counts['validation_status_blocked_count']} "
        f"usable_flag_conflict={counts['usable_flag_conflict_count']}",
        file=stdout,
    )
    print("Entries:", file=stdout)
    for entry in report["entries"]:
        issues = ", ".join(entry["issues"]) or "none"
        ticker = entry.get("ticker") or "UNKNOWN"
        snapshot_id = entry.get("snapshot_id") or "UNKNOWN"
        source_family = entry.get("source_family") or "unknown_source_family"
        print(
            f"- {ticker} | {snapshot_id} | {source_family} | "
            f"{entry['staging_validation_status']} | issues={issues}",
            file=stdout,
        )
    print(report["forbidden_side_effect_confirmation"], file=stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-validate-cached-source-snapshot-staging",
        description=(
            "Validate local manually staged cached-source snapshot acquisition "
            "manifests and emit a deterministic "
            "market-engine-cached-source-snapshot-staging-validation-v1 report."
        ),
    )
    parser.add_argument(
        "--staging-root",
        required=True,
        help="Local staging root containing cached-source snapshot artifacts.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=(),
        help="Optional ticker filter. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional local path where the JSON validation report should be written.",
    )
    parser.add_argument(
        "--validated-at",
        default=None,
        help="Optional deterministic UTC timestamp for the emitted report.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Print a compact human-readable summary instead of JSON.",
    )
    return parser


def _write_json_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
