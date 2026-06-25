from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.source_refresh.cached_source_snapshot_acquisition_dry_run import (
    build_cached_source_snapshot_acquisition_dry_run,
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
    report = build_cached_source_snapshot_acquisition_dry_run(
        tickers=args.ticker or (),
        source_families=args.source_family or (),
        output_root=args.output_root,
        dry_run_at=args.dry_run_at,
        batch_id=args.batch_id,
    )
    if args.output_json:
        try:
            _write_json_report(report, Path(args.output_json))
        except OSError as exc:
            print(f"ERROR: unable to write acquisition dry-run report: {exc}", file=stderr)
            return 2

    if args.human:
        render_human_readable_acquisition_dry_run(report, stdout=stdout)
    else:
        json.dump(report, stdout, indent=2, sort_keys=True)
        stdout.write("\n")
    return 0


def render_human_readable_acquisition_dry_run(
    report: Mapping[str, Any],
    *,
    stdout: TextIO,
) -> None:
    counts = report["counts"]
    print("CACHED-SOURCE SNAPSHOT ACQUISITION DRY-RUN", file=stdout)
    print(f"Report format: {report['report_format_version']}", file=stdout)
    print(f"Output root: {report['output_root']}", file=stdout)
    print(f"Dry-run at: {report['dry_run_at']}", file=stdout)
    print(
        "Counts: "
        f"total={counts['total_requested_entries']} "
        f"planned={counts['planned_entries']} "
        f"blocked={counts['blocked_entries']} "
        f"invalid_ticker={counts['invalid_ticker_count']} "
        f"unsupported_source_family={counts['unsupported_source_family_count']} "
        f"missing_ticker={counts['missing_ticker_count']} "
        f"missing_source_family={counts['missing_source_family_count']}",
        file=stdout,
    )
    print("Entries:", file=stdout)
    for entry in report["entries"]:
        issues = ", ".join(entry["issues"]) or "none"
        print(
            f"- {entry['ticker']} | {entry['source_family']} | "
            f"{entry['acquisition_dry_run_status']} | issues={issues}",
            file=stdout,
        )
    print(report["staging_validator_handoff"], file=stdout)
    print(report["forbidden_side_effect_confirmation"], file=stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-cached-source-snapshot-acquisition-dry-run",
        description=(
            "Plan cached-source snapshot acquisition intent locally and emit a "
            "deterministic "
            "market-engine-cached-source-snapshot-acquisition-dry-run-v1 report."
        ),
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        help="Requested ticker. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--source-family",
        action="append",
        default=None,
        help="Requested source family. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Local root where future acquired snapshots would be staged.",
    )
    parser.add_argument(
        "--dry-run-at",
        default=None,
        help="Optional deterministic UTC timestamp for the emitted report.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional deterministic batch identifier for proposed staging paths.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional local path where the JSON dry-run report should be written.",
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
