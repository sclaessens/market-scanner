from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.source_refresh.cached_source_snapshot_inventory import (
    build_cached_source_snapshot_inventory,
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
    report = build_cached_source_snapshot_inventory(
        input_root=args.input_root,
        inspected_at=args.inspected_at,
        tickers=args.ticker,
    )
    if args.output_json:
        try:
            _write_json_report(report, Path(args.output_json))
        except OSError as exc:
            print(f"ERROR: unable to write inventory report: {exc}", file=stderr)
            return 2

    if args.human:
        render_human_readable_inventory(report, stdout=stdout)
    else:
        json.dump(report, stdout, indent=2, sort_keys=True)
        stdout.write("\n")
    return 0


def render_human_readable_inventory(
    report: Mapping[str, Any],
    *,
    stdout: TextIO,
) -> None:
    counts = report["counts"]
    print("CACHED-SOURCE SNAPSHOT INVENTORY", file=stdout)
    print(f"Report format: {report['report_format_version']}", file=stdout)
    print(f"Input root: {report['input_root']}", file=stdout)
    print(f"Inspected at: {report['inspected_at']}", file=stdout)
    print(
        "Counts: "
        f"total={counts['total_inspected_entries']} "
        f"usable={counts['usable_entries']} "
        f"unusable={counts['unusable_entries']} "
        f"missing_manifest={counts['missing_manifest_count']} "
        f"malformed_manifest={counts['malformed_manifest_count']} "
        f"unknown_format={counts['unknown_format_count']} "
        f"missing_referenced_file={counts['missing_referenced_file_count']} "
        f"stale={counts['stale_count']}",
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
            f"{entry['inventory_status']} | issues={issues}",
            file=stdout,
        )
    print(report["forbidden_side_effect_confirmation"], file=stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-inventory-cached-source-snapshots",
        description=(
            "Inspect local cached-source snapshot acquisition manifests and emit "
            "a deterministic market-engine-cached-source-snapshot-inventory-v1 report."
        ),
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Local root containing cached-source snapshot artifacts to inspect.",
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
        help="Optional local path where the JSON inventory report should be written.",
    )
    parser.add_argument(
        "--inspected-at",
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
