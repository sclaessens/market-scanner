from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from market_engine.batch_status.artifact_discovery import discover_dry_run_artifacts
from market_engine.batch_status.status_index import (
    build_ticker_status_index,
    write_batch_status_outputs,
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
    tickers = _split_tickers(args.ticker)
    try:
        discovery = discover_dry_run_artifacts(
            args.artifact_root,
            tickers=tickers,
            max_artifacts=args.max_artifacts,
            include_invalid=args.include_invalid,
        )
        index = build_ticker_status_index(
            discovery,
            run_id=args.run_id,
            generated_at=args.generated_at,
        )
        output_dir = write_batch_status_outputs(
            index,
            discovery,
            output_root=args.output_root,
            run_id=args.run_id,
            allow_overwrite=args.allow_overwrite,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: unable to write batch status outputs: {exc}", file=stderr)
        return 2

    payload = {
        "run_id": args.run_id,
        "output_dir": output_dir.as_posix(),
        "summary": index["summary"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover existing Market Engine dry-run artifacts and write a "
            "deterministic ticker status index."
        )
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--max-artifacts", type=int, default=None)
    parser.add_argument("--ticker", action="append", default=())
    parser.add_argument("--include-invalid", action="store_true", default=True)
    return parser


def _split_tickers(values: Sequence[str]) -> tuple[str, ...]:
    tickers: list[str] = []
    for value in values:
        for item in value.split(","):
            cleaned = item.strip().upper()
            if cleaned:
                tickers.append(cleaned)
    return tuple(tickers)


if __name__ == "__main__":
    raise SystemExit(main())
