from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from market_engine.advice.advice_batch import run_advice_batch


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
        batch, output_dir = run_advice_batch(
            args.ticker_status_index,
            output_root=args.output_root,
            run_id=args.run_id,
            generated_at=args.generated_at,
            target_universe_path=args.target_universe,
            target_size=args.target_size,
            max_tickers=args.max_tickers,
            allow_overwrite=args.allow_overwrite,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: unable to write advice batch outputs: {exc}", file=stderr)
        return 2

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output_dir.as_posix(),
                "summary": batch["summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        file=stdout,
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produce deterministic Market Engine advice batch outputs from a "
            "ticker status index."
        )
    )
    parser.add_argument("--ticker-status-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--target-universe", default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
