from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from market_engine.evaluation.advice_outcome_refresh import run_advice_outcome_refresh


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
        refresh, output_dir = run_advice_outcome_refresh(
            args.evaluation_artifact,
            price_history_root=args.price_history_root,
            output_root=args.output_root,
            run_id=args.run_id,
            ticker_filter=_ticker_filter(args.tickers),
            allow_overwrite=args.allow_overwrite,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to refresh advice outcomes: {exc}", file=stderr)
        return 2

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output_dir.as_posix(),
                "summary": refresh["refresh_index"]["summary"],
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
            "Refresh unresolved Market Engine advice outcomes using local "
            "price-history snapshots only."
        )
    )
    parser.add_argument("--evaluation-artifact", required=True)
    parser.add_argument("--price-history-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def _ticker_filter(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    tickers = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    return tickers or None


if __name__ == "__main__":
    raise SystemExit(main())
