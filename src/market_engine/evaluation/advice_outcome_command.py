from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from market_engine.evaluation.advice_outcomes import (
    parse_horizons,
    run_advice_outcome_evaluation,
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
    try:
        evaluation, output_dir = run_advice_outcome_evaluation(
            args.advice_index,
            price_data_root=args.price_data_root,
            output_root=args.output_root,
            run_id=args.run_id,
            horizons=parse_horizons(args.horizons),
            allow_overwrite=args.allow_overwrite,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: unable to write advice outcome outputs: {exc}", file=stderr)
        return 2

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output_dir.as_posix(),
                "summary": evaluation["advice_outcome_index"]["summary"],
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
            "Evaluate deterministic Market Engine advice labels against local "
            "price-history outcomes."
        )
    )
    parser.add_argument("--advice-index", required=True)
    parser.add_argument("--price-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--horizons", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
