from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAIL_CLOSED_MESSAGE = (
    "Legacy full pipeline execution is disabled. "
    "Use the canonical app dry-run boundary configured in the active workflow."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed for legacy full market-scanner pipeline execution.",
    )
    parser.add_argument("--fundamentals-history-path", help="Optional raw fundamentals history input path.")
    parser.add_argument("--fundamental-metrics-output-path", help="Optional generated fundamental metrics output path.")
    parser.add_argument("--fundamental-analysis-output-path", help="Optional generated fundamental analysis output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fail closed so this legacy wrapper no longer invokes legacy runtime."""

    parse_args(argv if argv is not None else [])
    print(FAIL_CLOSED_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
