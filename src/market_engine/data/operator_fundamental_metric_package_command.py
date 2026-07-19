from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from market_engine.data.operator_fundamental_metric_package import (
    OperatorPackageConfigurationError,
    OperatorPackageInputError,
    prepare_operator_fundamental_metric_package,
)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = argparse.ArgumentParser(
        prog="market-engine-prepare-fundamental-metric-package",
        description="Prepare and fail-closed structurally validate a local operator fundamental metric package for explicit source-approval review.",
    )
    parser.add_argument("--input", required=True, help="Path to the immutable local operator JSON input.")
    parser.add_argument("--package-output", required=True, help="Path for the accepted ME-DATA07-compatible package.")
    parser.add_argument("--report-output", required=True, help="Path for the machine-readable validation report.")
    args = parser.parse_args(argv)
    try:
        package, report = prepare_operator_fundamental_metric_package(
            args.input,
            package_output_path=args.package_output,
            report_output_path=args.report_output,
        )
    except OperatorPackageConfigurationError as exc:
        print(f"ME-DATA08 configuration error: {exc}", file=stderr)
        return 2
    except OperatorPackageInputError as exc:
        print(f"ME-DATA08 input error: {exc}", file=stderr)
        return 2
    except OSError as exc:
        print(f"ME-DATA08 filesystem error: {exc}", file=stderr)
        return 3
    print(json.dumps({"package_id": report["package_id"], "status": report["status"], "downstream_consumability": report["downstream_consumability"], "report_output": Path(args.report_output).as_posix(), "package_output": Path(args.package_output).as_posix() if package else None}, indent=2, sort_keys=True), file=stdout)
    return 0 if package is not None else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
