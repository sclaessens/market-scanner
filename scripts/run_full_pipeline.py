from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, command: list[str]) -> None:
    command_text = " ".join(command)
    print(f"\nPipeline step started: {name}")
    print(f"Command: {command_text}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Pipeline step failed: {name}")
        print(f"Return code: {result.returncode}")
        sys.exit(result.returncode)

    print(f"Pipeline step completed: {name}")


def _build_run_scan_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "scripts/run_scan.py"]
    if args.fundamentals_history_path:
        command.extend(["--fundamentals-history-path", args.fundamentals_history_path])
    if args.fundamental_metrics_output_path:
        command.extend(["--fundamental-metrics-output-path", args.fundamental_metrics_output_path])
    if args.fundamental_analysis_output_path:
        command.extend(["--fundamental-analysis-output-path", args.fundamental_analysis_output_path])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the governed full market-scanner pipeline.")
    parser.add_argument("--fundamentals-history-path", help="Optional raw fundamentals history input path.")
    parser.add_argument("--fundamental-metrics-output-path", help="Optional generated fundamental metrics output path.")
    parser.add_argument("--fundamental-analysis-output-path", help="Optional generated fundamental analysis output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Governance-clean full pipeline wrapper.

    Sprint 0 rule:
    - scripts/run_scan.py owns the deterministic end-to-end flow:
      scanner → validation → context → fundamental → timing state →
      portfolio state → portfolio intelligence → Decision Engine →
      reporting → Telegram delivery.
    - Legacy watchlist action-updater steps are intentionally excluded because
      watchlist may only classify timing state, not emit capital actions.
    """
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("Pipeline run started: full pipeline")
    print(f"Started at: {started_at}")

    args = parse_args(argv if argv is not None else [])
    run_step(
        "1. Core end-to-end pipeline",
        _build_run_scan_command(args),
    )

    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("\nPipeline run completed: full pipeline")
    print(f"Completed at: {completed_at}")


if __name__ == "__main__":
    main(sys.argv[1:])
