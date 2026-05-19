from __future__ import annotations

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


def main() -> None:
    """
    Governance-clean full pipeline wrapper.

    Sprint 0 rule:
    - scripts/run_scan.py owns the deterministic end-to-end flow:
      scanner → validation → context → fundamental → timing state →
      portfolio state → portfolio intelligence → Decision Engine → reporting.
    - Legacy watchlist action-updater steps are intentionally excluded because
      watchlist may only classify timing state, not emit capital actions.
    """
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("Pipeline run started: full pipeline")
    print(f"Started at: {started_at}")

    run_step(
        "1. Core end-to-end pipeline",
        [sys.executable, "scripts/run_scan.py"],
    )

    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("\nPipeline run completed: full pipeline")
    print(f"Completed at: {completed_at}")


if __name__ == "__main__":
    main()
