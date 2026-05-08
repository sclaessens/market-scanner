from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, command: list[str]) -> None:
    print(f"\n===== {name} =====")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"❌ Error in step: {name}")
        sys.exit(result.returncode)

    print(f"✅ Completed: {name}")


def main() -> None:
    """
    Governance-clean full pipeline wrapper.

    Sprint 0 rule:
    - scripts/run_scan.py owns the deterministic end-to-end flow:
      scanner → validation → context → portfolio state → Decision Engine → reporting.
    - Legacy watchlist action-updater steps are intentionally excluded because
      watchlist may only classify timing state, not emit capital actions.
    """
    print("🚀 Starting governance-clean full pipeline...\n")

    run_step(
        "1. Core end-to-end pipeline",
        ["python", "scripts/run_scan.py"],
    )

    print("\n🎯 Governance-clean pipeline completed successfully!")


if __name__ == "__main__":
    main()
