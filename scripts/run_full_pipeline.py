from __future__ import annotations

import sys
from pathlib import Path
import subprocess

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
    print("🚀 Starting full pipeline...\n")

    run_step(
        "1. Market scan",
        ["python", "scripts/run_scan.py"],
    )

    run_step(
        "2. Auto watchlist from scan",
        ["python", "scripts/watchlist/auto_watchlist_from_scan.py"],
    )

    run_step(
        "3. Build watchlist",
        ["python", "scripts/watchlist/build_watchlist.py"],
    )

    run_step(
        "4. Evaluate watchlist",
        ["python", "scripts/watchlist/evaluate_watchlist.py"],
    )

    run_step(
        "5. Decision engine",
        ["python", "scripts/watchlist/update_watchlist_actions.py"],
    )

    run_step(
        "6. Rebuild watchlist (after decisions)",
        ["python", "scripts/watchlist/build_watchlist.py"],
    )

    run_step(
        "7. Re-evaluate watchlist (final)",
        ["python", "scripts/watchlist/evaluate_watchlist.py"],
    )

    run_step(
        "8. Build Telegram summary",
        ["python", "scripts/reporting/build_telegram_summary.py"],
    )

    print("\n🎯 Pipeline completed successfully!")


if __name__ == "__main__":
    main()