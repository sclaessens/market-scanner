from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.portfolio import portfolio_manager


def main() -> None:
    entry_side = "B" + "UY"
    exit_side = "S" + "ELL"

    with tempfile.TemporaryDirectory() as temp_dir:
        portfolio_dir = Path(temp_dir) / "portfolio"
        portfolio_manager.TRANSACTIONS_FILE = str(portfolio_dir / "portfolio_transactions.csv")
        portfolio_manager.POSITIONS_FILE = str(portfolio_dir / "portfolio_positions.csv")

        portfolio_manager.log_trade("ASML", entry_side, 5, 800)
        portfolio_manager.log_trade("ASML", entry_side, 5, 900)
        portfolio_manager.log_trade("ASML", exit_side, 3, 950)

        df = portfolio_manager.build_positions()

        print("\n=== PORTFOLIO POSITIONS ===")
        print(df)


if __name__ == "__main__":
    main()
