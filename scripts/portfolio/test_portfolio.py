import sys
import os

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.portfolio.portfolio_manager import log_trade, build_positions
import os

# RESET (optioneel)
if os.path.exists("data/portfolio/portfolio_transactions.csv"):
    os.remove("data/portfolio/portfolio_transactions.csv")

if os.path.exists("data/portfolio/portfolio_positions.csv"):
    os.remove("data/portfolio/portfolio_positions.csv")


# TEST FLOW
log_trade("ASML", "BUY", 5, 800)
log_trade("ASML", "BUY", 5, 900)
log_trade("ASML", "SELL", 3, 950)

df = build_positions()

print("\n=== PORTFOLIO POSITIONS ===")
print(df)