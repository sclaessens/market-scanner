import sys
import os

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.portfolio.portfolio_manager import log_trade, build_positions


# === CORE PARSER ===

def parse_trade_command(command: str):
    """
    Parse a trade command string and execute it.

    Example:
    BUY ASML 5 842.50
    SELL NVDA 2 190.00
    """

    parts = command.strip().split()

    if len(parts) != 4:
        raise ValueError("Invalid command format. Use: BUY TICKER QTY PRICE")

    side, ticker, quantity, price = parts

    side = side.upper()
    ticker = ticker.upper()

    if side not in ["BUY", "SELL"]:
        raise ValueError("Side must be BUY or SELL")

    try:
        quantity = float(quantity)
        price = float(price)
    except ValueError:
        raise ValueError("Quantity and price must be numbers")

    if quantity <= 0:
        raise ValueError("Quantity must be > 0")

    if price <= 0:
        raise ValueError("Price must be > 0")

    # === EXECUTE ===
    log_trade(ticker, side, quantity, price)
    positions = build_positions()

    return {
        "status": "success",
        "message": f"{side} {ticker} {quantity} @ {price}",
        "positions": positions
    }


# === CLI / TEST ===

if __name__ == "__main__":
    print("Enter trade command (e.g. BUY ASML 5 842.50)")

    while True:
        try:
            command = input(">> ")

            if command.lower() in ["exit", "quit"]:
                break

            result = parse_trade_command(command)

            print("\n✅ Trade executed:")
            print(result["message"])
            print("\n📊 Updated positions:")
            print(result["positions"])

        except Exception as e:
            print(f"\n❌ Error: {e}")