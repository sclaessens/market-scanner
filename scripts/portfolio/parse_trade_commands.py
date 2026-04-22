import sys
import os

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.portfolio.portfolio_manager import log_trade, build_positions


# === CORE PARSER ===

def parse_trade_command(command: str):
    """
    Smart parser for human-friendly trade commands.

    Supported formats:
    - BUY ASML 5 842.50
    - buy asml 5 842,50
    - buy asml 5 at 842.50
    - buy asml 5 @ 842.50
    - BUY NVDA 2 market
    """

    if not command:
        raise ValueError("Empty command")

    # =========================
    # NORMALIZE INPUT
    # =========================
    cmd = command.strip().lower()

    # Replace common separators
    cmd = cmd.replace("@", " ")
    cmd = cmd.replace(" at ", " ")

    parts = cmd.split()

    if len(parts) < 3:
        raise ValueError("Invalid command format")

    side = parts[0].upper()
    ticker = parts[1].upper()

    if side not in ["BUY", "SELL"]:
        raise ValueError("Side must be BUY or SELL")

    # =========================
    # QUANTITY
    # =========================
    try:
        quantity = float(parts[2])
    except ValueError:
        raise ValueError("Quantity must be a number")

    if quantity <= 0:
        raise ValueError("Quantity must be > 0")

    # =========================
    # PRICE
    # =========================
    price = None

    if len(parts) >= 4:
        raw_price = parts[3]

        if raw_price == "market":
            price = None
        else:
            try:
                price = float(raw_price.replace(",", "."))
            except ValueError:
                raise ValueError("Invalid price format")

    else:
        raise ValueError("Missing price")

    # =========================
    # HANDLE MARKET ORDERS
    # =========================
    if price is None:
        raise ValueError("Market orders not supported yet (no price)")

    # =========================
    # EXECUTE
    # =========================
    from scripts.portfolio.portfolio_manager import log_trade, build_positions

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