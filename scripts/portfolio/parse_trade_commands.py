import sys
import os

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.portfolio.portfolio_manager import log_trade


def parse_trade_command(command: str):
    """
    Parser voor trade commands.

    Supported formats:
    - BUY ASML 5 842.50
    - buy asml 5 842,50
    - buy asml 5 at 842.50
    - buy asml 5 @ 842.50
    - SELL ASML 2 870.00

    Belangrijk:
    Deze parser logt enkel de transactie in portfolio_transactions.csv.
    De actuele portfolio_positions.csv wordt later opgebouwd door build_portfolio.py.
    """

    if not command:
        raise ValueError("Empty command")

    cmd = command.strip().lower()
    cmd = cmd.replace("@", " ")
    cmd = cmd.replace(" at ", " ")

    parts = cmd.split()

    if len(parts) < 4:
        raise ValueError("Invalid command format. Expected: BUY TICKER QUANTITY PRICE")

    side = parts[0].upper()
    ticker = parts[1].upper()

    if side not in {"BUY", "SELL"}:
        raise ValueError("Side must be BUY or SELL")

    try:
        quantity = float(parts[2])
    except ValueError:
        raise ValueError("Quantity must be a number")

    if quantity <= 0:
        raise ValueError("Quantity must be > 0")

    raw_price = parts[3]

    if raw_price == "market":
        raise ValueError("Market orders not supported yet. Please provide an execution price.")

    try:
        price = float(raw_price.replace(",", "."))
    except ValueError:
        raise ValueError("Invalid price format")

    if price <= 0:
        raise ValueError("Price must be > 0")

    log_trade(ticker, side, quantity, price)

    return {
        "status": "success",
        "message": f"{side} {ticker} {quantity} @ {price}",
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "price": price,
    }


if __name__ == "__main__":
    print("Enter trade command (e.g. BUY ASML 5 842.50)")
    print("Type exit or quit to stop.")

    while True:
        try:
            command = input(">> ")

            if command.lower() in {"exit", "quit"}:
                break

            result = parse_trade_command(command)

            print("\n✅ Trade logged:")
            print(result["message"])
            print("\nℹ️ Portfolio positions will be rebuilt by build_portfolio.py during the scan.")

        except Exception as e:
            print(f"\n❌ Error: {e}")