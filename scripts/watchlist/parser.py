import os
import csv
from datetime import datetime

WATCHLIST_FILE = "data/watchlist/watchlist_transactions.csv"
LOG_FILE = "data/logs/telegram_command_log.csv"

os.makedirs("data/watchlist", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)


def parse_command(command: str):
    parts = command.strip().split()

    if len(parts) < 2:
        return None

    action = parts[0].upper()
    ticker = parts[1].upper()

    setup_type = None

    if action == "WATCH":
        if len(parts) >= 3:
            setup_type = parts[2].upper()
        else:
            setup_type = "GENERAL"

    elif action == "UNWATCH":
        setup_type = None

    else:
        return None

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": action,
        "setup_type": setup_type,
        "source": "manual",
        "note": ""
    }


def append_csv(file_path, row, header):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def log_command(command: str):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": command
    }

    append_csv(LOG_FILE, row, ["timestamp", "command"])


def process_command(command: str):
    parsed = parse_command(command)

    if not parsed:
        print(f"Invalid command: {command}")
        return

    append_csv(
        WATCHLIST_FILE,
        parsed,
        ["timestamp", "ticker", "action", "setup_type", "source", "note"]
    )

    log_command(command)

    print(f"Processed: {command}")


if __name__ == "__main__":
    # TEST COMMANDS (later vervangen door Telegram input)
    commands = [
        "WATCH ASML",
        "WATCH NVDA BREAKOUT",
        "UNWATCH ASML"
    ]

    for cmd in commands:
        process_command(cmd)
