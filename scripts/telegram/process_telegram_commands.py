from __future__ import annotations

import sys
import os

# 🔥 FIX IMPORT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import requests
from pathlib import Path

from scripts.portfolio.parse_trade_commands import parse_trade_command


# =========================
# PATHS & CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
OFFSET_FILE = PROJECT_ROOT / "data/logs/telegram_offset.txt"

TELEGRAM_API_BASE = "https://api.telegram.org"


# =========================
# ENV LOADER
# =========================

def load_env():
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


# =========================
# OFFSET HANDLING
# =========================

def load_offset() -> int:
    if not OFFSET_FILE.exists():
        return 0

    try:
        return int(OFFSET_FILE.read_text().strip())
    except:
        return 0


def save_offset(offset: int):
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(str(offset))


# =========================
# TELEGRAM API
# =========================

def get_updates(offset: int | None = None):
    token = get_env("TELEGRAM_BOT_TOKEN")

    url = f"{TELEGRAM_API_BASE}/bot{token}/getUpdates"

    params = {
        "timeout": 10
    }

    if offset:
        params["offset"] = offset

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram error: {data}")

    return data["result"]


def send_message(text: str):
    token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print(f"Error sending message: {e}")


# =========================
# CORE LOGIC
# =========================

def is_trade_command(text: str) -> bool:
    text = text.upper()
    return text.startswith("BUY ") or text.startswith("SELL ")


def process_commands():
    load_env()

    last_offset = load_offset()
    print(f"Last offset: {last_offset}")

    updates = get_updates(offset=last_offset + 1)

    if not updates:
        print("No new updates.")
        return

    max_update_id = last_offset

    for update in updates:
        update_id = update["update_id"]
        max_update_id = max(max_update_id, update_id)

        message = update.get("message")
        if not message:
            continue

        text = message.get("text", "").strip()
        if not text:
            continue

        print(f"\nReceived message:\n{text}")

        # 🔥 SPLIT MULTI-LINE COMMANDS
        commands = text.split("\n")

        for cmd in commands:
            cmd = cmd.strip()

            if not cmd:
                continue

            print(f"Processing command: {cmd}")

            if is_trade_command(cmd):
                try:
                    result = parse_trade_command(cmd)

                    send_message(f"✅ {result['message']} verwerkt")
                    print(f"Processed: {cmd}")

                except Exception as e:
                    error_msg = f"❌ Error: {cmd} → {e}"
                    send_message(error_msg)
                    print(error_msg)

            else:
                print(f"Ignored: {cmd}")

    # 🔥 CRUCIAAL: offset opslaan NA verwerking
    save_offset(max_update_id)
    print(f"\nNew offset saved: {max_update_id}")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    process_commands()