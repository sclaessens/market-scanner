from __future__ import annotations

import os
import requests
from pathlib import Path

from scripts.portfolio.parse_trade_commands import parse_trade_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

TELEGRAM_API_BASE = "https://api.telegram.org"


# === ENV LOADER ===

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


# === TELEGRAM ===

def get_updates(offset: int | None = None):
    token = get_env("TELEGRAM_BOT_TOKEN")

    url = f"{TELEGRAM_API_BASE}/bot{token}/getUpdates"

    params = {}
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

    requests.post(url, json=payload, timeout=20)


# === COMMAND PROCESSING ===

def process_commands():
    load_env()

    last_update_id = None

    updates = get_updates()

    if not updates:
        print("No new updates.")
        return

    for update in updates:
        last_update_id = update["update_id"]

        message = update.get("message")
        if not message:
            continue

        text = message.get("text", "").strip()

        if not text:
            continue

        print(f"Received: {text}")

        if text.startswith("BUY") or text.startswith("SELL"):
            try:
                result = parse_trade_command(text)

                send_message(f"✅ {result['message']} verwerkt")

            except Exception as e:
                send_message(f"❌ Error: {e}")

    # Markeer updates als gelezen
    if last_update_id:
        get_updates(offset=last_update_id + 1)


if __name__ == "__main__":
    process_commands()