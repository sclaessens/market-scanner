from __future__ import annotations

import os
from pathlib import Path
from typing import List

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
TELEGRAM_MESSAGE_FILE = PROJECT_ROOT / "reports" / "daily" / "telegram_message.txt"
TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4000


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """
    Laad een simpele .env file in os.environ.
    Verwacht lijnen in de vorm KEY=VALUE.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' ontbreekt. "
            f"Controleer je .env bestand."
        )
    return value


def read_telegram_message(file_path: Path = TELEGRAM_MESSAGE_FILE) -> str:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Telegram message file niet gevonden: {file_path}"
        )

    message = file_path.read_text(encoding="utf-8").strip()

    if not message:
        raise ValueError(
            f"Telegram message file is leeg: {file_path}"
        )

    return message


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Splits lange berichten op.
    Probeert eerst op lijnniveau te splitsen.
    """
    if len(text) <= max_length:
        return [text]

    parts: List[str] = []
    current_part = ""

    for line in text.splitlines(keepends=True):
        if len(current_part) + len(line) <= max_length:
            current_part += line
        else:
            if current_part.strip():
                parts.append(current_part.rstrip())
                current_part = ""

            while len(line) > max_length:
                parts.append(line[:max_length].rstrip())
                line = line[max_length:]

            current_part = line

    if current_part.strip():
        parts.append(current_part.rstrip())

    return parts


def send_message(text: str) -> None:
    load_env_file()

    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"
    message_parts = split_message(text)

    for part in message_parts:
        payload = {
            "chat_id": chat_id,
            "text": part,
            "disable_web_page_preview": True,
        }

        response = requests.post(url, json=payload, timeout=20)

        if response.status_code != 200:
            raise RuntimeError(
                f"Telegram API fout ({response.status_code}): {response.text}"
            )

        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API gaf fout terug: {data}")


def send_daily_summary() -> None:
    message = read_telegram_message()
    send_message(message)


if __name__ == "__main__":
    send_daily_summary()
    print("Telegram bericht succesvol verstuurd.")