from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from market_engine.data.local_market_data_universe import (
    DEFAULT_MIN_HISTORY_ROWS,
    DEFAULT_PRICE_HISTORY_ROOT,
    DEFAULT_REQUIRED_FORWARD_DATE,
    inspect_price_history,
)


def price_history_path(
    instrument: Mapping[str, Any],
    *,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
) -> Path:
    return Path(price_history_root) / f"{instrument['source_symbol']}.csv"


def load_price_history_metadata(
    instrument: Mapping[str, Any],
    *,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    required_forward_date: str = DEFAULT_REQUIRED_FORWARD_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
) -> dict[str, Any]:
    inspection = inspect_price_history(
        instrument,
        price_history_root=price_history_root,
        required_forward_date=required_forward_date,
        min_history_rows=min_history_rows,
    )
    return {
        **inspection,
        "path": price_history_path(instrument, price_history_root=price_history_root).as_posix(),
    }
