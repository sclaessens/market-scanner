from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION = (
    "market-engine-canonical-ticker-universe-v1"
)
CANONICAL_TICKER_UNIVERSE_PATH = Path(
    "data/market_engine/ticker_universe/ticker_universe.csv"
)

REQUIRED_CANONICAL_TICKER_UNIVERSE_COLUMNS = (
    "ticker",
    "name",
    "market",
    "asset_type",
    "active",
    "priority",
    "source_policy",
    "portfolio_relevant",
    "telegram_preview_eligible",
    "telegram_delivery_eligible",
    "notes",
)

ALLOWED_MARKETS = {"USA", "EURONEXT", "LSE", "XETRA", "OTHER"}
ALLOWED_ASSET_TYPES = {"equity", "fund", "etf", "index", "crypto", "other"}
ALLOWED_SOURCE_POLICIES = {
    "cached_source_only",
    "cached_source_required",
    "manual_review_only",
    "blocked",
}
DEFAULT_EXECUTION_SOURCE_POLICIES = {
    "cached_source_only",
    "cached_source_required",
}

_BOOLEAN_FIELDS = (
    "active",
    "portfolio_relevant",
    "telegram_preview_eligible",
    "telegram_delivery_eligible",
)
_TICKER_RE = re.compile(r"^[A-Z0-9.-]+$")


class CanonicalTickerUniverseValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalTickerUniverseEntry:
    contract_version: str
    source_path: str
    row_number: int
    ticker: str
    normalized_ticker: str
    name: str
    market: str
    asset_type: str
    active: bool
    priority: int
    source_policy: str
    portfolio_relevant: bool
    telegram_preview_eligible: bool
    telegram_delivery_eligible: bool
    notes: str
    metadata: dict[str, str]
    validation_state: str = "valid"


@dataclass(frozen=True)
class CanonicalTickerUniverseResult:
    contract_version: str
    source_path: str
    entries: tuple[CanonicalTickerUniverseEntry, ...]
    loaded_row_count: int
    selected_row_count: int
    excluded_inactive_count: int
    excluded_blocked_count: int
    excluded_manual_review_only_count: int
    include_inactive: bool
    validation_state: str = "valid"


def load_canonical_ticker_universe(
    path: str | Path = CANONICAL_TICKER_UNIVERSE_PATH,
    *,
    include_inactive: bool = False,
) -> CanonicalTickerUniverseResult:
    return validate_canonical_ticker_universe(
        path,
        include_inactive=include_inactive,
    )


def validate_canonical_ticker_universe(
    path: str | Path = CANONICAL_TICKER_UNIVERSE_PATH,
    *,
    include_inactive: bool = False,
) -> CanonicalTickerUniverseResult:
    source_path = Path(path)
    rows = _read_csv_rows(source_path)
    entries = tuple(_entry_from_row(source_path=source_path, row=row) for row in rows)
    _validate_duplicate_keys(entries)

    selected_entries = entries if include_inactive else tuple(
        entry
        for entry in entries
        if entry.active and entry.source_policy in DEFAULT_EXECUTION_SOURCE_POLICIES
    )
    ordered_entries = tuple(sorted(selected_entries, key=_entry_order_key))

    return CanonicalTickerUniverseResult(
        contract_version=CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION,
        source_path=source_path.as_posix(),
        entries=ordered_entries,
        loaded_row_count=len(entries),
        selected_row_count=len(ordered_entries),
        excluded_inactive_count=sum(1 for entry in entries if not entry.active),
        excluded_blocked_count=sum(
            1 for entry in entries if entry.source_policy == "blocked"
        ),
        excluded_manual_review_only_count=sum(
            1 for entry in entries if entry.source_policy == "manual_review_only"
        ),
        include_inactive=include_inactive,
    )


def _read_csv_rows(source_path: Path) -> tuple[dict[str, Any], ...]:
    if not source_path.exists():
        raise CanonicalTickerUniverseValidationError(
            f"Canonical ticker universe file not found: {source_path}"
        )
    try:
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                raw_header = next(reader)
            except StopIteration as exc:
                raise CanonicalTickerUniverseValidationError(
                    "Canonical ticker universe CSV is missing a header row."
                ) from exc
            header = _validate_header(raw_header)
            rows = []
            for row_number, raw_row in enumerate(reader, start=2):
                if _blank_row(raw_row):
                    continue
                values = [cell.strip() for cell in raw_row]
                if len(values) > len(header):
                    raise CanonicalTickerUniverseValidationError(
                        _error(row_number, "", "", "row has more values than header")
                    )
                padded = values + [""] * (len(header) - len(values))
                rows.append(
                    {
                        "row_number": row_number,
                        "values": dict(zip(header, padded, strict=True)),
                    }
                )
            return tuple(rows)
    except OSError as exc:
        raise CanonicalTickerUniverseValidationError(
            f"Unable to read canonical ticker universe CSV: {source_path}"
        ) from exc
    except csv.Error as exc:
        raise CanonicalTickerUniverseValidationError(
            f"Canonical ticker universe CSV is malformed: {source_path}: {exc}"
        ) from exc


def _validate_header(raw_header: Iterable[str]) -> tuple[str, ...]:
    header = tuple(cell.strip() for cell in raw_header)
    if not header or all(not cell for cell in header):
        raise CanonicalTickerUniverseValidationError(
            "Canonical ticker universe CSV is missing a header row."
        )
    if any(not cell for cell in header):
        raise CanonicalTickerUniverseValidationError(
            "Canonical ticker universe CSV contains an unnamed column."
        )
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise CanonicalTickerUniverseValidationError(
            "Canonical ticker universe CSV contains duplicate columns: "
            + ", ".join(duplicates)
        )
    missing = [
        column
        for column in REQUIRED_CANONICAL_TICKER_UNIVERSE_COLUMNS
        if column not in header
    ]
    if missing:
        raise CanonicalTickerUniverseValidationError(
            "Canonical ticker universe CSV missing required columns: "
            + ", ".join(missing)
        )
    return header


def _entry_from_row(
    *,
    source_path: Path,
    row: dict[str, Any],
) -> CanonicalTickerUniverseEntry:
    row_number = int(row["row_number"])
    values = row["values"]
    ticker = _required_text(values, "ticker", row_number=row_number)
    normalized_ticker = ticker.upper()
    if not _TICKER_RE.fullmatch(normalized_ticker):
        raise CanonicalTickerUniverseValidationError(
            _error(
                row_number,
                "ticker",
                ticker,
                "ticker must contain only uppercase letters, digits, dots or hyphens after normalization",
            )
        )
    name = _required_text(values, "name", row_number=row_number)
    market = _allowed_text(
        values,
        "market",
        allowed=ALLOWED_MARKETS,
        row_number=row_number,
        transform="upper",
    )
    asset_type = _allowed_text(
        values,
        "asset_type",
        allowed=ALLOWED_ASSET_TYPES,
        row_number=row_number,
        transform="lower",
    )
    active = _boolean(values, "active", row_number=row_number)
    priority = _priority(values, row_number=row_number)
    source_policy = _allowed_text(
        values,
        "source_policy",
        allowed=ALLOWED_SOURCE_POLICIES,
        row_number=row_number,
        transform="lower",
    )
    portfolio_relevant = _boolean(values, "portfolio_relevant", row_number=row_number)
    telegram_preview_eligible = _boolean(
        values,
        "telegram_preview_eligible",
        row_number=row_number,
    )
    telegram_delivery_eligible = _boolean(
        values,
        "telegram_delivery_eligible",
        row_number=row_number,
    )
    if telegram_delivery_eligible and not telegram_preview_eligible:
        raise CanonicalTickerUniverseValidationError(
            _error(
                row_number,
                "telegram_delivery_eligible",
                "true",
                "telegram delivery eligibility cannot override preview ineligibility",
            )
        )
    notes = str(values.get("notes", "")).strip()
    metadata = {
        key: str(value).strip()
        for key, value in values.items()
        if key not in REQUIRED_CANONICAL_TICKER_UNIVERSE_COLUMNS
    }
    return CanonicalTickerUniverseEntry(
        contract_version=CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION,
        source_path=source_path.as_posix(),
        row_number=row_number,
        ticker=normalized_ticker,
        normalized_ticker=normalized_ticker,
        name=name,
        market=market,
        asset_type=asset_type,
        active=active,
        priority=priority,
        source_policy=source_policy,
        portfolio_relevant=portfolio_relevant,
        telegram_preview_eligible=telegram_preview_eligible,
        telegram_delivery_eligible=telegram_delivery_eligible,
        notes=notes,
        metadata=metadata,
    )


def _required_text(
    values: dict[str, str],
    field_name: str,
    *,
    row_number: int,
) -> str:
    value = str(values.get(field_name, "")).strip()
    if not value:
        raise CanonicalTickerUniverseValidationError(
            _error(row_number, field_name, value, "required value is empty")
        )
    return value


def _allowed_text(
    values: dict[str, str],
    field_name: str,
    *,
    allowed: set[str],
    row_number: int,
    transform: str,
) -> str:
    value = _required_text(values, field_name, row_number=row_number)
    normalized = value.upper() if transform == "upper" else value.lower()
    if normalized not in allowed:
        raise CanonicalTickerUniverseValidationError(
            _error(
                row_number,
                field_name,
                value,
                "value is outside the allowed domain",
            )
        )
    return normalized


def _boolean(
    values: dict[str, str],
    field_name: str,
    *,
    row_number: int,
) -> bool:
    value = _required_text(values, field_name, row_number=row_number).lower()
    if value not in {"true", "false"}:
        raise CanonicalTickerUniverseValidationError(
            _error(
                row_number,
                field_name,
                value,
                "boolean value must be true or false",
            )
        )
    return value == "true"


def _priority(values: dict[str, str], *, row_number: int) -> int:
    raw_value = _required_text(values, "priority", row_number=row_number)
    try:
        priority = int(raw_value)
    except ValueError as exc:
        raise CanonicalTickerUniverseValidationError(
            _error(row_number, "priority", raw_value, "priority must be an integer")
        ) from exc
    if priority < 1:
        raise CanonicalTickerUniverseValidationError(
            _error(
                row_number,
                "priority",
                raw_value,
                "priority must be greater than or equal to 1",
            )
        )
    return priority


def _validate_duplicate_keys(entries: tuple[CanonicalTickerUniverseEntry, ...]) -> None:
    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (entry.normalized_ticker, entry.market)
        if key in seen:
            raise CanonicalTickerUniverseValidationError(
                _error(
                    entry.row_number,
                    "ticker",
                    entry.normalized_ticker,
                    f"duplicate ticker/market row also appears at row {seen[key]}",
                )
            )
        seen[key] = entry.row_number


def _entry_order_key(
    entry: CanonicalTickerUniverseEntry,
) -> tuple[int, str, str]:
    return (entry.priority, entry.ticker, entry.market)


def _blank_row(row: Iterable[str]) -> bool:
    return all(not str(cell).strip() for cell in row)


def _error(row_number: int, field_name: str, value: Any, reason: str) -> str:
    field = field_name or "<row>"
    return f"row {row_number}, field {field}, value {value!r}: {reason}"
