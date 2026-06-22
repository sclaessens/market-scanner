from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION = (
    "market-engine-editable-professional-swing-universe-v1"
)
PROFESSIONAL_SWING_UNIVERSE_PATH = Path(
    "data/market_engine/ticker_universe/professional_swing_universe/"
    "professional_swing_universe.csv"
)

REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS = (
    "ticker",
    "name",
    "market",
    "asset_type",
    "active",
    "universe_status",
    "source_policy_hint",
    "operator_priority",
    "swing_profile",
    "liquidity_profile",
    "volatility_profile",
    "market_cap_profile",
    "theme",
    "sector",
    "notes",
)

ALLOWED_PROFESSIONAL_SWING_MARKETS = {
    "USA",
    "EURONEXT",
    "LSE",
    "XETRA",
    "TSX",
    "ASX",
    "TSE",
    "HKEX",
    "OTHER",
    "UNKNOWN",
}
ALLOWED_PROFESSIONAL_SWING_ASSET_TYPES = {
    "equity",
    "fund",
    "etf",
    "index",
    "crypto",
    "other",
    "unknown",
}
ALLOWED_PROFESSIONAL_SWING_UNIVERSE_STATUSES = {
    "candidate",
    "watching",
    "research_required",
    "needs_source_mapping",
    "manual_review_only",
    "blocked",
    "rejected",
}
ALLOWED_PROFESSIONAL_SWING_SOURCE_POLICY_HINTS = {
    "cached_source_candidate",
    "source_mapping_required",
    "manual_review_only",
    "unsupported",
    "unknown",
}
ALLOWED_PROFESSIONAL_SWING_PROFILES = {
    "breakout",
    "pullback",
    "trend_continuation",
    "mean_reversion",
    "relative_strength",
    "earnings_momentum",
    "thematic_momentum",
    "quality_compounder",
    "turnaround",
    "speculative_growth",
    "unknown",
}
ALLOWED_PROFESSIONAL_SWING_LIQUIDITY_PROFILES = {"high", "medium", "low", "unknown"}
ALLOWED_PROFESSIONAL_SWING_VOLATILITY_PROFILES = {
    "low",
    "medium",
    "high",
    "extreme",
    "unknown",
}
ALLOWED_PROFESSIONAL_SWING_MARKET_CAP_PROFILES = {
    "mega_cap",
    "large_cap",
    "mid_cap",
    "small_cap",
    "micro_cap",
    "unknown",
}
DEFAULT_PROFESSIONAL_SWING_UNIVERSE_STATUSES = {"candidate", "watching"}
DEFAULT_PROFESSIONAL_SWING_SOURCE_POLICY_HINTS = {
    "cached_source_candidate",
    "unknown",
}

_TICKER_RE = re.compile(r"^[A-Z0-9.-]+$")
_NOTES_FIELD = "notes"


class ProfessionalSwingUniverseValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ProfessionalSwingUniverseEntry:
    contract_version: str
    source_path: str
    row_number: int
    ticker: str
    normalized_ticker: str
    name: str
    market: str
    asset_type: str
    active: bool
    universe_status: str
    source_policy_hint: str
    operator_priority: int
    swing_profile: str
    liquidity_profile: str
    volatility_profile: str
    market_cap_profile: str
    theme: str
    sector: str
    notes: str
    metadata: dict[str, str]
    validation_state: str = "valid"


@dataclass(frozen=True)
class ProfessionalSwingUniverseResult:
    contract_version: str
    source_path: str
    entries: tuple[ProfessionalSwingUniverseEntry, ...]
    loaded_row_count: int
    selected_row_count: int
    excluded_inactive_count: int
    excluded_universe_status_count: int
    excluded_source_policy_hint_count: int
    include_inactive: bool
    validation_state: str = "valid"


def load_professional_swing_universe(
    path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
    *,
    include_inactive: bool = False,
) -> ProfessionalSwingUniverseResult:
    return validate_professional_swing_universe(path, include_inactive=include_inactive)


def validate_professional_swing_universe(
    path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
    *,
    include_inactive: bool = False,
) -> ProfessionalSwingUniverseResult:
    source_path = Path(path)
    rows = _read_csv_rows(source_path)
    entries = tuple(_entry_from_row(source_path=source_path, row=row) for row in rows)
    _validate_duplicate_keys(entries)

    selected_entries = entries if include_inactive else tuple(
        entry
        for entry in entries
        if entry.active
        and entry.universe_status in DEFAULT_PROFESSIONAL_SWING_UNIVERSE_STATUSES
        and entry.source_policy_hint in DEFAULT_PROFESSIONAL_SWING_SOURCE_POLICY_HINTS
    )
    ordered_entries = tuple(sorted(selected_entries, key=_entry_order_key))

    return ProfessionalSwingUniverseResult(
        contract_version=EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
        source_path=source_path.as_posix(),
        entries=ordered_entries,
        loaded_row_count=len(entries),
        selected_row_count=len(ordered_entries),
        excluded_inactive_count=sum(1 for entry in entries if not entry.active),
        excluded_universe_status_count=sum(
            1
            for entry in entries
            if entry.universe_status not in DEFAULT_PROFESSIONAL_SWING_UNIVERSE_STATUSES
        ),
        excluded_source_policy_hint_count=sum(
            1
            for entry in entries
            if entry.source_policy_hint not in DEFAULT_PROFESSIONAL_SWING_SOURCE_POLICY_HINTS
        ),
        include_inactive=include_inactive,
    )


def _read_csv_rows(source_path: Path) -> tuple[dict[str, Any], ...]:
    if not source_path.exists():
        raise ProfessionalSwingUniverseValidationError(
            f"Professional Swing Universe file not found: {source_path}"
        )
    try:
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                raw_header = next(reader)
            except StopIteration as exc:
                raise ProfessionalSwingUniverseValidationError(
                    "Professional Swing Universe CSV is missing a header row."
                ) from exc
            header = _validate_header(raw_header)
            rows = []
            for row_number, raw_row in enumerate(reader, start=2):
                if _blank_row(raw_row):
                    continue
                values = [cell.strip() for cell in raw_row]
                if len(values) > len(header):
                    raise ProfessionalSwingUniverseValidationError(
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
        raise ProfessionalSwingUniverseValidationError(
            f"Unable to read Professional Swing Universe CSV: {source_path}"
        ) from exc
    except csv.Error as exc:
        raise ProfessionalSwingUniverseValidationError(
            f"Professional Swing Universe CSV is malformed: {source_path}: {exc}"
        ) from exc


def _validate_header(raw_header: Iterable[str]) -> tuple[str, ...]:
    header = tuple(cell.strip() for cell in raw_header)
    if not header or all(not cell for cell in header):
        raise ProfessionalSwingUniverseValidationError(
            "Professional Swing Universe CSV is missing a header row."
        )
    if any(not cell for cell in header):
        raise ProfessionalSwingUniverseValidationError(
            "Professional Swing Universe CSV contains an unnamed column."
        )
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise ProfessionalSwingUniverseValidationError(
            "Professional Swing Universe CSV contains duplicate columns: "
            + ", ".join(duplicates)
        )
    missing = [
        column
        for column in REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS
        if column not in header
    ]
    if missing:
        raise ProfessionalSwingUniverseValidationError(
            "Professional Swing Universe CSV missing required columns: "
            + ", ".join(missing)
        )
    return header


def _entry_from_row(
    *,
    source_path: Path,
    row: dict[str, Any],
) -> ProfessionalSwingUniverseEntry:
    row_number = int(row["row_number"])
    values = row["values"]
    ticker = _required_text(values, "ticker", row_number=row_number)
    normalized_ticker = ticker.upper()
    if not _TICKER_RE.fullmatch(normalized_ticker):
        raise ProfessionalSwingUniverseValidationError(
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
        allowed=ALLOWED_PROFESSIONAL_SWING_MARKETS,
        row_number=row_number,
        transform="upper",
    )
    asset_type = _allowed_text(
        values,
        "asset_type",
        allowed=ALLOWED_PROFESSIONAL_SWING_ASSET_TYPES,
        row_number=row_number,
        transform="lower",
    )
    active = _boolean(values, "active", row_number=row_number)
    universe_status = _allowed_text(
        values,
        "universe_status",
        allowed=ALLOWED_PROFESSIONAL_SWING_UNIVERSE_STATUSES,
        row_number=row_number,
        transform="lower",
    )
    source_policy_hint = _allowed_text(
        values,
        "source_policy_hint",
        allowed=ALLOWED_PROFESSIONAL_SWING_SOURCE_POLICY_HINTS,
        row_number=row_number,
        transform="lower",
    )
    operator_priority = _operator_priority(values, row_number=row_number)
    swing_profile = _allowed_text(
        values,
        "swing_profile",
        allowed=ALLOWED_PROFESSIONAL_SWING_PROFILES,
        row_number=row_number,
        transform="lower",
    )
    liquidity_profile = _allowed_text(
        values,
        "liquidity_profile",
        allowed=ALLOWED_PROFESSIONAL_SWING_LIQUIDITY_PROFILES,
        row_number=row_number,
        transform="lower",
    )
    volatility_profile = _allowed_text(
        values,
        "volatility_profile",
        allowed=ALLOWED_PROFESSIONAL_SWING_VOLATILITY_PROFILES,
        row_number=row_number,
        transform="lower",
    )
    market_cap_profile = _allowed_text(
        values,
        "market_cap_profile",
        allowed=ALLOWED_PROFESSIONAL_SWING_MARKET_CAP_PROFILES,
        row_number=row_number,
        transform="lower",
    )
    theme = _required_text(values, "theme", row_number=row_number)
    sector = _required_text(values, "sector", row_number=row_number)
    notes = str(values.get(_NOTES_FIELD, "")).strip()
    metadata = {
        key: str(value).strip()
        for key, value in values.items()
        if key not in REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS
    }
    return ProfessionalSwingUniverseEntry(
        contract_version=EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
        source_path=source_path.as_posix(),
        row_number=row_number,
        ticker=normalized_ticker,
        normalized_ticker=normalized_ticker,
        name=name,
        market=market,
        asset_type=asset_type,
        active=active,
        universe_status=universe_status,
        source_policy_hint=source_policy_hint,
        operator_priority=operator_priority,
        swing_profile=swing_profile,
        liquidity_profile=liquidity_profile,
        volatility_profile=volatility_profile,
        market_cap_profile=market_cap_profile,
        theme=theme,
        sector=sector,
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
    if not value and field_name != _NOTES_FIELD:
        raise ProfessionalSwingUniverseValidationError(
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
        raise ProfessionalSwingUniverseValidationError(
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
        raise ProfessionalSwingUniverseValidationError(
            _error(row_number, field_name, value, "boolean value must be true or false")
        )
    return value == "true"


def _operator_priority(values: dict[str, str], *, row_number: int) -> int:
    raw_value = _required_text(values, "operator_priority", row_number=row_number)
    try:
        operator_priority = int(raw_value)
    except ValueError as exc:
        raise ProfessionalSwingUniverseValidationError(
            _error(
                row_number,
                "operator_priority",
                raw_value,
                "operator_priority must be an integer",
            )
        ) from exc
    if operator_priority < 1:
        raise ProfessionalSwingUniverseValidationError(
            _error(
                row_number,
                "operator_priority",
                raw_value,
                "operator_priority must be greater than or equal to 1",
            )
        )
    return operator_priority


def _validate_duplicate_keys(entries: tuple[ProfessionalSwingUniverseEntry, ...]) -> None:
    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (entry.normalized_ticker, entry.market)
        if key in seen:
            raise ProfessionalSwingUniverseValidationError(
                _error(
                    entry.row_number,
                    "ticker",
                    entry.normalized_ticker,
                    f"duplicate ticker/market row also appears at row {seen[key]}",
                )
            )
        seen[key] = entry.row_number


def _entry_order_key(entry: ProfessionalSwingUniverseEntry) -> tuple[int, str, str]:
    return (entry.operator_priority, entry.ticker, entry.market)


def _blank_row(row: Iterable[str]) -> bool:
    return all(not str(cell).strip() for cell in row)


def _error(row_number: int, field_name: str, value: Any, reason: str) -> str:
    field = field_name or "<row>"
    return f"row {row_number}, field {field}, value {value!r}: {reason}"
