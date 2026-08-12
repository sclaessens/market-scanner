from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.data.local_market_data_universe import build_universe_snapshot
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    MarketEnginePortfolioContext,
    MarketEnginePortfolioPositionState,
)


LEDGER_SCHEMA_VERSION = "manual-portfolio-transaction-ledger-v1"
LEDGER_HEADER_SCHEMA_VERSION = "manual-portfolio-transaction-ledger-header-v1"
PREVIEW_SCHEMA_VERSION = "manual-portfolio-transaction-preview-v1"
PROJECTION_SCHEMA_VERSION = "market-engine-derived-positions-v1"
CANDIDATE_CONTEXT_SCHEMA_VERSION = "market-engine-portfolio-aware-candidate-context-v1"
SOURCE_TYPE = "manual_user_input"
PRIVATE_REPOSITORY_LEDGER_ROOT = Path("data/portfolio/private")
SUPPORTED_CURRENCIES = frozenset(
    {"AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "HKD", "JPY", "NOK", "SEK", "USD"}
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
CURRENCY = re.compile(r"^[A-Z]{3}$")
EVENT_FIELDS = frozenset(
    {
        "schema_version", "event_type", "transaction_id", "portfolio_id",
        "account_id", "instrument_id", "canonical_ticker", "transaction_type",
        "trade_date", "execution_timestamp", "quantity", "unit_price",
        "trade_currency", "fee", "broker_account_label", "source_type",
        "recorded_at", "note", "external_reference", "corrects_transaction_id",
        "reverses_transaction_id", "reason",
    }
)


class LedgerIssueCode(StrEnum):
    MISSING_TRANSACTION_ID = "MISSING_TRANSACTION_ID"
    DUPLICATE_TRANSACTION_ID = "DUPLICATE_TRANSACTION_ID"
    UNKNOWN_INSTRUMENT = "UNKNOWN_INSTRUMENT"
    AMBIGUOUS_TICKER = "AMBIGUOUS_TICKER"
    INSTRUMENT_IDENTITY_MISMATCH = "INSTRUMENT_IDENTITY_MISMATCH"
    UNAPPROVED_TICKER_ALIAS = "UNAPPROVED_TICKER_ALIAS"
    UNSUPPORTED_TRANSACTION_TYPE = "UNSUPPORTED_TRANSACTION_TYPE"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    UNSUPPORTED_CURRENCY = "UNSUPPORTED_CURRENCY"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    FEE_CURRENCY_AMBIGUITY = "FEE_CURRENCY_AMBIGUITY"
    INVALID_TRADE_DATE = "INVALID_TRADE_DATE"
    FUTURE_TRADE_DATE = "FUTURE_TRADE_DATE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    OVERSELL = "OVERSELL"
    DUPLICATE_REPLAY = "DUPLICATE_REPLAY"
    UNKNOWN_CORRECTION_TARGET = "UNKNOWN_CORRECTION_TARGET"
    TARGET_ALREADY_REVERSED = "TARGET_ALREADY_REVERSED"
    NONDETERMINISTIC_EVENT_ORDER = "NONDETERMINISTIC_EVENT_ORDER"
    UNSUPPORTED_SHORT_POSITION = "UNSUPPORTED_SHORT_POSITION"
    MALFORMED_DECIMAL = "MALFORMED_DECIMAL"
    MISSING_REQUIRED_VALUE = "MISSING_REQUIRED_VALUE"
    LEDGER_CORRUPT = "LEDGER_CORRUPT"
    LEDGER_INCOMPATIBLE = "LEDGER_INCOMPATIBLE"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    CONFIRMED_PREVIEW_MISMATCH = "CONFIRMED_PREVIEW_MISMATCH"
    PRIVATE_STORAGE_REQUIRED = "PRIVATE_STORAGE_REQUIRED"
    TRACKED_PRIVATE_DATA_PATH = "TRACKED_PRIVATE_DATA_PATH"
    LEDGER_PORTFOLIO_MISMATCH = "LEDGER_PORTFOLIO_MISMATCH"


class LedgerValidationError(ValueError):
    def __init__(self, code: LedgerIssueCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code


@dataclass(frozen=True)
class InstrumentIdentity:
    instrument_id: str
    canonical_ticker: str
    currency: str
    exchange: str
    approved_aliases: tuple[str, ...] = ()


class AuthoritativeInstrumentRegistry:
    def __init__(self, instruments: Sequence[Mapping[str, Any]]) -> None:
        by_id: dict[str, InstrumentIdentity] = {}
        by_symbol: dict[str, list[InstrumentIdentity]] = {}
        unapproved_aliases: set[str] = set()
        for raw in instruments:
            instrument_id = str(raw.get("instrument_id") or "").strip()
            ticker = str(raw.get("symbol") or raw.get("canonical_ticker") or "").upper().strip()
            currency = str(raw.get("currency") or "").upper().strip()
            exchange = str(raw.get("exchange") or raw.get("market") or "UNKNOWN").upper().strip()
            if not instrument_id or not ticker or not currency:
                continue
            aliases: list[str] = []
            source_symbol = str(raw.get("source_symbol") or "").upper().strip()
            mapping_status = str(raw.get("source_mapping_status") or "mapped")
            if source_symbol and source_symbol != ticker and mapping_status == "mapped":
                aliases.append(source_symbol)
            elif source_symbol and source_symbol != ticker:
                unapproved_aliases.add(source_symbol)
            identity = InstrumentIdentity(
                instrument_id=instrument_id,
                canonical_ticker=ticker,
                currency=currency,
                exchange=exchange,
                approved_aliases=tuple(sorted(set(aliases))),
            )
            if instrument_id in by_id:
                raise LedgerValidationError(
                    LedgerIssueCode.AMBIGUOUS_TICKER,
                    f"duplicate authoritative instrument ID: {instrument_id}",
                )
            by_id[instrument_id] = identity
            for symbol in (ticker, *identity.approved_aliases):
                by_symbol.setdefault(symbol, []).append(identity)
        self._by_id = by_id
        self._by_symbol = by_symbol
        self._unapproved_aliases = unapproved_aliases

    @classmethod
    def from_canonical_universe(
        cls,
        config_path: str | Path = "config/market_engine/universes/canonical_universe.json",
        *,
        price_history_root: str | Path = "data/processed",
    ) -> AuthoritativeInstrumentRegistry:
        snapshot = build_universe_snapshot(
            config_path=config_path,
            price_history_root=price_history_root,
        )
        return cls(snapshot["instruments"])

    def resolve(
        self,
        *,
        instrument_id: object = None,
        ticker: object = None,
    ) -> InstrumentIdentity:
        requested_id = str(instrument_id or "").strip()
        requested_ticker = str(ticker or "").upper().strip()
        by_id = self._by_id.get(requested_id) if requested_id else None
        candidates = self._by_symbol.get(requested_ticker, []) if requested_ticker else []
        if requested_ticker and len(candidates) > 1:
            raise LedgerValidationError(
                LedgerIssueCode.AMBIGUOUS_TICKER,
                f"ticker resolves to multiple authoritative instruments: {requested_ticker}",
            )
        by_ticker = candidates[0] if candidates else None
        if requested_id and by_id is None:
            raise LedgerValidationError(
                LedgerIssueCode.UNKNOWN_INSTRUMENT,
                f"unknown authoritative instrument ID: {requested_id}",
            )
        if requested_ticker and by_ticker is None:
            if requested_ticker in self._unapproved_aliases:
                raise LedgerValidationError(
                    LedgerIssueCode.UNAPPROVED_TICKER_ALIAS,
                    f"ticker alias is not approved for authoritative identity: {requested_ticker}",
                )
            canonical_match = [
                value
                for value in self._by_id.values()
                if requested_ticker in value.approved_aliases
            ]
            code = (
                LedgerIssueCode.AMBIGUOUS_TICKER
                if len(canonical_match) > 1
                else LedgerIssueCode.UNKNOWN_INSTRUMENT
            )
            raise LedgerValidationError(code, f"ticker is not an approved unique identity: {requested_ticker}")
        if by_id and by_ticker and by_id != by_ticker:
            raise LedgerValidationError(
                LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
                "instrument ID and ticker resolve to different authoritative records",
            )
        identity = by_id or by_ticker
        if identity is None:
            raise LedgerValidationError(
                LedgerIssueCode.MISSING_REQUIRED_VALUE,
                "instrument_id or ticker is required",
            )
        return identity


def normalize_transaction_preview(
    raw: Mapping[str, Any],
    *,
    registry: AuthoritativeInstrumentRegistry,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    now = _utc_timestamp(recorded_at or datetime.now(UTC).isoformat())
    event_type = str(raw.get("event_type") or "transaction").lower().strip()
    if event_type not in {"transaction", "correction", "reversal"}:
        raise LedgerValidationError(
            LedgerIssueCode.UNSUPPORTED_TRANSACTION_TYPE,
            f"unsupported ledger event type: {event_type}",
        )
    transaction_id = str(raw.get("transaction_id") or "").strip()
    if not transaction_id:
        raise LedgerValidationError(
            LedgerIssueCode.MISSING_TRANSACTION_ID,
            "transaction_id is required",
        )
    if not IDENTIFIER.fullmatch(transaction_id):
        raise LedgerValidationError(
            LedgerIssueCode.MISSING_TRANSACTION_ID,
            "transaction_id must be a stable safe identifier",
        )
    identity = registry.resolve(
        instrument_id=raw.get("instrument_id"),
        ticker=raw.get("ticker") or raw.get("canonical_ticker"),
    )
    portfolio_id = _required_identifier(raw, "portfolio_id")
    account_id = _required_identifier(raw, "account_id")
    target = None
    if event_type == "correction":
        target = _required_identifier(raw, "corrects_transaction_id")
    elif event_type == "reversal":
        target = _required_identifier(raw, "reverses_transaction_id")
    reason = _optional_text(raw.get("reason"))
    if event_type in {"correction", "reversal"} and not reason:
        raise LedgerValidationError(
            LedgerIssueCode.MISSING_REQUIRED_VALUE,
            "correction and reversal events require a reason",
        )

    if event_type == "reversal":
        transaction_type = None
        trade_date = None
        execution_timestamp = None
        quantity = None
        unit_price = None
        trade_currency = None
        fee = {"availability": "unavailable", "amount": None, "currency": None}
    else:
        transaction_type = str(raw.get("transaction_type") or "").upper().strip()
        if transaction_type not in {"BUY", "SELL"}:
            raise LedgerValidationError(
                LedgerIssueCode.UNSUPPORTED_TRANSACTION_TYPE,
                "transaction_type must be BUY or SELL as reported by the user",
            )
        trade_date = _trade_date(raw.get("trade_date"), now=now)
        execution_timestamp = _optional_execution_timestamp(
            raw.get("execution_timestamp"), trade_date=trade_date
        )
        quantity = _positive_decimal(raw.get("quantity"), field="quantity")
        unit_price = _nonnegative_decimal(raw.get("unit_price"), field="unit_price")
        trade_currency = str(raw.get("trade_currency") or "").upper().strip()
        _validate_currency(trade_currency)
        if trade_currency != identity.currency:
            raise LedgerValidationError(
                LedgerIssueCode.CURRENCY_MISMATCH,
                "trade currency does not match the authoritative instrument currency",
            )
        fee = _normalize_fee(raw.get("fee"), trade_currency=trade_currency)

    event = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "event_type": event_type,
        "transaction_id": transaction_id,
        "portfolio_id": portfolio_id,
        "account_id": account_id,
        "instrument_id": identity.instrument_id,
        "canonical_ticker": identity.canonical_ticker,
        "transaction_type": transaction_type,
        "trade_date": trade_date,
        "execution_timestamp": execution_timestamp,
        "quantity": quantity,
        "unit_price": unit_price,
        "trade_currency": trade_currency,
        "fee": fee,
        "broker_account_label": _optional_text(raw.get("broker_account_label")),
        "source_type": SOURCE_TYPE,
        "recorded_at": _utc_text(now),
        "note": _optional_text(raw.get("note")),
        "external_reference": _optional_text(raw.get("external_reference")),
        "corrects_transaction_id": target if event_type == "correction" else None,
        "reverses_transaction_id": target if event_type == "reversal" else None,
        "reason": reason,
    }
    digest = _sha256(_canonical_json(event))
    return {
        "preview_schema_version": PREVIEW_SCHEMA_VERSION,
        "event": event,
        "preview_digest": digest,
        "confirmation_token": digest,
        "confirmation_required": True,
    }


def confirm_and_append(
    preview: Mapping[str, Any],
    *,
    confirmation_token: str | None,
    ledger_path: str | Path,
    registry: AuthoritativeInstrumentRegistry,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    if not confirmation_token:
        raise LedgerValidationError(
            LedgerIssueCode.CONFIRMATION_REQUIRED,
            "the exact normalized preview must be explicitly confirmed",
        )
    if preview.get("preview_schema_version") != PREVIEW_SCHEMA_VERSION:
        raise LedgerValidationError(
            LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
            "preview contract is unsupported",
        )
    event = preview.get("event")
    if not isinstance(event, Mapping):
        raise LedgerValidationError(
            LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
            "preview event is missing",
        )
    digest = _sha256(_canonical_json(dict(event)))
    if (
        preview.get("preview_digest") != digest
        or preview.get("confirmation_token") != digest
        or confirmation_token != digest
    ):
        raise LedgerValidationError(
            LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
            "confirmation does not bind the exact normalized preview",
        )
    _validate_normalized_event(dict(event), registry=registry)
    path = validate_private_ledger_path(ledger_path, repository_root=repository_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _append_existing(path, dict(event))
    else:
        _create_ledger(path, dict(event))
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "append_status": "confirmed_event_appended",
        "transaction_id": event["transaction_id"],
        "event_digest": digest,
        "ledger_path": _redacted_path(path),
    }


def load_ledger(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_CORRUPT,
            "private ledger cannot be read completely",
        ) from exc
    if not lines:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "private ledger is empty")
    rows: list[Any] = []
    for index, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LedgerValidationError(
                LedgerIssueCode.LEDGER_CORRUPT,
                f"private ledger line {index} is malformed",
            ) from exc
    header = rows[0]
    if not isinstance(header, Mapping) or header.get("schema_version") != LEDGER_HEADER_SCHEMA_VERSION:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_INCOMPATIBLE,
            "private ledger header contract is unsupported",
        )
    events = rows[1:]
    if any(not isinstance(row, Mapping) for row in events):
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "ledger event is not an object")
    for row in events:
        _validate_loaded_event_shape(row)
    ids = [str(row.get("transaction_id") or "") for row in events]
    if not ids or any(not value for value in ids):
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "ledger contains an invalid event")
    if len(ids) != len(set(ids)):
        raise LedgerValidationError(
            LedgerIssueCode.DUPLICATE_REPLAY,
            "ledger contains a duplicate transaction ID",
        )
    if any(row.get("schema_version") != LEDGER_SCHEMA_VERSION for row in events):
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_INCOMPATIBLE,
            "ledger contains an incompatible event contract",
        )
    if any(row.get("portfolio_id") != header.get("portfolio_id") for row in events):
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_PORTFOLIO_MISMATCH,
            "ledger contains an event for another portfolio",
        )
    return {"header": dict(header), "events": [dict(row) for row in events]}


def rebuild_positions(
    ledger: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    loaded = load_ledger(ledger) if isinstance(ledger, (str, Path)) else ledger
    header = dict(loaded.get("header") or {})
    events = [dict(row) for row in loaded.get("events") or []]
    if header.get("schema_version") != LEDGER_HEADER_SCHEMA_VERSION:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_INCOMPATIBLE, "ledger header is invalid")
    active = _active_economic_events(events)
    active.sort(key=_economic_order)
    _validate_deterministic_order(active)
    states: dict[tuple[str, str, str], dict[str, Any]] = {}
    with localcontext() as context:
        context.prec = 50
        for event in active:
            key = (event["portfolio_id"], event["account_id"], event["instrument_id"])
            state = states.setdefault(key, _new_position_state(event))
            _apply_economic_event(state, event)
    all_events_by_position: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (event["portfolio_id"], event["account_id"], event["instrument_id"])
        all_events_by_position.setdefault(key, []).append(event)
    for key, rows in all_events_by_position.items():
        if key not in states:
            original = next((row for row in rows if row["event_type"] != "reversal"), None)
            if original is None:
                raise LedgerValidationError(
                    LedgerIssueCode.LEDGER_CORRUPT,
                    "reversal history has no referenced economic event",
                )
            states[key] = _new_position_state(original)
        state = states[key]
        economic_dates = [str(row["trade_date"]) for row in rows if row.get("trade_date")]
        state["first_trade_date"] = min(economic_dates)
        state["last_trade_date"] = max(economic_dates)
        ordered_rows = rows
        state["last_transaction_id"] = ordered_rows[-1]["transaction_id"]
        state["transaction_references"] = [row["transaction_id"] for row in ordered_rows]
    event_digest = _sha256(_canonical_json({"header": header, "events": events}))
    projection_timestamp = max((str(row["recorded_at"]) for row in events), default=None)
    positions = [_position_payload(value, event_digest, projection_timestamp) for value in states.values()]
    positions.sort(key=lambda row: (row["portfolio_id"], row["account_id"], row["instrument_id"]))
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "portfolio_id": header.get("portfolio_id"),
        "projection_timestamp": projection_timestamp,
        "ledger_digest": event_digest,
        "ledger_event_count": len(events),
        "active_transaction_count": len(active),
        "cost_basis_method": "moving_weighted_average_v1",
        "positions": positions,
        "portfolio_accounts": sorted({str(row["account_id"]) for row in events}),
        "calculation_status": "partial" if any(row["calculation_blockers"] for row in positions) else "complete",
    }


def build_transaction_derived_portfolio_context(
    projection: Mapping[str, Any],
    *,
    portfolio_id: str,
    account_id: str,
    instrument: InstrumentIdentity,
    context_run_id: str,
) -> MarketEnginePortfolioContext:
    if projection.get("schema_version") != PROJECTION_SCHEMA_VERSION:
        return _unknown_context(
            instrument=instrument,
            context_run_id=context_run_id,
            blockers=("PROJECTION_CONTRACT_INVALID",),
        )
    if projection.get("portfolio_id") != portfolio_id or account_id not in projection.get("portfolio_accounts", []):
        return _unknown_context(
            instrument=instrument,
            context_run_id=context_run_id,
            blockers=("PORTFOLIO_ACCOUNT_CONTEXT_UNKNOWN",),
        )
    match = next(
        (
            row
            for row in projection.get("positions", [])
            if row.get("portfolio_id") == portfolio_id
            and row.get("account_id") == account_id
            and row.get("instrument_id") == instrument.instrument_id
        ),
        None,
    )
    if match is None:
        position_state = MarketEnginePortfolioPositionState.NOT_HELD
        quantity = "0"
        weighted_average_cost = None
        realized_pnl = None
        transaction_count = 0
        references: tuple[str, ...] = ()
        blockers: tuple[str, ...] = ()
        calculation_status = "complete"
        transaction_currency = None
    else:
        position_state = (
            MarketEnginePortfolioPositionState.HELD
            if match["position_status"] == "open" and not match["calculation_blockers"]
            else MarketEnginePortfolioPositionState.PARTIALLY_KNOWN
            if match["calculation_blockers"]
            else MarketEnginePortfolioPositionState.CLOSED
        )
        quantity = match["quantity"]
        weighted_average_cost = match["weighted_average_cost"]
        realized_pnl = match["realized_profit_loss"]
        transaction_count = match["transaction_count"]
        references = tuple(match["transaction_references"])
        blockers = tuple(match["calculation_blockers"])
        calculation_status = match["calculation_status"]
        transaction_currency = match["transaction_currency"]
    missing = (
        "portfolio_base_currency",
        "current_market_value",
        "portfolio_total_value",
        "current_ticker_exposure_pct",
        "current_market_price",
        "unrealized_profit_loss",
    )
    return MarketEnginePortfolioContext(
        portfolio_context_format_version=MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
        portfolio_context_run_id=context_run_id,
        portfolio_snapshot_timestamp=str(projection.get("projection_timestamp") or ""),
        portfolio_base_currency="UNAVAILABLE",
        ticker=instrument.canonical_ticker,
        position_state=position_state,
        current_quantity=quantity,
        current_market_value=None,
        portfolio_total_value=None,
        current_ticker_exposure_pct=None,
        missing_portfolio_context_fields=missing,
        context_provenance={
            "source_contract": LEDGER_SCHEMA_VERSION,
            "projection_contract": PROJECTION_SCHEMA_VERSION,
            "portfolio_id": portfolio_id,
            "account_id": account_id,
            "instrument_id": instrument.instrument_id,
            "ledger_digest": projection.get("ledger_digest"),
            "transaction_references": references,
            "calculation_status": calculation_status,
            "calculation_blockers": blockers,
            "weighted_average_cost": weighted_average_cost,
            "transaction_currency": transaction_currency,
            "realized_profit_loss": realized_pnl,
            "transaction_count": transaction_count,
            "last_transaction_date": match["last_transaction_date"] if match else None,
            "market_price_status": "unavailable",
        },
    )


def build_non_actionable_candidate_context(
    *,
    instrument: InstrumentIdentity,
    portfolio_context: MarketEnginePortfolioContext,
) -> dict[str, Any]:
    if (
        portfolio_context.ticker != instrument.canonical_ticker
        or portfolio_context.context_provenance.get("instrument_id")
        != instrument.instrument_id
    ):
        raise LedgerValidationError(
            LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
            "portfolio context identity does not match the candidate instrument",
        )
    return {
        "schema_version": CANDIDATE_CONTEXT_SCHEMA_VERSION,
        "instrument_id": instrument.instrument_id,
        "canonical_ticker": instrument.canonical_ticker,
        "portfolio_position_state": (
            portfolio_context.position_state.value
            if isinstance(portfolio_context.position_state, MarketEnginePortfolioPositionState)
            else portfolio_context.position_state
        ),
        "current_quantity": portfolio_context.current_quantity,
        "weighted_average_cost": portfolio_context.context_provenance.get("weighted_average_cost"),
        "native_transaction_currency": portfolio_context.context_provenance.get("transaction_currency"),
        "realized_profit_loss": portfolio_context.context_provenance.get("realized_profit_loss"),
        "last_transaction_date": portfolio_context.context_provenance.get("last_transaction_date"),
        "transaction_count": portfolio_context.context_provenance.get("transaction_count"),
        "missing_context_fields": list(portfolio_context.missing_portfolio_context_fields),
        "provenance": dict(portfolio_context.context_provenance),
        "non_actionable_boundary": (
            "This context describes proven transaction-derived position state only; "
            "it creates no recommendation, allocation, sizing, or execution authority."
        ),
    }


def validate_private_ledger_path(
    ledger_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    path = Path(ledger_path).expanduser().resolve()
    repo = Path(repository_root).resolve() if repository_root is not None else _discover_repository_root()
    if repo is None:
        return path
    try:
        relative = path.relative_to(repo)
    except ValueError:
        return path
    allowed = PRIVATE_REPOSITORY_LEDGER_ROOT
    if relative != allowed and allowed not in relative.parents:
        raise LedgerValidationError(
            LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
            "repository-local live ledgers are allowed only below data/portfolio/private",
        )
    ignore_path = repo / ".gitignore"
    try:
        ignored = "/data/portfolio/private/" in ignore_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        ignored = False
    if not ignored:
        raise LedgerValidationError(
            LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
            "repository-local private ledger root is not protected by .gitignore",
        )
    if _git_tracks(repo, relative):
        raise LedgerValidationError(
            LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
            "refusing to write a private ledger to a Git-tracked path",
        )
    return path


def _create_ledger(path: Path, event: dict[str, Any]) -> None:
    header = {
        "schema_version": LEDGER_HEADER_SCHEMA_VERSION,
        "record_type": "ledger_header",
        "portfolio_id": event["portfolio_id"],
        "source_type": SOURCE_TYPE,
    }
    _validate_candidate_events([], event, portfolio_id=event["portfolio_id"])
    encoded = _canonical_json(header) + b"\n" + _canonical_json(event) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        _append_existing(path, event)
        return
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_existing(path: Path, event: dict[str, Any]) -> None:
    try:
        with path.open("r+", encoding="utf-8", newline="") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            text = handle.read()
            loaded = _load_ledger_text(text)
            _validate_candidate_events(
                loaded["events"],
                event,
                portfolio_id=str(loaded["header"]["portfolio_id"]),
            )
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(event).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
    except LedgerValidationError:
        raise
    except OSError as exc:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "ledger append failed") from exc


def _load_ledger_text(text: str) -> dict[str, Any]:
    temporary = text.splitlines()
    if not temporary:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "private ledger is empty")
    try:
        rows = [json.loads(line) for line in temporary]
    except (ValueError, json.JSONDecodeError) as exc:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "private ledger is malformed") from exc
    header, events = rows[0], rows[1:]
    if not isinstance(header, Mapping) or header.get("schema_version") != LEDGER_HEADER_SCHEMA_VERSION:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_INCOMPATIBLE, "ledger header is unsupported")
    if any(not isinstance(row, Mapping) for row in events):
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "ledger event is malformed")
    for row in events:
        _validate_loaded_event_shape(row)
    return {"header": dict(header), "events": [dict(row) for row in events]}


def _validate_candidate_events(events: list[dict[str, Any]], event: dict[str, Any], *, portfolio_id: str) -> None:
    if event["portfolio_id"] != portfolio_id:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_PORTFOLIO_MISMATCH,
            "confirmed event belongs to another portfolio",
        )
    ids = {str(row.get("transaction_id")) for row in events}
    if event["transaction_id"] in ids:
        raise LedgerValidationError(
            LedgerIssueCode.DUPLICATE_TRANSACTION_ID,
            "transaction ID already exists in the append-only ledger",
        )
    if event["event_type"] in {"correction", "reversal"}:
        target_key = "corrects_transaction_id" if event["event_type"] == "correction" else "reverses_transaction_id"
        target_id = event[target_key]
        target = next((row for row in events if row.get("transaction_id") == target_id), None)
        if target is None or target.get("event_type") == "reversal":
            raise LedgerValidationError(
                LedgerIssueCode.UNKNOWN_CORRECTION_TARGET,
                "correction or reversal target is unknown or not economic",
            )
        voided = _voided_event_ids(events)
        if target_id in voided:
            raise LedgerValidationError(
                LedgerIssueCode.TARGET_ALREADY_REVERSED,
                "correction or reversal target is already fully replaced or reversed",
            )
        identity_fields = ("portfolio_id", "account_id", "instrument_id")
        if any(event[field] != target.get(field) for field in identity_fields):
            raise LedgerValidationError(
                LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
                "correction or reversal identity differs from its target",
            )
    candidate = {
        "header": {
            "schema_version": LEDGER_HEADER_SCHEMA_VERSION,
            "portfolio_id": portfolio_id,
        },
        "events": [*events, event],
    }
    rebuild_positions(candidate)


def _active_economic_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids: set[str] = set()
    voided: set[str] = set()
    economic: list[dict[str, Any]] = []
    previous_recorded_at: str | None = None
    for event in events:
        recorded_at = str(event.get("recorded_at") or "")
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise LedgerValidationError(
                LedgerIssueCode.NONDETERMINISTIC_EVENT_ORDER,
                "ledger recorded timestamps are not append ordered",
            )
        previous_recorded_at = recorded_at
        transaction_id = str(event.get("transaction_id") or "")
        if not transaction_id or transaction_id in ids:
            raise LedgerValidationError(LedgerIssueCode.DUPLICATE_REPLAY, "duplicate ledger replay")
        ids.add(transaction_id)
        event_type = event.get("event_type")
        if event_type == "transaction":
            economic.append(event)
        elif event_type == "correction":
            target = str(event.get("corrects_transaction_id") or "")
            if target not in ids or target in voided:
                raise LedgerValidationError(LedgerIssueCode.UNKNOWN_CORRECTION_TARGET, "invalid correction target")
            voided.add(target)
            economic.append(event)
        elif event_type == "reversal":
            target = str(event.get("reverses_transaction_id") or "")
            if target not in ids or target in voided:
                raise LedgerValidationError(LedgerIssueCode.TARGET_ALREADY_REVERSED, "invalid reversal target")
            voided.add(target)
        else:
            raise LedgerValidationError(LedgerIssueCode.LEDGER_INCOMPATIBLE, "unsupported ledger event type")
    return [row for row in economic if row["transaction_id"] not in voided]


def _voided_event_ids(events: list[dict[str, Any]]) -> set[str]:
    voided: set[str] = set()
    for event in events:
        target = event.get("corrects_transaction_id") or event.get("reverses_transaction_id")
        if target:
            voided.add(str(target))
    return voided


def _economic_order(event: Mapping[str, Any]) -> tuple[str, str, str, str]:
    execution = str(event.get("execution_timestamp") or "")
    return (
        str(event.get("trade_date") or ""),
        execution,
        str(event.get("recorded_at") or ""),
        str(event.get("transaction_id") or ""),
    )


def _validate_deterministic_order(events: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            event["portfolio_id"],
            event["account_id"],
            event["instrument_id"],
            event["trade_date"],
        )
        groups.setdefault(key, []).append(event)
    for rows in groups.values():
        if len(rows) > 1 and any(row.get("execution_timestamp") is None for row in rows):
            kinds = {row["transaction_type"] for row in rows}
            if len(kinds) > 1:
                raise LedgerValidationError(
                    LedgerIssueCode.NONDETERMINISTIC_EVENT_ORDER,
                    "same-session purchase and sale require execution timestamps",
                )


def _new_position_state(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "portfolio_id": event["portfolio_id"],
        "account_id": event["account_id"],
        "instrument_id": event["instrument_id"],
        "canonical_ticker": event["canonical_ticker"],
        "currency": event["trade_currency"],
        "quantity": Decimal("0"),
        "remaining_cost_basis": Decimal("0"),
        "realized_profit_loss": Decimal("0"),
        "cumulative_fees": Decimal("0"),
        "cost_basis_known": True,
        "realized_known": True,
        "fees_known": True,
        "first_trade_date": event["trade_date"],
        "last_trade_date": event["trade_date"],
        "last_transaction_id": event["transaction_id"],
        "transaction_references": [],
    }


def _apply_economic_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    quantity = _decimal_from_normalized(event["quantity"])
    price = _decimal_from_normalized(event["unit_price"])
    fee_known = event["fee"]["availability"] == "available"
    fee = _decimal_from_normalized(event["fee"]["amount"]) if fee_known else None
    if event["transaction_type"] == "BUY":
        state["quantity"] += quantity
        if state["cost_basis_known"] and fee_known:
            state["remaining_cost_basis"] += quantity * price + fee
        else:
            state["cost_basis_known"] = False
        if not fee_known:
            state["fees_known"] = False
    else:
        if quantity > state["quantity"]:
            raise LedgerValidationError(LedgerIssueCode.OVERSELL, "sale exceeds available position")
        if state["quantity"] <= 0:
            raise LedgerValidationError(LedgerIssueCode.UNSUPPORTED_SHORT_POSITION, "short positions are unsupported")
        if state["cost_basis_known"]:
            average = state["remaining_cost_basis"] / state["quantity"]
            released_basis = average * quantity
            state["remaining_cost_basis"] -= released_basis
            if state["realized_known"] and fee_known:
                state["realized_profit_loss"] += quantity * price - fee - released_basis
            else:
                state["realized_known"] = False
        else:
            state["realized_known"] = False
        state["quantity"] -= quantity
        if state["quantity"] == 0 and state["cost_basis_known"]:
            state["remaining_cost_basis"] = Decimal("0")
        if not fee_known:
            state["fees_known"] = False
    if fee_known and state["fees_known"]:
        state["cumulative_fees"] += fee
    state["last_trade_date"] = max(state["last_trade_date"], event["trade_date"])
    state["last_transaction_id"] = event["transaction_id"]
    state["transaction_references"].append(event["transaction_id"])


def _position_payload(state: dict[str, Any], ledger_digest: str, timestamp: str | None) -> dict[str, Any]:
    blockers: list[str] = []
    if not state["fees_known"]:
        blockers.append("FEE_VALUE_UNAVAILABLE")
    if not state["cost_basis_known"]:
        blockers.append("COST_BASIS_UNAVAILABLE")
    if not state["realized_known"]:
        blockers.append("REALIZED_PROFIT_LOSS_UNAVAILABLE")
    weighted = None
    remaining = None
    if state["cost_basis_known"]:
        remaining = _decimal_text(state["remaining_cost_basis"])
        weighted = (
            _decimal_text(state["remaining_cost_basis"] / state["quantity"])
            if state["quantity"] > 0
            else "0"
        )
    return {
        "portfolio_id": state["portfolio_id"],
        "account_id": state["account_id"],
        "instrument_id": state["instrument_id"],
        "canonical_ticker": state["canonical_ticker"],
        "quantity": _decimal_text(state["quantity"]),
        "position_status": "open" if state["quantity"] > 0 else "closed",
        "weighted_average_cost": weighted,
        "cumulative_fees": _decimal_text(state["cumulative_fees"]) if state["fees_known"] else None,
        "remaining_cost_basis": remaining,
        "realized_profit_loss": _decimal_text(state["realized_profit_loss"]) if state["realized_known"] else None,
        "transaction_currency": state["currency"],
        "first_transaction_date": state["first_trade_date"],
        "last_transaction_date": state["last_trade_date"],
        "last_confirmed_transaction_id": state["last_transaction_id"],
        "transaction_count": len(state["transaction_references"]),
        "projection_timestamp": timestamp,
        "ledger_digest": ledger_digest,
        "transaction_references": list(state["transaction_references"]),
        "calculation_status": "partial" if blockers else "complete",
        "calculation_blockers": blockers,
    }


def _unknown_context(*, instrument: InstrumentIdentity, context_run_id: str, blockers: tuple[str, ...]) -> MarketEnginePortfolioContext:
    return MarketEnginePortfolioContext(
        portfolio_context_format_version=MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
        portfolio_context_run_id=context_run_id,
        portfolio_snapshot_timestamp="",
        portfolio_base_currency="UNAVAILABLE",
        ticker=instrument.canonical_ticker,
        position_state=MarketEnginePortfolioPositionState.UNKNOWN,
        current_quantity=None,
        current_market_value=None,
        portfolio_total_value=None,
        current_ticker_exposure_pct=None,
        missing_portfolio_context_fields=("position_state", "current_quantity"),
        context_provenance={
            "source_contract": LEDGER_SCHEMA_VERSION,
            "projection_contract": PROJECTION_SCHEMA_VERSION,
            "instrument_id": instrument.instrument_id,
            "calculation_status": "blocked",
            "calculation_blockers": blockers,
        },
    )


def _validate_normalized_event(event: dict[str, Any], *, registry: AuthoritativeInstrumentRegistry) -> None:
    if set(event) != EVENT_FIELDS or event.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise LedgerValidationError(LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH, "normalized event fields are invalid")
    identity = registry.resolve(instrument_id=event.get("instrument_id"), ticker=event.get("canonical_ticker"))
    if identity.instrument_id != event["instrument_id"] or identity.canonical_ticker != event["canonical_ticker"]:
        raise LedgerValidationError(LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH, "normalized identity no longer matches registry")
    if event.get("source_type") != SOURCE_TYPE:
        raise LedgerValidationError(LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH, "source type is not manual_user_input")
    normalized = normalize_transaction_preview(
        event,
        registry=registry,
        recorded_at=str(event.get("recorded_at") or ""),
    )["event"]
    if normalized != event:
        raise LedgerValidationError(
            LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
            "confirmed event is not the canonical normalized preview",
        )


def _validate_loaded_event_shape(event: Mapping[str, Any]) -> None:
    if set(event) != EVENT_FIELDS:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_CORRUPT,
            "ledger event fields are incomplete or unexpected",
        )
    if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_INCOMPATIBLE,
            "ledger contains an incompatible event contract",
        )
    if event.get("event_type") not in {"transaction", "correction", "reversal"}:
        raise LedgerValidationError(
            LedgerIssueCode.LEDGER_INCOMPATIBLE,
            "ledger contains an unsupported event type",
        )


def _normalize_fee(value: Any, *, trade_currency: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerValidationError(
            LedgerIssueCode.MISSING_REQUIRED_VALUE,
            "fee must explicitly declare available or unavailable",
        )
    availability = str(value.get("availability") or "").lower()
    if availability == "unavailable":
        if value.get("amount") is not None or value.get("currency") is not None:
            raise LedgerValidationError(
                LedgerIssueCode.FEE_CURRENCY_AMBIGUITY,
                "unavailable fees cannot carry an amount or currency",
            )
        return {"availability": "unavailable", "amount": None, "currency": None}
    if availability != "available":
        raise LedgerValidationError(
            LedgerIssueCode.MISSING_REQUIRED_VALUE,
            "fee availability must be explicit",
        )
    amount = _nonnegative_decimal(value.get("amount"), field="fee.amount")
    currency = str(value.get("currency") or "").upper().strip()
    _validate_currency(currency)
    if currency != trade_currency:
        raise LedgerValidationError(
            LedgerIssueCode.FEE_CURRENCY_AMBIGUITY,
            "fee currency must equal trade currency when no FX source exists",
        )
    return {"availability": "available", "amount": amount, "currency": currency}


def _required_identifier(raw: Mapping[str, Any], field: str) -> str:
    value = str(raw.get(field) or "").strip()
    if not value or not IDENTIFIER.fullmatch(value):
        raise LedgerValidationError(LedgerIssueCode.MISSING_REQUIRED_VALUE, f"{field} is required and must be a safe identifier")
    return value


def _positive_decimal(value: Any, *, field: str) -> str:
    parsed = _parse_decimal(value, field=field)
    if parsed <= 0:
        raise LedgerValidationError(LedgerIssueCode.INVALID_QUANTITY, f"{field} must be greater than zero")
    return _decimal_text(parsed)


def _nonnegative_decimal(value: Any, *, field: str) -> str:
    parsed = _parse_decimal(value, field=field)
    if parsed < 0:
        raise LedgerValidationError(LedgerIssueCode.INVALID_PRICE, f"{field} cannot be negative")
    return _decimal_text(parsed)


def _parse_decimal(value: Any, *, field: str) -> Decimal:
    if value is None or value == "":
        raise LedgerValidationError(LedgerIssueCode.MISSING_REQUIRED_VALUE, f"{field} is required")
    if isinstance(value, float):
        raise LedgerValidationError(LedgerIssueCode.MALFORMED_DECIMAL, f"{field} must not use binary floating point")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LedgerValidationError(LedgerIssueCode.MALFORMED_DECIMAL, f"{field} is malformed") from exc
    if not parsed.is_finite():
        raise LedgerValidationError(LedgerIssueCode.MALFORMED_DECIMAL, f"{field} must be finite")
    return parsed


def _decimal_from_normalized(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LedgerValidationError(LedgerIssueCode.LEDGER_CORRUPT, "ledger decimal is malformed") from exc


def _decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        return format(+value, "f")


def _trade_date(value: Any, *, now: datetime) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(LedgerIssueCode.INVALID_TRADE_DATE, "trade_date must be ISO YYYY-MM-DD") from exc
    if parsed > now.date():
        raise LedgerValidationError(LedgerIssueCode.FUTURE_TRADE_DATE, "trade_date cannot be in the future")
    return parsed.isoformat()


def _optional_execution_timestamp(value: Any, *, trade_date: str) -> str | None:
    if value in (None, ""):
        return None
    parsed = _utc_timestamp(str(value))
    if parsed.date().isoformat() != trade_date:
        raise LedgerValidationError(LedgerIssueCode.INVALID_TIMESTAMP, "execution timestamp date must equal trade date")
    return _utc_text(parsed)


def _utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LedgerValidationError(LedgerIssueCode.INVALID_TIMESTAMP, "timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise LedgerValidationError(LedgerIssueCode.INVALID_TIMESTAMP, "timestamp requires a timezone")
    return parsed.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_currency(value: str) -> None:
    if not CURRENCY.fullmatch(value) or value not in SUPPORTED_CURRENCIES:
        raise LedgerValidationError(LedgerIssueCode.UNSUPPORTED_CURRENCY, f"unsupported transaction currency: {value}")


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    return text or None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _discover_repository_root() -> Path | None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_tracks(repo: Path, relative: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return True
    return result.returncode == 0


def _redacted_path(path: Path) -> str:
    return f"<private-ledger>/{path.name}"


def run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        registry = AuthoritativeInstrumentRegistry.from_canonical_universe()
        if args.command == "preview":
            raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
            result = normalize_transaction_preview(raw, registry=registry)
        elif args.command == "confirm":
            preview = json.loads(Path(args.preview).read_text(encoding="utf-8"))
            result = confirm_and_append(
                preview,
                confirmation_token=args.confirmation_token,
                ledger_path=args.ledger,
                registry=registry,
            )
        else:
            result = rebuild_positions(args.ledger)
    except (OSError, ValueError, json.JSONDecodeError, LedgerValidationError) as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    json.dump(result, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-portfolio-ledger",
        description="Preview, confirm, append, and rebuild private manual portfolio transactions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    preview.add_argument("--input", required=True, help="Local JSON transaction input path.")
    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--preview", required=True, help="Exact normalized preview JSON path.")
    confirm.add_argument("--confirmation-token", required=True)
    confirm.add_argument("--ledger", required=True, help="User-controlled private ledger path.")
    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("--ledger", required=True, help="User-controlled private ledger path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv)


if __name__ == "__main__":
    raise SystemExit(main())
