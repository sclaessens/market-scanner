from __future__ import annotations

import json
import hashlib
import re
import subprocess
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import pytest

from market_engine.portfolio_review.manual_transaction_ledger import (
    CANDIDATE_CONTEXT_SCHEMA_VERSION,
    LEDGER_HEADER_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
    AuthoritativeInstrumentRegistry,
    LedgerIssueCode,
    LedgerValidationError,
    build_non_actionable_candidate_context,
    build_transaction_derived_portfolio_context,
    confirm_and_append,
    load_ledger,
    normalize_transaction_preview,
    rebuild_positions,
    run_command,
    validate_private_ledger_path,
)
from market_engine.portfolio_review import manual_transaction_ledger as ledger_module
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MarketEnginePortfolioPositionState,
)


RECORDED_AT = "2026-08-12T12:00:00Z"


@pytest.fixture
def registry() -> AuthoritativeInstrumentRegistry:
    return AuthoritativeInstrumentRegistry(
        [
            {
                "instrument_id": "equity:amd",
                "symbol": "AMD",
                "source_symbol": "AMD",
                "source_mapping_status": "mapped",
                "currency": "USD",
                "exchange": "XNAS",
            },
            {
                "instrument_id": "equity:brk.b",
                "symbol": "BRK.B",
                "source_symbol": "BRK-B",
                "source_mapping_status": "mapped",
                "currency": "USD",
                "exchange": "XNYS",
            },
        ]
    )


def _raw(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "transaction_id": "txn-001",
        "portfolio_id": "synthetic-portfolio",
        "account_id": "synthetic-account-a",
        "instrument_id": "equity:amd",
        "ticker": "AMD",
        "transaction_type": "BUY",
        "trade_date": "2026-08-10",
        "quantity": "10",
        "unit_price": "100.00",
        "trade_currency": "USD",
        "fee": {"availability": "available", "amount": "0", "currency": "USD"},
    }
    value.update(overrides)
    return value


def _preview(
    registry: AuthoritativeInstrumentRegistry,
    **overrides: object,
) -> dict[str, object]:
    return normalize_transaction_preview(
        _raw(**overrides),
        registry=registry,
        recorded_at=RECORDED_AT,
    )


def _append(
    ledger: Path,
    registry: AuthoritativeInstrumentRegistry,
    **overrides: object,
) -> dict[str, object]:
    preview = _preview(registry, **overrides)
    return confirm_and_append(
        preview,
        confirmation_token=str(preview["confirmation_token"]),
        ledger_path=ledger,
        registry=registry,
    )


def _position(ledger: Path) -> dict[str, object]:
    positions = rebuild_positions(ledger)["positions"]
    assert len(positions) == 1
    return positions[0]


def _rewrite_event(ledger: Path, index: int, **changes: object) -> None:
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[index + 1].update(changes)
    ledger.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True)


def _market_scanner_git_repo(path: Path, *, ignored: bool = True) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "remote", "add", "origin", "git@github.com:sclaessens/market-scanner.git")
    if ignored:
        (path / ".gitignore").write_text("/data/portfolio/private/\n", encoding="utf-8")
    return path


def _schema_errors(instance: object, schema: dict[str, object], path: str = "$") -> list[str]:
    errors: list[str] = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: const")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: enum")
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        type_checks = {
            "object": lambda value: isinstance(value, dict),
            "array": lambda value: isinstance(value, list),
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
            "null": lambda value: value is None,
        }
        if not any(type_checks[name](instance) for name in names):
            return [f"{path}: type"]
    if isinstance(instance, str):
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), instance) is None:
            errors.append(f"{path}: pattern")
        if len(instance) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: minLength")
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            errors.append(f"{path}: maxLength")
        try:
            if schema.get("format") == "date":
                date.fromisoformat(instance)
            elif schema.get("format") == "date-time":
                parsed = datetime.fromisoformat(instance.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("timezone required")
        except ValueError:
            errors.append(f"{path}: format")
    if isinstance(instance, int) and "minimum" in schema and instance < int(schema["minimum"]):
        errors.append(f"{path}: minimum")
    if isinstance(instance, dict):
        required = set(schema.get("required", []))
        errors.extend(f"{path}.{key}: required" for key in required - set(instance))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key}: additional" for key in set(instance) - set(properties))
        for key, child_schema in properties.items():
            if key in instance:
                errors.extend(_schema_errors(instance[key], child_schema, f"{path}.{key}"))
    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: minItems")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in instance}) != len(instance):
            errors.append(f"{path}: uniqueItems")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
    for clause in schema.get("allOf", []):
        if_schema = clause.get("if")
        if if_schema is None or not _schema_errors(instance, if_schema, path):
            errors.extend(_schema_errors(instance, clause.get("then", {}), path))
    return errors


def _assert_code(code: LedgerIssueCode, function, *args, **kwargs) -> None:
    with pytest.raises(LedgerValidationError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def test_valid_buy_requires_confirmation_and_appends_private_event(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "private" / "ledger.jsonl"
    preview = _preview(registry)

    result = confirm_and_append(
        preview,
        confirmation_token=str(preview["confirmation_token"]),
        ledger_path=ledger,
        registry=registry,
    )

    assert result["append_status"] == "confirmed_event_appended"
    assert result["ledger_path"] == "<private-ledger>/ledger.jsonl"
    loaded = load_ledger(ledger)
    assert loaded["header"]["schema_version"] == LEDGER_HEADER_SCHEMA_VERSION
    assert loaded["events"][0]["schema_version"] == LEDGER_SCHEMA_VERSION
    assert loaded["events"][0]["transaction_type"] == "BUY"
    assert ledger.stat().st_mode & 0o077 == 0
    assert _position(ledger)["quantity"] == "10"


def test_no_persistence_without_confirmation(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    preview = _preview(registry)

    _assert_code(
        LedgerIssueCode.CONFIRMATION_REQUIRED,
        confirm_and_append,
        preview,
        confirmation_token=None,
        ledger_path=ledger,
        registry=registry,
    )
    assert not ledger.exists()


def test_changed_preview_invalidates_confirmation(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    preview = _preview(registry)
    token = str(preview["confirmation_token"])
    changed = json.loads(json.dumps(preview))
    changed["event"]["quantity"] = "11"

    _assert_code(
        LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
        confirm_and_append,
        changed,
        confirmation_token=token,
        ledger_path=ledger,
        registry=registry,
    )
    assert not ledger.exists()


def test_multiple_buys_use_moving_weighted_average_with_buy_fees(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(
        ledger,
        registry,
        fee={"availability": "available", "amount": "10", "currency": "USD"},
    )
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        trade_date="2026-08-11",
        unit_price="120",
        fee={"availability": "available", "amount": "10", "currency": "USD"},
    )

    position = _position(ledger)
    assert position["quantity"] == "20"
    assert position["remaining_cost_basis"] == "2220.00"
    assert position["weighted_average_cost"] == "111.00"
    assert position["cumulative_fees"] == "20"


def test_partial_and_full_sell_preserve_moving_average_and_realized_result(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(
        ledger,
        registry,
        fee={"availability": "available", "amount": "10", "currency": "USD"},
    )
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        trade_date="2026-08-11",
        unit_price="120",
        fee={"availability": "available", "amount": "10", "currency": "USD"},
    )
    _append(
        ledger,
        registry,
        transaction_id="txn-003",
        transaction_type="SELL",
        trade_date="2026-08-12",
        quantity="5",
        unit_price="130",
        fee={"availability": "available", "amount": "5", "currency": "USD"},
    )
    partial = _position(ledger)
    assert partial["quantity"] == "15"
    assert partial["weighted_average_cost"] == "111.00"
    assert partial["remaining_cost_basis"] == "1665.00"
    assert partial["realized_profit_loss"] == "90.00"

    _append(
        ledger,
        registry,
        transaction_id="txn-004",
        transaction_type="SELL",
        trade_date="2026-08-12",
        execution_timestamp="2026-08-12T11:00:00Z",
        quantity="15",
        unit_price="140",
    )
    closed = _position(ledger)
    assert closed["position_status"] == "closed"
    assert closed["quantity"] == "0"
    assert closed["weighted_average_cost"] == "0"
    assert closed["remaining_cost_basis"] == "0"
    assert closed["realized_profit_loss"] == "525.00"


def test_oversell_rejected_without_partial_append(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    before = ledger.read_bytes()

    _assert_code(
        LedgerIssueCode.OVERSELL,
        _append,
        ledger,
        registry,
        transaction_id="txn-002",
        transaction_type="SELL",
        trade_date="2026-08-11",
        quantity="11",
    )
    assert ledger.read_bytes() == before


def test_explicit_zero_fee_differs_from_unavailable_fee(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    zero_ledger = tmp_path / "zero.jsonl"
    unavailable_ledger = tmp_path / "unknown.jsonl"
    _append(zero_ledger, registry)
    _append(
        unavailable_ledger,
        registry,
        fee={"availability": "unavailable", "amount": None, "currency": None},
    )

    zero = _position(zero_ledger)
    unavailable = _position(unavailable_ledger)
    assert zero["cumulative_fees"] == "0"
    assert zero["calculation_status"] == "complete"
    assert unavailable["cumulative_fees"] is None
    assert unavailable["weighted_average_cost"] is None
    assert unavailable["calculation_status"] == "partial"
    assert "FEE_VALUE_UNAVAILABLE" in unavailable["calculation_blockers"]


def test_known_cost_basis_restarts_after_unknown_fee_cycle_closes(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(
        ledger,
        registry,
        fee={"availability": "unavailable", "amount": None, "currency": None},
    )
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        transaction_type="SELL",
        trade_date="2026-08-11",
        quantity="10",
        unit_price="120",
    )
    closed = _position(ledger)
    assert closed["quantity"] == "0"
    assert closed["remaining_cost_basis"] == "0"

    _append(
        ledger,
        registry,
        transaction_id="txn-003",
        trade_date="2026-08-12",
        quantity="2",
        unit_price="50",
        fee={"availability": "available", "amount": "4", "currency": "USD"},
    )
    reopened = _position(ledger)
    assert reopened["quantity"] == "2"
    assert reopened["remaining_cost_basis"] == "104"
    assert reopened["weighted_average_cost"] == "52"
    assert reopened["cumulative_fees"] is None
    assert reopened["realized_profit_loss"] is None
    assert "COST_BASIS_UNAVAILABLE" not in reopened["calculation_blockers"]
    assert "FEE_VALUE_UNAVAILABLE" in reopened["calculation_blockers"]
    assert "REALIZED_PROFIT_LOSS_UNAVAILABLE" in reopened["calculation_blockers"]


def test_unknown_old_cost_basis_remains_unknown_after_partial_sale(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(
        ledger,
        registry,
        fee={"availability": "unavailable", "amount": None, "currency": None},
    )
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        transaction_type="SELL",
        trade_date="2026-08-11",
        quantity="5",
        unit_price="120",
    )
    partial = _position(ledger)
    assert partial["quantity"] == "5"
    assert partial["remaining_cost_basis"] is None
    assert partial["weighted_average_cost"] is None
    assert "COST_BASIS_UNAVAILABLE" in partial["calculation_blockers"]


def test_duplicate_transaction_and_duplicate_replay_fail_closed(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    before = ledger.read_bytes()
    _assert_code(LedgerIssueCode.DUPLICATE_TRANSACTION_ID, _append, ledger, registry)
    assert ledger.read_bytes() == before

    loaded = load_ledger(ledger)
    loaded["events"].append(dict(loaded["events"][0]))
    _assert_code(LedgerIssueCode.LEDGER_INCOMPATIBLE, rebuild_positions, loaded)


def test_rebuild_is_deterministic_idempotent_and_ignores_derived_files(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    first = rebuild_positions(ledger)
    fake_derived = tmp_path / "positions.json"
    fake_derived.write_text('{"quantity":"999999"}', encoding="utf-8")
    second = rebuild_positions(ledger)

    assert first == second
    assert first["schema_version"] == PROJECTION_SCHEMA_VERSION
    assert first["positions"][0]["quantity"] == "10"


def test_correction_replaces_original_without_editing_history(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    original = ledger.read_text(encoding="utf-8")
    _append(
        ledger,
        registry,
        event_type="correction",
        transaction_id="txn-correction-001",
        corrects_transaction_id="txn-001",
        reason="Synthetic input correction",
        quantity="5",
        unit_price="90",
    )

    loaded = load_ledger(ledger)
    assert len(loaded["events"]) == 2
    assert json.loads(original.splitlines()[1]) == loaded["events"][0]
    position = _position(ledger)
    assert position["quantity"] == "5"
    assert position["weighted_average_cost"] == "90"
    assert position["transaction_references"] == ["txn-001", "txn-correction-001"]
    assert position["transaction_count"] == 2


def test_full_reversal_and_unknown_or_duplicate_reversal(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    _append(
        ledger,
        registry,
        event_type="reversal",
        transaction_id="txn-reversal-001",
        reverses_transaction_id="txn-001",
        reason="Synthetic full reversal",
    )
    reversed_position = _position(ledger)
    assert reversed_position["position_status"] == "closed"
    assert reversed_position["quantity"] == "0"
    assert reversed_position["transaction_references"] == [
        "txn-001",
        "txn-reversal-001",
    ]
    assert reversed_position["last_confirmed_transaction_id"] == "txn-reversal-001"

    _assert_code(
        LedgerIssueCode.UNKNOWN_CORRECTION_TARGET,
        _append,
        ledger,
        registry,
        event_type="reversal",
        transaction_id="txn-reversal-unknown",
        reverses_transaction_id="unknown",
        reason="Synthetic unknown reversal",
    )
    _assert_code(
        LedgerIssueCode.TARGET_ALREADY_REVERSED,
        _append,
        ledger,
        registry,
        event_type="reversal",
        transaction_id="txn-reversal-002",
        reverses_transaction_id="txn-001",
        reason="Synthetic duplicate reversal",
    )


def test_same_ticker_in_two_accounts_remains_separate(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        account_id="synthetic-account-b",
        quantity="2",
    )
    projection = rebuild_positions(ledger)
    assert [(row["account_id"], row["quantity"]) for row in projection["positions"]] == [
        ("synthetic-account-a", "10"),
        ("synthetic-account-b", "2"),
    ]


def test_authoritative_identity_alias_and_ambiguity() -> None:
    registry = AuthoritativeInstrumentRegistry(
        [
            {
                "instrument_id": "equity:a",
                "symbol": "AAA.A",
                "source_symbol": "SHARED",
                "source_mapping_status": "mapped",
                "currency": "USD",
                "exchange": "XNYS",
            },
            {
                "instrument_id": "equity:b",
                "symbol": "BBB.B",
                "source_symbol": "SHARED",
                "source_mapping_status": "mapped",
                "currency": "USD",
                "exchange": "XNYS",
            },
        ]
    )
    _assert_code(LedgerIssueCode.AMBIGUOUS_TICKER, registry.resolve, ticker="SHARED")
    identity = registry.resolve(instrument_id="equity:a", ticker="AAA.A")
    assert identity.instrument_id == "equity:a"


def test_unknown_and_cross_instrument_identity_fail_with_specific_codes(
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    _assert_code(
        LedgerIssueCode.UNKNOWN_INSTRUMENT,
        _preview,
        registry,
        instrument_id="equity:unknown",
        ticker=None,
    )
    _assert_code(
        LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
        _preview,
        registry,
        instrument_id="equity:amd",
        ticker="BRK.B",
    )


def test_approved_alias_resolves_to_canonical_snapshot(
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    preview = _preview(
        registry,
        instrument_id=None,
        ticker="BRK-B",
        trade_currency="USD",
    )
    assert preview["event"]["instrument_id"] == "equity:brk.b"
    assert preview["event"]["canonical_ticker"] == "BRK.B"


def test_unapproved_legacy_alias_fails_closed() -> None:
    registry = AuthoritativeInstrumentRegistry(
        [
            {
                "instrument_id": "equity:legacy",
                "symbol": "LEGACY",
                "source_symbol": "LEGACY.OLD",
                "source_mapping_status": "unsupported",
                "currency": "USD",
                "exchange": "XNYS",
            }
        ]
    )
    _assert_code(
        LedgerIssueCode.UNAPPROVED_TICKER_ALIAS,
        registry.resolve,
        ticker="LEGACY.OLD",
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"transaction_id": ""}, LedgerIssueCode.MISSING_TRANSACTION_ID),
        ({"transaction_type": "HOLD"}, LedgerIssueCode.UNSUPPORTED_TRANSACTION_TYPE),
        ({"quantity": "0"}, LedgerIssueCode.INVALID_QUANTITY),
        ({"quantity": "-1"}, LedgerIssueCode.INVALID_QUANTITY),
        ({"unit_price": "-1"}, LedgerIssueCode.INVALID_PRICE),
        ({"quantity": "not-a-decimal"}, LedgerIssueCode.MALFORMED_DECIMAL),
        ({"quantity": 0.1}, LedgerIssueCode.MALFORMED_DECIMAL),
        ({"trade_date": "invalid"}, LedgerIssueCode.INVALID_TRADE_DATE),
        ({"trade_date": "2026-08-13"}, LedgerIssueCode.FUTURE_TRADE_DATE),
        ({"trade_currency": "XYZ"}, LedgerIssueCode.UNSUPPORTED_CURRENCY),
        ({"trade_currency": "EUR"}, LedgerIssueCode.CURRENCY_MISMATCH),
        (
            {"fee": {"availability": "available", "amount": "1", "currency": "EUR"}},
            LedgerIssueCode.FEE_CURRENCY_AMBIGUITY,
        ),
        (
            {"fee": {"availability": "unavailable", "amount": "0", "currency": "USD"}},
            LedgerIssueCode.FEE_CURRENCY_AMBIGUITY,
        ),
    ],
)
def test_specific_input_failures(
    registry: AuthoritativeInstrumentRegistry,
    overrides: dict[str, object],
    code: LedgerIssueCode,
) -> None:
    _assert_code(code, _preview, registry, **overrides)


def test_execution_timestamp_cannot_follow_trusted_preview_moment(
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    _assert_code(
        LedgerIssueCode.INVALID_TIMESTAMP,
        _preview,
        registry,
        trade_date="2026-08-12",
        execution_timestamp="2026-08-12T12:00:01Z",
    )
    equal = _preview(
        registry,
        trade_date="2026-08-12",
        execution_timestamp=RECORDED_AT,
    )
    assert equal["event"]["execution_timestamp"] == RECORDED_AT
    historical = _preview(
        registry,
        trade_date="2026-08-10",
        execution_timestamp="2026-08-10T15:30:00-04:00",
    )
    assert historical["event"]["execution_timestamp"] == "2026-08-10T19:30:00Z"
    _assert_code(
        LedgerIssueCode.INVALID_TIMESTAMP,
        _preview,
        registry,
        trade_date="2026-08-12",
        execution_timestamp="2026-08-12T13:30:00-01:00",
    )


def test_future_recorded_timestamp_and_changed_future_preview_fail_closed(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    with pytest.raises(LedgerValidationError) as caught:
        normalize_transaction_preview(
            _raw(trade_date="2026-08-12"),
            registry=registry,
            recorded_at="2999-01-01T00:00:00Z",
        )
    assert caught.value.code == LedgerIssueCode.INVALID_TIMESTAMP

    preview = _preview(registry, trade_date="2026-08-12")
    preview["event"]["execution_timestamp"] = "2026-08-12T12:00:01Z"
    digest = hashlib.sha256(
        json.dumps(preview["event"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    preview["preview_digest"] = digest
    preview["confirmation_token"] = digest
    _assert_code(
        LedgerIssueCode.CONFIRMED_PREVIEW_MISMATCH,
        confirm_and_append,
        preview,
        confirmation_token=digest,
        ledger_path=tmp_path / "ledger.jsonl",
        registry=registry,
    )


def test_loaded_impossible_execution_recorded_relationship_is_corrupt(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(
        ledger,
        registry,
        trade_date="2026-08-12",
        execution_timestamp="2026-08-12T11:00:00Z",
    )
    _rewrite_event(ledger, 0, execution_timestamp="2026-08-12T12:00:01Z")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, ledger)


def test_decimal_precision_is_serialized_without_binary_float(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry, quantity="0.00000001", unit_price="0.123456789123456789")
    position = _position(ledger)
    assert position["quantity"] == "0.00000001"
    assert position["weighted_average_cost"] == "0.123456789123456789"
    assert "float" not in ledger.read_text(encoding="utf-8")


def test_same_session_purchase_and_sale_require_execution_order(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    before = ledger.read_bytes()
    _assert_code(
        LedgerIssueCode.NONDETERMINISTIC_EVENT_ORDER,
        _append,
        ledger,
        registry,
        transaction_id="txn-002",
        transaction_type="SELL",
        quantity="1",
    )
    assert ledger.read_bytes() == before


def test_corrupt_and_incompatible_ledgers_fail_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"schema_version":', encoding="utf-8")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, corrupt)

    incompatible = tmp_path / "incompatible.jsonl"
    incompatible.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")
    _assert_code(LedgerIssueCode.LEDGER_INCOMPATIBLE, load_ledger, incompatible)


@pytest.mark.parametrize(
    "changes",
    [
        {"quantity": "-1"},
        {"quantity": "0"},
        {"quantity": 1.5},
        {"quantity": "NaN"},
        {"quantity": "1e2"},
        {"unit_price": "-1"},
        {"unit_price": "Infinity"},
        {"transaction_type": "HOLD"},
        {"source_type": "broker_import"},
        {"trade_currency": "XYZ"},
        {"fee": {"availability": "available", "amount": "-1", "currency": "USD"}},
        {"fee": {"availability": "unavailable", "amount": "0", "currency": "USD"}},
        {"recorded_at": "not-a-timestamp"},
        {"recorded_at": "2999-01-01T00:00:00Z"},
        {"unexpected": "field"},
    ],
)
def test_loaded_transaction_events_are_semantically_validated_fail_closed(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
    changes: dict[str, object],
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    _rewrite_event(ledger, 0, **changes)

    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, ledger)
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, rebuild_positions, ledger)


def test_nonstandard_json_nan_and_infinity_are_rejected_before_projection(tmp_path: Path) -> None:
    for constant in ("NaN", "Infinity"):
        ledger = tmp_path / f"{constant}.jsonl"
        header = {
            "schema_version": LEDGER_HEADER_SCHEMA_VERSION,
            "record_type": "ledger_header",
            "portfolio_id": "synthetic-portfolio",
            "source_type": "manual_user_input",
        }
        ledger.write_text(json.dumps(header) + f'\n{{"quantity":{constant}}}\n', encoding="utf-8")
        _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, ledger)


def test_loaded_reversal_correction_portfolio_and_reference_rules_fail_closed(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    reversal = tmp_path / "reversal.jsonl"
    _append(reversal, registry)
    _append(
        reversal,
        registry,
        event_type="reversal",
        transaction_id="txn-reversal",
        reverses_transaction_id="txn-001",
        reason="Synthetic reversal",
    )
    _rewrite_event(reversal, 1, quantity="1")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, reversal)

    correction = tmp_path / "correction.jsonl"
    _append(correction, registry)
    _append(
        correction,
        registry,
        event_type="correction",
        transaction_id="txn-correction",
        corrects_transaction_id="txn-001",
        reason="Synthetic correction",
    )
    _rewrite_event(correction, 1, quantity=None)
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, correction)

    wrong_portfolio = tmp_path / "wrong-portfolio.jsonl"
    _append(wrong_portfolio, registry)
    _rewrite_event(wrong_portfolio, 0, portfolio_id="other-portfolio")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, wrong_portfolio)

    malformed_reference = tmp_path / "malformed-reference.jsonl"
    _append(malformed_reference, registry)
    _append(
        malformed_reference,
        registry,
        event_type="reversal",
        transaction_id="txn-reversal",
        reverses_transaction_id="txn-001",
        reason="Synthetic reversal",
    )
    _rewrite_event(malformed_reference, 1, reverses_transaction_id="unknown")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, malformed_reference)


def test_loaded_header_requires_exact_authority_fields(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    rows = ledger.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["source_type"] = "broker_import"
    ledger.write_text(json.dumps(header) + "\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
    _assert_code(LedgerIssueCode.LEDGER_CORRUPT, load_ledger, ledger)


def test_repository_local_live_data_requires_ignored_private_path(tmp_path: Path) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    allowed = repo / "data/portfolio/private/ledger.jsonl"
    assert validate_private_ledger_path(allowed) == allowed.resolve()
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        repo / "data/portfolio/ledger.jsonl",
    )


def test_tracked_descendants_under_data_do_not_block_ignored_private_ledger(
    tmp_path: Path,
) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    reference = repo / "data/processed/reference.csv"
    reference.parent.mkdir(parents=True)
    reference.write_text("symbol,value\nSYNTHETIC,1\n", encoding="utf-8")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic Test")
    _git(repo, "add", ".gitignore", reference.relative_to(repo).as_posix())
    _git(repo, "commit", "-q", "-m", "synthetic tracked data")

    allowed = repo / "data/portfolio/private/ledger.jsonl"
    assert validate_private_ledger_path(allowed) == allowed.resolve()


def test_repository_local_private_path_fails_without_ignore_rule(tmp_path: Path) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo", ignored=False)
    _assert_code(
        LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
        validate_private_ledger_path,
        repo / "data/portfolio/private/ledger.jsonl",
    )


def test_private_path_detection_is_target_based_when_cwd_is_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    allowed = repo / "data/portfolio/private/ledger.jsonl"
    assert validate_private_ledger_path(allowed) == allowed.resolve()
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        repo / "artifacts/private-ledger.jsonl",
    )


def test_tracked_private_path_and_file_parent_conflicts_fail_closed(tmp_path: Path) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    tracked = repo / "data/portfolio/private/tracked.jsonl"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("synthetic", encoding="utf-8")
    _git(repo, "add", "-f", tracked.relative_to(repo).as_posix())
    _assert_code(LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH, validate_private_ledger_path, tracked)

    parent_file = repo / "data/portfolio/private/conflict"
    parent_file.write_text("synthetic", encoding="utf-8")
    _git(repo, "add", "-f", parent_file.relative_to(repo).as_posix())
    _assert_code(
        LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
        validate_private_ledger_path,
        parent_file / "ledger.jsonl",
    )

    untracked_parent_file = repo / "data/portfolio/private/untracked-conflict"
    untracked_parent_file.write_text("synthetic", encoding="utf-8")
    _assert_code(
        LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
        validate_private_ledger_path,
        untracked_parent_file / "ledger.jsonl",
    )


def test_other_git_repository_is_not_a_private_storage_location(tmp_path: Path) -> None:
    repo = tmp_path / "other-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "remote", "add", "origin", "git@github.com:example/other.git")
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        repo / "private/ledger.jsonl",
    )


def test_external_non_git_private_path_remains_supported(tmp_path: Path) -> None:
    external = tmp_path / "external/private/ledger.jsonl"
    assert validate_private_ledger_path(external) == external.resolve()


def test_git_file_worktree_is_detected_from_target_path(tmp_path: Path) -> None:
    main = _market_scanner_git_repo(tmp_path / "main")
    _git(main, "config", "user.email", "synthetic@example.invalid")
    _git(main, "config", "user.name", "Synthetic Test")
    _git(main, "add", ".gitignore")
    _git(main, "commit", "-q", "-m", "synthetic")
    worktree = tmp_path / "worktree"
    _git(main, "worktree", "add", "--detach", str(worktree))
    assert (worktree / ".git").is_file()
    target = worktree / "data/portfolio/private/ledger.jsonl"
    assert validate_private_ledger_path(target) == target.resolve()


def test_symlink_to_unsafe_repository_path_is_rejected(tmp_path: Path) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    unsafe = repo / "data/portfolio"
    unsafe.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    link = external / "linked-portfolio"
    link.symlink_to(unsafe, target_is_directory=True)
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        link / "ledger.jsonl",
    )


def test_git_detection_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _market_scanner_git_repo(tmp_path / "repo")
    monkeypatch.setattr(
        ledger_module,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 128, "", "failure"),
    )
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        repo / "data/portfolio/private/ledger.jsonl",
    )


def test_context_adapter_preserves_provenance_and_price_unavailability(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    instrument = registry.resolve(instrument_id="equity:amd")
    context = build_transaction_derived_portfolio_context(
        ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="synthetic-context-run",
    )
    candidate = build_non_actionable_candidate_context(
        instrument=instrument,
        portfolio_context=context,
    )

    assert context.position_state == MarketEnginePortfolioPositionState.HELD
    assert context.current_quantity == "10"
    assert context.current_market_value is None
    assert context.context_provenance["ledger_digest"] == rebuild_positions(ledger)["ledger_digest"]
    assert context.context_provenance["transaction_references"] == ("txn-001",)
    assert context.context_provenance["last_transaction_date"] == "2026-08-10"
    assert context.context_provenance["market_price_status"] == "unavailable"
    assert candidate["schema_version"] == CANDIDATE_CONTEXT_SCHEMA_VERSION
    assert candidate["portfolio_position_state"] == "held"
    forbidden = {"recommendation", "allocation", "position_size", "execution", "rank"}
    assert forbidden.isdisjoint(candidate)

    wrong_instrument = registry.resolve(instrument_id="equity:brk.b")
    _assert_code(
        LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
        build_non_actionable_candidate_context,
        instrument=wrong_instrument,
        portfolio_context=context,
    )


def test_projection_mapping_cannot_inject_quantity_or_ledger_digest(
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    instrument = registry.resolve(instrument_id="equity:amd")
    for forged in (
        {"schema_version": PROJECTION_SCHEMA_VERSION, "positions": [{"quantity": "999999"}]},
        {"schema_version": PROJECTION_SCHEMA_VERSION, "ledger_digest": "0" * 64},
    ):
        _assert_code(
            LedgerIssueCode.LEDGER_INCOMPATIBLE,
            build_transaction_derived_portfolio_context,
            forged,
            portfolio_id="synthetic-portfolio",
            account_id="synthetic-account-a",
            instrument=instrument,
            context_run_id="forged-context",
        )


def test_context_adapter_ignores_modified_derived_export_and_checks_identity(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    instrument = registry.resolve(instrument_id="equity:amd")
    derived_export = tmp_path / "positions.json"
    projection = rebuild_positions(ledger)
    projection["positions"][0]["quantity"] = "999999"
    projection["ledger_digest"] = "0" * 64
    derived_export.write_text(json.dumps(projection), encoding="utf-8")

    context = build_transaction_derived_portfolio_context(
        ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="ledger-backed",
    )
    assert context.current_quantity == "10"
    assert context.context_provenance["ledger_digest"] != "0" * 64

    _assert_code(
        LedgerIssueCode.LEDGER_INCOMPATIBLE,
        build_transaction_derived_portfolio_context,
        derived_export,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="derived-export",
    )

    _rewrite_event(ledger, 0, canonical_ticker="FORGED")
    _assert_code(
        LedgerIssueCode.INSTRUMENT_IDENTITY_MISMATCH,
        build_transaction_derived_portfolio_context,
        ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="identity-mismatch",
    )


def test_context_adapter_mismatched_portfolio_or_account_returns_unknown(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    instrument = registry.resolve(instrument_id="equity:amd")
    for portfolio_id, account_id in (
        ("other-portfolio", "synthetic-account-a"),
        ("synthetic-portfolio", "other-account"),
    ):
        context = build_transaction_derived_portfolio_context(
            ledger,
            portfolio_id=portfolio_id,
            account_id=account_id,
            instrument=instrument,
            context_run_id="mismatch",
        )
        assert context.position_state == MarketEnginePortfolioPositionState.UNKNOWN
        assert context.current_quantity is None


def test_context_adapter_distinguishes_not_held_closed_partial_and_unknown(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    instrument = registry.resolve(instrument_id="equity:amd")
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    _append(
        ledger,
        registry,
        transaction_id="txn-002",
        account_id="synthetic-account-b",
        quantity="2",
    )
    not_held = build_transaction_derived_portfolio_context(
        ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=registry.resolve(instrument_id="equity:brk.b"),
        context_run_id="not-held",
    )
    unknown = build_transaction_derived_portfolio_context(
        ledger,
        portfolio_id="synthetic-portfolio",
        account_id="unknown-account",
        instrument=instrument,
        context_run_id="unknown",
    )
    assert not_held.position_state == MarketEnginePortfolioPositionState.NOT_HELD
    assert not_held.current_quantity == "0"
    assert unknown.position_state == MarketEnginePortfolioPositionState.UNKNOWN

    partial_ledger = tmp_path / "partial.jsonl"
    _append(
        partial_ledger,
        registry,
        fee={"availability": "unavailable", "amount": None, "currency": None},
    )
    partial = build_transaction_derived_portfolio_context(
        partial_ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="partial",
    )
    assert partial.position_state == MarketEnginePortfolioPositionState.PARTIALLY_KNOWN

    closed_ledger = tmp_path / "closed.jsonl"
    _append(closed_ledger, registry)
    _append(
        closed_ledger,
        registry,
        transaction_id="txn-002",
        transaction_type="SELL",
        trade_date="2026-08-11",
        quantity="10",
    )
    closed = build_transaction_derived_portfolio_context(
        closed_ledger,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=instrument,
        context_run_id="closed",
    )
    assert closed.position_state == MarketEnginePortfolioPositionState.CLOSED


def test_module_has_no_provider_broker_network_notification_or_publication_side_effects() -> None:
    source = Path(__file__).parents[3] / "src/market_engine/portfolio_review/manual_transaction_ledger.py"
    text = source.read_text(encoding="utf-8")
    for marker in (
        "requests.",
        "urllib",
        "yfinance",
        "broker_api",
        "telegram",
        "market-data",
        "workflow_dispatch",
    ):
        assert marker not in text.lower()


def test_command_boundary_previews_then_confirms_exact_payload(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AuthoritativeInstrumentRegistry,
        "from_canonical_universe",
        classmethod(lambda cls: registry),
    )
    input_path = tmp_path / "synthetic-input.json"
    input_path.write_text(json.dumps(_raw()), encoding="utf-8")
    preview_output = StringIO()
    assert run_command(
        ["preview", "--input", str(input_path)],
        stdout=preview_output,
        stderr=StringIO(),
    ) == 0
    preview = json.loads(preview_output.getvalue())
    preview_path = tmp_path / "synthetic-preview.json"
    preview_path.write_text(json.dumps(preview), encoding="utf-8")
    ledger = tmp_path / "private-ledger.jsonl"
    confirm_output = StringIO()
    assert run_command(
        [
            "confirm",
            "--preview",
            str(preview_path),
            "--confirmation-token",
            preview["confirmation_token"],
            "--ledger",
            str(ledger),
        ],
        stdout=confirm_output,
        stderr=StringIO(),
    ) == 0
    response = json.loads(confirm_output.getvalue())
    assert response["ledger_path"] == "<private-ledger>/private-ledger.jsonl"
    assert "quantity" not in response
    assert rebuild_positions(ledger)["positions"][0]["quantity"] == "10"


def test_versioned_contract_schemas_match_runtime_payloads(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    root = Path(__file__).parents[3]
    event_schema = json.loads(
        (root / "config/market_engine/portfolio/manual_transaction_ledger_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    projection_schema = json.loads(
        (root / "config/market_engine/portfolio/derived_positions_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    event = _preview(registry)["event"]
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    projection = rebuild_positions(ledger)

    assert event_schema["$id"] == LEDGER_SCHEMA_VERSION
    assert set(event_schema["required"]) == set(event)
    assert _schema_errors(event, event_schema) == []
    invalid_event = dict(event)
    invalid_event["quantity"] = "-1"
    assert _schema_errors(invalid_event, event_schema)
    correction = _preview(
        registry,
        event_type="correction",
        transaction_id="txn-correction",
        corrects_transaction_id="txn-001",
        reason="Synthetic correction",
    )["event"]
    reversal = _preview(
        registry,
        event_type="reversal",
        transaction_id="txn-reversal",
        reverses_transaction_id="txn-001",
        reason="Synthetic reversal",
    )["event"]
    assert _schema_errors(correction, event_schema) == []
    assert _schema_errors(reversal, event_schema) == []
    invalid_reversal = dict(reversal)
    invalid_reversal["quantity"] = "1"
    assert _schema_errors(invalid_reversal, event_schema)
    invalid_correction = dict(correction)
    invalid_correction["quantity"] = None
    assert _schema_errors(invalid_correction, event_schema)
    assert projection_schema["$id"] == PROJECTION_SCHEMA_VERSION
    assert set(projection_schema["required"]) == set(projection)
    assert _schema_errors(projection, projection_schema) == []
    invalid_projection = dict(projection)
    invalid_projection["ledger_digest"] = "forged"
    assert _schema_errors(invalid_projection, projection_schema)
