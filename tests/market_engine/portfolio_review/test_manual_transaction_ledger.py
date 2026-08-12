from __future__ import annotations

import json
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
        execution_timestamp="2026-08-12T15:00:00Z",
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
    _assert_code(LedgerIssueCode.DUPLICATE_REPLAY, rebuild_positions, loaded)


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


def test_repository_local_live_data_requires_ignored_private_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("/data/portfolio/private/\n", encoding="utf-8")
    allowed = repo / "data/portfolio/private/ledger.jsonl"
    assert validate_private_ledger_path(allowed, repository_root=repo) == allowed.resolve()
    _assert_code(
        LedgerIssueCode.PRIVATE_STORAGE_REQUIRED,
        validate_private_ledger_path,
        repo / "data/portfolio/ledger.jsonl",
        repository_root=repo,
    )


def test_repository_local_private_path_fails_without_ignore_rule(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    _assert_code(
        LedgerIssueCode.TRACKED_PRIVATE_DATA_PATH,
        validate_private_ledger_path,
        repo / "data/portfolio/private/ledger.jsonl",
        repository_root=repo,
    )


def test_context_adapter_preserves_provenance_and_price_unavailability(
    tmp_path: Path,
    registry: AuthoritativeInstrumentRegistry,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _append(ledger, registry)
    projection = rebuild_positions(ledger)
    instrument = registry.resolve(instrument_id="equity:amd")
    context = build_transaction_derived_portfolio_context(
        projection,
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
    assert context.context_provenance["ledger_digest"] == projection["ledger_digest"]
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
    projection = rebuild_positions(ledger)
    not_held = build_transaction_derived_portfolio_context(
        projection,
        portfolio_id="synthetic-portfolio",
        account_id="synthetic-account-a",
        instrument=registry.resolve(instrument_id="equity:brk.b"),
        context_run_id="not-held",
    )
    unknown = build_transaction_derived_portfolio_context(
        projection,
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
        rebuild_positions(partial_ledger),
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
        rebuild_positions(closed_ledger),
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
    assert projection_schema["$id"] == PROJECTION_SCHEMA_VERSION
    assert set(projection_schema["required"]) == set(projection)
