from pathlib import Path

from market_scanner.portfolio import portfolio_contracts
from market_scanner.portfolio.portfolio_contracts import (
    PortfolioContractIssue,
    PortfolioContractIssueCode,
    PortfolioDatasetType,
    forbidden_portfolio_upstream_fields,
    portfolio_generated_dataset_types,
    portfolio_position_identity_fields,
    portfolio_source_dataset_types,
    portfolio_transaction_identity_fields,
    required_portfolio_position_fields,
    required_portfolio_transaction_fields,
    validate_portfolio_position_shape,
    validate_portfolio_transaction_shape,
)
from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    read_fixture_rows,
)


def _complete_transaction(**overrides):
    transaction = {
        "transaction_id": "txn-001",
        "portfolio_account": "synthetic-main",
        "symbol": "ALFA",
        "transaction_kind": "opening_position",
        "quantity_delta": "10",
        "cash_amount": "-1000.00",
        "currency": "EUR",
        "occurred_at": "2026-01-10",
        "source_reference": "reset_4_fixture",
    }
    transaction.update(overrides)
    return transaction


def _complete_position(**overrides):
    position = {
        "portfolio_account": "synthetic-main",
        "symbol": "ALFA",
        "quantity": "10",
        "average_cost": "100.00",
        "currency": "EUR",
        "source_reference": "normalized_manual_transactions",
    }
    position.update(overrides)
    return position


def test_portfolio_dataset_types_separate_source_from_generated_outputs():
    assert portfolio_source_dataset_types() == (
        PortfolioDatasetType.MANUAL_TRANSACTION_INPUT,
        PortfolioDatasetType.NORMALIZED_POSITION_INPUT,
    )
    assert portfolio_generated_dataset_types() == (
        PortfolioDatasetType.GENERATED_PORTFOLIO_REVIEW,
        PortfolioDatasetType.GENERATED_PORTFOLIO_CLASSIFICATION,
    )
    assert set(portfolio_source_dataset_types()).isdisjoint(
        portfolio_generated_dataset_types()
    )


def test_manual_transaction_required_fields_match_approved_v2_fixture():
    contract = next(
        contract
        for contract in APPROVED_FIXTURE_CONTRACTS
        if contract.name == "synthetic_portfolio_transactions"
    )

    assert required_portfolio_transaction_fields() == contract.required_columns
    assert validate_portfolio_transaction_shape(read_fixture_rows(contract)[0]) == ()


def test_normalized_position_required_fields_are_explicit():
    assert required_portfolio_position_fields() == (
        "portfolio_account",
        "symbol",
        "quantity",
        "average_cost",
        "currency",
        "source_reference",
    )


def test_portfolio_identity_fields_are_explicit():
    assert portfolio_transaction_identity_fields() == ("transaction_id",)
    assert portfolio_position_identity_fields() == ("portfolio_account", "symbol")


def test_portfolio_contracts_exclude_final_action_and_allocation_authority():
    transaction_fields = set(required_portfolio_transaction_fields())
    position_fields = set(required_portfolio_position_fields())

    for field_name in forbidden_portfolio_upstream_fields():
        assert field_name not in transaction_fields
        assert field_name not in position_fields


def test_missing_required_transaction_fields_are_reported_explicitly():
    transaction = _complete_transaction()
    transaction.pop("source_reference")

    issues = validate_portfolio_transaction_shape(transaction)

    assert issues == (
        PortfolioContractIssue(
            field_name="source_reference",
            issue_code=PortfolioContractIssueCode.MISSING_REQUIRED_FIELD,
            observed_value=None,
        ),
    )


def test_missing_numeric_portfolio_values_are_not_converted_to_zero():
    issues = validate_portfolio_transaction_shape(
        _complete_transaction(quantity_delta="")
    )

    assert issues == (
        PortfolioContractIssue(
            field_name="quantity_delta",
            issue_code=PortfolioContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_invalid_numeric_portfolio_values_are_reported_without_ingestion():
    issues = validate_portfolio_position_shape(_complete_position(quantity="many"))

    assert issues == (
        PortfolioContractIssue(
            field_name="quantity",
            issue_code=PortfolioContractIssueCode.INVALID_NUMERIC_VALUE,
            observed_value="many",
        ),
    )


def test_forbidden_portfolio_decision_fields_are_reported_as_issues():
    issues = validate_portfolio_transaction_shape(
        _complete_transaction(portfolio_action="rebalance")
    )

    assert issues == (
        PortfolioContractIssue(
            field_name="portfolio_action",
            issue_code=PortfolioContractIssueCode.FORBIDDEN_FIELD,
            observed_value="rebalance",
        ),
    )


def test_portfolio_contract_module_does_not_import_legacy_scripts():
    source = Path(portfolio_contracts.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source


def test_portfolio_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validate_portfolio_transaction_shape(_complete_transaction(cash_amount=None))
    validate_portfolio_position_shape(_complete_position(average_cost=""))

    assert list(tmp_path.iterdir()) == []
