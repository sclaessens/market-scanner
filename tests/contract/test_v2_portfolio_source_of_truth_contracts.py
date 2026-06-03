from importlib import reload
from pathlib import Path

from market_scanner.portfolio import portfolio_source_contracts
from market_scanner.portfolio.portfolio_source_contracts import (
    PortfolioSourceContractIssue,
    PortfolioSourceContractIssueCode,
    PortfolioSourceDatasetRole,
    forbidden_portfolio_display_authority_fields,
    forbidden_portfolio_source_fields,
    generated_dataset_roles,
    is_portfolio_source_of_truth,
    manual_source_dataset_roles,
    portfolio_dataset_roles,
    reporting_display_dataset_roles,
    required_manual_position_fields,
    required_manual_transaction_fields,
    required_portfolio_display_input_fields,
    validate_manual_position_source_shape,
    validate_manual_transaction_source_shape,
    validate_portfolio_display_input_shape,
)


def _manual_position(**overrides):
    record = {
        "portfolio_id": "synthetic-main",
        "ticker": "ASML",
        "quantity": "10",
        "currency": "EUR",
        "source_type": "manual_source",
        "as_of_date": "2026-06-02",
    }
    record.update(overrides)
    return record


def _manual_transaction(**overrides):
    record = {
        "portfolio_id": "synthetic-main",
        "transaction_id": "txn-001",
        "ticker": "ASML",
        "transaction_type": "purchase",
        "quantity": "10",
        "price": "100.00",
        "currency": "EUR",
        "transaction_date": "2026-01-15",
        "source_type": "manual_source",
    }
    record.update(overrides)
    return record


def _display_input(**overrides):
    record = {
        "ticker": "ASML",
        "profit_loss_percent_display": "+8.4%",
        "current_price_display": "€XXX",
        "target_price_display": "€XXX",
        "action_status": "REVIEW",
        "currency": "EUR",
        "source_reference": "synthetic-source-row-001",
    }
    record.update(overrides)
    return record


def test_portfolio_dataset_roles_are_explicit_and_complete():
    assert portfolio_dataset_roles() == (
        PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS,
        PortfolioSourceDatasetRole.MANUAL_SOURCE_POSITIONS,
        PortfolioSourceDatasetRole.NORMALIZED_POSITIONS,
        PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_REVIEW,
        PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_INTELLIGENCE,
        PortfolioSourceDatasetRole.REPORTING_DISPLAY_INPUT,
    )


def test_manual_source_positions_are_source_of_truth_for_holdings():
    assert PortfolioSourceDatasetRole.MANUAL_SOURCE_POSITIONS in (
        manual_source_dataset_roles()
    )
    assert is_portfolio_source_of_truth(
        PortfolioSourceDatasetRole.MANUAL_SOURCE_POSITIONS
    )
    assert validate_manual_position_source_shape(_manual_position()) == ()


def test_manual_source_transactions_are_source_records_not_generated_output():
    assert PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS in (
        manual_source_dataset_roles()
    )
    assert PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS not in (
        generated_dataset_roles()
    )
    assert is_portfolio_source_of_truth(
        PortfolioSourceDatasetRole.MANUAL_SOURCE_TRANSACTIONS
    )
    assert validate_manual_transaction_source_shape(_manual_transaction()) == ()


def test_generated_portfolio_outputs_are_not_source_of_truth():
    for role in (
        PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_REVIEW,
        PortfolioSourceDatasetRole.GENERATED_PORTFOLIO_INTELLIGENCE,
    ):
        assert role in generated_dataset_roles()
        assert not is_portfolio_source_of_truth(role)


def test_reporting_and_telegram_display_inputs_are_not_source_of_truth():
    assert reporting_display_dataset_roles() == (
        PortfolioSourceDatasetRole.REPORTING_DISPLAY_INPUT,
    )
    assert not is_portfolio_source_of_truth(
        PortfolioSourceDatasetRole.REPORTING_DISPLAY_INPUT
    )


def test_required_manual_position_fields_exist():
    assert required_manual_position_fields() == (
        "portfolio_id",
        "ticker",
        "quantity",
        "currency",
        "source_type",
        "as_of_date",
    )


def test_required_manual_transaction_fields_exist():
    assert required_manual_transaction_fields() == (
        "portfolio_id",
        "transaction_id",
        "ticker",
        "transaction_type",
        "quantity",
        "price",
        "currency",
        "transaction_date",
        "source_type",
    )


def test_required_display_input_fields_align_with_telegram_ux_needs():
    display_fields = required_portfolio_display_input_fields()

    assert display_fields == (
        "ticker",
        "profit_loss_percent_display",
        "current_price_display",
        "target_price_display",
        "action_status",
        "currency",
        "source_reference",
    )
    assert validate_portfolio_display_input_shape(_display_input()) == ()


def test_display_values_are_supplied_upstream_and_not_calculation_fields():
    display_fields = set(required_portfolio_display_input_fields())

    for forbidden in (
        "profit_loss_calculation",
        "current_price_fetch",
        "target_price_calculation",
    ):
        assert forbidden not in display_fields
        assert forbidden in forbidden_portfolio_display_authority_fields()


def test_missing_manual_source_values_remain_explicit():
    issues = validate_manual_position_source_shape(_manual_position(quantity=""))

    assert issues == (
        PortfolioSourceContractIssue(
            field_name="quantity",
            issue_code=PortfolioSourceContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_missing_display_values_remain_explicit_not_zero():
    issues = validate_portfolio_display_input_shape(
        _display_input(target_price_display="")
    )

    assert issues == (
        PortfolioSourceContractIssue(
            field_name="target_price_display",
            issue_code=PortfolioSourceContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_forbidden_source_fields_are_reported_explicitly():
    issues = validate_manual_transaction_source_shape(
        _manual_transaction(action_status="REVIEW")
    )

    assert issues == (
        PortfolioSourceContractIssue(
            field_name="action_status",
            issue_code=PortfolioSourceContractIssueCode.FORBIDDEN_FIELD,
            observed_value="REVIEW",
        ),
    )


def test_source_contracts_do_not_include_final_action_or_allocation_authority():
    manual_fields = set(required_manual_position_fields()) | set(
        required_manual_transaction_fields()
    )

    for field_name in forbidden_portfolio_source_fields():
        assert field_name not in manual_fields


def test_display_input_contract_does_not_include_allocation_or_ranking_authority():
    display_fields = set(required_portfolio_display_input_fields())

    for field_name in forbidden_portfolio_display_authority_fields():
        assert field_name not in display_fields


def test_portfolio_source_contract_module_does_not_import_legacy_scripts():
    source = Path(portfolio_source_contracts.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source


def test_importing_source_contract_module_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(portfolio_source_contracts)

    assert list(tmp_path.iterdir()) == []


def test_source_contract_helpers_create_no_files_or_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validate_manual_position_source_shape(_manual_position(quantity=None))
    validate_manual_transaction_source_shape(_manual_transaction(price=None))
    validate_portfolio_display_input_shape(_display_input(current_price_display=None))

    assert list(tmp_path.iterdir()) == []
    assert not Path("reports/daily/telegram_message.txt").exists()
