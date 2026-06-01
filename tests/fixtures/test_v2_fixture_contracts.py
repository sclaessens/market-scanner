from pathlib import Path

from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    FORBIDDEN_NON_DECISION_FIXTURE_VALUES,
    read_fixture_rows,
)


def test_approved_fixture_files_exist_and_are_not_empty():
    for contract in APPROVED_FIXTURE_CONTRACTS:
        assert contract.path.exists()
        assert contract.path.stat().st_size > 0


def test_approved_fixture_files_have_required_columns():
    for contract in APPROVED_FIXTURE_CONTRACTS:
        rows = read_fixture_rows(contract)

        assert rows
        assert set(contract.required_columns).issubset(rows[0].keys())


def test_approved_fixture_values_do_not_encode_final_decision_terms():
    for contract in APPROVED_FIXTURE_CONTRACTS:
        rows = read_fixture_rows(contract)
        values = [column for row in rows for column in row.keys()]
        values.extend(value for row in rows for value in row.values())

        normalized_values = {
            str(value).strip().lower() for value in values if str(value).strip()
        }

        assert normalized_values.isdisjoint(FORBIDDEN_NON_DECISION_FIXTURE_VALUES)


def test_source_data_fixture_keeps_missing_values_explicit():
    source_contract = next(
        contract
        for contract in APPROVED_FIXTURE_CONTRACTS
        if contract.name == "synthetic_source_data_readiness"
    )
    rows = read_fixture_rows(source_contract)
    review_required_rows = [
        row for row in rows if row["readiness_state"] == "review_required"
    ]

    assert review_required_rows
    for row in review_required_rows:
        assert row["metric_value"] == ""
        assert row["missing_value_policy"] == "missing_not_zero"
        assert row["review_required_reason"]


def test_fixture_import_and_reading_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    before = set(Path(tmp_path).iterdir())
    monkeypatch.chdir(tmp_path)

    for contract in APPROVED_FIXTURE_CONTRACTS:
        read_fixture_rows(contract)

    after = set(Path(tmp_path).iterdir())

    assert after == before
