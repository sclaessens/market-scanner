from market_scanner.shared.data_contracts import APPROVED_FIXTURE_CONTRACTS


def test_reset_4_registers_only_approved_fixture_contracts():
    assert {contract.classification for contract in APPROVED_FIXTURE_CONTRACTS} == {
        "approved_fixture"
    }
    assert {contract.source for contract in APPROVED_FIXTURE_CONTRACTS} == {"synthetic"}


def test_reset_4_fixture_contracts_use_v2_fixture_paths():
    for contract in APPROVED_FIXTURE_CONTRACTS:
        assert contract.relative_path.startswith("data/fixtures/v2/")
        assert "processed" not in contract.relative_path
        assert "reports" not in contract.relative_path
        assert "raw" not in contract.relative_path
        assert "local" not in contract.relative_path
