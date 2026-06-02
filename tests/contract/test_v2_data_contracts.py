from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    DataLifecycleStage,
)


def test_reset_10b_lifecycle_stages_are_defined():
    assert {stage.name for stage in DataLifecycleStage} == {
        "RAW_SOURCE",
        "NORMALIZED_INPUT",
        "FIXTURE_INPUT",
        "GENERATED_OUTPUT",
        "REPORTING_OUTPUT",
        "LOCAL_ONLY",
    }


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


def test_reset_10b_fixture_contracts_are_fixture_input_only():
    assert {
        contract.lifecycle_stage for contract in APPROVED_FIXTURE_CONTRACTS
    } == {DataLifecycleStage.FIXTURE_INPUT}


def test_reset_10b_fixture_contracts_do_not_approve_production_paths():
    production_path_terms = {
        "data/raw/",
        "data/normalized/",
        "data/generated/",
        "data/local/",
        "data/processed/",
        "data/portfolio/",
        "data/watchlist/",
        "reports/",
    }

    for contract in APPROVED_FIXTURE_CONTRACTS:
        assert all(term not in contract.relative_path for term in production_path_terms)
