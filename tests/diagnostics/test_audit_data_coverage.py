from __future__ import annotations

from pathlib import Path


LEGACY_DIAGNOSTICS_AUDIT_MODULE_PATH = Path(
    "archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py"
)

DIAGNOSTICS_COVERAGE_CONTRACT = {
    "target_universe": {
        "required_fields": {
            "target_mode",
            "target_total_tickers",
            "target_total_ticker_date_rows",
            "explicit_target_date_source",
        }
    },
    "portfolio_metadata": {
        "required_fields": {
            "metadata_complete_count",
            "metadata_partial_count",
            "metadata_missing_count",
            "metadata_invalid_count",
            "metadata_coverage_percentage",
            "metadata_freshness_distribution",
        }
    },
    "fundamentals": {
        "required_fields": {
            "fundamentals_sufficient_count",
            "fundamentals_partial_count",
            "fundamentals_missing_count",
            "fundamentals_invalid_count",
            "fundamentals_coverage_percentage",
            "ticker_date_match_success_count",
            "ticker_date_match_failure_count",
            "date_mismatch_count",
            "diagnostics",
        }
    },
}

FORBIDDEN_DIAGNOSTICS_AUTHORITY_TERMS = {
    "ranking",
    "scoring",
    "tradeability",
    "tradeable",
    "allocation",
    "urgency",
    "conviction",
    "buy",
    "sell",
}


def test_diagnostics_audit_script_is_archived_reference_only():
    assert LEGACY_DIAGNOSTICS_AUDIT_MODULE_PATH == Path(
        "archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py"
    )


def test_diagnostics_coverage_contract_keeps_expected_sections():
    assert set(DIAGNOSTICS_COVERAGE_CONTRACT) == {
        "target_universe",
        "portfolio_metadata",
        "fundamentals",
    }


def test_portfolio_metadata_contract_tracks_completeness_and_freshness_only():
    metadata_fields = DIAGNOSTICS_COVERAGE_CONTRACT["portfolio_metadata"][
        "required_fields"
    ]

    assert {
        "metadata_complete_count",
        "metadata_partial_count",
        "metadata_missing_count",
        "metadata_invalid_count",
        "metadata_coverage_percentage",
        "metadata_freshness_distribution",
    }.issubset(metadata_fields)


def test_fundamentals_contract_tracks_source_coverage_and_diagnostics_only():
    fundamentals_fields = DIAGNOSTICS_COVERAGE_CONTRACT["fundamentals"]["required_fields"]

    assert {
        "fundamentals_sufficient_count",
        "fundamentals_partial_count",
        "fundamentals_missing_count",
        "fundamentals_invalid_count",
        "fundamentals_coverage_percentage",
        "ticker_date_match_success_count",
        "ticker_date_match_failure_count",
        "date_mismatch_count",
        "diagnostics",
    }.issubset(fundamentals_fields)


def test_diagnostics_contract_has_no_investment_or_ranking_authority():
    contract_text = " ".join(
        sorted(
            field
            for section in DIAGNOSTICS_COVERAGE_CONTRACT.values()
            for field in section["required_fields"]
        )
    ).lower()

    for forbidden_term in FORBIDDEN_DIAGNOSTICS_AUTHORITY_TERMS:
        assert forbidden_term not in contract_text


def test_active_code_no_longer_imports_diagnostics_audit_script():
    for root in (Path("src"), Path("tests"), Path(".github")):
        if not root.exists():
            continue

        for path in root.rglob("*.py"):
            if path == Path("tests/diagnostics/test_audit_data_coverage.py"):
                continue

            source = path.read_text(encoding="utf-8")
            assert "from scripts.diagnostics import audit_data_coverage" not in source
            assert "import scripts.diagnostics" not in source
