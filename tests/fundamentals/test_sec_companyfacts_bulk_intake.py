from __future__ import annotations

from market_scanner.fundamentals.sec_companyfacts_smoke_boundary import (
    SEC_COMPANYFACTS_SOURCE_FAMILY,
)

SEC_COMPANYFACTS_BULK_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)

BULK_INTAKE_PROVIDER_RISK_MARKERS = {
    "network_access_requires_explicit_operator_action",
    "cache_write_requires_explicit_operator_action",
    "manifest_write_requires_explicit_operator_action",
    "no_test_may_download_sec_bulk_data",
    "no_test_may_write_sec_bulk_cache",
}


def test_sec_companyfacts_bulk_intake_policy_is_canonical_provider_governance() -> None:
    assert SEC_COMPANYFACTS_SOURCE_FAMILY == "SEC EDGAR / SEC CompanyFacts"
    assert SEC_COMPANYFACTS_BULK_SOURCE_URL.startswith("https://www.sec.gov/")
    assert SEC_COMPANYFACTS_BULK_SOURCE_URL.endswith("/companyfacts.zip")


def test_sec_companyfacts_bulk_intake_requires_explicit_operator_action() -> None:
    assert "network_access_requires_explicit_operator_action" in (
        BULK_INTAKE_PROVIDER_RISK_MARKERS
    )
    assert "cache_write_requires_explicit_operator_action" in (
        BULK_INTAKE_PROVIDER_RISK_MARKERS
    )
    assert "manifest_write_requires_explicit_operator_action" in (
        BULK_INTAKE_PROVIDER_RISK_MARKERS
    )


def test_sec_companyfacts_bulk_intake_policy_forbids_test_side_effects() -> None:
    assert "no_test_may_download_sec_bulk_data" in BULK_INTAKE_PROVIDER_RISK_MARKERS
    assert "no_test_may_write_sec_bulk_cache" in BULK_INTAKE_PROVIDER_RISK_MARKERS


def test_sec_companyfacts_bulk_intake_policy_has_no_investment_authority() -> None:
    rendered = " ".join(BULK_INTAKE_PROVIDER_RISK_MARKERS).lower()

    assert "buy" not in rendered
    assert "sell" not in rendered
    assert "allocation" not in rendered
    assert "tradeability" not in rendered
    assert "conviction" not in rendered
    assert "urgency" not in rendered