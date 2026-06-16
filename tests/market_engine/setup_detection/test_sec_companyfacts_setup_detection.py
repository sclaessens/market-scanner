from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION,
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationObservation,
    SecCompanyFactsDerivedCashGenerationObservationSet,
    SecCompanyFactsDerivedCashGenerationState,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
    SecCompanyFactsFundamentalObservation,
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservationSet,
    SecCompanyFactsFundamentalObservationState,
)
from market_engine.setup_detection.sec_companyfacts_setup_detection import (
    SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION,
    SecCompanyFactsSetupCategory,
    SecCompanyFactsSetupState,
    build_sec_companyfacts_setup_detection,
    persist_sec_companyfacts_setup_detection,
)


def test_complete_positive_evidence_produces_setup_detection_output():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(),
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )

    assert setup_detection.setup_detection_format_version == (
        SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION
    )
    assert setup_detection.input_contracts == (
        SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
        SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION,
    )
    assert _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP).state == (
        SecCompanyFactsSetupState.SETUP_DETECTED
    )
    assert _item(
        setup_detection,
        SecCompanyFactsSetupCategory.FUNDAMENTAL_AVAILABILITY_SETUP,
    ).state == SecCompanyFactsSetupState.SETUP_DETECTED
    cash_item = _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP)
    assert "FREE_CASH_FLOW_DERIVATION" in cash_item.derived_observation_references
    assert "OPERATING_CASH_FLOW_SOURCE_VALUE" in cash_item.source_observation_references


def test_partial_evidence_produces_partially_detected_setup():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(net_income_state=SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE),
        _derived_observation_set(state=SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE),
        setup_detection_run_id="setup-run-1",
    )

    assert _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP).state == (
        SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED
    )
    assert _item(
        setup_detection,
        SecCompanyFactsSetupCategory.PROFITABILITY_EVIDENCE_SETUP,
    ).state == SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED


def test_missing_required_observations_produce_blocked_setup():
    fundamental_set = _fundamental_observation_set(include_revenue=False)
    setup_detection = build_sec_companyfacts_setup_detection(
        fundamental_set,
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )

    revenue_item = _item(setup_detection, SecCompanyFactsSetupCategory.REVENUE_EVIDENCE_SETUP)
    availability_item = _item(
        setup_detection,
        SecCompanyFactsSetupCategory.FUNDAMENTAL_AVAILABILITY_SETUP,
    )
    assert revenue_item.state == SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
    assert revenue_item.missing_observations == ("REVENUE_SOURCE_PRESENCE",)
    assert availability_item.state == SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
    assert "REVENUE_SOURCE_PRESENCE" in availability_item.missing_observations


def test_conflicted_evidence_produces_conflicted_setup():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(
            operating_cash_flow_state=(
                SecCompanyFactsFundamentalObservationState.NEGATIVE_SOURCE_VALUE
            ),
            operating_cash_flow=-10,
        ),
        _derived_observation_set(state=SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE),
        setup_detection_run_id="setup-run-1",
    )

    cash_item = _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP)
    assert cash_item.state == SecCompanyFactsSetupState.SETUP_CONFLICTED
    assert cash_item.setup_evidence["conflicted_source_observations"] == (
        "OPERATING_CASH_FLOW_SOURCE_VALUE",
    )


def test_unsupported_input_contract_fails_closed():
    fundamental_set = replace(
        _fundamental_observation_set(),
        observation_format_version="unsupported",
    )

    with pytest.raises(ValueError, match="unsupported SEC CompanyFacts Fundamental Observation contract"):
        build_sec_companyfacts_setup_detection(
            fundamental_set,
            _derived_observation_set(),
            setup_detection_run_id="setup-run-1",
        )


def test_numeric_zero_is_preserved_and_not_treated_as_missing():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(
            net_income=0,
            net_income_state=SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE,
            operating_cash_flow=0,
            operating_cash_flow_state=SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE,
        ),
        _derived_observation_set(
            state=SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE,
            free_cash_flow=0,
        ),
        setup_detection_run_id="setup-run-1",
    )

    profitability_item = _item(
        setup_detection,
        SecCompanyFactsSetupCategory.PROFITABILITY_EVIDENCE_SETUP,
    )
    cash_item = _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP)
    assert profitability_item.setup_evidence["net_income"] == 0
    assert profitability_item.missing_observations == ()
    assert cash_item.setup_evidence["derived_values"]["free_cash_flow"] == 0
    assert cash_item.missing_observations == ()


def test_source_and_derived_references_are_preserved():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(),
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )

    cash_item = _item(setup_detection, SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP)
    source_ref = cash_item.source_observation_references["OPERATING_CASH_FLOW_SOURCE_VALUE"]
    derived_ref = cash_item.derived_observation_references["FREE_CASH_FLOW_DERIVATION"]
    assert source_ref["source_values"]["operating_cash_flow"] == 30
    assert derived_ref["derived_values"]["free_cash_flow"] == 25
    assert setup_detection.source_refresh_snapshot_id == "NVDA_companyfacts"
    assert setup_detection.source_context_state == "AVAILABLE"


def test_forbidden_action_authority_terms_are_not_emitted_in_setup_messages():
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(),
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )
    messages = " ".join(item.message for item in setup_detection.setup_items)
    forbidden_terms = (
        "BUY",
        "SELL",
        "HOLD",
        "target price",
        "rating",
        "score",
        "ranking",
        "conviction",
        "urgency",
        "tradeability",
        "allocation",
        "position sizing",
        "execution",
        "order",
    )
    assert not any(term in messages for term in forbidden_terms)


def test_persistence_writes_json_under_temporary_root(tmp_path):
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(),
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )

    output_path = persist_sec_companyfacts_setup_detection(
        setup_detection,
        run_id="setup-run-1",
        root_dir=tmp_path,
    )

    assert output_path == tmp_path / "setup-run-1" / "NVDA" / "setup_detection.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["setup_detection_format_version"] == (
        SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION
    )
    assert payload["setup_items"][0]["category"]


def test_persistence_refuses_overwrite(tmp_path):
    setup_detection = build_sec_companyfacts_setup_detection(
        _fundamental_observation_set(),
        _derived_observation_set(),
        setup_detection_run_id="setup-run-1",
    )
    persist_sec_companyfacts_setup_detection(
        setup_detection,
        run_id="setup-run-1",
        root_dir=tmp_path,
    )

    with pytest.raises(FileExistsError):
        persist_sec_companyfacts_setup_detection(
            setup_detection,
            run_id="setup-run-1",
            root_dir=tmp_path,
        )


def test_active_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _item(setup_detection, category: SecCompanyFactsSetupCategory):
    return next(item for item in setup_detection.setup_items if item.category == category)


def _fundamental_observation_set(
    *,
    include_revenue: bool = True,
    net_income: int | None = 20,
    net_income_state: SecCompanyFactsFundamentalObservationState = (
        SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
    ),
    operating_cash_flow: int | None = 30,
    operating_cash_flow_state: SecCompanyFactsFundamentalObservationState = (
        SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
    ),
) -> SecCompanyFactsFundamentalObservationSet:
    observations = [
        _fundamental_observation(
            SecCompanyFactsFundamentalObservationCategory.SOURCE_CONTEXT_AVAILABILITY,
            SecCompanyFactsFundamentalObservationState.PRESENT,
            ("revenue", "net_income", "operating_cash_flow", "capital_expenditures"),
            {
                "revenue": 100,
                "net_income": net_income,
                "operating_cash_flow": operating_cash_flow,
                "capital_expenditures": 5,
            },
        ),
        _fundamental_observation(
            SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE,
            net_income_state,
            ("net_income",),
            {"net_income": net_income},
        ),
        _fundamental_observation(
            SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE,
            operating_cash_flow_state,
            ("operating_cash_flow",),
            {"operating_cash_flow": operating_cash_flow},
        ),
        _fundamental_observation(
            SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE,
            SecCompanyFactsFundamentalObservationState.PRESENT,
            ("capital_expenditures",),
            {"capital_expenditures": 5},
        ),
        _fundamental_observation(
            SecCompanyFactsFundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
            SecCompanyFactsFundamentalObservationState.PRESENT,
            ("operating_cash_flow", "capital_expenditures"),
            {"operating_cash_flow": operating_cash_flow, "capital_expenditures": 5},
        ),
    ]
    if include_revenue:
        observations.insert(
            1,
            _fundamental_observation(
                SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE,
                SecCompanyFactsFundamentalObservationState.PRESENT,
                ("revenue",),
                {"revenue": 100},
            ),
        )

    return SecCompanyFactsFundamentalObservationSet(
        ticker="NVDA",
        cik="0001045810",
        provider_name="SEC_COMPANYFACTS",
        observation_format_version=SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
        source_context_format_version="sec-companyfacts-source-context-v1",
        source_context_state="AVAILABLE",
        source_context_reference="data/market_engine/source_contexts/run/NVDA/source_context.json",
        source_refresh_snapshot_id="NVDA_companyfacts",
        source_refresh_fetched_at="2026-06-15T12:00:00+00:00",
        source_refresh_payload_format_version="sec-companyfacts-raw-v1",
        observations=tuple(observations),
    )


def _derived_observation_set(
    *,
    state: SecCompanyFactsDerivedCashGenerationState = (
        SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
    ),
    free_cash_flow: int | None = 25,
) -> SecCompanyFactsDerivedCashGenerationObservationSet:
    return SecCompanyFactsDerivedCashGenerationObservationSet(
        ticker="NVDA",
        cik="0001045810",
        provider_name="SEC_COMPANYFACTS",
        derived_observation_format_version=(
            SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
        ),
        fundamental_observation_format_version=(
            SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION
        ),
        source_context_format_version="sec-companyfacts-source-context-v1",
        source_context_state="AVAILABLE",
        source_refresh_snapshot_id="NVDA_companyfacts",
        source_refresh_fetched_at="2026-06-15T12:00:00+00:00",
        source_refresh_payload_format_version="sec-companyfacts-raw-v1",
        observations=(
            SecCompanyFactsDerivedCashGenerationObservation(
                ticker="NVDA",
                cik="0001045810",
                provider_name="SEC_COMPANYFACTS",
                category=(
                    SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
                ),
                state=state,
                message="Derived cash-generation source value is observed.",
                formula="operating_cash_flow - capital_expenditures",
                derived_values={"free_cash_flow": free_cash_flow},
                required_source_fields=("operating_cash_flow", "capital_expenditures"),
                missing_source_fields=() if free_cash_flow is not None else ("operating_cash_flow",),
                source_observation_references={},
            ),
        ),
    )


def _fundamental_observation(
    category: SecCompanyFactsFundamentalObservationCategory,
    state: SecCompanyFactsFundamentalObservationState,
    canonical_fields: tuple[str, ...],
    source_values: dict[str, object | None],
) -> SecCompanyFactsFundamentalObservation:
    missing_source_fields = tuple(
        field_name
        for field_name, value in source_values.items()
        if value is None
    )
    return SecCompanyFactsFundamentalObservation(
        ticker="NVDA",
        cik="0001045810",
        provider_name="SEC_COMPANYFACTS",
        category=category,
        state=state,
        message="Synthetic source-grounded observation.",
        source_context_state="AVAILABLE",
        canonical_fields=canonical_fields,
        source_values=source_values,
        source_references={
            field_name: {"sec_tag_selected": "SyntheticTag", "unit": "USD"}
            for field_name, value in source_values.items()
            if value is not None
        },
        missing_source_fields=missing_source_fields,
    )
