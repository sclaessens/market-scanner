from __future__ import annotations

import json
from dataclasses import asdict

from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION,
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationState,
    build_sec_companyfacts_derived_cash_generation_observations,
    persist_sec_companyfacts_derived_cash_generation_observations,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    build_sec_companyfacts_fundamental_observations,
)
from market_engine.source_context.sec_companyfacts_context import (
    build_sec_companyfacts_source_context_from_snapshot_path,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


def test_builds_positive_free_cash_flow_from_fundamental_observations(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    observations_by_category = _observations_by_category(derived_set)

    free_cash_flow_observation = observations_by_category[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]

    assert derived_set.ticker == "NVDA"
    assert derived_set.cik == "0001045810"
    assert derived_set.provider_name == "SEC_COMPANYFACTS"
    assert (
        derived_set.derived_observation_format_version
        == SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
    )
    assert (
        derived_set.fundamental_observation_format_version
        == fundamental_observation_set.observation_format_version
    )
    assert (
        derived_set.source_context_format_version
        == fundamental_observation_set.source_context_format_version
    )
    assert derived_set.source_context_state == "AVAILABLE"
    assert derived_set.source_refresh_snapshot_id == "nvda_companyfacts"
    assert derived_set.source_refresh_fetched_at == "2026-06-15T12:00:00Z"

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
    )
    assert free_cash_flow_observation.formula == "operating_cash_flow - capital_expenditures"
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": 25}
    assert free_cash_flow_observation.required_source_fields == (
        "operating_cash_flow",
        "capital_expenditures",
    )
    assert free_cash_flow_observation.missing_source_fields == ()
    assert (
        SecCompanyFactsDerivedCashGenerationCategory.CASH_GENERATION_DERIVATION_LIMITATION
        not in observations_by_category
    )


def test_builds_negative_free_cash_flow(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(5, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(30, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    free_cash_flow_observation = _observations_by_category(derived_set)[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
    )
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": -25}


def test_builds_zero_free_cash_flow(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(10, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(10, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    free_cash_flow_observation = _observations_by_category(derived_set)[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE
    )
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": 0}


def test_zero_operating_cash_flow_remains_present_for_derivation(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(0, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(10, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    free_cash_flow_observation = _observations_by_category(derived_set)[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
    )
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": -10}
    assert free_cash_flow_observation.missing_source_fields == ()


def test_missing_operating_cash_flow_limits_derivation(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(10, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    observations_by_category = _observations_by_category(derived_set)

    free_cash_flow_observation = observations_by_category[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]
    limitation_observation = observations_by_category[
        SecCompanyFactsDerivedCashGenerationCategory.CASH_GENERATION_DERIVATION_LIMITATION
    ]

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA
    )
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": None}
    assert free_cash_flow_observation.missing_source_fields == ("operating_cash_flow",)

    assert limitation_observation.state == SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED
    assert limitation_observation.missing_source_fields == ("operating_cash_flow",)


def test_missing_capex_limits_derivation(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    observations_by_category = _observations_by_category(derived_set)

    free_cash_flow_observation = observations_by_category[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]
    limitation_observation = observations_by_category[
        SecCompanyFactsDerivedCashGenerationCategory.CASH_GENERATION_DERIVATION_LIMITATION
    ]

    assert (
        free_cash_flow_observation.state
        == SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA
    )
    assert free_cash_flow_observation.derived_values == {"free_cash_flow": None}
    assert free_cash_flow_observation.missing_source_fields == ("capital_expenditures",)

    assert limitation_observation.state == SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED
    assert limitation_observation.missing_source_fields == ("capital_expenditures",)


def test_source_observation_references_are_preserved(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    free_cash_flow_observation = _observations_by_category(derived_set)[
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    ]

    assert set(free_cash_flow_observation.source_observation_references) == {
        "operating_cash_flow",
        "capital_expenditures",
    }

    operating_cash_flow_reference = (
        free_cash_flow_observation.source_observation_references["operating_cash_flow"]
    )
    capex_reference = (
        free_cash_flow_observation.source_observation_references["capital_expenditures"]
    )

    assert operating_cash_flow_reference.category == "OPERATING_CASH_FLOW_SOURCE_VALUE"
    assert operating_cash_flow_reference.source_values == {"operating_cash_flow": 30}
    assert operating_cash_flow_reference.source_references["operating_cash_flow"][
        "sec_tag_selected"
    ] == "NetCashProvidedByUsedInOperatingActivities"

    assert capex_reference.category == "CAPEX_SOURCE_PRESENCE"
    assert capex_reference.source_values == {"capital_expenditures": 5}
    assert capex_reference.source_references["capital_expenditures"][
        "sec_tag_selected"
    ] == "PaymentsToAcquirePropertyPlantAndEquipment"


def test_persistence_writes_json_without_overwrite(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )

    observation_path = persist_sec_companyfacts_derived_cash_generation_observations(
        derived_set,
        run_id="derived-cash-generation-run",
        root_dir=tmp_path / "derived_observations" / "cash_generation",
    )

    assert (
        observation_path
        == tmp_path
        / "derived_observations"
        / "cash_generation"
        / "derived-cash-generation-run"
        / "NVDA"
        / "derived_cash_generation_observations.json"
    )

    payload = json.loads(observation_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "NVDA"
    assert (
        payload["derived_observation_format_version"]
        == SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
    )
    assert payload["source_context_state"] == "AVAILABLE"
    assert payload["observations"][0]["category"] == "FREE_CASH_FLOW_DERIVATION"
    assert payload["observations"][0]["derived_values"] == {"free_cash_flow": 25}

    try:
        persist_sec_companyfacts_derived_cash_generation_observations(
            derived_set,
            run_id="derived-cash-generation-run",
            root_dir=tmp_path / "derived_observations" / "cash_generation",
        )
    except FileExistsError as error:
        assert (
            "refusing to overwrite existing SEC CompanyFacts Derived Cash Generation Observations"
            in str(error)
        )
    else:
        raise AssertionError("expected FileExistsError for existing derived observations output")


def test_derived_cash_generation_does_not_emit_analysis_or_decision_authority(tmp_path):
    fundamental_observation_set = _fundamental_observations(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_observation_set
    )
    payload = asdict(derived_set)

    forbidden_authority_fields = {
        "BUY",
        "SELL",
        "HOLD",
        "recommendation",
        "rating",
        "score",
        "rank",
        "ranking",
        "conviction",
        "urgency",
        "tradeability",
        "allocation",
        "position_size",
        "position_sizing",
        "execution",
        "target_price",
        "portfolio_action",
        "decision",
        "telegram",
        "delivery",
        "report_instruction",
        "fcf_yield",
        "valuation",
        "margin",
        "growth",
        "peer_comparison",
        "trend_analysis",
    }

    assert forbidden_authority_fields.isdisjoint(payload)
    for observation in payload["observations"]:
        assert forbidden_authority_fields.isdisjoint(observation)
        assert all(
            forbidden_term not in observation["message"].lower()
            for forbidden_term in forbidden_authority_fields
        )


def test_derived_observation_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _fundamental_observations(tmp_path, raw_payload: dict[str, object]):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=raw_payload,
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        snapshot_id="nvda_companyfacts",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )
    source_context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)
    return build_sec_companyfacts_fundamental_observations(source_context)


def _observations_by_category(derived_set):
    return {
        observation.category: observation
        for observation in derived_set.observations
    }


def _payload(facts: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": values}}
                for tag, values in facts.items()
            }
        }
    }


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }