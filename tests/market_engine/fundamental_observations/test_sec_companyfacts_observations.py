from __future__ import annotations

import json
from dataclasses import asdict

from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservationState,
    build_sec_companyfacts_fundamental_observations,
    persist_sec_companyfacts_fundamental_observations,
)
from market_engine.source_context.sec_companyfacts_context import (
    SecCompanyFactsContextFieldState,
    SecCompanyFactsContextState,
    build_sec_companyfacts_source_context_from_snapshot_path,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


def test_builds_observations_from_available_source_context(tmp_path):
    source_context = _source_context(
        tmp_path,
        _complete_payload(),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)

    assert observation_set.ticker == "NVDA"
    assert observation_set.cik == "0001045810"
    assert observation_set.provider_name == "SEC_COMPANYFACTS"
    assert (
        observation_set.observation_format_version
        == SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION
    )
    assert observation_set.source_context_format_version == source_context.context_format_version
    assert observation_set.source_context_state == "AVAILABLE"
    assert observation_set.source_refresh_snapshot_id == source_context.source_refresh_snapshot_id
    assert observation_set.source_refresh_fetched_at == source_context.source_refresh_fetched_at
    assert observation_set.source_refresh_payload_format_version == source_context.source_refresh_payload_format_version

    observations_by_category = _observations_by_category(observation_set)

    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.SOURCE_CONTEXT_AVAILABILITY
    ].state == SecCompanyFactsFundamentalObservationState.PRESENT
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE
    ].state == SecCompanyFactsFundamentalObservationState.PRESENT
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE
    ].state == SecCompanyFactsFundamentalObservationState.PRESENT
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS
    ].state == SecCompanyFactsFundamentalObservationState.PRESENT
    assert SecCompanyFactsFundamentalObservationCategory.DATA_LIMITATION not in observations_by_category


def test_partial_source_context_emits_missing_data_and_data_limitation(tmp_path):
    source_context = _source_context(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            }
        ),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    observations_by_category = _observations_by_category(observation_set)

    assert observation_set.source_context_state == "PARTIAL"
    assert source_context.source_context_state == SecCompanyFactsContextState.PARTIAL

    capex_observation = observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE
    ]
    assert capex_observation.state == SecCompanyFactsFundamentalObservationState.MISSING_DATA
    assert capex_observation.missing_source_fields == ("capital_expenditures",)

    cash_generation_observation = observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS
    ]
    assert cash_generation_observation.state == SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED
    assert cash_generation_observation.missing_source_fields == ("capital_expenditures",)

    data_limitation = observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.DATA_LIMITATION
    ]
    assert data_limitation.state == SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED
    assert data_limitation.missing_source_fields == ("capital_expenditures",)


def test_missing_source_context_emits_not_assessed_and_missing_data(tmp_path):
    source_context = _source_context(
        tmp_path,
        _payload({"GrossProfit": [_fact(40, "2025-12-31")]}),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    observations_by_category = _observations_by_category(observation_set)

    assert observation_set.source_context_state == "MISSING"
    assert source_context.source_context_state == SecCompanyFactsContextState.MISSING

    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.SOURCE_CONTEXT_AVAILABILITY
    ].state == SecCompanyFactsFundamentalObservationState.NOT_ASSESSED
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE
    ].state == SecCompanyFactsFundamentalObservationState.MISSING_DATA
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.MISSING_DATA
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.MISSING_DATA
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE
    ].state == SecCompanyFactsFundamentalObservationState.MISSING_DATA
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.DATA_LIMITATION
    ].state == SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED


def test_positive_negative_zero_and_missing_source_values_are_handled(tmp_path):
    source_context = _source_context(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(0, "2025-12-31")],
                "NetIncomeLoss": [_fact(-20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(0, "2025-12-31")],
            }
        ),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    observations_by_category = _observations_by_category(observation_set)

    assert source_context.field_states["revenue"] == SecCompanyFactsContextFieldState.PRESENT
    assert source_context.canonical_fields["revenue"] == 0

    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.NEGATIVE_SOURCE_VALUE
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE
    ].state == SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE
    assert observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE
    ].state == SecCompanyFactsFundamentalObservationState.MISSING_DATA


def test_observations_preserve_source_values_and_provenance(tmp_path):
    source_context = _source_context(
        tmp_path,
        _complete_payload(),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    observations_by_category = _observations_by_category(observation_set)

    revenue_observation = observations_by_category[
        SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE
    ]

    assert revenue_observation.source_values == {"revenue": 100}
    assert revenue_observation.source_references["revenue"] == {
        "sec_tag_selected": "Revenues",
        "provider_name": "SEC_COMPANYFACTS",
        "taxonomy_namespace": "us-gaap",
        "unit": "USD",
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "filing_form": "10-K",
        "filing_date": "2026-02-15",
        "period_start_date": "2025-01-01",
        "period_end_date": "2025-12-31",
        "accession_number": "0000000000-2025-000001",
        "frame": "CY2025",
        "selection_reason": "primary approved tag selected",
        "fallback_alias_used": None,
    }


def test_observations_do_not_emit_derived_calculations(tmp_path):
    source_context = _source_context(
        tmp_path,
        _complete_payload(),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    payload = asdict(observation_set)

    forbidden_derived_fields = {
        "free_cash_flow",
        "fcf",
        "growth",
        "margin",
        "ratio",
        "valuation",
        "yield",
        "capital_intensity",
    }

    assert forbidden_derived_fields.isdisjoint(payload)
    for observation in payload["observations"]:
        assert forbidden_derived_fields.isdisjoint(observation)
        assert all(
            forbidden_term not in observation["message"].lower()
            for forbidden_term in forbidden_derived_fields
        )


def test_observations_do_not_emit_decision_or_delivery_authority(tmp_path):
    source_context = _source_context(
        tmp_path,
        _complete_payload(),
    )

    observation_set = build_sec_companyfacts_fundamental_observations(source_context)
    payload = asdict(observation_set)

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
    }

    assert forbidden_authority_fields.isdisjoint(payload)
    for observation in payload["observations"]:
        assert forbidden_authority_fields.isdisjoint(observation)


def test_observation_persistence_writes_json_without_overwrite(tmp_path):
    source_context = _source_context(
        tmp_path,
        _complete_payload(),
    )
    observation_set = build_sec_companyfacts_fundamental_observations(source_context)

    observation_path = persist_sec_companyfacts_fundamental_observations(
        observation_set,
        run_id="fundamental-observation-run",
        root_dir=tmp_path / "fundamental_observations",
    )

    assert (
        observation_path
        == tmp_path
        / "fundamental_observations"
        / "fundamental-observation-run"
        / "NVDA"
        / "fundamental_observations.json"
    )

    payload = json.loads(observation_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "NVDA"
    assert payload["observation_format_version"] == SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION
    assert payload["source_context_state"] == "AVAILABLE"
    assert payload["observations"][0]["category"] == "SOURCE_CONTEXT_AVAILABILITY"

    try:
        persist_sec_companyfacts_fundamental_observations(
            observation_set,
            run_id="fundamental-observation-run",
            root_dir=tmp_path / "fundamental_observations",
        )
    except FileExistsError as error:
        assert "refusing to overwrite existing SEC CompanyFacts Fundamental Observations" in str(error)
    else:
        raise AssertionError("expected FileExistsError for existing observations output")


def test_observation_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _source_context(tmp_path, raw_payload: dict[str, object]):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=raw_payload,
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        snapshot_id="nvda_companyfacts",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )
    return build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)


def _observations_by_category(observation_set):
    return {
        observation.category: observation
        for observation in observation_set.observations
    }


def _complete_payload() -> dict[str, object]:
    return _payload(
        {
            "Revenues": [_fact(100, "2025-12-31")],
            "NetIncomeLoss": [_fact(20, "2025-12-31")],
            "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
        }
    )


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