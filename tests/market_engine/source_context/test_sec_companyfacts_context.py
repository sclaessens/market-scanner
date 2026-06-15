from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from market_engine.source_context.sec_companyfacts_context import (
    SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION,
    SecCompanyFactsContextBuildError,
    SecCompanyFactsContextFieldState,
    SecCompanyFactsContextState,
    build_sec_companyfacts_source_context_from_snapshot_path,
    persist_sec_companyfacts_source_context,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


def test_builds_available_source_context_from_cached_raw_snapshot(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_complete_payload(),
        ticker="nvda",
        cik="1045810",
        run_id="source-refresh-run",
        snapshot_id="nvda_companyfacts",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)

    assert context.ticker == "NVDA"
    assert context.cik == "0001045810"
    assert context.source_name == "sec_companyfacts"
    assert context.provider_name == "SEC_COMPANYFACTS"
    assert context.context_format_version == SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION
    assert context.source_context_state == SecCompanyFactsContextState.AVAILABLE
    assert context.source_refresh_snapshot_id == "nvda_companyfacts"
    assert context.source_refresh_fetched_at == "2026-06-15T12:00:00Z"
    assert context.source_refresh_snapshot_path == snapshot_path.as_posix()
    assert context.missing_canonical_fields == ()
    assert context.canonical_fields == {
        "revenue": 100,
        "net_income": 20,
        "operating_cash_flow": 30,
        "capital_expenditures": 5,
    }
    assert context.field_states == {
        "revenue": SecCompanyFactsContextFieldState.PRESENT,
        "net_income": SecCompanyFactsContextFieldState.PRESENT,
        "operating_cash_flow": SecCompanyFactsContextFieldState.PRESENT,
        "capital_expenditures": SecCompanyFactsContextFieldState.PRESENT,
    }


def test_partial_source_context_preserves_missing_fields(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            }
        ),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)

    assert context.source_context_state == SecCompanyFactsContextState.PARTIAL
    assert context.canonical_fields["capital_expenditures"] is None
    assert context.field_states["capital_expenditures"] == SecCompanyFactsContextFieldState.MISSING
    assert context.fields["capital_expenditures"].state == SecCompanyFactsContextFieldState.MISSING
    assert context.missing_canonical_fields == ("capital_expenditures",)


def test_missing_source_context_does_not_convert_missing_values_to_zero(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_payload({"GrossProfit": [_fact(40, "2025-12-31")]}),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)

    assert context.source_context_state == SecCompanyFactsContextState.MISSING
    assert context.canonical_fields == {
        "revenue": None,
        "net_income": None,
        "operating_cash_flow": None,
        "capital_expenditures": None,
    }
    assert set(context.missing_canonical_fields) == {
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
    }


def test_zero_value_is_present_not_missing(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_payload({"Revenues": [_fact(0, "2025-12-31")]}),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)

    assert context.source_context_state == SecCompanyFactsContextState.PARTIAL
    assert context.canonical_fields["revenue"] == 0
    assert context.field_states["revenue"] == SecCompanyFactsContextFieldState.PRESENT


def test_source_context_preserves_sec_provenance_and_period_metadata(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_complete_payload(),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        snapshot_id="nvda_companyfacts",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)
    revenue = context.fields["revenue"]

    assert revenue.sec_tag_selected == "Revenues"
    assert revenue.provider_name == "SEC_COMPANYFACTS"
    assert revenue.taxonomy_namespace == "us-gaap"
    assert revenue.unit == "USD"
    assert revenue.raw_value == 100
    assert revenue.fiscal_year == 2025
    assert revenue.fiscal_period == "FY"
    assert revenue.filing_form == "10-K"
    assert revenue.filing_date == "2026-02-15"
    assert revenue.period_start_date == "2025-01-01"
    assert revenue.period_end_date == "2025-12-31"
    assert revenue.accession_number == "0000000000-2025-000001"
    assert revenue.frame == "CY2025"
    assert revenue.selection_reason == "primary approved tag selected"


def test_context_persistence_writes_source_context_json(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_complete_payload(),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )
    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)

    context_path = persist_sec_companyfacts_source_context(
        context,
        run_id="source-context-run",
        root_dir=tmp_path / "source_contexts",
    )

    assert context_path == tmp_path / "source_contexts" / "source-context-run" / "NVDA" / "source_context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "NVDA"
    assert payload["source_context_state"] == "AVAILABLE"
    assert payload["field_states"]["revenue"] == "PRESENT"
    assert payload["fields"]["revenue"]["sec_tag_selected"] == "Revenues"


def test_cached_snapshot_errors_are_wrapped_as_context_build_errors(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(SecCompanyFactsContextBuildError):
        build_sec_companyfacts_source_context_from_snapshot_path(missing_path)


def test_entity_mismatch_is_controlled_context_build_error(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_complete_payload(),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    with pytest.raises(SecCompanyFactsContextBuildError):
        build_sec_companyfacts_source_context_from_snapshot_path(
            snapshot_path,
            expected_ticker="AMD",
        )


def test_source_context_does_not_emit_analysis_or_decision_authority(tmp_path):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=_complete_payload(),
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )

    context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)
    payload = asdict(context)

    forbidden_fields = {
        "BUY",
        "SELL",
        "HOLD",
        "recommendation",
        "allocation",
        "ranking",
        "score",
        "conviction",
        "urgency",
        "tradeability",
        "position_sizing",
        "execution",
        "free_cash_flow",
        "growth",
        "margin",
        "observation",
        "analysis",
        "telegram",
        "delivery",
    }
    assert forbidden_fields.isdisjoint(payload)
    for field_payload in payload["fields"].values():
        assert forbidden_fields.isdisjoint(field_payload)


def test_source_context_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


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
