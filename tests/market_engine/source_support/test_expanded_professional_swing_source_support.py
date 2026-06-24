from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from market_engine.candidate_classification import (
    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_provider_error,
    persist_sec_companyfacts_raw_snapshot,
)
from market_engine.source_support import (
    EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
    ExpandedProfessionalSwingSourceSupportError,
    ProfessionalSwingSourceSupportStatus,
    classify_expanded_professional_swing_universe_source_support,
    expanded_to_plain_dict,
)
from market_engine.ticker_universe import build_professional_swing_universe_expansion


HEADER = (
    "ticker,name,market,asset_type,active,universe_status,source_policy_hint,"
    "operator_priority,swing_profile,liquidity_profile,volatility_profile,"
    "market_cap_profile,theme,sector,notes"
)


def test_expanded_candidate_with_cached_source_is_supported_cached(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("NVDA", priority=1)], [_candidate("AMD", name="AMD")])
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr06-fixtures",
        ticker="AMD",
        cik="0000002488",
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-24T12:00:00+00:00",
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=source_root,
    )

    assert result.format_version == EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION
    amd = _entry_by_ticker(result.entries, "AMD")
    assert amd.status == ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    assert amd.asset_type == "equity"
    assert amd.universe_entry_origin == "expansion_candidate"
    assert amd.source_candidate_reference == "operator_report:AMD"
    assert amd.source_support.source_artifacts[0].ticker == "AMD"


def test_expanded_candidate_without_snapshot_is_missing_snapshot(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("NVDA", priority=1)], [_candidate("VRT", name="Vertiv")])

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )

    vrt = _entry_by_ticker(result.entries, "VRT")
    assert vrt.status == ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT.value
    assert vrt.universe_entry_origin == "expansion_candidate"
    assert vrt.source_support.source_artifacts == ()


def test_manual_review_only_existing_universe_entry_remains_manual_review_only(tmp_path: Path) -> None:
    expansion = _build_expansion(
        tmp_path,
        [
            _row(
                "MANUAL",
                name="Manual Review",
                priority=1,
                universe_status="manual_review_only",
                source_policy_hint="manual_review_only",
            )
        ],
        [],
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )

    manual = result.entries[0]
    assert manual.status == ProfessionalSwingSourceSupportStatus.MANUAL_REVIEW_ONLY.value
    assert manual.universe_entry_origin == "existing_universe"
    assert manual.source_support.provider_errors == ()


def test_unsupported_sec_companyfacts_state_remains_unsupported(tmp_path: Path) -> None:
    expansion = _build_expansion(
        tmp_path,
        [_row("NOPE", name="Unsupported", priority=1)],
        [],
    )
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_provider_error(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr06-fixtures",
        ticker="NOPE",
        cik=None,
        error_type="UnsupportedTickerError",
        error_message="no approved SEC CompanyFacts CIK",
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=source_root,
    )

    nope = result.entries[0]
    assert nope.status == ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS.value
    assert nope.source_support.provider_errors[0].error_type == "UnsupportedTickerError"


def test_malformed_candidate_source_artifact_is_explicit(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("NVDA", priority=1)], [_candidate("BAD", name="Bad")])
    malformed_path = (
        _source_root(tmp_path)
        / "sec_companyfacts"
        / "me-sr06-fixtures"
        / "raw"
        / "BAD_companyfacts.json"
    )
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text("{not json", encoding="utf-8")

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )

    bad = _entry_by_ticker(result.entries, "BAD")
    assert bad.status == ProfessionalSwingSourceSupportStatus.MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT.value
    assert bad.source_support.source_artifacts[0].error_type == "SecCompanyFactsSnapshotJsonError"


def test_existing_universe_entries_remain_classifiable(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("MSFT", priority=1)], [])
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr06-fixtures",
        ticker="MSFT",
        cik="0000789019",
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-24T12:00:00+00:00",
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=source_root,
    )

    msft = result.entries[0]
    assert msft.status == ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    assert msft.universe_entry_origin == "existing_universe"
    assert msft.universe_entry_provenance["row_number"] == 2


def test_newly_included_me_uni09_candidate_entries_remain_classifiable(tmp_path: Path) -> None:
    expansion = _build_expansion(
        tmp_path,
        [_row("NVDA", priority=1)],
        [_candidate("AVGO", name="Broadcom"), _candidate("VRT", name="Vertiv")],
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )

    candidate_entries = [entry for entry in result.entries if entry.universe_entry_origin == "expansion_candidate"]
    assert [entry.ticker for entry in candidate_entries] == ["AVGO", "VRT"]
    assert {entry.status for entry in candidate_entries} == {
        ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT.value
    }


def test_deterministic_ordering_follows_expanded_universe_order(tmp_path: Path) -> None:
    expansion = _build_expansion(
        tmp_path,
        [_row("NVDA", priority=1), _row("MSFT", priority=2)],
        [_candidate("VRT", name="Vertiv"), _candidate("AMD", name="AMD")],
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )

    assert [entry.ticker for entry in result.entries] == ["NVDA", "MSFT", "AMD", "VRT"]


def test_summary_counts_are_explicit(tmp_path: Path) -> None:
    expansion = _build_expansion(
        tmp_path,
        [
            _row("MSFT", priority=1),
            _row(
                "MANUAL",
                name="Manual",
                priority=2,
                universe_status="manual_review_only",
                source_policy_hint="manual_review_only",
            ),
            _row("NOPE", name="Unsupported", priority=3),
        ],
        [_candidate("AMD", name="AMD"), _candidate("VRT", name="Vertiv")],
    )
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr06-fixtures",
        ticker="AMD",
        cik="0000002488",
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-24T12:00:00+00:00",
    )
    persist_sec_companyfacts_provider_error(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr06-fixtures",
        ticker="NOPE",
        cik=None,
        error_type="UnsupportedTickerError",
        error_message="no approved SEC CompanyFacts CIK",
    )

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=source_root,
    )

    assert result.summary_counts == {
        "total_expanded_universe_entries": 5,
        "supported_cached": 1,
        "missing_snapshot": 2,
        "unsupported_sec_companyfacts": 1,
        "missing_required_source_field": 0,
        "malformed_or_unreadable_source_artifact": 0,
        "ambiguous_identity": 0,
        "manual_review_only": 1,
        "excluded": 0,
        "blocked_unsupported_or_manual_review_total": 4,
    }


def test_unsupported_input_contract_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ExpandedProfessionalSwingSourceSupportError):
        classify_expanded_professional_swing_universe_source_support(
            expansion_result={"format_version": "unsupported"},
            source_snapshot_root=_source_root(tmp_path),
        )


def test_duplicate_ticker_market_in_expanded_output_fails_closed(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("NVDA", priority=1)], [_candidate("AMD", name="AMD")])
    duplicate_expansion = replace(
        expansion,
        final_universe_entries=expansion.final_universe_entries + (expansion.final_universe_entries[-1],),
    )

    with pytest.raises(ExpandedProfessionalSwingSourceSupportError, match="Duplicate"):
        classify_expanded_professional_swing_universe_source_support(
            expansion_result=duplicate_expansion,
            source_snapshot_root=_source_root(tmp_path),
        )


def test_expanded_source_support_has_no_provider_network_or_action_dependencies() -> None:
    import market_engine.source_support.expanded_professional_swing as expanded

    module_names = set(expanded.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names


def test_normal_output_has_no_forbidden_action_authority_language(tmp_path: Path) -> None:
    expansion = _build_expansion(tmp_path, [_row("NVDA", priority=1)], [_candidate("AMD", name="AMD")])

    result = classify_expanded_professional_swing_universe_source_support(
        expansion_result=expansion,
        source_snapshot_root=_source_root(tmp_path),
    )
    output = json.dumps(expanded_to_plain_dict(result), sort_keys=True).lower()

    assert "buy" not in output
    assert "sell" not in output
    assert "target price" not in output
    assert "allocation" not in output
    assert "position size" not in output
    assert "ranking" not in output
    assert "scoring" not in output
    assert "conviction" not in output
    assert "urgency" not in output
    assert "tradeability" not in output


def _build_expansion(
    tmp_path: Path,
    universe_rows: list[str],
    candidates: list[dict[str, object]],
):
    return build_professional_swing_universe_expansion(
        existing_universe_path=_write_universe(tmp_path, universe_rows),
        candidate_classification_path=_write_candidate_summary(tmp_path, candidates),
    )


def _source_root(tmp_path: Path) -> Path:
    return tmp_path / "source_snapshots"


def _write_universe(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_candidate_summary(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "candidate_classification_summary.json"
    path.write_text(
        json.dumps(
            {
                "candidate_classification_format_version": (
                    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION
                ),
                "candidate_classification_run_id": "candidate-fixture",
                "per_ticker_classifications": records,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _candidate(ticker: str, *, name: str | None = None) -> dict[str, object]:
    return {
        "ticker": ticker,
        "source_candidate_id": f"candidate:{ticker}",
        "candidate_bucket": "ready_for_manual_candidate_review",
        "candidate_rationale": "Local readable output has coherent review-only context.",
        "evidence_references": [
            {
                "reference_type": "operator_report",
                "reference": f"operator_report:{ticker}",
            }
        ],
        "blocking_reasons": [],
        "safety_flags": {
            "actionable_language_detected": False,
            "unsupported_input_detected": False,
            "malformed_input_detected": False,
            "stale_data_detected": False,
            "blocked_state_detected": False,
        },
        "proposed_universe_entry": {
            "ticker": ticker,
            "name": name or f"{ticker} Example",
            "market": "USA",
            "asset_type": "equity",
            "active": "true",
            "universe_status": "candidate",
            "source_policy_hint": "cached_source_candidate",
            "operator_priority": "",
            "swing_profile": "unknown",
            "liquidity_profile": "unknown",
            "volatility_profile": "unknown",
            "market_cap_profile": "unknown",
            "theme": "candidate_classification",
            "sector": "unknown",
            "notes": "non-actionable candidate classification source",
        },
    }


def _row(
    ticker: str,
    *,
    name: str | None = None,
    priority: int,
    universe_status: str = "candidate",
    source_policy_hint: str = "cached_source_candidate",
) -> str:
    return (
        f"{ticker},{name or ticker},USA,equity,true,{universe_status},{source_policy_hint},"
        f"{priority},trend_continuation,high,high,mega_cap,theme,technology,"
    )


def _entry_by_ticker(entries, ticker: str):
    return next(entry for entry in entries if entry.ticker == ticker)


def _companyfacts_payload(
    *,
    revenue: int = 100,
    net_income: int = 20,
    operating_cash_flow: int = 30,
    capital_expenditures: int = 5,
) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(revenue)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(net_income)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(operating_cash_flow)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(capital_expenditures)]}
                },
            }
        }
    }


def _fact(value: int) -> dict[str, object]:
    return {
        "val": value,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "accn": "0000000000-25-000001",
        "frame": "CY2025",
    }
