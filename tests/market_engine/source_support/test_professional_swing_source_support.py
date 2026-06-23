from __future__ import annotations

from pathlib import Path

import pytest

from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_provider_error,
    persist_sec_companyfacts_raw_snapshot,
)
from market_engine.source_support import (
    PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
    ProfessionalSwingSourceSupportError,
    ProfessionalSwingSourceSupportStatus,
    classify_professional_swing_universe_source_support,
)


HEADER = (
    "ticker,name,market,asset_type,active,universe_status,source_policy_hint,"
    "operator_priority,swing_profile,liquidity_profile,volatility_profile,"
    "market_cap_profile,theme,sector,notes"
)


def test_fully_supported_ticker_is_classified_supported_cached(tmp_path: Path) -> None:
    universe_path = _write_universe(
        tmp_path,
        [
            _row("NVDA", priority=1),
        ],
    )
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr05-fixtures",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=source_root,
    )

    assert result.format_version == PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION
    assert result.supported_count == 1
    assert result.entries[0].ticker == "NVDA"
    assert result.entries[0].status == ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    assert result.entries[0].missing_required_source_fields == ()
    assert [field.canonical_field for field in result.entries[0].required_source_fields] == [
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
    ]
    assert result.entries[0].source_artifacts[0].source_name == "sec_companyfacts"
    assert result.entries[0].source_artifacts[0].ticker == "NVDA"
    assert result.entries[0].universe_entry_reference["row_number"] == 2


def test_unsupported_ticker_uses_local_provider_error_without_provider_call(tmp_path: Path) -> None:
    universe_path = _write_universe(
        tmp_path,
        [_row("NOPE", name="Unsupported", priority=1)],
    )
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_provider_error(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr05-fixtures",
        ticker="NOPE",
        cik=None,
        error_type="UnsupportedTickerError",
        error_message="no approved SEC CompanyFacts CIK",
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=source_root,
    )

    entry = result.entries[0]
    assert entry.status == ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS.value
    assert entry.provider_errors[0].error_type == "UnsupportedTickerError"
    assert "provider_errors.csv" in entry.provider_errors[0].source_path


def test_missing_snapshot_is_explicit(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("MSFT", priority=1)])

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=_source_root(tmp_path),
    )

    assert result.missing_snapshot_count == 1
    assert result.entries[0].status == ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT.value
    assert result.entries[0].source_artifacts == ()


def test_missing_required_source_field_is_explicit(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("AMD", priority=1)])
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr05-fixtures",
        ticker="AMD",
        cik="0000002488",
        raw_payload=_companyfacts_payload(omit=("capital_expenditures",)),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=source_root,
    )

    entry = result.entries[0]
    assert entry.status == ProfessionalSwingSourceSupportStatus.MISSING_REQUIRED_SOURCE_FIELD.value
    assert entry.missing_required_source_fields == ("capital_expenditures",)
    assert next(
        field
        for field in entry.required_source_fields
        if field.canonical_field == "capital_expenditures"
    ).present is False


def test_malformed_artifact_fails_closed(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("BAD", priority=1)])
    malformed_path = (
        _source_root(tmp_path)
        / "sec_companyfacts"
        / "me-sr05-fixtures"
        / "raw"
        / "BAD_companyfacts.json"
    )
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text("{not json", encoding="utf-8")

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=_source_root(tmp_path),
    )

    entry = result.entries[0]
    assert entry.status == ProfessionalSwingSourceSupportStatus.MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT.value
    assert entry.source_artifacts[0].error_type == "SecCompanyFactsSnapshotJsonError"


def test_numeric_zero_is_preserved_as_supported_evidence(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("ZERO", priority=1)])
    source_root = _source_root(tmp_path)
    persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr05-fixtures",
        ticker="ZERO",
        cik="0000000001",
        raw_payload=_companyfacts_payload(
            revenue=0,
            net_income=0,
            operating_cash_flow=0,
            capital_expenditures=0,
        ),
        fetched_at="2026-06-15T12:00:00+00:00",
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=source_root,
    )

    entry = result.entries[0]
    assert entry.status == ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    assert entry.numeric_zero_evidence_present is True
    assert {field.source_value for field in entry.required_source_fields} == {0}
    assert entry.missing_required_source_fields == ()


def test_manual_review_and_excluded_rows_are_not_treated_as_missing_snapshots(
    tmp_path: Path,
) -> None:
    universe_path = _write_universe(
        tmp_path,
        [
            _row(
                "MANUAL",
                name="Manual",
                priority=1,
                universe_status="manual_review_only",
                source_policy_hint="manual_review_only",
            ),
            _row(
                "BLOCK",
                name="Blocked",
                priority=2,
                universe_status="blocked",
                source_policy_hint="unsupported",
            ),
        ],
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=_source_root(tmp_path),
    )

    assert [entry.status for entry in result.entries] == [
        ProfessionalSwingSourceSupportStatus.MANUAL_REVIEW_ONLY.value,
        ProfessionalSwingSourceSupportStatus.EXCLUDED.value,
    ]
    assert result.missing_snapshot_count == 0


def test_ambiguous_identity_is_explicit(tmp_path: Path) -> None:
    universe_path = _write_universe(tmp_path, [_row("DUP", priority=1)])
    source_root = _source_root(tmp_path)
    for run_id, cik in (("run-a", "0000000001"), ("run-b", "0000000002")):
        persist_sec_companyfacts_raw_snapshot(
            root_dir=source_root / "sec_companyfacts",
            run_id=run_id,
            ticker="DUP",
            cik=cik,
            raw_payload=_companyfacts_payload(),
            fetched_at="2026-06-15T12:00:00+00:00",
        )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=source_root,
    )

    assert result.entries[0].status == ProfessionalSwingSourceSupportStatus.AMBIGUOUS_IDENTITY.value
    assert len(result.entries[0].source_artifacts) == 2


def test_output_order_is_deterministic(tmp_path: Path) -> None:
    universe_path = _write_universe(
        tmp_path,
        [
            _row("MSFT", priority=2),
            _row("AMD", priority=1),
            _row("NVDA", priority=1),
        ],
    )

    result = classify_professional_swing_universe_source_support(
        universe_path=universe_path,
        source_snapshot_root=_source_root(tmp_path),
    )

    assert [entry.ticker for entry in result.entries] == ["AMD", "NVDA", "MSFT"]


def test_invalid_universe_fails_closed(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("ticker\nNVDA\n", encoding="utf-8")

    with pytest.raises(ProfessionalSwingSourceSupportError):
        classify_professional_swing_universe_source_support(
            universe_path=bad_path,
            source_snapshot_root=_source_root(tmp_path),
        )


def test_source_support_classifier_has_no_provider_network_or_legacy_dependencies() -> None:
    import market_engine.source_support.professional_swing as professional_swing

    module_names = set(professional_swing.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names


def _source_root(tmp_path: Path) -> Path:
    return tmp_path / "source_snapshots"


def _write_universe(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


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


def _companyfacts_payload(
    *,
    revenue: int = 100,
    net_income: int = 20,
    operating_cash_flow: int = 30,
    capital_expenditures: int = 5,
    omit: tuple[str, ...] = (),
) -> dict[str, object]:
    facts: dict[str, object] = {}
    if "revenue" not in omit:
        facts["Revenues"] = {"units": {"USD": [_fact(revenue)]}}
    if "net_income" not in omit:
        facts["NetIncomeLoss"] = {"units": {"USD": [_fact(net_income)]}}
    if "operating_cash_flow" not in omit:
        facts["NetCashProvidedByUsedInOperatingActivities"] = {
            "units": {"USD": [_fact(operating_cash_flow)]}
        }
    if "capital_expenditures" not in omit:
        facts["PaymentsToAcquirePropertyPlantAndEquipment"] = {
            "units": {"USD": [_fact(capital_expenditures)]}
        }
    return {"facts": {"us-gaap": facts}}


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
