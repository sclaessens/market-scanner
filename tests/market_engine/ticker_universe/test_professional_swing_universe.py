from __future__ import annotations

from pathlib import Path

import pytest

from market_engine.ticker_universe import (
    EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
    PROFESSIONAL_SWING_UNIVERSE_PATH,
    ProfessionalSwingUniverseValidationError,
    load_professional_swing_universe,
)


HEADER = (
    "ticker,name,market,asset_type,active,universe_status,source_policy_hint,"
    "operator_priority,swing_profile,liquidity_profile,volatility_profile,"
    "market_cap_profile,theme,sector,notes"
)


def _write_csv(tmp_path: Path, body: str, *, header: str = HEADER) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(header + "\n" + body, encoding="utf-8")
    return path


def test_valid_minimal_professional_swing_csv_loads_default_candidate_rows(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,"
        "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
    )

    result = load_professional_swing_universe(path)

    assert result.contract_version == EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION
    assert result.loaded_row_count == 1
    assert result.selected_row_count == 1
    assert result.entries[0].ticker == "NVDA"
    assert result.entries[0].validation_state == "valid"


def test_valid_professional_swing_csv_preserves_optional_metadata(
    tmp_path: Path,
) -> None:
    header = HEADER + ",exchange,currency,provider_ticker,source_support_status"
    path = _write_csv(
        tmp_path,
        "MSFT,Microsoft,USA,equity,true,candidate,unknown,1,quality_compounder,"
        "high,low,mega_cap,cloud_ai,technology,notes,NASDAQ,USD,MSFT,unknown\n",
        header=header,
    )

    result = load_professional_swing_universe(path)

    assert result.entries[0].metadata == {
        "exchange": "NASDAQ",
        "currency": "USD",
        "provider_ticker": "MSFT",
        "source_support_status": "unknown",
    }


def test_default_selection_excludes_inactive_manual_source_mapping_blocked_and_rejected_rows(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,trend_continuation,high,high,mega_cap,ai_infrastructure,technology,",
                "MSFT,Microsoft,USA,equity,false,candidate,cached_source_candidate,2,quality_compounder,high,low,mega_cap,cloud_ai,technology,",
                "ASML,ASML Holding,EURONEXT,equity,true,needs_source_mapping,source_mapping_required,3,quality_compounder,high,medium,large_cap,semiconductor_equipment,technology,",
                "COST,Costco,USA,equity,true,manual_review_only,manual_review_only,4,quality_compounder,high,low,mega_cap,quality_compounder,consumer_defensive,",
                "AMD,AMD,USA,equity,true,blocked,unsupported,5,relative_strength,high,high,large_cap,ai_compute,technology,",
                "TSLA,Tesla,USA,equity,true,rejected,cached_source_candidate,6,relative_strength,high,extreme,mega_cap,ev_autonomy,consumer_cyclical,",
            ]
        ),
    )

    result = load_professional_swing_universe(path)

    assert [entry.ticker for entry in result.entries] == ["NVDA"]
    assert result.loaded_row_count == 6
    assert result.excluded_inactive_count == 1
    assert result.excluded_universe_status_count == 4
    assert result.excluded_source_policy_hint_count == 3


def test_include_inactive_returns_all_valid_rows_with_deterministic_order(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "msft,Microsoft,usa,equity,false,candidate,cached_source_candidate,2,quality_compounder,high,low,mega_cap,cloud_ai,technology,",
                " nvda ,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,trend_continuation,high,high,mega_cap,ai_infrastructure,technology,",
                "amd,AMD,USA,equity,true,watching,unknown,1,relative_strength,high,high,large_cap,ai_compute,technology,",
                "cost,Costco,USA,equity,true,manual_review_only,manual_review_only,3,quality_compounder,high,low,mega_cap,quality_compounder,consumer_defensive,",
            ]
        ),
    )

    result = load_professional_swing_universe(path, include_inactive=True)

    assert [entry.ticker for entry in result.entries] == ["AMD", "NVDA", "MSFT", "COST"]
    assert [entry.operator_priority for entry in result.entries] == [1, 1, 2, 3]
    assert result.include_inactive is True


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ProfessionalSwingUniverseValidationError, match="not found"):
        load_professional_swing_universe(tmp_path / "missing.csv")


def test_missing_required_column_fails_closed(tmp_path: Path) -> None:
    header = HEADER.replace("ticker,", "")
    path = _write_csv(
        tmp_path,
        "NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
        header=header,
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="ticker"):
        load_professional_swing_universe(path)


def test_empty_required_value_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,"
        "trend_continuation,high,high,mega_cap,,technology,\n",
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="theme"):
        load_professional_swing_universe(path)


def test_notes_column_may_be_blank(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,"
        "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
    )

    result = load_professional_swing_universe(path)

    assert result.entries[0].notes == ""


def test_duplicate_ticker_market_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,trend_continuation,high,high,mega_cap,ai_infrastructure,technology,",
                "nvda,NVIDIA Duplicate,USA,equity,true,watching,cached_source_candidate,2,relative_strength,high,high,mega_cap,ai_infrastructure,technology,",
            ]
        ),
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="duplicate"):
        load_professional_swing_universe(path)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("market", "NASDAQ"),
        ("asset_type", "stock"),
        ("universe_status", "approved"),
        ("source_policy_hint", "live_refresh"),
        ("swing_profile", "buy_the_dip"),
        ("liquidity_profile", "thin"),
        ("volatility_profile", "wild"),
        ("market_cap_profile", "giant"),
    ],
)
def test_invalid_allowed_value_fields_fail_closed(
    tmp_path: Path,
    column: str,
    bad_value: str,
) -> None:
    values = {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "market": "USA",
        "asset_type": "equity",
        "active": "true",
        "universe_status": "candidate",
        "source_policy_hint": "cached_source_candidate",
        "operator_priority": "1",
        "swing_profile": "trend_continuation",
        "liquidity_profile": "high",
        "volatility_profile": "high",
        "market_cap_profile": "mega_cap",
        "theme": "ai_infrastructure",
        "sector": "technology",
        "notes": "",
    }
    values[column] = bad_value
    path = _write_csv(tmp_path, ",".join(values[name] for name in HEADER.split(",")))

    with pytest.raises(ProfessionalSwingUniverseValidationError, match=column):
        load_professional_swing_universe(path)


@pytest.mark.parametrize("bad_active", ["yes", "1", "maybe", ""])
def test_invalid_active_field_fails_closed(tmp_path: Path, bad_active: str) -> None:
    path = _write_csv(
        tmp_path,
        f"NVDA,NVIDIA,USA,equity,{bad_active},candidate,cached_source_candidate,1,"
        "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="active"):
        load_professional_swing_universe(path)


@pytest.mark.parametrize("bad_priority", ["0", "-1", "high", "1.5", ""])
def test_invalid_operator_priority_fails_closed(tmp_path: Path, bad_priority: str) -> None:
    path = _write_csv(
        tmp_path,
        f"NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,{bad_priority},"
        "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="operator_priority"):
        load_professional_swing_universe(path)


def test_invalid_ticker_format_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "NV DA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,"
        "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,\n",
    )

    with pytest.raises(ProfessionalSwingUniverseValidationError, match="ticker"):
        load_professional_swing_universe(path)


def test_no_provider_network_or_decision_engine_dependencies_are_imported() -> None:
    import market_engine.ticker_universe.professional_swing as professional_swing

    module_names = set(professional_swing.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names


def test_current_professional_swing_universe_loads_and_excludes_mapping_and_manual_rows() -> None:
    selected = load_professional_swing_universe(PROFESSIONAL_SWING_UNIVERSE_PATH)
    all_rows = load_professional_swing_universe(
        PROFESSIONAL_SWING_UNIVERSE_PATH,
        include_inactive=True,
    )

    assert all_rows.loaded_row_count == 53
    assert selected.selected_row_count == 45
    assert [entry.ticker for entry in selected.entries[:2]] == ["NVDA", "AMD"]
    assert "ASML" not in [entry.ticker for entry in selected.entries]
    assert "MSTR" not in [entry.ticker for entry in selected.entries]
    assert "HO" not in [entry.ticker for entry in selected.entries]

    asml_entry = next(entry for entry in all_rows.entries if entry.ticker == "ASML")
    assert asml_entry.universe_status == "needs_source_mapping"
    assert asml_entry.source_policy_hint == "source_mapping_required"

    ho_entry = next(entry for entry in all_rows.entries if entry.ticker == "HO")
    assert ho_entry.universe_status == "manual_review_only"
    assert ho_entry.source_policy_hint == "manual_review_only"
