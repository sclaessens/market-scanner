from __future__ import annotations

from pathlib import Path

import pytest

from market_engine.ticker_universe import (
    CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION,
    CanonicalTickerUniverseValidationError,
    load_canonical_ticker_universe,
)


HEADER = (
    "ticker,name,market,asset_type,active,priority,source_policy,"
    "portfolio_relevant,telegram_preview_eligible,telegram_delivery_eligible,notes"
)


def _write_csv(tmp_path: Path, body: str, *, header: str = HEADER) -> Path:
    path = tmp_path / "ticker_universe.csv"
    path.write_text(header + "\n" + body, encoding="utf-8")
    return path


def test_valid_minimal_csv_loads_active_cached_source_rows(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "NVDA,NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,\n",
    )

    result = load_canonical_ticker_universe(path)

    assert result.contract_version == CANONICAL_TICKER_UNIVERSE_CONTRACT_VERSION
    assert result.loaded_row_count == 1
    assert result.selected_row_count == 1
    assert result.entries[0].ticker == "NVDA"
    assert result.entries[0].validation_state == "valid"


def test_valid_csv_preserves_optional_metadata(tmp_path: Path) -> None:
    header = HEADER + ",sector,theme,provider_ticker"
    path = _write_csv(
        tmp_path,
        "MSFT,Microsoft,USA,equity,true,1,cached_source_required,true,true,false,"
        "notes,Technology,Cloud,MSFT\n",
        header=header,
    )

    result = load_canonical_ticker_universe(path)

    assert result.entries[0].metadata == {
        "sector": "Technology",
        "theme": "Cloud",
        "provider_ticker": "MSFT",
    }


def test_default_selection_excludes_inactive_blocked_and_manual_review_rows(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "NVDA,NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,",
                "MSFT,Microsoft,USA,equity,false,2,cached_source_only,true,true,false,",
                "COST,Costco,USA,equity,true,3,manual_review_only,true,true,false,",
                "AMD,AMD,USA,equity,true,4,blocked,true,true,false,",
            ]
        ),
    )

    result = load_canonical_ticker_universe(path)

    assert [entry.ticker for entry in result.entries] == ["NVDA"]
    assert result.loaded_row_count == 4
    assert result.excluded_inactive_count == 1
    assert result.excluded_manual_review_only_count == 1
    assert result.excluded_blocked_count == 1


def test_include_inactive_returns_all_valid_rows_with_deterministic_order(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "msft,Microsoft,usa,equity,false,2,cached_source_only,true,true,false,",
                " nvda ,NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,",
                "amd,AMD,USA,equity,true,1,cached_source_required,true,true,false,",
                "cost,Costco,USA,equity,true,3,manual_review_only,true,true,false,",
            ]
        ),
    )

    result = load_canonical_ticker_universe(path, include_inactive=True)

    assert [entry.ticker for entry in result.entries] == ["AMD", "NVDA", "MSFT", "COST"]
    assert [entry.priority for entry in result.entries] == [1, 1, 2, 3]
    assert result.include_inactive is True


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(CanonicalTickerUniverseValidationError, match="not found"):
        load_canonical_ticker_universe(tmp_path / "missing.csv")


def test_missing_required_column_fails_closed(tmp_path: Path) -> None:
    header = HEADER.replace("ticker,", "")
    path = _write_csv(
        tmp_path,
        "NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,\n",
        header=header,
    )

    with pytest.raises(CanonicalTickerUniverseValidationError, match="ticker"):
        load_canonical_ticker_universe(path)


def test_empty_ticker_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        ",NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,\n",
    )

    with pytest.raises(CanonicalTickerUniverseValidationError, match="ticker"):
        load_canonical_ticker_universe(path)


def test_duplicate_ticker_market_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "\n".join(
            [
                "NVDA,NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,",
                "nvda,NVIDIA Duplicate,USA,equity,true,2,cached_source_required,true,true,false,",
            ]
        ),
    )

    with pytest.raises(CanonicalTickerUniverseValidationError, match="duplicate"):
        load_canonical_ticker_universe(path)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("active", "yes"),
        ("portfolio_relevant", "1"),
        ("telegram_preview_eligible", "maybe"),
        ("telegram_delivery_eligible", "send"),
    ],
)
def test_invalid_boolean_fields_fail_closed(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
) -> None:
    values = {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "market": "USA",
        "asset_type": "equity",
        "active": "true",
        "priority": "1",
        "source_policy": "cached_source_only",
        "portfolio_relevant": "true",
        "telegram_preview_eligible": "true",
        "telegram_delivery_eligible": "false",
        "notes": "",
    }
    values[field_name] = bad_value
    path = _write_csv(tmp_path, ",".join(values[column] for column in HEADER.split(",")))

    with pytest.raises(CanonicalTickerUniverseValidationError, match=field_name):
        load_canonical_ticker_universe(path)


@pytest.mark.parametrize("bad_priority", ["0", "-1", "high", "1.5", ""])
def test_invalid_priority_fails_closed(tmp_path: Path, bad_priority: str) -> None:
    path = _write_csv(
        tmp_path,
        f"NVDA,NVIDIA,USA,equity,true,{bad_priority},cached_source_only,true,true,false,\n",
    )

    with pytest.raises(CanonicalTickerUniverseValidationError, match="priority"):
        load_canonical_ticker_universe(path)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("market", "NASDAQ"),
        ("asset_type", "stock"),
        ("source_policy", "live_refresh"),
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
        "priority": "1",
        "source_policy": "cached_source_only",
        "portfolio_relevant": "true",
        "telegram_preview_eligible": "true",
        "telegram_delivery_eligible": "false",
        "notes": "",
    }
    values[column] = bad_value
    path = _write_csv(tmp_path, ",".join(values[name] for name in HEADER.split(",")))

    with pytest.raises(CanonicalTickerUniverseValidationError, match=column):
        load_canonical_ticker_universe(path)


def test_invalid_ticker_format_fails_closed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "NV DA,NVIDIA,USA,equity,true,1,cached_source_only,true,true,false,\n",
    )

    with pytest.raises(CanonicalTickerUniverseValidationError, match="ticker"):
        load_canonical_ticker_universe(path)


def test_delivery_eligibility_cannot_override_preview_ineligibility(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path,
        "NVDA,NVIDIA,USA,equity,true,1,cached_source_only,true,false,true,\n",
    )

    with pytest.raises(
        CanonicalTickerUniverseValidationError,
        match="delivery eligibility",
    ):
        load_canonical_ticker_universe(path)


def test_no_provider_network_or_decision_engine_dependencies_are_imported() -> None:
    import market_engine.ticker_universe.canonical as canonical

    module_names = set(canonical.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names
