from __future__ import annotations

from pathlib import Path

import pytest

from market_engine.run.editable_universe_runtime_input import (
    EDITABLE_UNIVERSE_RUNTIME_INPUT_FORMAT_VERSION,
    EditableUniverseRuntimeInputError,
    build_cached_source_batch_argv_from_professional_swing_universe,
    build_professional_swing_runtime_input,
    selected_tickers_from_runtime_input,
)
from market_engine.ticker_universe import (
    EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
)


def test_professional_swing_universe_builds_local_runtime_input(tmp_path: Path) -> None:
    path = _write_professional_swing_universe(tmp_path)

    runtime_input = build_professional_swing_runtime_input(path)

    assert runtime_input.format_version == EDITABLE_UNIVERSE_RUNTIME_INPUT_FORMAT_VERSION
    assert runtime_input.source_contract_version == (
        EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION
    )
    assert runtime_input.requested_tickers == ("AMD", "NVDA", "VRT")
    assert runtime_input.loaded_row_count == 5
    assert runtime_input.selected_row_count == 3
    assert runtime_input.excluded_source_mapping_required_tickers == ("ASML",)
    assert runtime_input.excluded_manual_review_only_tickers == ("HO",)
    assert runtime_input.source_policy_hint_authority == (
        "operator_hint_not_source_support_authority"
    )
    assert runtime_input.canonical_promotion_authority is False
    assert runtime_input.provider_call_authority is False


def test_runtime_input_payload_is_serializable_and_preserves_boundaries(tmp_path: Path) -> None:
    runtime_input = build_professional_swing_runtime_input(
        _write_professional_swing_universe(tmp_path)
    )

    payload = runtime_input.to_payload()

    assert payload["requested_tickers"] == ("AMD", "NVDA", "VRT")
    assert payload["runtime_input_authority"] == (
        "local_cached_source_batch_requested_tickers_only"
    )
    assert payload["provider_call_authority"] is False
    assert payload["canonical_promotion_authority"] is False


def test_professional_swing_universe_builds_cached_source_batch_argv(
    tmp_path: Path,
) -> None:
    path = _write_professional_swing_universe(tmp_path)

    argv = build_cached_source_batch_argv_from_professional_swing_universe(
        path=path,
        source_snapshot_root="data/market_engine/source_snapshots",
        portfolio_context="data/market_engine/portfolio_contexts/local_portfolio_context.json",
        batch_id="me-uni07-test",
        generated_at="2026-06-22T13:00:00Z",
        ticker_limit=2,
        write_local_artifacts=True,
        artifact_output_root="artifacts/market_engine",
        emit_json=True,
    )

    assert argv == (
        "--tickers",
        "AMD,NVDA,VRT",
        "--source-snapshot-root",
        "data/market_engine/source_snapshots",
        "--portfolio-context",
        "data/market_engine/portfolio_contexts/local_portfolio_context.json",
        "--batch-id",
        "me-uni07-test",
        "--generated-at",
        "2026-06-22T13:00:00Z",
        "--ticker-limit",
        "2",
        "--write-local-artifacts",
        "--artifact-output-root",
        "artifacts/market_engine",
        "--emit-json",
    )


def test_selected_tickers_from_runtime_input_applies_positive_limit(tmp_path: Path) -> None:
    runtime_input = build_professional_swing_runtime_input(
        _write_professional_swing_universe(tmp_path)
    )

    assert selected_tickers_from_runtime_input(runtime_input, ticker_limit=2) == (
        "AMD",
        "NVDA",
    )


def test_selected_tickers_from_runtime_input_rejects_zero_limit(tmp_path: Path) -> None:
    runtime_input = build_professional_swing_runtime_input(
        _write_professional_swing_universe(tmp_path)
    )

    with pytest.raises(EditableUniverseRuntimeInputError, match="ticker_limit"):
        selected_tickers_from_runtime_input(runtime_input, ticker_limit=0)


def test_missing_professional_swing_universe_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EditableUniverseRuntimeInputError, match="Unable to load"):
        build_professional_swing_runtime_input(tmp_path / "missing.csv")


def _write_professional_swing_universe(tmp_path: Path) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(
        "\n".join(
            [
                "ticker,name,market,asset_type,active,universe_status,source_policy_hint,operator_priority,swing_profile,liquidity_profile,volatility_profile,market_cap_profile,theme,sector,notes",
                "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,2,trend_continuation,high,high,mega_cap,ai_infrastructure,technology,",
                "AMD,Advanced Micro Devices,USA,equity,true,watching,unknown,1,relative_strength,high,high,large_cap,ai_compute,technology,",
                "ASML,ASML Holding,EURONEXT,equity,true,needs_source_mapping,source_mapping_required,3,quality_compounder,high,medium,large_cap,semiconductor_equipment,technology,",
                "HO,Thales,EURONEXT,equity,true,manual_review_only,manual_review_only,4,quality_compounder,medium,medium,large_cap,defence_security,industrials,",
                "VRT,Vertiv,USA,equity,true,watching,cached_source_candidate,5,thematic_momentum,high,high,large_cap,data_center_infrastructure,industrials,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
