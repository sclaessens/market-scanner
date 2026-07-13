from __future__ import annotations

from market_engine.data.complete_local_market_dataset import (
    _extract_batch_frame,
    _to_yfinance_symbol,
    build_acceptance_result,
)

import pandas as pd


def test_yfinance_symbol_normalizes_class_share_provider_syntax() -> None:
    assert _to_yfinance_symbol("BRK.B") == "BRK-B"
    assert _to_yfinance_symbol("MOG.A") == "MOG-A"


def test_extract_batch_frame_returns_requested_symbol_from_multiindex() -> None:
    frame = pd.DataFrame(
        {
            ("AAA", "Open"): [1.0],
            ("AAA", "Close"): [2.0],
            ("BBB", "Open"): [3.0],
            ("BBB", "Close"): [4.0],
        }
    )

    extracted = _extract_batch_frame(frame, "BBB", False)

    assert list(extracted.columns) == ["Open", "Close"]
    assert extracted.iloc[0]["Close"] == 4.0


def test_acceptance_result_requires_universe_coverage_and_current_history() -> None:
    result = build_acceptance_result(
        universe={"summary": {"total_instruments": 952}},
        coverage={
            "summary": {
                "total_canonical_instruments": 952,
                "valid": 946,
                "insufficient": 6,
                "missing": 0,
                "invalid": 0,
                "unsupported": 0,
            }
        },
        acquisition={"summary": {"valid_current_snapshot": 950}},
    )

    assert result["checks"]["canonical_universe_gt_900"] is True
    assert result["checks"]["valid_history_gt_90pct"] is True
    assert result["checks"]["current_history_materially_current"] is False
    assert result["status"] == "operational_dataset_partial"
