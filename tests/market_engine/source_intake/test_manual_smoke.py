from __future__ import annotations

import pytest

from market_engine.source_intake import manual_smoke


def test_sec_companyfacts_manual_smoke_requires_explicit_tickers():
    with pytest.raises(SystemExit) as error:
        manual_smoke.main(["--provider", "sec-companyfacts"])

    assert "requires --tickers" in str(error.value)


def test_sec_companyfacts_manual_smoke_enforces_max_tickers():
    with pytest.raises(SystemExit) as error:
        manual_smoke.main(
            [
                "--provider",
                "sec-companyfacts",
                "--tickers",
                "NVDA",
                "AMD",
                "--max-tickers",
                "1",
            ]
        )

    assert "refusing to run 2 tickers" in str(error.value)
