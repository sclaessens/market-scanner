from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.candidate_classification import (
    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION,
)
from market_engine.ticker_universe import (
    PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
    ProfessionalSwingUniverseExpansionError,
    build_professional_swing_universe_expansion,
)


HEADER = (
    "ticker,name,market,asset_type,active,universe_status,source_policy_hint,"
    "operator_priority,swing_profile,liquidity_profile,volatility_profile,"
    "market_cap_profile,theme,sector,notes"
)


def _write_universe(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "professional_swing_universe.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _base_universe(tmp_path: Path) -> Path:
    return _write_universe(
        tmp_path,
        [
            "NVDA,NVIDIA,USA,equity,true,candidate,cached_source_candidate,1,"
            "trend_continuation,high,high,mega_cap,ai_infrastructure,technology,",
            "MSFT,Microsoft,USA,equity,true,watching,unknown,2,"
            "quality_compounder,high,low,mega_cap,cloud_ai,technology,",
        ],
    )


def _candidate(
    ticker: str,
    *,
    bucket: str = "ready_for_manual_candidate_review",
    name: str | None = None,
    market: str = "USA",
    asset_type: str = "equity",
    universe_status: str = "candidate",
    source_policy_hint: str = "cached_source_candidate",
    blocking_reasons: list[str] | None = None,
    safety_flags: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "candidate_bucket": bucket,
        "candidate_rationale": "Local readable output has coherent review-only context.",
        "evidence_references": [
            {
                "reference_type": "operator_report",
                "reference": f"operator_report:{ticker}",
            }
        ],
        "blocking_reasons": blocking_reasons or [],
        "safety_flags": safety_flags
        or {
            "actionable_language_detected": False,
            "unsupported_input_detected": False,
            "malformed_input_detected": False,
            "stale_data_detected": False,
            "blocked_state_detected": False,
        },
        "proposed_universe_entry": {
            "ticker": ticker,
            "name": name or f"{ticker} Example",
            "market": market,
            "asset_type": asset_type,
            "active": "true",
            "universe_status": universe_status,
            "source_policy_hint": source_policy_hint,
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


def test_eligible_candidate_is_included_and_existing_entries_are_preserved(
    tmp_path: Path,
) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )

    assert result.format_version == PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION
    assert result.existing_universe_count == 2
    assert result.candidate_count == 1
    assert result.included_count == 1
    assert result.excluded_count == 0
    assert result.resulting_universe_count == 3
    assert result.included_candidate_entries[0].ticker == "AMD"
    assert result.included_candidate_entries[0].reason == "eligible_non_actionable_candidate"
    assert [entry["ticker"] for entry in result.final_universe_entries] == [
        "NVDA",
        "MSFT",
        "AMD",
    ]
    assert result.summary_counts["included_count"] == 1
    assert result.non_actionable_boundary is True


def test_candidate_already_present_is_marked_duplicate_not_added(
    tmp_path: Path,
) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("NVDA", name="NVIDIA")],
    )

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )

    assert result.included_count == 0
    assert result.duplicate_count == 1
    assert result.resulting_universe_count == 2
    assert result.duplicate_candidate_entries[0].already_present is True
    assert result.duplicate_candidate_entries[0].reason == (
        "already_present_in_professional_swing_universe"
    )


def test_duplicate_candidate_input_is_not_added_twice(tmp_path: Path) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("AMD", name="AMD"), _candidate("AMD", name="AMD")],
    )

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )

    assert result.included_count == 1
    assert result.duplicate_count == 1
    assert result.resulting_universe_count == 3
    assert result.duplicate_candidate_entries[0].reason == "duplicate_candidate_input"


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (
            _candidate(
                "ASML",
                name="ASML Holding",
                universe_status="manual_review_only",
                source_policy_hint="manual_review_only",
            ),
            "manual_review_only",
        ),
        (
            _candidate(
                "TSM",
                name="Taiwan Semiconductor",
                blocking_reasons=["ambiguous_identity"],
            ),
            "ambiguous_identity",
        ),
        (
            _candidate(
                "ARM",
                name="Arm Holdings",
                bucket="requires_source_coverage_review",
                blocking_reasons=["missing_source_coverage"],
            ),
            "ineligible_candidate_bucket:requires_source_coverage_review",
        ),
        (
            _candidate(
                "ETF1",
                name="Example Fund",
                asset_type="etf",
            ),
            "unsupported_asset_type:etf",
        ),
    ],
)
def test_ineligible_candidates_are_excluded_with_explicit_reasons(
    tmp_path: Path,
    record: dict[str, object],
    reason: str,
) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(tmp_path, [record])

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )

    assert result.included_count == 0
    assert result.excluded_count == 1
    assert result.excluded_candidate_entries[0].reason == reason


def test_malformed_candidate_missing_required_entry_field_fails_closed(
    tmp_path: Path,
) -> None:
    universe_path = _base_universe(tmp_path)
    candidate = _candidate("AMD", name="")
    candidate["proposed_universe_entry"]["name"] = ""
    candidate_path = _write_candidate_summary(tmp_path, [candidate])

    with pytest.raises(ProfessionalSwingUniverseExpansionError, match="company name"):
        build_professional_swing_universe_expansion(
            existing_universe_path=universe_path,
            candidate_classification_path=candidate_path,
        )


def test_unknown_candidate_classification_status_fails_closed(tmp_path: Path) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("AMD", bucket="secret_priority_candidate")],
    )

    with pytest.raises(ProfessionalSwingUniverseExpansionError, match="Unknown"):
        build_professional_swing_universe_expansion(
            existing_universe_path=universe_path,
            candidate_classification_path=candidate_path,
        )


def test_unsupported_candidate_classification_format_fails_closed(
    tmp_path: Path,
) -> None:
    universe_path = _base_universe(tmp_path)
    path = tmp_path / "candidate_classification_summary.json"
    path.write_text(
        json.dumps(
            {
                "candidate_classification_format_version": "unsupported",
                "per_ticker_classifications": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfessionalSwingUniverseExpansionError, match="unsupported"):
        build_professional_swing_universe_expansion(
            existing_universe_path=universe_path,
            candidate_classification_path=path,
        )


def test_deterministic_candidate_ordering_and_priorities(tmp_path: Path) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(
        tmp_path,
        [
            _candidate("VRT", name="Vertiv"),
            _candidate("AMD", name="AMD"),
            _candidate("AVGO", name="Broadcom"),
        ],
    )

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )

    assert [item.ticker for item in result.included_candidate_entries] == [
        "AMD",
        "AVGO",
        "VRT",
    ]
    assert [entry["ticker"] for entry in result.final_universe_entries] == [
        "NVDA",
        "MSFT",
        "AMD",
        "AVGO",
        "VRT",
    ]
    assert [entry["operator_priority"] for entry in result.final_universe_entries] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_operator_approval_filter_excludes_unapproved_candidates(
    tmp_path: Path,
) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(
        tmp_path,
        [_candidate("AMD", name="AMD"), _candidate("AVGO", name="Broadcom")],
    )

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
        operator_approved_tickers=("AMD",),
    )

    assert result.included_count == 1
    assert result.included_candidate_entries[0].ticker == "AMD"
    assert result.excluded_candidate_entries[0].ticker == "AVGO"
    assert result.excluded_candidate_entries[0].reason == "not_operator_approved"


def test_no_provider_network_or_action_authority_dependencies_are_imported() -> None:
    import market_engine.ticker_universe.professional_swing_expansion as expansion

    module_names = set(expansion.__dict__)

    assert "requests" not in module_names
    assert "urllib" not in module_names
    assert "socket" not in module_names
    assert "subprocess" not in module_names
    assert "yfinance" not in module_names
    assert "telegram" not in module_names
    assert "market_scanner" not in module_names


def test_generated_summary_uses_non_actionable_language(tmp_path: Path) -> None:
    universe_path = _base_universe(tmp_path)
    candidate_path = _write_candidate_summary(tmp_path, [_candidate("AMD", name="AMD")])

    result = build_professional_swing_universe_expansion(
        existing_universe_path=universe_path,
        candidate_classification_path=candidate_path,
    )
    summary_text = json.dumps(result.to_summary_payload(), sort_keys=True).lower()

    assert "buy" not in summary_text
    assert "sell" not in summary_text
    assert "target price" not in summary_text
    assert "allocation" not in summary_text
    assert "position size" not in summary_text
    assert "ranking" not in summary_text
    assert "conviction" not in summary_text
    assert "urgency" not in summary_text
