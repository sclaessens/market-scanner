from __future__ import annotations

from market_scanner.analysis.analysis_boundary import build_analysis_plan


def test_fundamental_analysis_contract_is_now_canonical_evidence_only() -> None:
    plan = build_analysis_plan()

    assert plan.canonical_owner == "src/market_scanner/analysis/"
    assert tuple(stage.name for stage in plan.stages) == (
        "fundamental_evidence_review",
        "profile_evidence_review",
        "limitation_review",
    )


def test_fundamental_analysis_contract_has_no_final_decision_authority() -> None:
    for stage in build_analysis_plan().stages:
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False