from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from market_engine.governor.evaluation import (
    GovernorEvaluation,
    evaluate_governor_evidence,
    to_plain_dict,
)
from market_engine.governor.recommendation import (
    GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION,
    RecommendationEligibilityState,
)


RUN_FORMAT = "market-engine-me-gv05-governor-recommendation-mapping-v1"
NEXT_SPRINT = (
    "ME-GV06 - Implement buy-zone and position-management explanation contract"
)


class MeGv05RunnerError(ValueError):
    """Raised when the governed recommendation dry-run cannot proceed safely."""


def run_governor_recommendation_mapping(
    *,
    input_path: str | Path,
    run_id: str,
    evaluation_timestamp: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-GV05 artifact root already exists: {root}")
    input_reference = _display_path(source_path)
    cases = _load_cases(source_path)
    evaluations = tuple(
        evaluate_governor_evidence(
            case,
            evaluation_timestamp=evaluation_timestamp,
            input_reference=input_reference,
        )
        for case in cases
    )
    _assert_evaluation_invariants(evaluations)
    summary = _summary(evaluations)
    _assert_summary_invariants(summary)
    result = {
        "run_format": RUN_FORMAT,
        "recommendation_contract_version": (
            GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION
        ),
        "run_id": _required_text(run_id, "run_id"),
        "evaluation_timestamp": _required_text(
            evaluation_timestamp,
            "evaluation_timestamp",
        ),
        "input_reference": input_reference,
        "evaluations": tuple(to_plain_dict(item) for item in evaluations),
        "summary": summary,
        "reserved_state_counts": {
            "actionable": summary["actionable_count"],
            "actionable_review": summary["actionable_review_count"],
            "recommendation_state_ready": (
                summary["recommendation_state_ready_count"]
            ),
            "decision_ready": summary["decision_ready_count"],
            "de_ready": summary["de_ready_count"],
        },
        "forbidden_side_effects_confirmed": {
            "provider_calls_performed": False,
            "network_used": False,
            "source_acquisition_performed": False,
            "snapshot_import_performed": False,
            "production_report_written": False,
            "telegram_or_email_sent": False,
            "portfolio_written": False,
            "watchlist_written": False,
            "broker_action_performed": False,
            "decision_engine_invoked": False,
        },
        "authority_boundary": (
            "Governed recommendation-state interpretation only. Every result "
            "is non-actionable. Buy-zone, position-management, allocation, "
            "execution, delivery, broker, and Decision Engine authority remain "
            "unavailable."
        ),
        "next_sprint": NEXT_SPRINT,
    }
    normalized = json.loads(json.dumps(result, sort_keys=True))
    root.mkdir(parents=True)
    _write_json(root / "governor_recommendation_mapping.json", normalized)
    (root / "governor_recommendation_mapping_report.md").write_text(
        _markdown_report(normalized),
        encoding="utf-8",
    )
    return normalized


def _summary(evaluations: Sequence[GovernorEvaluation]) -> dict[str, Any]:
    recommendations = [item.recommendation_state for item in evaluations]
    factors = [
        factor
        for evaluation in evaluations
        for factor in evaluation.factor_evaluations
    ]
    reason_counts = Counter(
        reason
        for recommendation in recommendations
        for reason in recommendation["reason_codes"]
    )
    return {
        "evaluations_total": len(evaluations),
        "recommendation_eligible_count": sum(
            item["eligibility_state"]
            == RecommendationEligibilityState.ELIGIBLE
            for item in recommendations
        ),
        "recommendation_ineligible_count": sum(
            item["eligibility_state"]
            == RecommendationEligibilityState.INELIGIBLE
            for item in recommendations
        ),
        "counts_by_recommendation_state": dict(
            sorted(Counter(item["state"].value for item in recommendations).items())
        ),
        "counts_by_eligibility_state": dict(
            sorted(
                Counter(
                    item["eligibility_state"].value
                    for item in recommendations
                ).items()
            )
        ),
        "recommendation_reason_code_counts": dict(sorted(reason_counts.items())),
        "hard_conflict_block_count": reason_counts[
            "blocked_unresolved_hard_conflict"
        ],
        "data_confidence_block_count": reason_counts[
            "ineligible_data_confidence_below_threshold"
        ],
        "critical_factor_coverage_block_count": reason_counts[
            "ineligible_critical_factor_coverage"
        ],
        "actionable_count": sum(
            bool(item["actionable"]) for item in recommendations
        ),
        "actionable_review_count": sum(
            bool(item.authority_boundary["actionable_review"])
            for item in evaluations
        ),
        "recommendation_state_ready_count": sum(
            bool(item["recommendation_state_ready"])
            for item in recommendations
        ),
        "decision_ready_count": sum(
            bool(item.authority_boundary["decision_ready"])
            for item in evaluations
        ),
        "de_ready_count": sum(
            bool(item.authority_boundary["de_ready"])
            for item in evaluations
        ),
        "non_null_weight_count": sum(
            factor.weight is not None for factor in factors
        ),
        "non_null_weighted_score_count": sum(
            factor.weighted_score is not None for factor in factors
        ),
        "non_null_overall_score_count": sum(
            item.overall_evaluation["score"] is not None
            or item.overall_evaluation["weighted_score"] is not None
            for item in evaluations
        ),
        "non_null_rank_count": sum(
            item.overall_evaluation["rank"] is not None
            for item in evaluations
        ),
    }


def _assert_evaluation_invariants(
    evaluations: Sequence[GovernorEvaluation],
) -> None:
    for evaluation in evaluations:
        recommendation = evaluation.recommendation_state
        if (
            recommendation["actionable"]
            or recommendation["recommendation_state_ready"]
            or recommendation["decision_engine_ready"]
        ):
            raise MeGv05RunnerError(
                "ME-GV05 reserved recommendation authority became reachable"
            )
        if any(
            factor.weight is not None or factor.weighted_score is not None
            for factor in evaluation.factor_evaluations
        ):
            raise MeGv05RunnerError("ME-GV05 factor weights must remain null")
        if any(
            evaluation.overall_evaluation[field] is not None
            for field in ("score", "score_scale", "weighted_score", "rank")
        ):
            raise MeGv05RunnerError(
                "ME-GV05 overall score and rank must remain null"
            )
        if (
            evaluation.buy_zone_explanation["execution_authorized"]
            or evaluation.buy_zone_explanation["stop_order_authorized"]
            or evaluation.position_management_explanation[
                "portfolio_mutation_authorized"
            ]
            or evaluation.position_management_explanation[
                "order_generation_authorized"
            ]
        ):
            raise MeGv05RunnerError(
                "ME-GV05 explanation authority became reachable"
            )


def _assert_summary_invariants(summary: Mapping[str, Any]) -> None:
    zero_fields = (
        "actionable_count",
        "actionable_review_count",
        "recommendation_state_ready_count",
        "decision_ready_count",
        "de_ready_count",
        "non_null_weight_count",
        "non_null_weighted_score_count",
        "non_null_overall_score_count",
        "non_null_rank_count",
    )
    if any(summary[field] != 0 for field in zero_fields):
        raise MeGv05RunnerError(
            "ME-GV05 reserved authority or aggregation state became reachable"
        )


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# ME-GV05 Governor Recommendation-State Mapping",
        "",
        "## Purpose and Contract Identity",
        "",
        (
            "Map approved Governor factor states and scores through an explicit "
            "eligibility gate into a governed, non-actionable interpretation."
        ),
        "",
        f"- Contract: `{result['recommendation_contract_version']}`",
        f"- Run ID: `{result['run_id']}`",
        f"- Timestamp: `{result['evaluation_timestamp']}`",
        f"- Input: `{result['input_reference']}`",
        "",
        "## Recommendation Eligibility Summary",
        "",
        f"- Eligible: {summary['recommendation_eligible_count']}",
        f"- Ineligible: {summary['recommendation_ineligible_count']}",
        "",
        "## Recommendation State Counts",
        "",
        "| State | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{state}` | {count} |"
        for state, count in summary["counts_by_recommendation_state"].items()
    )
    lines.extend(
        [
            "",
            "## Mapping Rule Summary",
            "",
            (
                "Eligibility requires a completed non-actionable Governor "
                "evaluation, approved non-actionable Recommendation Review, "
                "scored fundamentals, growth, risk, and data confidence, no "
                "critical score limitation, no hard conflict, and data "
                "confidence of at least 75."
            ),
            "",
            (
                "Directional mapping uses explicit per-factor thresholds. It "
                "does not create or use an overall numeric score or rank."
            ),
            "",
            "## Per-Case Recommendation State",
            "",
            (
                "| Ticker | Eligibility | State | Reason codes | "
                "Blocking factors |"
            ),
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for evaluation in result["evaluations"]:
        recommendation = evaluation["recommendation_state"]
        lines.append(
            f"| {evaluation['ticker']} | "
            f"`{recommendation['eligibility_state']}` | "
            f"`{recommendation['state']}` | "
            f"{', '.join(recommendation['reason_codes'])} | "
            f"{', '.join(recommendation['blocking_factors']) or 'none'} |"
        )
    lines.extend(["", "## Supporting Factor Scores", ""])
    for evaluation in result["evaluations"]:
        recommendation = evaluation["recommendation_state"]
        scores = ", ".join(
            f"{item['factor']}={item['score']}"
            for item in recommendation["supporting_factor_scores"]
        )
        lines.append(f"- {evaluation['ticker']}: {scores or 'none'}")
    lines.extend(
        [
            "",
            "## Data Confidence, Conflict, and Risk Boundaries",
            "",
            (
                "Data confidence is an explicit eligibility threshold and "
                "never a multiplier. Hard conflicts block eligibility; soft "
                "conflicts remain visible and cap favorable mappings at "
                "`watch`. Higher risk-factor scores mean a more favorable "
                "lower-risk profile; low risk scores constrain the mapping."
            ),
            "",
            "## Missing Valuation and Portfolio-Fit Handling",
            "",
            (
                "Missing valuation remains missing and prevents a complete "
                "Governor evaluation. Portfolio fit remains blocked without "
                "approved portfolio context and is disclosed as a limitation; "
                "it creates no portfolio or allocation authority."
            ),
            "",
            "## Null, Actionability, and Authority Confirmations",
            "",
            f"- Actionable results: {summary['actionable_count']}",
            f"- Decision-ready results: {summary['decision_ready_count']}",
            f"- DE-ready results: {summary['de_ready_count']}",
            (
                f"- Non-null factor weights: "
                f"{summary['non_null_weight_count']}"
            ),
            (
                f"- Non-null weighted factor scores: "
                f"{summary['non_null_weighted_score_count']}"
            ),
            (
                f"- Non-null overall scores: "
                f"{summary['non_null_overall_score_count']}"
            ),
            f"- Non-null ranks: {summary['non_null_rank_count']}",
            (
                "- Buy-zone and position-management authority: "
                "non-executable and non-mutating"
            ),
            "",
            result["authority_boundary"],
            "",
            "## Next Sprint",
            "",
            f"`{result['next_sprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _load_cases(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MeGv05RunnerError(
            f"Governor recommendation input is unreadable: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MeGv05RunnerError(
            "Governor recommendation input is malformed JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MeGv05RunnerError(
            "Governor recommendation input must be a JSON object"
        )
    raw_cases = payload.get("cases", [payload])
    if not isinstance(raw_cases, list) or not raw_cases or not all(
        isinstance(item, Mapping) for item in raw_cases
    ):
        raise MeGv05RunnerError(
            "Governor recommendation cases must be non-empty JSON objects"
        )
    return tuple(
        sorted(
            raw_cases,
            key=lambda item: (
                str(item.get("ticker") or ""),
                str(item.get("market") or ""),
                str(item.get("evaluation_id") or ""),
            ),
        )
    )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MeGv05RunnerError(
            f"{field_name} must be non-empty text without padding"
        )
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic non-actionable ME-GV05 mapping."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-timestamp", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = run_governor_recommendation_mapping(
        input_path=args.input,
        run_id=args.run_id,
        evaluation_timestamp=args.evaluation_timestamp,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
