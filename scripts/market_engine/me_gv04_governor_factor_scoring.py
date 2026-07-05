from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from market_engine.governor.evaluation import (
    GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION,
    FactorFamily,
    GovernorEvaluation,
    evaluate_governor_evidence,
    to_plain_dict,
)
from market_engine.governor.scoring import (
    GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION,
    SCORE_SCALE,
)
RUN_FORMAT = "market-engine-me-gv04-governor-factor-scoring-v1"
FIXTURE_BATCH_FORMAT = "market-engine-governor-evidence-fixture-batch-v1"
NEXT_SPRINT = (
    "ME-GV05 - Implement recommendation-state mapping under approved boundary"
)


class MeGv04RunnerError(ValueError):
    """Raised when ME-GV04 scoring violates its non-actionable boundary."""


def run_governor_factor_scoring(
    *,
    input_path: str | Path,
    run_id: str,
    evaluation_timestamp: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-GV04 artifact root already exists: {root}")
    payload = _load_fixture(source_path)
    input_reference = _display_path(source_path)
    evaluations = tuple(
        evaluate_governor_evidence(
            case,
            evaluation_timestamp=evaluation_timestamp,
            input_reference=input_reference,
        )
        for case in _ordered_cases(payload)
    )
    _assert_evaluation_invariants(evaluations)
    summary = _summary(evaluations)
    _assert_summary_invariants(summary)
    result = {
        "run_format": RUN_FORMAT,
        "contract_version": GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION,
        "score_contract_version": (
            GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION
        ),
        "score_scale": dict(SCORE_SCALE),
        "run_id": _required_text(run_id, "run_id"),
        "evaluation_timestamp": _required_text(
            evaluation_timestamp,
            "evaluation_timestamp",
        ),
        "input_reference": input_reference,
        "input_contract": payload["fixture_contract"],
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
            "Local non-actionable factor scoring only. Factor weights, "
            "weighted scores, overall score, rank, recommendation mapping, "
            "buy-zone, position-management, allocation, execution, delivery, "
            "and Decision Engine authority remain unavailable."
        ),
        "next_sprint": NEXT_SPRINT,
    }
    normalized = json.loads(json.dumps(result, sort_keys=True))
    root.mkdir(parents=True)
    _write_json(root / "governor_factor_scoring.json", normalized)
    (root / "governor_factor_scoring_report.md").write_text(
        _markdown_report(normalized),
        encoding="utf-8",
    )
    return normalized


def _summary(evaluations: Sequence[GovernorEvaluation]) -> dict[str, Any]:
    factors = [
        factor
        for evaluation in evaluations
        for factor in evaluation.factor_evaluations
    ]
    factor_state_counts = Counter(item.state.value for item in factors)
    scored = [item for item in factors if item.score is not None]
    return {
        "evaluations_total": len(evaluations),
        "scored_factor_count": len(scored),
        "unscored_factor_count": len(factors) - len(scored),
        "score_null_count": sum(item.score is None for item in factors),
        "counts_by_factor": {
            family.value: {
                "evaluations": sum(item.factor is family for item in factors),
                "scored": sum(
                    item.factor is family and item.score is not None
                    for item in factors
                ),
            }
            for family in FactorFamily
        },
        "counts_by_factor_state": dict(sorted(factor_state_counts.items())),
        "counts_by_score_contract": {
            GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION: len(scored)
        },
        "conflict_blocked_score_count": sum(
            bool(item.conflicting_evidence_references)
            and item.score is None
            for item in factors
        ),
        "actionable_count": sum(
            bool(item.recommendation_state["actionable"])
            for item in evaluations
        ),
        "actionable_review_count": sum(
            bool(item.authority_boundary["actionable_review"])
            for item in evaluations
        ),
        "recommendation_state_ready_count": sum(
            bool(item.recommendation_state["recommendation_state_ready"])
            for item in evaluations
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
            item.weight is not None for item in factors
        ),
        "non_null_weighted_score_count": sum(
            item.weighted_score is not None for item in factors
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
        for factor in evaluation.factor_evaluations:
            if factor.score is not None and factor.state.value != "evaluable":
                raise MeGv04RunnerError(
                    "Only evaluable factors may receive a score"
                )
            if factor.weight is not None or factor.weighted_score is not None:
                raise MeGv04RunnerError(
                    "ME-GV04 factor weights must remain null"
                )
        overall = evaluation.overall_evaluation
        if any(
            overall[field] is not None
            for field in ("score", "score_scale", "weighted_score", "rank")
        ):
            raise MeGv04RunnerError(
                "ME-GV04 overall score and rank must remain null"
            )
        if (
            evaluation.recommendation_state["state"]
            != "blocked_not_authorized"
            or evaluation.recommendation_state["actionable"]
            or evaluation.recommendation_state["decision_engine_ready"]
        ):
            raise MeGv04RunnerError(
                "ME-GV04 recommendation boundary became reachable"
            )
        if (
            evaluation.buy_zone_explanation["state"]
            != "blocked_not_authorized"
            or evaluation.position_management_explanation["state"]
            != "blocked_not_authorized"
        ):
            raise MeGv04RunnerError(
                "ME-GV04 explanation boundary became reachable"
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
        raise MeGv04RunnerError(
            "ME-GV04 reserved authority or aggregation state became reachable"
        )


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    scale = result["score_scale"]
    lines = [
        "# ME-GV04 Governor Factor Scoring Dry-Run",
        "",
        "## Purpose",
        "",
        (
            "Apply deterministic factor-specific scoring to explicit approved "
            "analysis evidence while preserving every non-actionable boundary."
        ),
        "",
        "## Input Evidence and Run Identity",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Timestamp: `{result['evaluation_timestamp']}`",
        f"- Input: `{result['input_reference']}`",
        f"- Evaluations: {summary['evaluations_total']}",
        "",
        "## Score Contract and Scale",
        "",
        f"- Contract: `{result['score_contract_version']}`",
        (
            f"- Scale: {scale['minimum']} to {scale['maximum']}; "
            f"midpoint {scale['midpoint']}"
        ),
        (
            "- Direction: higher means a more favorable factor assessment; "
            "for risk, higher means a more favorable lower-risk profile."
        ),
        "- Precision: two decimal places, half-up.",
        "",
        "## Factor-State Summary",
        "",
        "| State | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{state}` | {count} |"
        for state, count in summary["counts_by_factor_state"].items()
    )
    lines.extend(
        [
            "",
            "## Scored-Factor Summary",
            "",
            f"- Scored factors: {summary['scored_factor_count']}",
            f"- Unscored factors: {summary['unscored_factor_count']}",
            f"- Null scores: {summary['score_null_count']}",
            (
                "- Conflict-blocked scores: "
                f"{summary['conflict_blocked_score_count']}"
            ),
            "",
            "## Factor Scores per Evaluation",
            "",
            "| Ticker | Factor | State | Score | Score limitations |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for evaluation in result["evaluations"]:
        for factor in evaluation["factor_evaluations"]:
            score = (
                f"{factor['score']:.2f}"
                if factor["score"] is not None
                else "null"
            )
            limitations = ", ".join(factor["score_limitations"]) or "none"
            lines.append(
                f"| {evaluation['ticker']} | `{factor['factor']}` | "
                f"`{factor['state']}` | {score} | {limitations} |"
            )
    lines.extend(["", "## Score Component Breakdown", ""])
    for evaluation in result["evaluations"]:
        for factor in evaluation["factor_evaluations"]:
            if not factor["score_components"]:
                continue
            lines.extend(
                [
                    f"### {evaluation['ticker']} / {factor['factor']}",
                    "",
                    (
                        "| Component | Evidence | Input | Rule | "
                        "Normalized | Contribution | Limitations |"
                    ),
                    "| --- | --- | ---: | --- | ---: | ---: | --- |",
                ]
            )
            for component in factor["score_components"]:
                limitations = (
                    ", ".join(component["limitations"]) or "none"
                )
                lines.append(
                    f"| `{component['component_id']}` | "
                    f"`{component['evidence_reference']}` | "
                    f"{component['input_value']} | "
                    f"`{component['normalization_rule']}` | "
                    f"{component['normalized_value']:.2f} | "
                    f"{component['normalized_contribution']:.2f} | "
                    f"{limitations} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Unscored Factors and Conflict Handling",
            "",
            (
                "Non-evaluable factors remain null. Unsupported factor scorers "
                "also remain null even when evidence state is `evaluable`. "
                "Conflicts remain visible and are never silently averaged."
            ),
            "",
            "## Null and Reserved-State Confirmations",
            "",
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
        ]
    )
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in result["reserved_state_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Recommendation, Buy-Zone, and Authority Boundary",
            "",
            (
                "Recommendation state, buy-zone explanation, and position "
                "management remain `blocked_not_authorized` for every case."
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


def _load_fixture(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MeGv04RunnerError(
            f"Governor scoring input fixture is unreadable: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MeGv04RunnerError(
            "Governor scoring input fixture is malformed JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MeGv04RunnerError(
            "Governor scoring input fixture must be a JSON object"
        )
    if payload.get("fixture_contract") != FIXTURE_BATCH_FORMAT:
        raise MeGv04RunnerError(
            "Governor scoring input fixture contract is unsupported"
        )
    return payload


def _ordered_cases(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise MeGv04RunnerError(
            "Governor scoring input cases must be a non-empty list"
        )
    if not all(isinstance(item, Mapping) for item in raw_cases):
        raise MeGv04RunnerError(
            "Governor scoring input cases must be JSON objects"
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
        raise MeGv04RunnerError(
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
        description="Run deterministic non-actionable ME-GV04 factor scoring."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-timestamp", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = run_governor_factor_scoring(
        input_path=args.input,
        run_id=args.run_id,
        evaluation_timestamp=args.evaluation_timestamp,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
