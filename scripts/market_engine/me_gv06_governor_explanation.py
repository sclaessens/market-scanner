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
from market_engine.governor.explanation import (
    GOVERNOR_EXPLANATION_CONTRACT_VERSION,
)


RUN_FORMAT = "market-engine-me-gv06-governor-explanation-v1"
NEXT_SPRINT = (
    "ME-DS01 - Define Dispatch Station output contract for Governor reports"
)


class MeGv06RunnerError(ValueError):
    """Raised when the Governor explanation dry-run cannot proceed safely."""


def run_governor_explanation(
    *,
    input_path: str | Path,
    run_id: str,
    evaluation_timestamp: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-GV06 artifact root already exists: {root}")
    input_reference = _display_path(source_path)
    evaluations = tuple(
        evaluate_governor_evidence(
            case,
            evaluation_timestamp=evaluation_timestamp,
            input_reference=input_reference,
        )
        for case in _load_cases(source_path)
    )
    _assert_evaluation_invariants(evaluations)
    summary = _summary(evaluations)
    _assert_summary_invariants(summary)
    result = {
        "run_format": RUN_FORMAT,
        "explanation_contract_version": GOVERNOR_EXPLANATION_CONTRACT_VERSION,
        "run_id": _required_text(run_id, "run_id"),
        "evaluation_timestamp": _required_text(
            evaluation_timestamp,
            "evaluation_timestamp",
        ),
        "input_reference": input_reference,
        "evaluations": tuple(to_plain_dict(item) for item in evaluations),
        "summary": summary,
        "reserved_state_counts": {
            "recommendation_state_ready": (
                summary["recommendation_state_ready_count"]
            ),
            "actionable": summary["actionable_count"],
            "actionable_review": summary["actionable_review_count"],
            "decision_ready": summary["decision_ready_count"],
            "de_ready": summary["de_ready_count"],
        },
        "forbidden_side_effects_confirmed": {
            "provider_calls_performed": False,
            "network_used": False,
            "source_acquisition_performed": False,
            "production_report_written": False,
            "telegram_or_email_sent": False,
            "portfolio_written": False,
            "watchlist_written": False,
            "broker_action_performed": False,
            "decision_engine_invoked": False,
        },
        "authority_boundary": (
            "Evidence-backed explanation only. No order, stop placement, "
            "position sizing, portfolio mutation, allocation, broker action, "
            "delivery, or Decision Engine authority is invoked."
        ),
        "next_sprint": NEXT_SPRINT,
    }
    normalized = json.loads(json.dumps(result, sort_keys=True))
    root.mkdir(parents=True)
    _write_json(root / "governor_explanation.json", normalized)
    (root / "governor_explanation_report.md").write_text(
        _markdown_report(normalized),
        encoding="utf-8",
    )
    return normalized


def _summary(evaluations: Sequence[GovernorEvaluation]) -> dict[str, Any]:
    buy_zones = [item.buy_zone_explanation for item in evaluations]
    positions = [item.position_management_explanation for item in evaluations]
    factors = [
        factor
        for evaluation in evaluations
        for factor in evaluation.factor_evaluations
    ]
    buy_reasons = Counter(
        reason for item in buy_zones for reason in item["reason_codes"]
    )
    position_reasons = Counter(
        reason for item in positions for reason in item["reason_codes"]
    )
    return {
        "evaluations_total": len(evaluations),
        "buy_zone_eligible_count": sum(
            item["eligibility_state"] == "eligible" for item in buy_zones
        ),
        "buy_zone_ineligible_count": sum(
            item["eligibility_state"] == "ineligible" for item in buy_zones
        ),
        "counts_by_buy_zone_state": dict(
            sorted(Counter(item["state"].value for item in buy_zones).items())
        ),
        "buy_zone_reason_code_counts": dict(sorted(buy_reasons.items())),
        "position_management_eligible_count": sum(
            item["eligibility_state"] == "eligible" for item in positions
        ),
        "position_management_ineligible_count": sum(
            item["eligibility_state"] == "ineligible" for item in positions
        ),
        "counts_by_position_management_state": dict(
            sorted(Counter(item["state"].value for item in positions).items())
        ),
        "position_management_reason_code_counts": dict(
            sorted(position_reasons.items())
        ),
        "stale_price_block_count": buy_reasons["blocked_stale_price_context"],
        "hard_conflict_block_count": buy_reasons[
            "blocked_unresolved_hard_price_conflict"
        ],
        "missing_price_evidence_count": buy_reasons[
            "missing_approved_price_context"
        ],
        "missing_position_context_count": position_reasons[
            "missing_approved_position_context"
        ],
        "execution_authorized_count": sum(
            bool(item["execution_authorized"]) for item in buy_zones
        ),
        "portfolio_mutation_authorized_count": sum(
            bool(item["portfolio_mutation_authorized"]) for item in positions
        ),
        "order_generation_authorized_count": sum(
            bool(item["order_generation_authorized"]) for item in positions
        ),
        "actionable_count": sum(
            bool(item.recommendation_state["actionable"])
            for item in evaluations
        ),
        "recommendation_state_ready_count": sum(
            bool(item.recommendation_state["recommendation_state_ready"])
            for item in evaluations
        ),
        "actionable_review_count": sum(
            bool(item.authority_boundary["actionable_review"])
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
        buy_zone = evaluation.buy_zone_explanation
        position = evaluation.position_management_explanation
        if (
            buy_zone["execution_authorized"]
            or buy_zone["stop_order_authorized"]
            or buy_zone["decision_engine_ready"]
            or position["portfolio_mutation_authorized"]
            or position["order_generation_authorized"]
            or position["decision_engine_ready"]
        ):
            raise MeGv06RunnerError(
                "ME-GV06 explanation authority became reachable"
            )
        if (
            evaluation.recommendation_state["actionable"]
            or evaluation.recommendation_state["decision_engine_ready"]
        ):
            raise MeGv06RunnerError(
                "ME-GV06 recommendation authority became reachable"
            )
        if any(
            factor.weight is not None or factor.weighted_score is not None
            for factor in evaluation.factor_evaluations
        ):
            raise MeGv06RunnerError("ME-GV06 factor weights must remain null")
        if any(
            evaluation.overall_evaluation[field] is not None
            for field in ("score", "score_scale", "weighted_score", "rank")
        ):
            raise MeGv06RunnerError(
                "ME-GV06 overall score and rank must remain null"
            )


def _assert_summary_invariants(summary: Mapping[str, Any]) -> None:
    zero_fields = (
        "execution_authorized_count",
        "portfolio_mutation_authorized_count",
        "order_generation_authorized_count",
        "actionable_count",
        "recommendation_state_ready_count",
        "actionable_review_count",
        "decision_ready_count",
        "de_ready_count",
        "non_null_weight_count",
        "non_null_weighted_score_count",
        "non_null_overall_score_count",
        "non_null_rank_count",
    )
    if any(summary[field] != 0 for field in zero_fields):
        raise MeGv06RunnerError(
            "ME-GV06 reserved authority or aggregation state became reachable"
        )


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# ME-GV06 Governor Buy-Zone and Position-Management Explanation",
        "",
        "## Purpose, Contract, and Run Identity",
        "",
        (
            "Explain approved price/setup and position-review context without "
            "creating execution, order, mutation, or Decision Engine authority."
        ),
        "",
        f"- Contract: `{result['explanation_contract_version']}`",
        f"- Run ID: `{result['run_id']}`",
        f"- Timestamp: `{result['evaluation_timestamp']}`",
        f"- Input: `{result['input_reference']}`",
        "",
        "## Buy-Zone Eligibility Summary",
        "",
        f"- Eligible: {summary['buy_zone_eligible_count']}",
        f"- Ineligible: {summary['buy_zone_ineligible_count']}",
        "",
        "## Buy-Zone State Counts",
        "",
        "| State | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{state}` | {count} |"
        for state, count in summary["counts_by_buy_zone_state"].items()
    )
    lines.extend(
        [
            "",
            "## Position-Management Eligibility Summary",
            "",
            f"- Eligible: {summary['position_management_eligible_count']}",
            f"- Ineligible: {summary['position_management_ineligible_count']}",
            "",
            "## Position-Management State Counts",
            "",
            "| State | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{state}` | {count} |"
        for state, count in summary[
            "counts_by_position_management_state"
        ].items()
    )
    lines.extend(
        [
            "",
            "## Per-Evaluation Explanation",
            "",
            (
                "| Ticker | Recommendation | Buy-zone | Position management | "
                "Buy-zone reasons | Position reasons |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for evaluation in result["evaluations"]:
        buy_zone = evaluation["buy_zone_explanation"]
        position = evaluation["position_management_explanation"]
        lines.append(
            f"| {evaluation['ticker']} | "
            f"`{evaluation['recommendation_state']['state']}` | "
            f"`{buy_zone['state']}` | `{position['state']}` | "
            f"{', '.join(buy_zone['reason_codes'])} | "
            f"{', '.join(position['reason_codes'])} |"
        )
    lines.extend(["", "## Approved Price Conditions and Invalidation", ""])
    for evaluation in result["evaluations"]:
        buy_zone = evaluation["buy_zone_explanation"]
        lines.extend(
            [
                f"### {evaluation['ticker']}",
                "",
                (
                    "- Approved price references: "
                    + (
                        ", ".join(buy_zone["approved_price_references"])
                        or "none"
                    )
                ),
                (
                    "- Pullback condition: "
                    f"`{buy_zone['pullback_condition']['state']}`"
                ),
                (
                    "- Breakout condition: "
                    f"`{buy_zone['breakout_condition']['state']}`"
                ),
                (
                    "- Invalidation context: "
                    f"`{buy_zone['invalidation_context']['state']}`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Guardrails and Missing Context",
            "",
            (
                "Stale or unprovenanced price evidence fails closed. Hard "
                "conflicts block explanation and soft conflicts remain visible. "
                "Risk and data confidence are explicit gates. Missing valuation "
                "remains a limitation and never becomes a target price. Missing "
                "position context never becomes a held position."
            ),
            "",
            "## Authority and Null Confirmations",
            "",
            (
                f"- Execution-authorized: "
                f"{summary['execution_authorized_count']}"
            ),
            (
                f"- Portfolio-mutation-authorized: "
                f"{summary['portfolio_mutation_authorized_count']}"
            ),
            (
                f"- Order-generation-authorized: "
                f"{summary['order_generation_authorized_count']}"
            ),
            f"- Actionable: {summary['actionable_count']}",
            f"- Decision-ready: {summary['decision_ready_count']}",
            f"- DE-ready: {summary['de_ready_count']}",
            f"- Non-null weights: {summary['non_null_weight_count']}",
            (
                f"- Non-null weighted scores: "
                f"{summary['non_null_weighted_score_count']}"
            ),
            (
                f"- Non-null overall scores: "
                f"{summary['non_null_overall_score_count']}"
            ),
            f"- Non-null ranks: {summary['non_null_rank_count']}",
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
        raise MeGv06RunnerError(
            f"Governor explanation input is unreadable: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MeGv06RunnerError(
            "Governor explanation input is malformed JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MeGv06RunnerError(
            "Governor explanation input must be a JSON object"
        )
    raw_cases = payload.get("cases", [payload])
    if not isinstance(raw_cases, list) or not raw_cases or not all(
        isinstance(item, Mapping) for item in raw_cases
    ):
        raise MeGv06RunnerError(
            "Governor explanation cases must be non-empty JSON objects"
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
        raise MeGv06RunnerError(
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
        description="Run deterministic non-executable ME-GV06 explanation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-timestamp", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = run_governor_explanation(
        input_path=args.input,
        run_id=args.run_id,
        evaluation_timestamp=args.evaluation_timestamp,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
