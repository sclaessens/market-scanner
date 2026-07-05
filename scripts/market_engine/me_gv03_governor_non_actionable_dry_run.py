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


RUN_FORMAT = "market-engine-me-gv03-governor-dry-run-v1"
FIXTURE_BATCH_FORMAT = "market-engine-governor-evidence-fixture-batch-v1"
NEXT_SPRINT = "ME-GV04 - Implement factor scoring from approved analysis evidence"


class MeGv03RunnerError(ValueError):
    """Raised when the local Governor dry-run cannot proceed safely."""


def run_governor_non_actionable_dry_run(
    *,
    input_path: str | Path,
    run_id: str,
    evaluation_timestamp: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    root = Path(artifact_root)
    if root.exists():
        raise FileExistsError(f"ME-GV03 artifact root already exists: {root}")
    payload = _load_fixture(source_path)
    cases = _ordered_cases(payload)
    input_reference = _display_path(source_path)
    evaluations = tuple(
        evaluate_governor_evidence(
            case,
            evaluation_timestamp=evaluation_timestamp,
            input_reference=input_reference,
        )
        for case in cases
    )
    summary = _summary(evaluations)
    _assert_reserved_states(summary)
    result = {
        "run_format": RUN_FORMAT,
        "contract_version": GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION,
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
            "Local Governor evidence-readiness dry-run only. No scoring, "
            "recommendation mapping, buy-zone, position-management, allocation, "
            "execution, delivery, or Decision Engine authority is invoked."
        ),
        "next_sprint": NEXT_SPRINT,
    }
    normalized = json.loads(json.dumps(result, sort_keys=True))
    root.mkdir(parents=True)
    _write_json(root / "governor_evaluation.json", normalized)
    (root / "governor_evaluation_report.md").write_text(
        _markdown_report(normalized),
        encoding="utf-8",
    )
    return normalized


def _summary(
    evaluations: Sequence[GovernorEvaluation],
) -> dict[str, Any]:
    evaluation_state_counts = Counter(
        item.evaluation_state.value for item in evaluations
    )
    factor_state_counts = Counter(
        factor.state.value
        for item in evaluations
        for factor in item.factor_evaluations
    )
    counts_by_factor = {
        factor.value: dict(
            sorted(
                Counter(
                    item.state.value
                    for evaluation in evaluations
                    for item in evaluation.factor_evaluations
                    if item.factor is factor
                ).items()
            )
        )
        for factor in FactorFamily
    }
    blocked_reason_counts = Counter(
        reason
        for item in evaluations
        for reason in item.blocked_reasons
    )
    factor_payloads = [
        factor
        for item in evaluations
        for factor in item.factor_evaluations
    ]
    return {
        "evaluations_total": len(evaluations),
        "counts_by_evaluation_state": dict(
            sorted(evaluation_state_counts.items())
        ),
        "counts_by_factor_state": dict(sorted(factor_state_counts.items())),
        "counts_by_factor": counts_by_factor,
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
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
        "non_null_score_count": sum(
            factor.score is not None
            or factor.score_scale is not None
            or factor.weighted_score is not None
            for factor in factor_payloads
        )
        + sum(
            item.overall_evaluation["score"] is not None
            or item.overall_evaluation["score_scale"] is not None
            or item.overall_evaluation["weighted_score"] is not None
            for item in evaluations
        ),
        "non_null_weight_count": sum(
            factor.weight is not None for factor in factor_payloads
        ),
        "non_null_rank_count": sum(
            item.overall_evaluation["rank"] is not None
            for item in evaluations
        ),
    }


def _assert_reserved_states(summary: Mapping[str, Any]) -> None:
    zero_fields = (
        "actionable_count",
        "actionable_review_count",
        "recommendation_state_ready_count",
        "decision_ready_count",
        "de_ready_count",
        "non_null_score_count",
        "non_null_weight_count",
        "non_null_rank_count",
    )
    if any(summary[field] != 0 for field in zero_fields):
        raise MeGv03RunnerError(
            "ME-GV03 reserved authority or scoring state became reachable"
        )


def _load_fixture(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MeGv03RunnerError(
            f"Governor input fixture is unreadable: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MeGv03RunnerError("Governor input fixture is malformed JSON") from exc
    if not isinstance(payload, Mapping):
        raise MeGv03RunnerError("Governor input fixture must be a JSON object")
    if payload.get("fixture_contract") != FIXTURE_BATCH_FORMAT:
        raise MeGv03RunnerError("Governor input fixture contract is unsupported")
    return payload


def _ordered_cases(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise MeGv03RunnerError("Governor input cases must be a non-empty list")
    if not all(isinstance(item, Mapping) for item in raw_cases):
        raise MeGv03RunnerError("Governor input cases must be JSON objects")
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


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# ME-GV03 Governor Non-Actionable Dry-Run",
        "",
        "## Purpose",
        "",
        (
            "Classify approved deterministic evidence sufficiency through the "
            "ME-GV01 contract shape and ME-GV02 factor taxonomy."
        ),
        "",
        "## Input Evidence and Run Identity",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Timestamp: `{result['evaluation_timestamp']}`",
        f"- Input: `{result['input_reference']}`",
        f"- Evaluations: {summary['evaluations_total']}",
        "",
        "## Evaluation Summary",
        "",
        "| State | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{state}` | {count} |"
        for state, count in summary["counts_by_evaluation_state"].items()
    )
    lines.extend(
        [
            "",
            "## Per-Case Evaluation State",
            "",
            "| Ticker | Market | Evaluation state | Blocked reasons |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["evaluations"]:
        reasons = ", ".join(item["blocked_reasons"]) or "-"
        lines.append(
            f"| {item['ticker']} | {item['market']} | "
            f"{item['evaluation_state']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Factor-State Matrix",
            "",
            "| Ticker | "
            + " | ".join(factor.value for factor in FactorFamily)
            + " |",
            "| --- | " + " | ".join("---" for _ in FactorFamily) + " |",
        ]
    )
    for item in result["evaluations"]:
        states = {
            factor["factor"]: factor["state"]
            for factor in item["factor_evaluations"]
        }
        lines.append(
            f"| {item['ticker']} | "
            + " | ".join(states[factor.value] for factor in FactorFamily)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Missing Evidence, Blockers, and Conflicts",
            "",
        ]
    )
    for item in result["evaluations"]:
        conflicts = sorted(
            {
                reference
                for factor in item["factor_evaluations"]
                for reference in factor["conflicting_evidence_references"]
            }
        )
        lines.extend(
            [
                f"### {item['ticker']}",
                "",
                "- Missing: "
                + (", ".join(item["missing_evidence"]) or "none"),
                "- Blocked: "
                + (", ".join(item["blocked_reasons"]) or "none"),
                "- Conflict references: "
                + (", ".join(conflicts) or "none"),
                "- Limitations: "
                + (", ".join(item["risk_and_limitations"]) or "none"),
                "",
            ]
        )
    lines.extend(
        [
            "## Score-Null and Reserved-State Confirmation",
            "",
            f"- Non-null score fields: {summary['non_null_score_count']}",
            f"- Non-null weight fields: {summary['non_null_weight_count']}",
            f"- Non-null rank fields: {summary['non_null_rank_count']}",
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
                "Recommendation output remains non-actionable; buy-zone "
                "explanation and position management remain "
                "`blocked_not_authorized` for every case."
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


def _required_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise MeGv03RunnerError(
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
        description="Run deterministic non-actionable ME-GV03 evaluation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-timestamp", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = run_governor_non_actionable_dry_run(
        input_path=args.input,
        run_id=args.run_id,
        evaluation_timestamp=args.evaluation_timestamp,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
