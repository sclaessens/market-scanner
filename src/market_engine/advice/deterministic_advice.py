from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from market_engine.advice.setup_price_market_context import (
    extract_setup_price_market_context,
)


ADVICE_INDEX_SCHEMA_VERSION = "market-engine-advice-index-v1"
ADVICE_INDEX_ARTIFACT_TYPE = "market-engine-deterministic-advice-index"
ADVICE_SUMMARY_SCHEMA_VERSION = "market-engine-advice-summary-v1"
UNABLE_TO_ADVISE_SCHEMA_VERSION = "market-engine-unable-to-advise-v1"
MANIFEST_SCHEMA_VERSION = "market-engine-advice-run-manifest-v1"

ADVICE_LABELS = (
    "buy_candidate",
    "wait_for_price",
    "watchlist",
    "avoid_for_now",
    "hold_existing",
    "take_loss_review",
    "unable_to_advise",
)
CONFIDENCE_VALUES = ("low", "medium", "high")
ADVICE_READINESS_VALUES = ("not_ready", "partial", "ready")

WATCHLIST_OR_BETTER = {
    "buy_candidate",
    "wait_for_price",
    "watchlist",
    "hold_existing",
}
ACTIONABLE_OUTPUTS = {
    "buy_candidate",
    "wait_for_price",
    "watchlist",
    "avoid_for_now",
    "hold_existing",
    "take_loss_review",
}


def build_advice_index(
    ticker_status_index_path: str | Path,
    *,
    run_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    input_path = Path(ticker_status_index_path)
    ticker_status_index = json.loads(input_path.read_text(encoding="utf-8"))
    emitted_at = generated_at or _generated_at_utc()
    advice_rows = [
        _advice_for_ticker(row)
        for row in sorted(
            ticker_status_index.get("tickers") or (),
            key=lambda item: str(item.get("ticker") or ""),
        )
    ]
    summary = _summary(advice_rows)
    return {
        "schema_version": ADVICE_INDEX_SCHEMA_VERSION,
        "artifact_type": ADVICE_INDEX_ARTIFACT_TYPE,
        "run_id": run_id,
        "generated_at": emitted_at,
        "input": {
            "ticker_status_index_path": input_path.as_posix(),
            "ticker_status_index_run_id": ticker_status_index.get("run_id"),
            "ticker_status_index_sha256": _sha256(input_path),
        },
        "summary": summary,
        "tickers": advice_rows,
    }


def write_advice_outputs(
    advice_index: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    allow_overwrite: bool = False,
) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _advice_summary_payload(advice_index)
    unable = _unable_to_advise_payload(advice_index)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-deterministic-advice-run-manifest",
        "run_id": run_id,
        "generated_at": advice_index.get("generated_at"),
        "outputs": {
            "advice_index_json": "advice_index.json",
            "advice_index_md": "advice_index.md",
            "advice_summary_json": "advice_summary.json",
            "unable_to_advise_json": "unable_to_advise.json",
        },
        "baseline_guardrail": {
            "openai_api_required": False,
            "provider_invocation_allowed": False,
            "source_acquisition_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "advice_labels_produced": True,
        },
        "next_baseline_sprint": "ME-ADV02 - 500-ticker advice batch output",
    }

    _write_json(output_dir / "advice_index.json", advice_index)
    (output_dir / "advice_index.md").write_text(
        render_advice_markdown(advice_index),
        encoding="utf-8",
    )
    _write_json(output_dir / "advice_summary.json", summary)
    _write_json(output_dir / "unable_to_advise.json", unable)
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def render_advice_markdown(advice_index: Mapping[str, Any]) -> str:
    summary = advice_index.get("summary") or {}
    advice_counts = summary.get("advice_counts") or {}
    rows = [
        "# Market Engine Advice Index",
        "",
        f"Run ID: `{advice_index.get('run_id')}`",
        f"Generated at: `{advice_index.get('generated_at')}`",
        f"Input ticker status index: `{(advice_index.get('input') or {}).get('ticker_status_index_path')}`",
        "",
        "## Summary",
        "",
        "| Advice | Count |",
        "|---|---:|",
    ]
    for label in ADVICE_LABELS:
        rows.append(f"| {label} | {advice_counts.get(label, 0)} |")
    rows.extend(
        [
            "",
            "## Advice Table",
            "",
            "| Ticker | Advice | Confidence | Readiness | Setup | Trend | Price position | Risk | Primary reason | Missing for buy candidate | Next action |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in advice_index.get("tickers") or ():
        setup_context = _mapping(row.get("setup_price_market_context"))
        rows.append(
            "| "
            + " | ".join(
                (
                    _md(row.get("ticker")),
                    _md(row.get("advice")),
                    _md(row.get("confidence")),
                    _md(row.get("advice_readiness")),
                    _md(setup_context.get("setup_state")),
                    _md(setup_context.get("trend_state")),
                    _md(setup_context.get("price_position")),
                    _md(setup_context.get("risk_state")),
                    _md(row.get("primary_reason")),
                    _md(", ".join(row.get("missing_for_buy_candidate") or ())),
                    _md(row.get("next_action")),
                )
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _advice_for_ticker(row: Mapping[str, Any]) -> dict[str, Any]:
    artifact_payload, artifact_error = _read_artifact_payload(row.get("artifact_path"))
    context = _AdviceContext(row=row, payload=artifact_payload, artifact_error=artifact_error)
    advice = _apply_rules(context)
    return {
        "ticker": _string(row.get("ticker")) or "UNKNOWN",
        "advice": advice["advice"],
        "confidence": advice["confidence"],
        "advice_readiness": advice["advice_readiness"],
        "primary_reason": advice["primary_reason"],
        "reasons": advice["reasons"],
        "blockers": advice["blockers"],
        "missing_for_buy_candidate": advice["missing_for_buy_candidate"],
        "next_action": advice["next_action"],
        "setup_price_market_context": context.setup_price_market_context,
        "source_status": _string(row.get("status")),
        "readiness_level": _string(row.get("readiness_level")),
        "context_stale": bool(row.get("context_stale")),
        "actionable_review_allowed": bool(row.get("actionable_review_allowed")),
        "decision_engine_ready": bool(row.get("decision_engine_ready")),
        "artifact_path": _string(row.get("artifact_path")),
        "artifact_sha256": _string(row.get("artifact_sha256")),
    }


def _apply_rules(context: "_AdviceContext") -> dict[str, Any]:
    if context.invalid_or_missing_artifact:
        return _result(
            "unable_to_advise",
            "low",
            "not_ready",
            "No valid canonical artifact is available for deterministic advice.",
            ["invalid_or_missing_artifact"],
            context.blockers,
            context.missing_for_buy_candidate or ["valid_dry_run_artifact"],
            "Produce a valid dry-run artifact before assigning an advice label.",
        )

    if context.missing_fundamental_context:
        return _result(
            "unable_to_advise",
            "low",
            "not_ready",
            "Fundamental context is missing, so deterministic advice is not possible.",
            ["valid_artifact_available", "missing_fundamental_context"],
            context.blockers,
            context.missing_for_buy_candidate or ["fundamental_context"],
            "Collect fundamental context before assigning an advice label.",
        )

    if context.setup_price_market_context_invalid:
        return _result(
            "unable_to_advise",
            "low",
            "not_ready",
            "Setup/price/market context is invalid.",
            ["valid_artifact_available", "setup_price_market_context_invalid"],
            context.blockers,
            context.missing_for_buy_candidate or ["valid_setup_price_market_context"],
            "Repair setup/price/market context before assigning an advice label.",
        )

    if context.has_serious_negative_signal:
        return _result(
            "avoid_for_now",
            "medium",
            "partial",
            "The artifact contains explicit unsupported or conflicting evidence flags.",
            ["valid_artifact_available", "serious_unsupported_or_conflict_signal"],
            context.blockers,
            context.missing_for_buy_candidate,
            "Avoid for now and resolve the unsupported or conflicting evidence first.",
        )

    if context.setup_price_market_context_usable:
        if context.setup_avoid_signal:
            return _result(
                "avoid_for_now",
                "medium",
                "partial",
                "Setup/price/market context indicates a weak, downtrend, breakdown, or elevated-risk setup.",
                ["valid_artifact_available", "setup_price_market_context_available", "setup_avoid_signal"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Avoid for now until setup, price position, or risk improves.",
            )
        if context.setup_wait_for_price_signal:
            return _result(
                "wait_for_price",
                "medium",
                "partial",
                "Setup/price/market context is constructive, but price is above the preferred entry zone.",
                ["valid_artifact_available", "setup_price_market_context_available", "price_above_preferred_entry"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Wait for price to return closer to the preferred entry zone.",
            )
        if context.setup_buy_candidate_signal and not context.has_hard_blockers:
            return _result(
                "buy_candidate",
                "medium",
                "partial",
                "Setup/price/market context is constructive and price is near a reasonable entry zone.",
                ["valid_artifact_available", "setup_price_market_context_available", "constructive_setup_price_context"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Review manually as a buy candidate; no order is created.",
            )
        if context.setup_context_unknown_or_partial:
            return _result(
                "watchlist",
                "low",
                "partial",
                "Setup/price/market context is partial or inconclusive.",
                ["valid_artifact_available", "setup_price_market_context_partial"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Keep on watchlist while improving setup/price/market evidence.",
            )

    if context.has_existing_position:
        if context.position_loss_or_high_risk:
            return _result(
                "take_loss_review",
                "medium",
                "partial",
                "Existing position context indicates loss or elevated risk review is needed.",
                ["valid_artifact_available", "existing_position", "loss_or_high_risk_position"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Review whether the existing position should be reduced or exited manually.",
            )
        return _result(
            "hold_existing",
            "medium",
            "partial",
            "Existing position context is present without a loss or elevated-risk flag.",
            ["valid_artifact_available", "existing_position", "no_loss_or_high_risk_position_flag"],
            context.blockers,
            context.missing_for_buy_candidate,
            "Hold the existing position for manual review; do not create an order.",
        )

    if context.context_stale:
        if context.setup_price_missing or context.missing_for_buy_candidate:
            return _result(
                "unable_to_advise",
                "low",
                "not_ready",
                "Context is stale and important data is missing.",
                ["valid_artifact_available", "context_stale", "important_data_missing"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Refresh or collect missing context before assigning an advice label.",
            )
        return _result(
            "watchlist",
            "low",
            "partial",
            "Context is stale, but blockers are explicit and the artifact can remain on watch.",
            ["valid_artifact_available", "context_stale", "explicit_blockers_available"],
            context.blockers,
            context.missing_for_buy_candidate,
            "Keep on watchlist until context freshness is restored.",
        )

    if context.actionable_or_de_ready:
        if context.setup_price_present and not context.has_missing_data and not context.has_hard_blockers:
            return _result(
                "buy_candidate",
                "medium",
                "ready",
                "Actionable review is available with setup/price context and no missing data.",
                ["valid_artifact_available", "actionable_review_available", "setup_price_context_present"],
                context.blockers,
                [],
                "Review manually as a buy candidate; no order is created.",
            )
        if context.setup_price_present:
            return _result(
                "wait_for_price",
                "medium",
                "partial",
                "Setup/price context exists, but uncertainty or blockers remain.",
                ["valid_artifact_available", "actionable_review_available", "setup_price_context_present", "uncertainty_remains"],
                context.blockers,
                context.missing_for_buy_candidate,
                "Wait for better price/context confirmation before buy-candidate review.",
            )

    if context.setup_price_missing and not context.context_stale:
        return _result(
            "watchlist",
            "low",
            "partial",
            "Valid non-stale artifact exists, but setup/price and portfolio context are missing.",
            ["valid_artifact_available", "context_not_stale", "partial_analysis_available", "setup_price_context_missing"],
            context.blockers,
            context.missing_for_buy_candidate,
            "Collect setup/price context before considering buy_candidate.",
        )

    if context.blocked_but_usable:
        return _result(
            "watchlist",
            "low",
            "partial",
            "The artifact is blocked, but it is valid, non-stale, partial analysis with explicit blockers.",
            ["valid_artifact_available", "context_not_stale", "blocked_partial_analysis_with_known_blockers"],
            context.blockers,
            context.missing_for_buy_candidate,
            "Keep on watchlist and resolve the explicit blockers.",
        )

    return _result(
        "watchlist",
        "low",
        "partial",
        "Valid artifact exists, but there is not enough complete evidence for a stronger advice label.",
        ["valid_artifact_available", "insufficient_complete_evidence_for_buy_candidate"],
        context.blockers,
        context.missing_for_buy_candidate,
        "Keep on watchlist while collecting stronger setup and portfolio context.",
    )


class _AdviceContext:
    def __init__(
        self,
        *,
        row: Mapping[str, Any],
        payload: Mapping[str, Any],
        artifact_error: str | None,
    ) -> None:
        self.row = row
        self.payload = payload
        self.artifact_error = artifact_error
        self.setup_price_market_context = extract_setup_price_market_context(
            row,
            payload,
        ).to_payload()
        self.blockers = _unique_strings(
            row.get("blocked_reasons"),
            row.get("readiness_blocked_reasons"),
            [row.get("blocked_stage")] if row.get("blocked_stage") else [],
            [artifact_error] if artifact_error else [],
            self.setup_price_market_context.get("blocked_reasons"),
        )

    @property
    def invalid_or_missing_artifact(self) -> bool:
        return (
            self.artifact_error is not None
            or self.row.get("status") == "invalid_artifact"
            or not self.row.get("artifact_path")
        )

    @property
    def context_stale(self) -> bool:
        return bool(self.row.get("context_stale"))

    @property
    def actionable_or_de_ready(self) -> bool:
        return bool(self.row.get("actionable_review_allowed")) or bool(
            self.row.get("decision_engine_ready")
        )

    @property
    def setup_price_missing(self) -> bool:
        if self.setup_price_market_context_usable:
            return False
        return "setup_price_market" in _strings(
            self.row.get("evidence_families_missing")
        ) or "missing_setup_or_price_context" in _strings(
            self.row.get("readiness_blocked_reasons")
        )

    @property
    def setup_price_present(self) -> bool:
        return self.setup_price_market_context_usable or (
            not self.setup_price_missing and "setup_price_context" in _strings(
            self.payload.get("available_context_families")
            )
        )

    @property
    def has_missing_data(self) -> bool:
        return bool(_strings(self.row.get("missing_data_summary")))

    @property
    def has_hard_blockers(self) -> bool:
        blockers = set(_strings(self.row.get("blocked_reasons")))
        allowed_blockers = {"Stage preserves an upstream blocked state."}
        stage = self.row.get("blocked_stage")
        if stage and stage != "portfolio_review":
            return True
        return bool(blockers - allowed_blockers)

    @property
    def setup_price_market_context_usable(self) -> bool:
        return self.setup_price_market_context.get("context_status") in {
            "available",
            "partial",
        }

    @property
    def setup_price_market_context_invalid(self) -> bool:
        return self.setup_price_market_context.get("context_status") == "invalid"

    @property
    def setup_avoid_signal(self) -> bool:
        context = self.setup_price_market_context
        return (
            context.get("trend_state") == "downtrend"
            or context.get("setup_state") == "weak_setup"
            or context.get("price_position") == "below_support_or_breakdown"
            or context.get("risk_state") in {"elevated", "high"}
        )

    @property
    def setup_wait_for_price_signal(self) -> bool:
        context = self.setup_price_market_context
        return (
            context.get("trend_state") == "uptrend"
            and context.get("setup_state") in {"breakout_candidate", "pullback_watch"}
            and context.get("price_position") == "above_preferred_entry"
        )

    @property
    def setup_buy_candidate_signal(self) -> bool:
        context = self.setup_price_market_context
        return (
            context.get("trend_state") == "uptrend"
            and context.get("setup_state") in {"breakout_candidate", "pullback_watch"}
            and context.get("price_position") in {"near_entry_zone", "fair_zone"}
            and context.get("risk_state") in {"normal", "unknown"}
        )

    @property
    def setup_context_unknown_or_partial(self) -> bool:
        context = self.setup_price_market_context
        return (
            context.get("context_status") == "partial"
            or context.get("setup_state") == "unknown"
            or context.get("price_position") == "unknown"
        )

    @property
    def missing_fundamental_context(self) -> bool:
        if "fundamental_context" in _strings(self.row.get("missing_data_summary")):
            return True
        if "fundamental_context" in _strings(self.row.get("evidence_families_missing")):
            return True
        if self.payload.get("fundamental_context_required") is False:
            return False
        provenance = _mapping(self.payload.get("provenance_summary"))
        fundamental = _mapping(provenance.get("fundamental_observations"))
        if fundamental:
            return False
        stage_results = self.payload.get("stage_results")
        if isinstance(stage_results, list):
            return not any(
                _mapping(stage).get("stage_name") == "fundamental_observations"
                and _mapping(stage).get("status") == "completed"
                for stage in stage_results
            )
        return self.payload.get("fundamental_context_present") is False

    @property
    def has_serious_negative_signal(self) -> bool:
        text = " ".join(
            _unique_strings(
                self.row.get("blocked_reasons"),
                self.row.get("readiness_blocked_reasons"),
                self.row.get("missing_data_summary"),
                self.payload.get("advice_flags"),
            )
        ).lower()
        return any(token in text for token in ("unsupported", "conflict", "severe_risk"))

    @property
    def has_existing_position(self) -> bool:
        portfolio = _mapping(self.payload.get("portfolio_context"))
        return bool(portfolio.get("existing_position") or portfolio.get("position_state") == "held")

    @property
    def position_loss_or_high_risk(self) -> bool:
        portfolio = _mapping(self.payload.get("portfolio_context"))
        return bool(
            portfolio.get("loss_review_required")
            or portfolio.get("risk_state") in {"high", "loss_review"}
            or _number(portfolio.get("unrealized_return_pct")) is not None
            and _number(portfolio.get("unrealized_return_pct")) < -10
        )

    @property
    def blocked_but_usable(self) -> bool:
        return (
            self.row.get("status") == "blocked"
            and not self.context_stale
            and self.row.get("readiness_level") == "partial_analysis"
            and bool(self.blockers)
        )

    @property
    def missing_for_buy_candidate(self) -> list[str]:
        missing: list[str] = []
        if self.setup_price_missing:
            missing.append("setup_price_market_context")
        missing.extend(_strings(self.setup_price_market_context.get("missing")))
        if "portfolio_context" in _strings(self.row.get("missing_data_summary")):
            missing.append("portfolio_context")
        if self.missing_fundamental_context:
            missing.append("fundamental_context")
        for item in _strings(self.row.get("evidence_families_missing")):
            if item == "setup_price_market":
                continue
            missing.append(item)
        return sorted(set(missing))


def _result(
    advice: str,
    confidence: str,
    advice_readiness: str,
    primary_reason: str,
    reasons: list[str],
    blockers: list[str],
    missing_for_buy_candidate: list[str],
    next_action: str,
) -> dict[str, Any]:
    if advice not in ADVICE_LABELS:
        raise ValueError(f"unsupported advice label: {advice}")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"unsupported confidence: {confidence}")
    if advice_readiness not in ADVICE_READINESS_VALUES:
        raise ValueError(f"unsupported advice readiness: {advice_readiness}")
    return {
        "advice": advice,
        "confidence": confidence,
        "advice_readiness": advice_readiness,
        "primary_reason": primary_reason,
        "reasons": reasons,
        "blockers": blockers,
        "missing_for_buy_candidate": missing_for_buy_candidate,
        "next_action": next_action,
    }


def _advice_summary_payload(advice_index: Mapping[str, Any]) -> dict[str, Any]:
    summary = advice_index.get("summary") or {}
    return {
        "schema_version": ADVICE_SUMMARY_SCHEMA_VERSION,
        "run_id": advice_index.get("run_id"),
        "tickers_total": summary.get("tickers_total", 0),
        "advice_counts": summary.get("advice_counts", {}),
        "confidence_counts": summary.get("confidence_counts", {}),
        "actionable_output_count": summary.get("actionable_output_count", 0),
        "watchlist_or_better_count": summary.get("watchlist_or_better_count", 0),
        "unable_to_advise_count": summary.get("unable_to_advise_count", 0),
        "top_missing_for_buy_candidate": summary.get("top_missing_for_buy_candidate", {}),
        "setup_price_market_context_counts": summary.get(
            "setup_price_market_context_counts",
            {},
        ),
        "next_baseline_sprint": "ME-ADV02 - 500-ticker advice batch output",
    }


def _unable_to_advise_payload(advice_index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": UNABLE_TO_ADVISE_SCHEMA_VERSION,
        "run_id": advice_index.get("run_id"),
        "tickers": [
            {
                "ticker": row.get("ticker"),
                "reason": row.get("primary_reason"),
                "missing_for_advice": row.get("missing_for_buy_candidate") or [],
            }
            for row in advice_index.get("tickers") or ()
            if row.get("advice") == "unable_to_advise"
        ],
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    advice_counts = Counter(row["advice"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    missing_counts: Counter[str] = Counter()
    setup_context_counts: Counter[str] = Counter()
    for row in rows:
        missing_counts.update(row.get("missing_for_buy_candidate") or ())
        setup_context = _mapping(row.get("setup_price_market_context"))
        setup_context_counts.update([_string(setup_context.get("context_status")) or "missing"])
    return {
        "tickers_total": len(rows),
        "advice_counts": {label: advice_counts.get(label, 0) for label in ADVICE_LABELS},
        "confidence_counts": {
            value: confidence_counts.get(value, 0) for value in CONFIDENCE_VALUES
        },
        "buy_candidate_count": advice_counts.get("buy_candidate", 0),
        "wait_for_price_count": advice_counts.get("wait_for_price", 0),
        "watchlist_count": advice_counts.get("watchlist", 0),
        "avoid_for_now_count": advice_counts.get("avoid_for_now", 0),
        "hold_existing_count": advice_counts.get("hold_existing", 0),
        "take_loss_review_count": advice_counts.get("take_loss_review", 0),
        "unable_to_advise_count": advice_counts.get("unable_to_advise", 0),
        "actionable_output_count": sum(advice_counts.get(label, 0) for label in ACTIONABLE_OUTPUTS),
        "watchlist_or_better_count": sum(advice_counts.get(label, 0) for label in WATCHLIST_OR_BETTER),
        "top_missing_for_buy_candidate": dict(sorted(missing_counts.items())),
        "setup_price_market_context_counts": {
            value: setup_context_counts.get(value, 0)
            for value in ("available", "partial", "missing", "invalid")
        },
    }


def _read_artifact_payload(path_value: Any) -> tuple[Mapping[str, Any], str | None]:
    path_text = _string(path_value)
    if not path_text:
        return {}, "missing_artifact_path"
    path = Path(path_text)
    if not path.exists():
        return {}, "artifact_path_not_found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, "artifact_json_invalid"
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return {}, "artifact_payload_missing"
    return payload, None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _unique_strings(*values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for item in _strings(value):
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _md(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value).replace("|", "\\|")
