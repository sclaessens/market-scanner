from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION = (
    "market-engine-end-to-end-dry-run-v1"
)

MARKET_ENGINE_END_TO_END_DRY_RUN_BOUNDARY = (
    "The end-to-end dry-run validates and summarizes approved Market Engine "
    "contract flow only; it does not fetch live data, deliver messages, mutate "
    "portfolio or watchlist state, schedule work, or create action/allocation authority."
)

APPROVED_DRY_RUN_INPUT_MODES = (
    "synthetic_contract_fixture",
    "local_snapshot_fixture",
    "explicit_in_memory_payload",
)

FORBIDDEN_DRY_RUN_FIELDS = (
    "buy_instruction",
    "sell_instruction",
    "hold_instruction",
    "allocation_advice",
    "target_weight",
    "target_weights",
    "target_price",
    "position_size",
    "position_sizing",
    "order_generation",
    "execution_instruction",
    "broker_ready_payload",
    "trade_ticket",
    "urgency",
    "conviction",
    "ranking",
    "best_pick",
    "watchlist_mutation",
    "portfolio_mutation",
    "email_delivery",
    "notification_delivery",
    "scheduler_behavior",
    "production_report_write",
    "live_provider_fetch",
    "live_market_data_fetch",
)

_STAGE_CONTRACT_SPECS: tuple[dict[str, str], ...] = (
    {
        "stage_name": "source_context",
        "label": "Source Context",
        "version_field": "source_context_format_version",
        "expected_version": "sec-companyfacts-source-context-v1",
    },
    {
        "stage_name": "fundamental_observations",
        "label": "Fundamental Observations",
        "version_field": "fundamental_observations_format_version",
        "expected_version": "sec-companyfacts-fundamental-observations-v1",
    },
    {
        "stage_name": "derived_observations",
        "label": "Derived Observations",
        "version_field": "derived_observations_format_version",
        "expected_version": "sec-companyfacts-derived-cash-generation-observations-v1",
    },
    {
        "stage_name": "setup_detection",
        "label": "Setup Detection",
        "version_field": "setup_detection_format_version",
        "expected_version": "sec-companyfacts-setup-detection-v1",
    },
    {
        "stage_name": "analysis_review",
        "label": "Analysis Review",
        "version_field": "analysis_review_format_version",
        "expected_version": "sec-companyfacts-analysis-review-v1",
    },
    {
        "stage_name": "recommendation_review",
        "label": "Recommendation Review",
        "version_field": "recommendation_review_format_version",
        "expected_version": "sec-companyfacts-recommendation-review-v1",
    },
    {
        "stage_name": "portfolio_review",
        "label": "Portfolio Review",
        "version_field": "portfolio_review_format_version",
        "expected_version": "sec-companyfacts-portfolio-review-v1",
    },
    {
        "stage_name": "decision_engine_handoff",
        "label": "Decision Engine handoff",
        "version_field": "handoff_format_version",
        "expected_version": "market-engine-decision-engine-handoff-v1",
    },
    {
        "stage_name": "delivery_reporting",
        "label": "Delivery / Reporting",
        "version_field": "report_format_version",
        "expected_version": "market-engine-delivery-report-v1",
    },
)

REQUIRED_DRY_RUN_STAGE_NAMES = tuple(
    spec["stage_name"] for spec in _STAGE_CONTRACT_SPECS
) + ("dry_run_summary",)


class MarketEngineEndToEndDryRunStageStatus(str, Enum):
    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    BLOCKED = "blocked"
    UNSUPPORTED_INPUT = "unsupported_input"
    CONTRACT_VIOLATION = "contract_violation"


class MarketEngineEndToEndDryRunState(str, Enum):
    DRY_RUN_COMPLETED = "dry_run_completed"
    DRY_RUN_COMPLETED_WITH_LIMITATIONS = "dry_run_completed_with_limitations"
    DRY_RUN_BLOCKED = "dry_run_blocked"
    DRY_RUN_UNSUPPORTED_INPUT = "dry_run_unsupported_input"
    DRY_RUN_CONTRACT_VIOLATION = "dry_run_contract_violation"


@dataclass(frozen=True)
class MarketEngineEndToEndDryRunStageResult:
    stage_name: str
    stage_label: str
    expected_contract_version: str | None
    observed_contract_version: str | None
    status: MarketEngineEndToEndDryRunStageStatus
    fixture_backed: bool
    provenance_references: dict[str, Any] = field(default_factory=dict)
    missing_data_markers: tuple[str, ...] = field(default_factory=tuple)
    stale_data_markers: tuple[str, ...] = field(default_factory=tuple)
    numeric_zero_evidence: dict[str, Any] = field(default_factory=dict)
    blocked_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MarketEngineEndToEndDryRun:
    dry_run_format_version: str
    dry_run_id: str
    generated_at: str | None
    input_mode: str
    ticker: str
    cik: str
    provider_name: str
    run_state: MarketEngineEndToEndDryRunState
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...]
    blocked_stage: str | None
    blocked_reasons: tuple[str, ...]
    missing_data_summary: tuple[str, ...]
    stale_data_summary: tuple[str, ...]
    numeric_zero_evidence_summary: dict[str, Any]
    provenance_summary: dict[str, Any]
    delivery_report_reference: dict[str, Any]
    forbidden_side_effect_confirmation: str
    authority_boundary_confirmation: str
    audit_metadata: dict[str, Any]
    non_execution_boundary: str = MARKET_ENGINE_END_TO_END_DRY_RUN_BOUNDARY

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_state"] = self.run_state.value
        payload["stage_results"] = [
            {
                **asdict(stage_result),
                "status": stage_result.status.value,
            }
            for stage_result in self.stage_results
        ]
        return payload


def build_market_engine_end_to_end_dry_run(
    stage_payloads: Mapping[str, Any] | None,
    *,
    dry_run_id: str,
    input_mode: str,
    generated_at: str | None = None,
) -> MarketEngineEndToEndDryRun:
    input_mode_reasons = _input_mode_validation_reasons(input_mode)
    if input_mode_reasons:
        stage_results = _not_started_results()
        return _dry_run(
            dry_run_id=dry_run_id,
            generated_at=generated_at,
            input_mode=input_mode,
            run_state=MarketEngineEndToEndDryRunState.DRY_RUN_UNSUPPORTED_INPUT,
            stage_results=stage_results,
            blocked_stage="input_mode",
            blocked_reasons=input_mode_reasons,
        )

    if not isinstance(stage_payloads, Mapping):
        stage_results = _not_started_results()
        return _dry_run(
            dry_run_id=dry_run_id,
            generated_at=generated_at,
            input_mode=input_mode,
            run_state=MarketEngineEndToEndDryRunState.DRY_RUN_CONTRACT_VIOLATION,
            stage_results=stage_results,
            blocked_stage="stage_payloads",
            blocked_reasons=("Dry-run stage payloads must be a mapping.",),
        )

    stage_results: list[MarketEngineEndToEndDryRunStageResult] = []
    blocked_stage: str | None = None
    blocked_reasons: tuple[str, ...] = ()
    terminal_state: MarketEngineEndToEndDryRunState | None = None

    for spec in _STAGE_CONTRACT_SPECS:
        if blocked_stage is not None:
            stage_results.append(_not_started_result(spec))
            continue

        result = _inspect_stage_payload(stage_payloads, spec, input_mode)
        stage_results.append(result)

        if result.status == MarketEngineEndToEndDryRunStageStatus.CONTRACT_VIOLATION:
            blocked_stage = result.stage_name
            blocked_reasons = result.blocked_reasons
            terminal_state = MarketEngineEndToEndDryRunState.DRY_RUN_CONTRACT_VIOLATION
        elif result.status == MarketEngineEndToEndDryRunStageStatus.UNSUPPORTED_INPUT:
            blocked_stage = result.stage_name
            blocked_reasons = result.blocked_reasons
            terminal_state = MarketEngineEndToEndDryRunState.DRY_RUN_UNSUPPORTED_INPUT
        elif result.status == MarketEngineEndToEndDryRunStageStatus.BLOCKED:
            blocked_stage = result.stage_name
            blocked_reasons = result.blocked_reasons
            terminal_state = MarketEngineEndToEndDryRunState.DRY_RUN_BLOCKED

    stage_results.append(
        _dry_run_summary_result(
            blocked_stage=blocked_stage,
            blocked_reasons=blocked_reasons,
            terminal_state=terminal_state,
        )
    )

    run_state = terminal_state or _completed_run_state(stage_results)

    return _dry_run(
        dry_run_id=dry_run_id,
        generated_at=generated_at,
        input_mode=input_mode,
        run_state=run_state,
        stage_results=tuple(stage_results),
        blocked_stage=blocked_stage,
        blocked_reasons=blocked_reasons,
        ticker=_first_text(stage_payloads, "ticker"),
        cik=_first_text(stage_payloads, "cik"),
        provider_name=_first_text(stage_payloads, "provider_name"),
    )


def _input_mode_validation_reasons(input_mode: str) -> tuple[str, ...]:
    if input_mode not in APPROVED_DRY_RUN_INPUT_MODES:
        return ("Dry-run input mode is unsupported.",)
    return ()


def _inspect_stage_payload(
    stage_payloads: Mapping[str, Any],
    spec: Mapping[str, str],
    input_mode: str,
) -> MarketEngineEndToEndDryRunStageResult:
    stage_name = spec["stage_name"]
    stage_label = spec["label"]
    expected_version = spec["expected_version"]
    version_field = spec["version_field"]
    payload = stage_payloads.get(stage_name)

    if payload is None:
        return MarketEngineEndToEndDryRunStageResult(
            stage_name=stage_name,
            stage_label=stage_label,
            expected_contract_version=expected_version,
            observed_contract_version=None,
            status=MarketEngineEndToEndDryRunStageStatus.BLOCKED,
            fixture_backed=_fixture_backed(payload, input_mode),
            blocked_reasons=(f"{stage_label} payload is missing.",),
        )

    if not isinstance(payload, Mapping):
        return MarketEngineEndToEndDryRunStageResult(
            stage_name=stage_name,
            stage_label=stage_label,
            expected_contract_version=expected_version,
            observed_contract_version=None,
            status=MarketEngineEndToEndDryRunStageStatus.CONTRACT_VIOLATION,
            fixture_backed=_fixture_backed(payload, input_mode),
            blocked_reasons=(f"{stage_label} payload must be a mapping.",),
        )

    observed_version = payload.get(version_field)
    missing_markers = _collect_markers(payload, "missing")
    stale_markers = _collect_markers(payload, "stale")
    numeric_zero_evidence: dict[str, Any] = {}
    _collect_zero_values(stage_name, payload, numeric_zero_evidence)
    provenance_references = _collect_provenance(payload)
    blocked_reasons = _payload_blocked_reasons(payload)

    if _contains_forbidden_semantics(payload):
        return MarketEngineEndToEndDryRunStageResult(
            stage_name=stage_name,
            stage_label=stage_label,
            expected_contract_version=expected_version,
            observed_contract_version=_safe_text(observed_version),
            status=MarketEngineEndToEndDryRunStageStatus.CONTRACT_VIOLATION,
            fixture_backed=_fixture_backed(payload, input_mode),
            provenance_references=provenance_references,
            missing_data_markers=missing_markers,
            stale_data_markers=stale_markers,
            numeric_zero_evidence=numeric_zero_evidence,
            blocked_reasons=(f"{stage_label} payload contains prohibited dry-run semantics.",),
        )

    if observed_version != expected_version:
        return MarketEngineEndToEndDryRunStageResult(
            stage_name=stage_name,
            stage_label=stage_label,
            expected_contract_version=expected_version,
            observed_contract_version=_safe_text(observed_version),
            status=MarketEngineEndToEndDryRunStageStatus.UNSUPPORTED_INPUT,
            fixture_backed=_fixture_backed(payload, input_mode),
            provenance_references=provenance_references,
            missing_data_markers=missing_markers,
            stale_data_markers=stale_markers,
            numeric_zero_evidence=numeric_zero_evidence,
            blocked_reasons=(f"{stage_label} contract version is unsupported.",),
        )

    if blocked_reasons:
        return MarketEngineEndToEndDryRunStageResult(
            stage_name=stage_name,
            stage_label=stage_label,
            expected_contract_version=expected_version,
            observed_contract_version=_safe_text(observed_version),
            status=MarketEngineEndToEndDryRunStageStatus.BLOCKED,
            fixture_backed=_fixture_backed(payload, input_mode),
            provenance_references=provenance_references,
            missing_data_markers=missing_markers,
            stale_data_markers=stale_markers,
            numeric_zero_evidence=numeric_zero_evidence,
            blocked_reasons=blocked_reasons,
        )

    status = MarketEngineEndToEndDryRunStageStatus.COMPLETED
    if missing_markers or stale_markers:
        status = MarketEngineEndToEndDryRunStageStatus.COMPLETED_WITH_LIMITATIONS

    return MarketEngineEndToEndDryRunStageResult(
        stage_name=stage_name,
        stage_label=stage_label,
        expected_contract_version=expected_version,
        observed_contract_version=_safe_text(observed_version),
        status=status,
        fixture_backed=_fixture_backed(payload, input_mode),
        provenance_references=provenance_references,
        missing_data_markers=missing_markers,
        stale_data_markers=stale_markers,
        numeric_zero_evidence=numeric_zero_evidence,
        blocked_reasons=(),
    )


def _dry_run_summary_result(
    *,
    blocked_stage: str | None,
    blocked_reasons: tuple[str, ...],
    terminal_state: MarketEngineEndToEndDryRunState | None,
) -> MarketEngineEndToEndDryRunStageResult:
    if terminal_state == MarketEngineEndToEndDryRunState.DRY_RUN_CONTRACT_VIOLATION:
        status = MarketEngineEndToEndDryRunStageStatus.CONTRACT_VIOLATION
    elif terminal_state == MarketEngineEndToEndDryRunState.DRY_RUN_UNSUPPORTED_INPUT:
        status = MarketEngineEndToEndDryRunStageStatus.UNSUPPORTED_INPUT
    elif terminal_state == MarketEngineEndToEndDryRunState.DRY_RUN_BLOCKED:
        status = MarketEngineEndToEndDryRunStageStatus.BLOCKED
    else:
        status = MarketEngineEndToEndDryRunStageStatus.COMPLETED
    return MarketEngineEndToEndDryRunStageResult(
        stage_name="dry_run_summary",
        stage_label="Dry-run summary",
        expected_contract_version=MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
        observed_contract_version=MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
        status=status,
        fixture_backed=False,
        provenance_references={"blocked_stage": blocked_stage},
        blocked_reasons=blocked_reasons,
    )


def _not_started_results() -> tuple[MarketEngineEndToEndDryRunStageResult, ...]:
    results = [_not_started_result(spec) for spec in _STAGE_CONTRACT_SPECS]
    results.append(
        MarketEngineEndToEndDryRunStageResult(
            stage_name="dry_run_summary",
            stage_label="Dry-run summary",
            expected_contract_version=MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
            observed_contract_version=None,
            status=MarketEngineEndToEndDryRunStageStatus.NOT_STARTED,
            fixture_backed=False,
        )
    )
    return tuple(results)


def _not_started_result(spec: Mapping[str, str]) -> MarketEngineEndToEndDryRunStageResult:
    return MarketEngineEndToEndDryRunStageResult(
        stage_name=spec["stage_name"],
        stage_label=spec["label"],
        expected_contract_version=spec["expected_version"],
        observed_contract_version=None,
        status=MarketEngineEndToEndDryRunStageStatus.NOT_STARTED,
        fixture_backed=False,
    )


def _completed_run_state(
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...] | list[MarketEngineEndToEndDryRunStageResult],
) -> MarketEngineEndToEndDryRunState:
    if any(
        stage_result.missing_data_markers or stage_result.stale_data_markers
        for stage_result in stage_results
    ):
        return MarketEngineEndToEndDryRunState.DRY_RUN_COMPLETED_WITH_LIMITATIONS
    return MarketEngineEndToEndDryRunState.DRY_RUN_COMPLETED


def _dry_run(
    *,
    dry_run_id: str,
    generated_at: str | None,
    input_mode: str,
    run_state: MarketEngineEndToEndDryRunState,
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...],
    blocked_stage: str | None,
    blocked_reasons: tuple[str, ...],
    ticker: str = "",
    cik: str = "",
    provider_name: str = "",
) -> MarketEngineEndToEndDryRun:
    return MarketEngineEndToEndDryRun(
        dry_run_format_version=MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
        dry_run_id=dry_run_id,
        generated_at=generated_at,
        input_mode=input_mode,
        ticker=ticker,
        cik=cik,
        provider_name=provider_name,
        run_state=run_state,
        stage_results=stage_results,
        blocked_stage=blocked_stage,
        blocked_reasons=blocked_reasons,
        missing_data_summary=_summary_markers(stage_results, "missing"),
        stale_data_summary=_summary_markers(stage_results, "stale"),
        numeric_zero_evidence_summary=_summary_numeric_zero_evidence(stage_results),
        provenance_summary=_summary_provenance(stage_results),
        delivery_report_reference=_delivery_report_reference(stage_results),
        forbidden_side_effect_confirmation=(
            "No provider, market-data, broker, message-delivery, scheduler, portfolio, "
            "watchlist, production-report, or execution side effects are performed."
        ),
        authority_boundary_confirmation=(
            "Decision Engine remains the only future action/allocation authority; "
            "this dry-run emits integration-review output only."
        ),
        audit_metadata={
            "required_stage_names": REQUIRED_DRY_RUN_STAGE_NAMES,
            "approved_input_modes": APPROVED_DRY_RUN_INPUT_MODES,
            "blocked_stage": blocked_stage,
            "blocked_reasons": blocked_reasons,
            "non_execution_boundary": MARKET_ENGINE_END_TO_END_DRY_RUN_BOUNDARY,
        },
    )


def _payload_blocked_reasons(payload: Mapping[str, Any]) -> tuple[str, ...]:
    direct_reasons = payload.get("blocked_reasons") or payload.get("blocked_unavailable_reasons")
    if direct_reasons:
        return tuple(str(reason) for reason in direct_reasons)
    state_values = [
        value
        for key, value in payload.items()
        if isinstance(value, str) and key.endswith(("state", "status"))
    ]
    if any("blocked" in value.lower() for value in state_values):
        return ("Stage preserves an upstream blocked state.",)
    return ()


def _collect_markers(payload: Mapping[str, Any], marker_type: str) -> tuple[str, ...]:
    markers: list[str] = []
    terms = ("missing", "unavailable") if marker_type == "missing" else ("stale",)
    for key, value in payload.items():
        normalized_key = key.lower()
        if any(term in normalized_key for term in terms):
            if isinstance(value, (list, tuple, set)):
                markers.extend(str(item) for item in value)
            elif isinstance(value, str):
                markers.append(value)
            elif isinstance(value, Mapping):
                markers.extend(str(nested_key) for nested_key in value.keys())
    return tuple(dict.fromkeys(marker for marker in markers if marker))


def _collect_zero_values(prefix: str, value: Any, evidence: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            _collect_zero_values(f"{prefix}.{key}", nested_value, evidence)
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _collect_zero_values(f"{prefix}[{index}]", nested_value, evidence)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        evidence[prefix] = value


def _collect_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = key.lower()
        if (
            "provenance" in normalized_key
            or "reference" in normalized_key
            or "references" in normalized_key
            or normalized_key.endswith("_id")
        ):
            provenance[key] = value
    return provenance


def _contains_forbidden_semantics(payload: Mapping[str, Any]) -> bool:
    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
                if normalized_key in {
                    "forbidden_language_guardrails",
                    "forbidden_side_effect_confirmation",
                    "authority_boundary_confirmation",
                    "non_execution_boundary",
                    "boundary_notes",
                    "audit_metadata",
                }:
                    continue
                if normalized_key in FORBIDDEN_DRY_RUN_FIELDS:
                    return True
                if visit(nested_value):
                    return True
        elif isinstance(value, (list, tuple, set)):
            return any(visit(item) for item in value)
        return False

    return visit(payload)


def _fixture_backed(payload: Any, input_mode: str) -> bool:
    if input_mode == "synthetic_contract_fixture":
        return True
    if isinstance(payload, Mapping):
        return bool(payload.get("fixture_backed", input_mode == "local_snapshot_fixture"))
    return False


def _summary_markers(
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...],
    marker_type: str,
) -> tuple[str, ...]:
    markers: list[str] = []
    for stage_result in stage_results:
        if marker_type == "missing":
            markers.extend(stage_result.missing_data_markers)
        else:
            markers.extend(stage_result.stale_data_markers)
    return tuple(dict.fromkeys(markers))


def _summary_numeric_zero_evidence(
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for stage_result in stage_results:
        evidence.update(stage_result.numeric_zero_evidence)
    return evidence


def _summary_provenance(
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...],
) -> dict[str, Any]:
    return {
        stage_result.stage_name: stage_result.provenance_references
        for stage_result in stage_results
        if stage_result.provenance_references
    }


def _delivery_report_reference(
    stage_results: tuple[MarketEngineEndToEndDryRunStageResult, ...],
) -> dict[str, Any]:
    for stage_result in stage_results:
        if stage_result.stage_name == "delivery_reporting":
            return stage_result.provenance_references
    return {}


def _first_text(stage_payloads: Mapping[str, Any], key: str) -> str:
    for payload in stage_payloads.values():
        if isinstance(payload, Mapping) and isinstance(payload.get(key), str):
            return payload[key]
    return ""


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
