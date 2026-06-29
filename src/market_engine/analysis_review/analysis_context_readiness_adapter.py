from __future__ import annotations

from typing import Any, Mapping

from market_engine.analysis_review.analysis_context_readiness import (
    AnalysisContextEvidenceFamily,
    AnalysisContextReadinessResult,
    classify_analysis_context_readiness,
)
from market_engine.analysis_review.company_profile_analysis_context import (
    COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION,
)
from market_engine.analysis_review.sec_companyfacts_analysis_review import (
    SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
)
from market_engine.decision_engine_handoff.sec_companyfacts_handoff import (
    MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
)
from market_engine.delivery_reporting.sec_companyfacts_delivery_report import (
    MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
)
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
)
from market_engine.setup_detection.sec_companyfacts_setup_detection import (
    SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION,
)
from market_engine.source_context.company_profile_context import (
    COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
)
from market_engine.source_context.sec_companyfacts_context import (
    SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION,
)


_SOURCE_CONTEXT_VERSIONS = {
    COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
    SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION,
}


def classify_analysis_context_readiness_from_stage_payloads(
    stage_payloads: Mapping[str, Any] | None,
) -> AnalysisContextReadinessResult:
    if not isinstance(stage_payloads, Mapping):
        return classify_analysis_context_readiness(None)

    source_context = _mapping(stage_payloads.get("source_context"))
    fundamental_observations = _mapping(
        stage_payloads.get("fundamental_observations")
    )
    setup_detection = _mapping(stage_payloads.get("setup_detection"))
    analysis_review = _mapping(stage_payloads.get("analysis_review"))
    portfolio_review = _mapping(stage_payloads.get("portfolio_review"))
    decision_engine_handoff = _mapping(
        stage_payloads.get("decision_engine_handoff")
    )
    delivery_reporting = _mapping(stage_payloads.get("delivery_reporting"))

    evidence_families: set[AnalysisContextEvidenceFamily] = set()

    profile_source_present = _profile_source_context_present(
        source_context=source_context,
        analysis_review=analysis_review,
    )
    combined_profile_present = _combined_company_profile_context_present(
        source_context=source_context,
        analysis_review=analysis_review,
    )
    profile_present = profile_source_present or combined_profile_present
    fundamentals_present = _fundamental_context_present(
        fundamental_observations
    )
    setup_present = _setup_context_present(setup_detection)

    if profile_present:
        evidence_families.add(AnalysisContextEvidenceFamily.COMPANY_PROFILE)
    if fundamentals_present:
        evidence_families.add(AnalysisContextEvidenceFamily.FUNDAMENTALS)
    if setup_present:
        evidence_families.add(AnalysisContextEvidenceFamily.SETUP_PRICE_MARKET)
    if _portfolio_context_present(portfolio_review):
        evidence_families.add(AnalysisContextEvidenceFamily.PORTFOLIO_CONTEXT)
    if _delivery_handoff_context_present(
        decision_engine_handoff=decision_engine_handoff,
        delivery_reporting=delivery_reporting,
    ):
        evidence_families.add(
            AnalysisContextEvidenceFamily.DELIVERY_REPORTING_HANDOFF
        )

    relevant_payloads = tuple(
        payload
        for payload in (
            source_context,
            fundamental_observations,
            setup_detection,
            analysis_review,
        )
        if payload is not None
    )
    context_stale = any(
        _contains_stale_context(payload) for payload in relevant_payloads
    )
    provenance_valid = _provenance_is_valid(
        source_context=source_context,
        fundamental_observations=(
            fundamental_observations if fundamentals_present else None
        ),
        setup_detection=setup_detection if setup_present else None,
        analysis_review=analysis_review if profile_present else None,
        profile_source_present=profile_source_present,
    )
    if provenance_valid and not context_stale:
        evidence_families.add(
            AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS
        )

    return classify_analysis_context_readiness(
        evidence_families,
        provenance_valid=provenance_valid,
        context_stale=context_stale,
    )


def _profile_source_context_present(
    *,
    source_context: Mapping[str, Any] | None,
    analysis_review: Mapping[str, Any] | None,
) -> bool:
    return (
        _stage_contract_is_usable(
            source_context,
            version_field="source_context_format_version",
            expected_version=COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
        )
        and source_context.get("consumption_state") == "consumed"
        and _stage_contract_is_usable(
            analysis_review,
            version_field="analysis_review_format_version",
            expected_version=COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION,
        )
        and analysis_review.get("input_family") == "company_profile"
        and analysis_review.get("context_state") == "descriptive_context_available"
    )


def _combined_company_profile_context_present(
    *,
    source_context: Mapping[str, Any] | None,
    analysis_review: Mapping[str, Any] | None,
) -> bool:
    if not _stage_contract_is_usable(
        analysis_review,
        version_field="analysis_review_format_version",
        expected_version=SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
    ):
        return False
    profile_context = _mapping(analysis_review.get("company_profile_context"))
    return (
        source_context is not None
        and profile_context is not None
        and profile_context.get("analysis_review_format_version")
        == COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION
        and profile_context.get("input_family") == "company_profile"
        and profile_context.get("context_state") == "descriptive_context_available"
        and _text(profile_context.get("source_refresh_snapshot_id")) is not None
        and _non_empty_mapping(profile_context.get("provenance"))
        and _text(profile_context.get("ticker")) == _text(source_context.get("ticker"))
    )


def _portfolio_context_present(
    portfolio_review: Mapping[str, Any] | None,
) -> bool:
    if not _stage_contract_is_usable(
        portfolio_review,
        version_field="portfolio_review_format_version",
        expected_version=SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
    ):
        return False
    if (
        portfolio_review.get("portfolio_context_format_version")
        == MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION
    ):
        return True
    context_reference = _mapping(
        portfolio_review.get("portfolio_context_reference")
    )
    return (
        context_reference is not None
        and context_reference.get("portfolio_context_format_version")
        == MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION
    )


def _fundamental_context_present(
    fundamental_observations: Mapping[str, Any] | None,
) -> bool:
    if not _stage_contract_is_usable(
        fundamental_observations,
        version_field="fundamental_observations_format_version",
        expected_version=SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
    ):
        return False
    return _items_are_complete(
        fundamental_observations,
        item_field="observations",
        incomplete_states={
            "missing_data",
            "not_assessed",
            "source_limited",
        },
    )


def _setup_context_present(
    setup_detection: Mapping[str, Any] | None,
) -> bool:
    if not _stage_contract_is_usable(
        setup_detection,
        version_field="setup_detection_format_version",
        expected_version=SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION,
    ):
        return False
    return _items_are_complete(
        setup_detection,
        item_field="setup_items",
        incomplete_states={
            "setup_blocked_by_missing_data",
            "setup_conflicted",
            "setup_not_assessed",
            "setup_partially_detected",
            "setup_requires_human_review",
        },
    )


def _items_are_complete(
    payload: Mapping[str, Any],
    *,
    item_field: str,
    incomplete_states: set[str],
) -> bool:
    if item_field not in payload:
        return True
    items = payload.get(item_field)
    if not isinstance(items, (list, tuple)) or not items:
        return False
    for item in items:
        if not isinstance(item, Mapping):
            return False
        state = _text(item.get("state"))
        if state is None or state.lower() in incomplete_states:
            return False
    return True


def _delivery_handoff_context_present(
    *,
    decision_engine_handoff: Mapping[str, Any] | None,
    delivery_reporting: Mapping[str, Any] | None,
) -> bool:
    return _stage_contract_is_usable(
        decision_engine_handoff,
        version_field="handoff_format_version",
        expected_version=MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
    ) and _stage_contract_is_usable(
        delivery_reporting,
        version_field="report_format_version",
        expected_version=MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION,
    )


def _provenance_is_valid(
    *,
    source_context: Mapping[str, Any] | None,
    fundamental_observations: Mapping[str, Any] | None,
    setup_detection: Mapping[str, Any] | None,
    analysis_review: Mapping[str, Any] | None,
    profile_source_present: bool,
) -> bool:
    if not _source_context_is_usable(source_context):
        return False

    source_snapshot_id = _text(source_context.get("source_refresh_snapshot_id"))
    if source_snapshot_id is None:
        return False

    if profile_source_present:
        compatibility_gate = _mapping(source_context.get("compatibility_gate"))
        if (
            not _non_empty_mapping(source_context.get("provenance"))
            or _text(source_context.get("manifest_path")) is None
            or compatibility_gate is None
            or compatibility_gate.get("allowed") is not True
            or not _non_empty_mapping(source_context.get("cached_source_reference"))
        ):
            return False
    elif not (
        source_context.get("fixture_backed") is True
        or _non_empty_mapping(source_context.get("cached_source_reference"))
    ):
        return False

    payloads = tuple(
        payload
        for payload in (
            fundamental_observations,
            setup_detection,
            analysis_review,
        )
        if payload is not None
    )
    source_ticker = _text(source_context.get("ticker"))
    if source_ticker is None:
        return False

    for payload in payloads:
        if _text(payload.get("ticker")) != source_ticker:
            return False
        if not _has_lineage_reference(payload, source_snapshot_id):
            return False
    return True


def _source_context_is_usable(
    source_context: Mapping[str, Any] | None,
) -> bool:
    return (
        source_context is not None
        and source_context.get("source_context_format_version")
        in _SOURCE_CONTEXT_VERSIONS
        and not _payload_is_blocked(source_context)
    )


def _has_lineage_reference(
    payload: Mapping[str, Any],
    source_snapshot_id: str,
) -> bool:
    if payload.get("source_refresh_snapshot_id") == source_snapshot_id:
        return True
    source_reference = _mapping(payload.get("source_context_reference"))
    if (
        source_reference is not None
        and source_reference.get("source_refresh_snapshot_id") == source_snapshot_id
    ):
        return True
    return any(
        _non_empty_mapping(payload.get(key))
        for key in (
            "cached_source_reference",
            "derived_observations_reference",
            "setup_detection_reference",
        )
    )


def _stage_contract_is_usable(
    payload: Mapping[str, Any] | None,
    *,
    version_field: str,
    expected_version: str,
) -> bool:
    return (
        payload is not None
        and payload.get(version_field) == expected_version
        and not _payload_is_blocked(payload)
    )


def _payload_is_blocked(payload: Mapping[str, Any]) -> bool:
    direct_reasons = payload.get("blocked_reasons") or payload.get(
        "blocked_unavailable_reasons"
    )
    if direct_reasons:
        return True
    blocked_states = {
        "blocked",
        "contract_violation",
        "invalid",
        "not_started",
        "unsupported_input",
    }
    return any(
        value.lower() in blocked_states or value.lower().startswith("blocked_")
        for key, value in payload.items()
        if isinstance(value, str) and key.lower().endswith(("state", "status"))
    )


def _contains_stale_context(value: Any, *, parent_key: str = "") -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_stale_context(
                nested_value,
                parent_key=str(key).lower(),
            )
            for key, nested_value in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        if "stale" in parent_key and value:
            return True
        return any(
            _contains_stale_context(item, parent_key=parent_key) for item in value
        )
    if "stale" not in parent_key:
        return False
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "false",
            "fresh",
            "none",
            "not_stale",
        }
    return bool(value)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _non_empty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
