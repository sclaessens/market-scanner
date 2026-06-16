from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION,
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationObservation,
    SecCompanyFactsDerivedCashGenerationObservationSet,
    SecCompanyFactsDerivedCashGenerationState,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
    SecCompanyFactsFundamentalObservation,
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservationSet,
    SecCompanyFactsFundamentalObservationState,
)


SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION = "sec-companyfacts-setup-detection-v1"
SEC_COMPANYFACTS_SETUP_DETECTION_ROOT = Path("data/market_engine/setup_detections")

NON_ACTIONABLE_SETUP_DETECTION_BOUNDARY = (
    "Setup Detection describes evidence patterns only and is non-actionable."
)


class SecCompanyFactsSetupCategory(str, Enum):
    CASH_GENERATION_SETUP = "cash_generation_setup"
    FUNDAMENTAL_AVAILABILITY_SETUP = "fundamental_availability_setup"
    PROFITABILITY_EVIDENCE_SETUP = "profitability_evidence_setup"
    REVENUE_EVIDENCE_SETUP = "revenue_evidence_setup"
    BALANCE_SHEET_EVIDENCE_SETUP = "balance_sheet_evidence_setup"
    DATA_LIMITATION_SETUP = "data_limitation_setup"
    NOT_ASSESSED_SETUP = "not_assessed_setup"


class SecCompanyFactsSetupState(str, Enum):
    SETUP_DETECTED = "setup_detected"
    SETUP_PARTIALLY_DETECTED = "setup_partially_detected"
    SETUP_NOT_DETECTED = "setup_not_detected"
    SETUP_CONFLICTED = "setup_conflicted"
    SETUP_BLOCKED_BY_MISSING_DATA = "setup_blocked_by_missing_data"
    SETUP_NOT_ASSESSED = "setup_not_assessed"


@dataclass(frozen=True)
class SecCompanyFactsSetupDetectionItem:
    category: SecCompanyFactsSetupCategory
    state: SecCompanyFactsSetupState
    message: str
    input_observation_families: tuple[str, ...]
    required_observations: tuple[str, ...]
    missing_observations: tuple[str, ...]
    source_observation_references: dict[str, dict[str, Any]]
    derived_observation_references: dict[str, dict[str, Any]]
    setup_evidence: dict[str, Any]
    setup_limitations: tuple[str, ...]
    non_actionable_boundary: str = NON_ACTIONABLE_SETUP_DETECTION_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsSetupDetection:
    ticker: str
    cik: str
    provider_name: str
    setup_detection_format_version: str
    fundamental_observation_format_version: str
    derived_observation_format_version: str
    source_context_format_version: str
    source_context_state: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    setup_detection_run_id: str
    input_contracts: tuple[str, ...]
    setup_items: tuple[SecCompanyFactsSetupDetectionItem, ...]
    non_actionable_boundary: str = NON_ACTIONABLE_SETUP_DETECTION_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_setup_detection(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    *,
    setup_detection_run_id: str,
) -> SecCompanyFactsSetupDetection:
    _validate_input_contracts(
        fundamental_observation_set,
        derived_cash_generation_observation_set,
    )
    _validate_observation_set_alignment(
        fundamental_observation_set,
        derived_cash_generation_observation_set,
    )

    fundamental_by_category = _fundamental_observations_by_category(
        fundamental_observation_set
    )
    derived_by_category = _derived_observations_by_category(
        derived_cash_generation_observation_set
    )

    setup_items = (
        _fundamental_availability_setup(fundamental_by_category),
        _revenue_evidence_setup(fundamental_by_category),
        _profitability_evidence_setup(fundamental_by_category),
        _cash_generation_setup(
            fundamental_by_category=fundamental_by_category,
            derived_by_category=derived_by_category,
        ),
        _balance_sheet_evidence_setup(),
        _data_limitation_setup(
            fundamental_by_category=fundamental_by_category,
            derived_by_category=derived_by_category,
        ),
    )

    return SecCompanyFactsSetupDetection(
        ticker=fundamental_observation_set.ticker,
        cik=fundamental_observation_set.cik,
        provider_name=fundamental_observation_set.provider_name,
        setup_detection_format_version=SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION,
        fundamental_observation_format_version=(
            fundamental_observation_set.observation_format_version
        ),
        derived_observation_format_version=(
            derived_cash_generation_observation_set.derived_observation_format_version
        ),
        source_context_format_version=fundamental_observation_set.source_context_format_version,
        source_context_state=fundamental_observation_set.source_context_state,
        source_refresh_snapshot_id=fundamental_observation_set.source_refresh_snapshot_id,
        source_refresh_fetched_at=fundamental_observation_set.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            fundamental_observation_set.source_refresh_payload_format_version
        ),
        setup_detection_run_id=setup_detection_run_id,
        input_contracts=(
            SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION,
            SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION,
        ),
        setup_items=setup_items,
    )


def persist_sec_companyfacts_setup_detection(
    setup_detection: SecCompanyFactsSetupDetection,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_SETUP_DETECTION_ROOT
    setup_detection_dir = root / run_id / setup_detection.ticker
    setup_detection_dir.mkdir(parents=True, exist_ok=True)
    setup_detection_path = setup_detection_dir / "setup_detection.json"

    if setup_detection_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Setup Detection: "
            f"{setup_detection_path}"
        )

    setup_detection_path.write_text(
        json.dumps(_to_jsonable(setup_detection), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return setup_detection_path


def _validate_input_contracts(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> None:
    if (
        fundamental_observation_set.observation_format_version
        != SEC_COMPANYFACTS_FUNDAMENTAL_OBSERVATION_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported SEC CompanyFacts Fundamental Observation contract: "
            f"{fundamental_observation_set.observation_format_version}"
        )
    if (
        derived_cash_generation_observation_set.derived_observation_format_version
        != SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported SEC CompanyFacts Derived Cash Generation Observation contract: "
            f"{derived_cash_generation_observation_set.derived_observation_format_version}"
        )


def _validate_observation_set_alignment(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> None:
    mismatches = []
    for field_name in (
        "ticker",
        "cik",
        "provider_name",
        "source_context_format_version",
        "source_context_state",
        "source_refresh_snapshot_id",
        "source_refresh_fetched_at",
        "source_refresh_payload_format_version",
    ):
        if getattr(fundamental_observation_set, field_name) != getattr(
            derived_cash_generation_observation_set,
            field_name,
        ):
            mismatches.append(field_name)
    if mismatches:
        raise ValueError(
            "fundamental and derived observation sets do not align on: "
            + ", ".join(mismatches)
        )


def _fundamental_availability_setup(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> SecCompanyFactsSetupDetectionItem:
    required_observations = (
        "SOURCE_CONTEXT_AVAILABILITY",
        "REVENUE_SOURCE_PRESENCE",
        "NET_INCOME_SOURCE_VALUE",
        "OPERATING_CASH_FLOW_SOURCE_VALUE",
        "CAPEX_SOURCE_PRESENCE",
        "CASH_GENERATION_SOURCE_COMPLETENESS",
    )
    missing_observations = _missing_fundamental_observations(
        fundamental_by_category,
        required_observations,
    )
    referenced_observations = _fundamental_references(
        fundamental_by_category,
        required_observations,
    )

    if missing_observations:
        state = SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
        message = "Fundamental availability setup is blocked because required observations are missing."
    elif _has_limited_fundamental_observations(fundamental_by_category):
        state = SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED
        message = "Fundamental availability setup is partially detected with upstream limitations."
    else:
        state = SecCompanyFactsSetupState.SETUP_DETECTED
        message = "Required fundamental observations are available for setup detection."

    return _setup_item(
        category=SecCompanyFactsSetupCategory.FUNDAMENTAL_AVAILABILITY_SETUP,
        state=state,
        message=message,
        input_observation_families=("ME-FO",),
        required_observations=required_observations,
        missing_observations=missing_observations,
        source_observation_references=referenced_observations,
        derived_observation_references={},
        setup_evidence={
            "available_observations": tuple(sorted(referenced_observations)),
        },
        setup_limitations=missing_observations,
    )


def _revenue_evidence_setup(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> SecCompanyFactsSetupDetectionItem:
    required_observations = ("REVENUE_SOURCE_PRESENCE",)
    revenue = fundamental_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.REVENUE_SOURCE_PRESENCE
    )
    if revenue is None:
        return _missing_setup_item(
            category=SecCompanyFactsSetupCategory.REVENUE_EVIDENCE_SETUP,
            required_observations=required_observations,
            message="Revenue evidence setup is blocked because the revenue observation is missing.",
            input_observation_families=("ME-FO",),
        )

    if revenue.state == SecCompanyFactsFundamentalObservationState.PRESENT:
        state = SecCompanyFactsSetupState.SETUP_DETECTED
        message = "Revenue evidence setup is detected from available source observation."
    elif revenue.state == SecCompanyFactsFundamentalObservationState.MISSING_DATA:
        state = SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
        message = "Revenue evidence setup is blocked because revenue source data is missing."
    else:
        state = SecCompanyFactsSetupState.SETUP_NOT_ASSESSED
        message = "Revenue evidence setup is not assessed from the current observation state."

    return _setup_item(
        category=SecCompanyFactsSetupCategory.REVENUE_EVIDENCE_SETUP,
        state=state,
        message=message,
        input_observation_families=("ME-FO",),
        required_observations=required_observations,
        missing_observations=() if state != SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA else required_observations,
        source_observation_references={
            "REVENUE_SOURCE_PRESENCE": _fundamental_observation_reference(revenue)
        },
        derived_observation_references={},
        setup_evidence={
            "revenue": revenue.source_values.get("revenue"),
            "observation_state": revenue.state.value,
        },
        setup_limitations=revenue.missing_source_fields,
    )


def _profitability_evidence_setup(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> SecCompanyFactsSetupDetectionItem:
    required_observations = ("NET_INCOME_SOURCE_VALUE",)
    net_income = fundamental_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.NET_INCOME_SOURCE_VALUE
    )
    if net_income is None:
        return _missing_setup_item(
            category=SecCompanyFactsSetupCategory.PROFITABILITY_EVIDENCE_SETUP,
            required_observations=required_observations,
            message="Profitability evidence setup is blocked because the net income observation is missing.",
            input_observation_families=("ME-FO",),
        )

    if net_income.state == SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE:
        state = SecCompanyFactsSetupState.SETUP_DETECTED
        message = "Profitability evidence setup is detected from positive net income source value."
    elif net_income.state == SecCompanyFactsFundamentalObservationState.ZERO_SOURCE_VALUE:
        state = SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED
        message = "Profitability evidence setup is partially detected from zero net income source value."
    elif net_income.state == SecCompanyFactsFundamentalObservationState.NEGATIVE_SOURCE_VALUE:
        state = SecCompanyFactsSetupState.SETUP_NOT_DETECTED
        message = "Profitability evidence setup is not detected from negative net income source value."
    elif net_income.state == SecCompanyFactsFundamentalObservationState.MISSING_DATA:
        state = SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
        message = "Profitability evidence setup is blocked because net income source data is missing."
    else:
        state = SecCompanyFactsSetupState.SETUP_NOT_ASSESSED
        message = "Profitability evidence setup is not assessed from the current observation state."

    return _setup_item(
        category=SecCompanyFactsSetupCategory.PROFITABILITY_EVIDENCE_SETUP,
        state=state,
        message=message,
        input_observation_families=("ME-FO",),
        required_observations=required_observations,
        missing_observations=() if state != SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA else required_observations,
        source_observation_references={
            "NET_INCOME_SOURCE_VALUE": _fundamental_observation_reference(net_income)
        },
        derived_observation_references={},
        setup_evidence={
            "net_income": net_income.source_values.get("net_income"),
            "observation_state": net_income.state.value,
        },
        setup_limitations=net_income.missing_source_fields,
    )


def _cash_generation_setup(
    *,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsSetupDetectionItem:
    required_observations = ("FREE_CASH_FLOW_DERIVATION",)
    free_cash_flow = derived_by_category.get(
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    )
    if free_cash_flow is None:
        return _missing_setup_item(
            category=SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP,
            required_observations=required_observations,
            message="Cash generation setup is blocked because the derived observation is missing.",
            input_observation_families=("ME-DO",),
        )

    conflicted_source_observations = _cash_generation_conflicts(
        free_cash_flow,
        fundamental_by_category,
    )
    if conflicted_source_observations:
        state = SecCompanyFactsSetupState.SETUP_CONFLICTED
        message = "Cash generation setup is conflicted by upstream observation states."
        missing_observations: tuple[str, ...] = ()
    elif (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
    ):
        state = SecCompanyFactsSetupState.SETUP_DETECTED
        message = "Cash generation setup is detected from positive derived cash-generation evidence."
        missing_observations = ()
    elif (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE
    ):
        state = SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED
        message = "Cash generation setup is partially detected from zero derived cash-generation evidence."
        missing_observations = ()
    elif (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
    ):
        state = SecCompanyFactsSetupState.SETUP_NOT_DETECTED
        message = "Cash generation setup is not detected from negative derived cash-generation evidence."
        missing_observations = ()
    elif free_cash_flow.state in {
        SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
        SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
    }:
        state = SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
        message = "Cash generation setup is blocked because required source observations are missing."
        missing_observations = required_observations
    else:
        state = SecCompanyFactsSetupState.SETUP_NOT_ASSESSED
        message = "Cash generation setup is not assessed from the current derived observation state."
        missing_observations = ()

    return _setup_item(
        category=SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP,
        state=state,
        message=message,
        input_observation_families=("ME-DO", "ME-FO"),
        required_observations=required_observations,
        missing_observations=missing_observations,
        source_observation_references={
            category.value: _fundamental_observation_reference(observation)
            for category, observation in fundamental_by_category.items()
            if category
            in {
                SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE,
                SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE,
            }
        },
        derived_observation_references={
            "FREE_CASH_FLOW_DERIVATION": _derived_observation_reference(free_cash_flow)
        },
        setup_evidence={
            "derived_values": free_cash_flow.derived_values,
            "derived_state": free_cash_flow.state.value,
            "conflicted_source_observations": conflicted_source_observations,
        },
        setup_limitations=(
            free_cash_flow.missing_source_fields + conflicted_source_observations
        ),
    )


def _balance_sheet_evidence_setup() -> SecCompanyFactsSetupDetectionItem:
    return _setup_item(
        category=SecCompanyFactsSetupCategory.BALANCE_SHEET_EVIDENCE_SETUP,
        state=SecCompanyFactsSetupState.SETUP_NOT_ASSESSED,
        message="Balance-sheet evidence setup is not assessed by the current SEC CompanyFacts setup detection scope.",
        input_observation_families=("ME-FO",),
        required_observations=(),
        missing_observations=(),
        source_observation_references={},
        derived_observation_references={},
        setup_evidence={},
        setup_limitations=("balance_sheet_observations_not_in_current_input_contract",),
    )


def _data_limitation_setup(
    *,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsSetupDetectionItem:
    limited_fundamental = _limited_fundamental_observation_names(fundamental_by_category)
    limited_derived = _limited_derived_observation_names(derived_by_category)
    limitations = limited_fundamental + limited_derived
    if limitations:
        state = SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA
        message = "Setup detection is limited because one or more upstream observations are limited."
    else:
        state = SecCompanyFactsSetupState.SETUP_NOT_DETECTED
        message = "No upstream data limitation setup is detected."

    return _setup_item(
        category=SecCompanyFactsSetupCategory.DATA_LIMITATION_SETUP,
        state=state,
        message=message,
        input_observation_families=("ME-FO", "ME-DO"),
        required_observations=(),
        missing_observations=limitations,
        source_observation_references={
            category.value: _fundamental_observation_reference(observation)
            for category, observation in fundamental_by_category.items()
            if category.value in limited_fundamental
        },
        derived_observation_references={
            category.value: _derived_observation_reference(observation)
            for category, observation in derived_by_category.items()
            if category.value in limited_derived
        },
        setup_evidence={"limited_observations": limitations},
        setup_limitations=limitations,
    )


def _missing_setup_item(
    *,
    category: SecCompanyFactsSetupCategory,
    required_observations: tuple[str, ...],
    message: str,
    input_observation_families: tuple[str, ...],
) -> SecCompanyFactsSetupDetectionItem:
    return _setup_item(
        category=category,
        state=SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA,
        message=message,
        input_observation_families=input_observation_families,
        required_observations=required_observations,
        missing_observations=required_observations,
        source_observation_references={},
        derived_observation_references={},
        setup_evidence={},
        setup_limitations=required_observations,
    )


def _setup_item(
    *,
    category: SecCompanyFactsSetupCategory,
    state: SecCompanyFactsSetupState,
    message: str,
    input_observation_families: tuple[str, ...],
    required_observations: tuple[str, ...],
    missing_observations: tuple[str, ...],
    source_observation_references: dict[str, dict[str, Any]],
    derived_observation_references: dict[str, dict[str, Any]],
    setup_evidence: dict[str, Any],
    setup_limitations: tuple[str, ...],
) -> SecCompanyFactsSetupDetectionItem:
    return SecCompanyFactsSetupDetectionItem(
        category=category,
        state=state,
        message=message,
        input_observation_families=input_observation_families,
        required_observations=required_observations,
        missing_observations=missing_observations,
        source_observation_references=source_observation_references,
        derived_observation_references=derived_observation_references,
        setup_evidence=setup_evidence,
        setup_limitations=setup_limitations,
    )


def _cash_generation_conflicts(
    free_cash_flow: SecCompanyFactsDerivedCashGenerationObservation,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> tuple[str, ...]:
    operating_cash_flow = fundamental_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE
    )
    if operating_cash_flow is None:
        return ()

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
        and operating_cash_flow.state
        == SecCompanyFactsFundamentalObservationState.NEGATIVE_SOURCE_VALUE
    ):
        return ("OPERATING_CASH_FLOW_SOURCE_VALUE",)

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
        and operating_cash_flow.state
        == SecCompanyFactsFundamentalObservationState.POSITIVE_SOURCE_VALUE
    ):
        return ("OPERATING_CASH_FLOW_SOURCE_VALUE",)

    return ()


def _has_limited_fundamental_observations(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> bool:
    limited_states = {
        SecCompanyFactsFundamentalObservationState.MISSING_DATA,
        SecCompanyFactsFundamentalObservationState.NOT_ASSESSED,
        SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED,
    }
    return any(observation.state in limited_states for observation in fundamental_by_category.values())


def _limited_fundamental_observation_names(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> tuple[str, ...]:
    limited_states = {
        SecCompanyFactsFundamentalObservationState.MISSING_DATA,
        SecCompanyFactsFundamentalObservationState.NOT_ASSESSED,
        SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED,
    }
    return tuple(
        category.value
        for category, observation in fundamental_by_category.items()
        if observation.state in limited_states
    )


def _limited_derived_observation_names(
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> tuple[str, ...]:
    limited_states = {
        SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
        SecCompanyFactsDerivedCashGenerationState.NOT_ASSESSED,
        SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
    }
    return tuple(
        category.value
        for category, observation in derived_by_category.items()
        if observation.state in limited_states
    )


def _missing_fundamental_observations(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    required_observations: tuple[str, ...],
) -> tuple[str, ...]:
    present = {category.value for category in fundamental_by_category}
    return tuple(
        observation_name
        for observation_name in required_observations
        if observation_name not in present
    )


def _fundamental_references(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    categories: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        category.value: _fundamental_observation_reference(observation)
        for category, observation in fundamental_by_category.items()
        if category.value in categories
    }


def _fundamental_observation_reference(
    observation: SecCompanyFactsFundamentalObservation,
) -> dict[str, Any]:
    return {
        "category": observation.category.value,
        "state": observation.state.value,
        "canonical_fields": observation.canonical_fields,
        "source_values": observation.source_values,
        "source_references": observation.source_references,
        "missing_source_fields": observation.missing_source_fields,
        "observation_format_version": observation.observation_format_version,
    }


def _derived_observation_reference(
    observation: SecCompanyFactsDerivedCashGenerationObservation,
) -> dict[str, Any]:
    return {
        "category": observation.category.value,
        "state": observation.state.value,
        "formula": observation.formula,
        "derived_values": observation.derived_values,
        "required_source_fields": observation.required_source_fields,
        "missing_source_fields": observation.missing_source_fields,
        "source_observation_references": {
            source_field: asdict(reference)
            for source_field, reference in observation.source_observation_references.items()
        },
        "derived_observation_format_version": observation.derived_observation_format_version,
    }


def _fundamental_observations_by_category(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
) -> dict[
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservation,
]:
    return {
        observation.category: observation
        for observation in fundamental_observation_set.observations
    }


def _derived_observations_by_category(
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> dict[
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationObservation,
]:
    return {
        observation.category: observation
        for observation in derived_cash_generation_observation_set.observations
    }


def _to_jsonable(
    setup_detection: SecCompanyFactsSetupDetection,
) -> dict[str, Any]:
    payload = asdict(setup_detection)
    payload["setup_items"] = [
        {
            **asdict(item),
            "category": item.category.value,
            "state": item.state.value,
        }
        for item in setup_detection.setup_items
    ]
    return payload
