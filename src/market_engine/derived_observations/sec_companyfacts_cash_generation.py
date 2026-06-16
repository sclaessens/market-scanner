from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SecCompanyFactsFundamentalObservation,
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservationSet,
    SecCompanyFactsFundamentalObservationState,
)


SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION = (
    "sec-companyfacts-derived-cash-generation-observations-v1"
)
SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_ROOT = Path(
    "data/market_engine/derived_observations/cash_generation"
)

NON_DECISION_DERIVED_OBSERVATION_BOUNDARY = (
    "Derived Observations are computed source-grounded outputs only."
)


class SecCompanyFactsDerivedCashGenerationCategory(str, Enum):
    FREE_CASH_FLOW_DERIVATION = "FREE_CASH_FLOW_DERIVATION"
    CASH_GENERATION_DERIVATION_LIMITATION = "CASH_GENERATION_DERIVATION_LIMITATION"


class SecCompanyFactsDerivedCashGenerationState(str, Enum):
    DERIVED_POSITIVE_SOURCE_VALUE = "DERIVED_POSITIVE_SOURCE_VALUE"
    DERIVED_NEGATIVE_SOURCE_VALUE = "DERIVED_NEGATIVE_SOURCE_VALUE"
    DERIVED_ZERO_SOURCE_VALUE = "DERIVED_ZERO_SOURCE_VALUE"
    MISSING_SOURCE_DATA = "MISSING_SOURCE_DATA"
    NOT_ASSESSED = "NOT_ASSESSED"
    SOURCE_LIMITED = "SOURCE_LIMITED"


@dataclass(frozen=True)
class SecCompanyFactsDerivedSourceObservationReference:
    category: str
    state: str
    canonical_fields: tuple[str, ...]
    source_values: dict[str, Any | None]
    source_references: dict[str, dict[str, Any | None]]
    missing_source_fields: tuple[str, ...]


@dataclass(frozen=True)
class SecCompanyFactsDerivedCashGenerationObservation:
    ticker: str
    cik: str
    provider_name: str
    category: SecCompanyFactsDerivedCashGenerationCategory
    state: SecCompanyFactsDerivedCashGenerationState
    message: str
    formula: str | None
    derived_values: dict[str, Any | None]
    required_source_fields: tuple[str, ...]
    missing_source_fields: tuple[str, ...]
    source_observation_references: dict[str, SecCompanyFactsDerivedSourceObservationReference]
    derived_observation_format_version: str = (
        SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
    )
    non_decision_boundary: str = NON_DECISION_DERIVED_OBSERVATION_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsDerivedCashGenerationObservationSet:
    ticker: str
    cik: str
    provider_name: str
    derived_observation_format_version: str
    fundamental_observation_format_version: str
    source_context_format_version: str
    source_context_state: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    observations: tuple[SecCompanyFactsDerivedCashGenerationObservation, ...]
    non_decision_boundary: str = NON_DECISION_DERIVED_OBSERVATION_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_derived_cash_generation_observations(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
) -> SecCompanyFactsDerivedCashGenerationObservationSet:
    observations_by_category = _observations_by_category(fundamental_observation_set)

    operating_cash_flow_observation = observations_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.OPERATING_CASH_FLOW_SOURCE_VALUE
    )
    capital_expenditures_observation = observations_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.CAPEX_SOURCE_PRESENCE
    )

    free_cash_flow_observation = _free_cash_flow_observation(
        fundamental_observation_set,
        operating_cash_flow_observation=operating_cash_flow_observation,
        capital_expenditures_observation=capital_expenditures_observation,
    )

    observations = [free_cash_flow_observation]

    limitation_observation = _cash_generation_derivation_limitation(
        fundamental_observation_set,
        operating_cash_flow_observation=operating_cash_flow_observation,
        capital_expenditures_observation=capital_expenditures_observation,
    )
    if limitation_observation is not None:
        observations.append(limitation_observation)

    return SecCompanyFactsDerivedCashGenerationObservationSet(
        ticker=fundamental_observation_set.ticker,
        cik=fundamental_observation_set.cik,
        provider_name=fundamental_observation_set.provider_name,
        derived_observation_format_version=(
            SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_FORMAT_VERSION
        ),
        fundamental_observation_format_version=(
            fundamental_observation_set.observation_format_version
        ),
        source_context_format_version=fundamental_observation_set.source_context_format_version,
        source_context_state=fundamental_observation_set.source_context_state,
        source_refresh_snapshot_id=fundamental_observation_set.source_refresh_snapshot_id,
        source_refresh_fetched_at=fundamental_observation_set.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            fundamental_observation_set.source_refresh_payload_format_version
        ),
        observations=tuple(observations),
    )


def persist_sec_companyfacts_derived_cash_generation_observations(
    observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_DERIVED_CASH_GENERATION_ROOT
    observation_dir = root / run_id / observation_set.ticker
    observation_dir.mkdir(parents=True, exist_ok=True)
    observation_path = observation_dir / "derived_cash_generation_observations.json"

    if observation_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Derived Cash Generation Observations: "
            f"{observation_path}"
        )

    observation_path.write_text(
        json.dumps(_to_jsonable(observation_set), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return observation_path


def _free_cash_flow_observation(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    *,
    operating_cash_flow_observation: SecCompanyFactsFundamentalObservation | None,
    capital_expenditures_observation: SecCompanyFactsFundamentalObservation | None,
) -> SecCompanyFactsDerivedCashGenerationObservation:
    required_source_fields = ("operating_cash_flow", "capital_expenditures")
    missing_source_fields = _missing_source_fields(
        operating_cash_flow_observation=operating_cash_flow_observation,
        capital_expenditures_observation=capital_expenditures_observation,
    )

    source_observation_references = _source_observation_references(
        operating_cash_flow_observation=operating_cash_flow_observation,
        capital_expenditures_observation=capital_expenditures_observation,
    )

    if missing_source_fields:
        return SecCompanyFactsDerivedCashGenerationObservation(
            ticker=fundamental_observation_set.ticker,
            cik=fundamental_observation_set.cik,
            provider_name=fundamental_observation_set.provider_name,
            category=SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION,
            state=SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
            message=(
                "Free cash flow derivation is not assessed because required source fields are missing."
            ),
            formula="operating_cash_flow - capital_expenditures",
            derived_values={"free_cash_flow": None},
            required_source_fields=required_source_fields,
            missing_source_fields=missing_source_fields,
            source_observation_references=source_observation_references,
        )

    operating_cash_flow = _source_value(
        operating_cash_flow_observation,
        "operating_cash_flow",
    )
    capital_expenditures = _source_value(
        capital_expenditures_observation,
        "capital_expenditures",
    )

    free_cash_flow = operating_cash_flow - capital_expenditures

    if free_cash_flow > 0:
        state = SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
        message = "Free cash flow derived source value is positive."
    elif free_cash_flow < 0:
        state = SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
        message = "Free cash flow derived source value is negative."
    else:
        state = SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE
        message = "Free cash flow derived source value is zero."

    return SecCompanyFactsDerivedCashGenerationObservation(
        ticker=fundamental_observation_set.ticker,
        cik=fundamental_observation_set.cik,
        provider_name=fundamental_observation_set.provider_name,
        category=SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION,
        state=state,
        message=message,
        formula="operating_cash_flow - capital_expenditures",
        derived_values={"free_cash_flow": free_cash_flow},
        required_source_fields=required_source_fields,
        missing_source_fields=(),
        source_observation_references=source_observation_references,
    )


def _cash_generation_derivation_limitation(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    *,
    operating_cash_flow_observation: SecCompanyFactsFundamentalObservation | None,
    capital_expenditures_observation: SecCompanyFactsFundamentalObservation | None,
) -> SecCompanyFactsDerivedCashGenerationObservation | None:
    missing_source_fields = _missing_source_fields(
        operating_cash_flow_observation=operating_cash_flow_observation,
        capital_expenditures_observation=capital_expenditures_observation,
    )

    if not missing_source_fields:
        return None

    return SecCompanyFactsDerivedCashGenerationObservation(
        ticker=fundamental_observation_set.ticker,
        cik=fundamental_observation_set.cik,
        provider_name=fundamental_observation_set.provider_name,
        category=(
            SecCompanyFactsDerivedCashGenerationCategory.CASH_GENERATION_DERIVATION_LIMITATION
        ),
        state=SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
        message=(
            "Cash-generation derivation is limited because required source fields are missing."
        ),
        formula=None,
        derived_values={},
        required_source_fields=("operating_cash_flow", "capital_expenditures"),
        missing_source_fields=missing_source_fields,
        source_observation_references=_source_observation_references(
            operating_cash_flow_observation=operating_cash_flow_observation,
            capital_expenditures_observation=capital_expenditures_observation,
        ),
    )


def _missing_source_fields(
    *,
    operating_cash_flow_observation: SecCompanyFactsFundamentalObservation | None,
    capital_expenditures_observation: SecCompanyFactsFundamentalObservation | None,
) -> tuple[str, ...]:
    missing_fields: list[str] = []

    if not _has_present_source_value(
        operating_cash_flow_observation,
        "operating_cash_flow",
    ):
        missing_fields.append("operating_cash_flow")

    if not _has_present_source_value(
        capital_expenditures_observation,
        "capital_expenditures",
    ):
        missing_fields.append("capital_expenditures")

    return tuple(missing_fields)


def _has_present_source_value(
    observation: SecCompanyFactsFundamentalObservation | None,
    field_name: str,
) -> bool:
    if observation is None:
        return False

    if field_name in observation.missing_source_fields:
        return False

    if field_name not in observation.source_values:
        return False

    return observation.source_values[field_name] is not None


def _source_value(
    observation: SecCompanyFactsFundamentalObservation | None,
    field_name: str,
) -> Any:
    if observation is None:
        raise ValueError(f"missing observation for source field: {field_name}")

    if field_name not in observation.source_values:
        raise ValueError(f"missing source value for source field: {field_name}")

    value = observation.source_values[field_name]
    if value is None:
        raise ValueError(f"source value is None for source field: {field_name}")

    return value


def _source_observation_references(
    *,
    operating_cash_flow_observation: SecCompanyFactsFundamentalObservation | None,
    capital_expenditures_observation: SecCompanyFactsFundamentalObservation | None,
) -> dict[str, SecCompanyFactsDerivedSourceObservationReference]:
    references: dict[str, SecCompanyFactsDerivedSourceObservationReference] = {}

    if operating_cash_flow_observation is not None:
        references["operating_cash_flow"] = _source_observation_reference(
            operating_cash_flow_observation
        )

    if capital_expenditures_observation is not None:
        references["capital_expenditures"] = _source_observation_reference(
            capital_expenditures_observation
        )

    return references


def _source_observation_reference(
    observation: SecCompanyFactsFundamentalObservation,
) -> SecCompanyFactsDerivedSourceObservationReference:
    return SecCompanyFactsDerivedSourceObservationReference(
        category=observation.category.value,
        state=observation.state.value,
        canonical_fields=observation.canonical_fields,
        source_values=observation.source_values,
        source_references=observation.source_references,
        missing_source_fields=observation.missing_source_fields,
    )


def _observations_by_category(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
) -> dict[
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservation,
]:
    return {
        observation.category: observation
        for observation in fundamental_observation_set.observations
    }


def _to_jsonable(
    observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> dict[str, Any]:
    payload = asdict(observation_set)
    payload["observations"] = [
        {
            **asdict(observation),
            "category": observation.category.value,
            "state": observation.state.value,
            "source_observation_references": {
                source_field: asdict(reference)
                for source_field, reference in observation.source_observation_references.items()
            },
        }
        for observation in observation_set.observations
    ]
    return payload