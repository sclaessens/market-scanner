from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from market_engine.analysis_review.sec_companyfacts_analysis_review import (
    build_sec_companyfacts_analysis_review,
)
from market_engine.decision_engine_handoff.sec_companyfacts_handoff import (
    build_market_engine_decision_engine_handoff,
)
from market_engine.delivery_reporting.sec_companyfacts_delivery_report import (
    build_market_engine_delivery_report,
)
from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    build_sec_companyfacts_derived_cash_generation_observations,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    build_sec_companyfacts_fundamental_observations,
)
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MarketEnginePortfolioContext,
    build_sec_companyfacts_portfolio_review,
)
from market_engine.recommendation_review.sec_companyfacts_recommendation_review import (
    build_sec_companyfacts_recommendation_review,
)
from market_engine.setup_detection.sec_companyfacts_setup_detection import (
    build_sec_companyfacts_setup_detection,
)
from market_engine.source_context.sec_companyfacts_context import (
    SecCompanyFactsContextBuildError,
    build_sec_companyfacts_source_context_from_snapshot_path,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    SEC_COMPANYFACTS_SNAPSHOT_ROOT,
)


MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION = (
    "market-engine-cached-source-local-execution-input-v1"
)
CACHED_SOURCE_SNAPSHOT_INPUT_MODE = "cached_source_snapshot"


class CachedSourceLocalExecutionError(ValueError):
    pass


def build_cached_source_local_execution_stage_payloads(
    *,
    source_snapshot_path: str | Path,
    source_snapshot_root: str | Path = SEC_COMPANYFACTS_SNAPSHOT_ROOT,
    dry_run_id: str,
    generated_at: str | None,
    portfolio_context_payload: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    snapshot_path = _validated_snapshot_path(
        source_snapshot_path=Path(source_snapshot_path),
        source_snapshot_root=Path(source_snapshot_root),
    )
    try:
        source_context = build_sec_companyfacts_source_context_from_snapshot_path(
            snapshot_path,
        )
        fundamental_observations = build_sec_companyfacts_fundamental_observations(
            source_context,
        )
        derived_observations = build_sec_companyfacts_derived_cash_generation_observations(
            fundamental_observations,
        )
        setup_detection = build_sec_companyfacts_setup_detection(
            fundamental_observations,
            derived_observations,
            setup_detection_run_id=f"{dry_run_id}-setup-detection",
        )
        analysis_review = build_sec_companyfacts_analysis_review(
            fundamental_observations,
            derived_observations,
            setup_detection,
        )
        recommendation_review = build_sec_companyfacts_recommendation_review(
            analysis_review,
            recommendation_review_run_id=f"{dry_run_id}-recommendation-review",
        )
        portfolio_context = _portfolio_context(portfolio_context_payload)
        portfolio_review = build_sec_companyfacts_portfolio_review(
            recommendation_review,
            portfolio_context,
            portfolio_review_run_id=f"{dry_run_id}-portfolio-review",
            created_at=generated_at,
        )
        decision_engine_handoff = build_market_engine_decision_engine_handoff(
            portfolio_review,
            handoff_run_id=f"{dry_run_id}-decision-engine-handoff",
            created_at=generated_at,
        )
        delivery_report = build_market_engine_delivery_report(
            decision_engine_handoff,
            report_id=f"{dry_run_id}-delivery-report",
            generated_at=generated_at,
        )
    except SecCompanyFactsContextBuildError as exc:
        raise CachedSourceLocalExecutionError(str(exc)) from exc
    except ValueError as exc:
        raise CachedSourceLocalExecutionError(
            f"Cached-source local execution could not build downstream contracts: {exc}"
        ) from exc

    stage_payloads = {
        "source_context": _source_context_payload(
            source_context,
            snapshot_path=snapshot_path,
        ),
        "fundamental_observations": _fundamental_observations_payload(
            fundamental_observations,
        ),
        "derived_observations": _derived_observations_payload(derived_observations),
        "setup_detection": _jsonable(setup_detection),
        "analysis_review": _jsonable(analysis_review),
        "recommendation_review": _jsonable(recommendation_review),
        "portfolio_review": _jsonable(portfolio_review),
        "decision_engine_handoff": decision_engine_handoff.to_payload(),
        "delivery_reporting": delivery_report.to_payload(),
    }
    _add_cached_source_provenance(
        stage_payloads,
        snapshot_path=snapshot_path,
        source_snapshot_root=Path(source_snapshot_root).resolve(),
    )
    return stage_payloads


def load_cached_source_local_execution_stage_payloads(
    path: str | Path,
    *,
    dry_run_id: str,
    generated_at: str | None,
) -> Mapping[str, Any]:
    payload = _read_json_object(Path(path))
    observed_version = payload.get("cached_source_local_execution_input_format_version")
    if observed_version != MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION:
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution input must use "
            f"{MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION}."
        )
    if payload.get("input_mode") != CACHED_SOURCE_SNAPSHOT_INPUT_MODE:
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution input_mode must be cached_source_snapshot."
        )
    if payload.get("non_production_local_execution") is not True:
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution input must set "
            "non_production_local_execution=true."
        )
    snapshot_path = payload.get("source_snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution input requires source_snapshot_path."
        )
    snapshot_root = payload.get("source_snapshot_root") or SEC_COMPANYFACTS_SNAPSHOT_ROOT
    if not isinstance(snapshot_root, (str, Path)):
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution source_snapshot_root must be a path string."
        )
    portfolio_context_payload = payload.get("portfolio_context")
    if portfolio_context_payload is not None and not isinstance(
        portfolio_context_payload,
        Mapping,
    ):
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution portfolio_context must be an object."
        )

    return build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=snapshot_root,
        dry_run_id=dry_run_id,
        generated_at=generated_at,
        portfolio_context_payload=portfolio_context_payload,
    )


def load_portfolio_context_payload(path: str | Path) -> Mapping[str, Any]:
    return _read_json_object(Path(path))


def _validated_snapshot_path(
    *,
    source_snapshot_path: Path,
    source_snapshot_root: Path,
) -> Path:
    if any(part == ".." for part in source_snapshot_root.parts):
        raise CachedSourceLocalExecutionError(
            "Cached-source snapshot root may not contain parent traversal."
        )
    root = source_snapshot_root.resolve()
    snapshot_path = source_snapshot_path.resolve()
    try:
        snapshot_path.relative_to(root)
    except ValueError as exc:
        raise CachedSourceLocalExecutionError(
            "Cached-source snapshot path must stay under the configured snapshot root."
        ) from exc
    return snapshot_path


def _portfolio_context(
    payload: Mapping[str, Any] | None,
) -> MarketEnginePortfolioContext | None:
    if payload is None:
        return None
    required_fields = (
        "portfolio_context_format_version",
        "portfolio_context_run_id",
        "portfolio_snapshot_timestamp",
        "portfolio_base_currency",
        "ticker",
        "position_state",
    )
    missing = [
        field_name
        for field_name in required_fields
        if payload.get(field_name) in (None, "")
    ]
    if missing:
        raise CachedSourceLocalExecutionError(
            "Cached-source portfolio context missing required fields: "
            + ", ".join(missing)
        )
    return MarketEnginePortfolioContext(
        portfolio_context_format_version=str(payload["portfolio_context_format_version"]),
        portfolio_context_run_id=str(payload["portfolio_context_run_id"]),
        portfolio_snapshot_timestamp=str(payload["portfolio_snapshot_timestamp"]),
        portfolio_base_currency=str(payload["portfolio_base_currency"]),
        ticker=str(payload["ticker"]),
        position_state=str(payload["position_state"]),
        current_quantity=payload.get("current_quantity"),
        current_market_value=payload.get("current_market_value"),
        portfolio_total_value=payload.get("portfolio_total_value"),
        current_ticker_exposure_pct=payload.get("current_ticker_exposure_pct"),
        exposure_buckets=_mapping(payload.get("exposure_buckets")),
        concentration_thresholds=_mapping(payload.get("concentration_thresholds")),
        policy_constraints=_mapping(payload.get("policy_constraints")),
        missing_portfolio_context_fields=tuple(
            payload.get("missing_portfolio_context_fields") or ()
        ),
        stale_portfolio_context_fields=tuple(
            payload.get("stale_portfolio_context_fields") or ()
        ),
        context_provenance=_mapping(payload.get("context_provenance")),
    )


def _source_context_payload(source_context: Any, *, snapshot_path: Path) -> dict[str, Any]:
    payload = _jsonable(source_context)
    payload["source_context_format_version"] = payload["context_format_version"]
    payload["cached_source_snapshot_path"] = snapshot_path.as_posix()
    return payload


def _fundamental_observations_payload(observation_set: Any) -> dict[str, Any]:
    payload = _jsonable(observation_set)
    payload["fundamental_observations_format_version"] = payload[
        "observation_format_version"
    ]
    payload["fundamental_observations_run_id"] = (
        f"{payload['source_refresh_snapshot_id']}-fundamental-observations"
    )
    return payload


def _derived_observations_payload(observation_set: Any) -> dict[str, Any]:
    payload = _jsonable(observation_set)
    payload["derived_observations_format_version"] = payload[
        "derived_observation_format_version"
    ]
    payload["derived_observations_run_id"] = (
        f"{payload['source_refresh_snapshot_id']}-derived-observations"
    )
    return payload


def _add_cached_source_provenance(
    stage_payloads: dict[str, Any],
    *,
    snapshot_path: Path,
    source_snapshot_root: Path,
) -> None:
    try:
        snapshot_reference = snapshot_path.relative_to(source_snapshot_root).as_posix()
    except ValueError:
        snapshot_reference = snapshot_path.as_posix()
    cached_source_reference = {
        "input_mode": CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
        "source_snapshot_path": snapshot_path.as_posix(),
        "source_snapshot_reference": snapshot_reference,
        "source_snapshot_root": source_snapshot_root.as_posix(),
        "cached_source_local_execution_input_format_version": (
            MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION
        ),
    }
    for payload in stage_payloads.values():
        if isinstance(payload, dict):
            payload["cached_source_reference"] = cached_source_reference


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CachedSourceLocalExecutionError(
            f"Unable to read cached-source local execution JSON: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CachedSourceLocalExecutionError(
            f"Cached-source local execution JSON is invalid: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CachedSourceLocalExecutionError(
            "Cached-source local execution JSON must contain an object at the top level."
        )
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CachedSourceLocalExecutionError("Expected a JSON object.")
    return dict(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            _jsonable_key(key): _jsonable(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_jsonable(nested_value) for nested_value in value)
    if isinstance(value, list):
        return [_jsonable(nested_value) for nested_value in value]
    return value


def _jsonable_key(key: Any) -> str:
    if isinstance(key, Enum):
        return str(key.value)
    return str(key)
