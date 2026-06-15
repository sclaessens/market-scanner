from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.source_intake.sec_companyfacts_fields import (
    SEC_COMPANYFACTS_PROVIDER_NAME,
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    SecCompanyFactsMappedField,
    map_sec_companyfacts_fields,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    SEC_COMPANYFACTS_SOURCE_NAME,
    SecCompanyFactsRawSnapshot,
    SecCompanyFactsSnapshotError,
    load_sec_companyfacts_raw_snapshot,
)


SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION = "sec-companyfacts-source-context-v1"
SEC_COMPANYFACTS_SOURCE_CONTEXT_ROOT = Path("data/market_engine/source_contexts/fundamentals")


class SecCompanyFactsContextState(str, Enum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    INVALID = "INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    UNSUPPORTED = "UNSUPPORTED"


class SecCompanyFactsContextFieldState(str, Enum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


class SecCompanyFactsContextBuildError(Exception):
    """Raised when a SEC CompanyFacts Source Context cannot be built safely."""


@dataclass(frozen=True)
class SecCompanyFactsContextField:
    canonical_field_name: str
    state: SecCompanyFactsContextFieldState
    raw_value: Any | None = None
    sec_tag_selected: str | None = None
    provider_name: str | None = None
    taxonomy_namespace: str | None = None
    unit: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    filing_form: str | None = None
    filing_date: str | None = None
    period_start_date: str | None = None
    period_end_date: str | None = None
    accession_number: str | None = None
    frame: str | None = None
    selection_reason: str | None = None
    fallback_alias_used: str | None = None


@dataclass(frozen=True)
class SecCompanyFactsSourceContext:
    ticker: str
    cik: str
    source_name: str
    provider_name: str
    context_format_version: str
    source_context_state: SecCompanyFactsContextState
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    source_refresh_snapshot_path: str | None
    canonical_fields: dict[str, Any | None]
    field_states: dict[str, SecCompanyFactsContextFieldState]
    fields: dict[str, SecCompanyFactsContextField]
    missing_canonical_fields: tuple[str, ...]
    unsupported_canonical_fields: tuple[str, ...] = ()
    invalid_canonical_fields: tuple[str, ...] = ()
    mode: str = "source-context-only"
    forbidden_authority_boundary: str = (
        "Source Context may expose source availability, selected source values, provenance, "
        "and missingness only. It must not emit analysis, observations, recommendations, "
        "portfolio review, delivery output, or Decision Engine authority."
    )
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_source_context_from_snapshot_path(
    snapshot_path: Path,
    *,
    expected_ticker: str | None = None,
    expected_cik: str | None = None,
) -> SecCompanyFactsSourceContext:
    try:
        snapshot = load_sec_companyfacts_raw_snapshot(
            snapshot_path,
            expected_ticker=expected_ticker,
            expected_cik=expected_cik,
        )
    except SecCompanyFactsSnapshotError as error:
        raise SecCompanyFactsContextBuildError(
            f"cannot build SEC CompanyFacts Source Context from snapshot: {error}"
        ) from error
    return build_sec_companyfacts_source_context_from_snapshot(snapshot)


def build_sec_companyfacts_source_context_from_snapshot(
    snapshot: SecCompanyFactsRawSnapshot,
) -> SecCompanyFactsSourceContext:
    if snapshot.source_name != SEC_COMPANYFACTS_SOURCE_NAME:
        raise SecCompanyFactsContextBuildError(
            f"unsupported source snapshot for SEC CompanyFacts Source Context: {snapshot.source_name}"
        )

    mapped_fields = map_sec_companyfacts_fields(snapshot.raw_payload)
    fields = {
        field_name: _context_field(field_name, mapped_field)
        for field_name, mapped_field in mapped_fields.items()
    }
    canonical_fields = {
        field_name: context_field.raw_value
        for field_name, context_field in fields.items()
    }
    field_states = {
        field_name: context_field.state
        for field_name, context_field in fields.items()
    }
    missing_canonical_fields = tuple(
        field_name
        for field_name in SEC_COMPANYFACTS_REQUIRED_FIELDS
        if field_states.get(field_name) == SecCompanyFactsContextFieldState.MISSING
    )

    return SecCompanyFactsSourceContext(
        ticker=snapshot.ticker,
        cik=snapshot.cik,
        source_name=snapshot.source_name,
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        context_format_version=SEC_COMPANYFACTS_SOURCE_CONTEXT_FORMAT_VERSION,
        source_context_state=_context_state(field_states),
        source_refresh_snapshot_id=snapshot.snapshot_id,
        source_refresh_fetched_at=snapshot.fetched_at,
        source_refresh_payload_format_version=snapshot.payload_format_version,
        source_refresh_snapshot_path=snapshot.path.as_posix() if snapshot.path is not None else None,
        canonical_fields=canonical_fields,
        field_states=field_states,
        fields=fields,
        missing_canonical_fields=missing_canonical_fields,
    )


def persist_sec_companyfacts_source_context(
    context: SecCompanyFactsSourceContext,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_SOURCE_CONTEXT_ROOT
    context_dir = root / run_id / context.ticker
    context_dir.mkdir(parents=True, exist_ok=True)
    context_path = context_dir / "source_context.json"
    if context_path.exists():
        raise FileExistsError(f"refusing to overwrite existing SEC CompanyFacts Source Context: {context_path}")
    context_path.write_text(
        json.dumps(_to_jsonable(context), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return context_path


def _context_field(
    field_name: str,
    mapped_field: SecCompanyFactsMappedField | None,
) -> SecCompanyFactsContextField:
    if mapped_field is None:
        return SecCompanyFactsContextField(
            canonical_field_name=field_name,
            state=SecCompanyFactsContextFieldState.MISSING,
        )
    return SecCompanyFactsContextField(
        canonical_field_name=field_name,
        state=SecCompanyFactsContextFieldState.PRESENT,
        raw_value=mapped_field.raw_value,
        sec_tag_selected=mapped_field.sec_tag_selected,
        provider_name=mapped_field.provider_name,
        taxonomy_namespace=mapped_field.taxonomy_namespace,
        unit=mapped_field.unit,
        fiscal_year=mapped_field.fiscal_year,
        fiscal_period=mapped_field.fiscal_period,
        filing_form=mapped_field.filing_form,
        filing_date=mapped_field.filing_date,
        period_start_date=mapped_field.period_start_date,
        period_end_date=mapped_field.period_end_date,
        accession_number=mapped_field.accession_number,
        frame=mapped_field.frame,
        selection_reason=mapped_field.selection_reason,
        fallback_alias_used=mapped_field.fallback_alias_used,
    )


def _context_state(
    field_states: dict[str, SecCompanyFactsContextFieldState],
) -> SecCompanyFactsContextState:
    present_count = sum(
        1
        for field_name in SEC_COMPANYFACTS_REQUIRED_FIELDS
        if field_states.get(field_name) == SecCompanyFactsContextFieldState.PRESENT
    )
    if present_count == len(SEC_COMPANYFACTS_REQUIRED_FIELDS):
        return SecCompanyFactsContextState.AVAILABLE
    if present_count > 0:
        return SecCompanyFactsContextState.PARTIAL
    return SecCompanyFactsContextState.MISSING


def _to_jsonable(context: SecCompanyFactsSourceContext) -> dict[str, Any]:
    payload = asdict(context)
    payload["source_context_state"] = context.source_context_state.value
    payload["field_states"] = {
        field_name: state.value
        for field_name, state in context.field_states.items()
    }
    payload["fields"] = {
        field_name: {
            **asdict(context_field),
            "state": context_field.state.value,
        }
        for field_name, context_field in context.fields.items()
    }
    return payload
