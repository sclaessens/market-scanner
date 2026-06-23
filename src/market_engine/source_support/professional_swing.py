from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from market_engine.source_intake.sec_companyfacts_fields import (
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
from market_engine.ticker_universe.professional_swing import (
    EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
    PROFESSIONAL_SWING_UNIVERSE_PATH,
    ProfessionalSwingUniverseEntry,
    ProfessionalSwingUniverseValidationError,
    load_professional_swing_universe,
)


PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION = (
    "market-engine-professional-swing-source-support-v1"
)
DEFAULT_SOURCE_SNAPSHOT_ROOT = Path("data/market_engine/source_snapshots")


class ProfessionalSwingSourceSupportError(ValueError):
    """Raised when source-support classification cannot be performed safely."""


class ProfessionalSwingSourceSupportStatus(StrEnum):
    SUPPORTED_CACHED = "supported_cached"
    MISSING_SNAPSHOT = "missing_snapshot"
    UNSUPPORTED_SEC_COMPANYFACTS = "unsupported_sec_companyfacts"
    MISSING_REQUIRED_SOURCE_FIELD = "missing_required_source_field"
    MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT = "malformed_or_unreadable_source_artifact"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    MANUAL_REVIEW_ONLY = "manual_review_only"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class SourceArtifactReference:
    source_name: str
    source_path: str
    ticker: str | None
    cik: str | None
    snapshot_id: str | None
    payload_format_version: str | None
    fetched_at: str | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RequiredSourceFieldSupport:
    canonical_field: str
    present: bool
    source_value: Any
    sec_tag_selected: str | None
    taxonomy_namespace: str | None
    unit: str | None
    filing_form: str | None
    fiscal_year: int | None
    period_end_date: str | None
    accession_number: str | None


@dataclass(frozen=True)
class ProfessionalSwingTickerSourceSupport:
    ticker: str
    name: str
    market: str
    active: bool
    universe_status: str
    source_policy_hint: str
    operator_priority: int
    status: str
    reason: str
    required_source_fields: tuple[RequiredSourceFieldSupport, ...]
    missing_required_source_fields: tuple[str, ...]
    source_artifacts: tuple[SourceArtifactReference, ...]
    provider_errors: tuple[SourceArtifactReference, ...]
    numeric_zero_evidence_present: bool
    universe_entry_reference: dict[str, Any]


@dataclass(frozen=True)
class ProfessionalSwingSourceSupportResult:
    format_version: str
    universe_contract_version: str
    universe_source_path: str
    source_snapshot_root: str
    required_source_fields: tuple[str, ...]
    entries: tuple[ProfessionalSwingTickerSourceSupport, ...]
    supported_count: int
    unsupported_count: int
    missing_snapshot_count: int
    missing_required_source_field_count: int
    malformed_or_unreadable_count: int
    ambiguous_identity_count: int
    manual_review_only_count: int
    excluded_count: int
    classification_boundary: str


def classify_professional_swing_universe_source_support(
    *,
    universe_path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
    source_snapshot_root: str | Path = DEFAULT_SOURCE_SNAPSHOT_ROOT,
) -> ProfessionalSwingSourceSupportResult:
    try:
        universe = load_professional_swing_universe(universe_path, include_inactive=True)
    except ProfessionalSwingUniverseValidationError as exc:
        raise ProfessionalSwingSourceSupportError(
            "Professional Swing Universe source-support classification failed because "
            "the universe CSV is invalid."
        ) from exc

    root = Path(source_snapshot_root)
    inventory = _discover_sec_companyfacts_artifacts(root)
    entries = tuple(
        _classify_entry(entry=entry, inventory=inventory)
        for entry in universe.entries
    )
    return ProfessionalSwingSourceSupportResult(
        format_version=PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
        universe_contract_version=EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
        universe_source_path=universe.source_path,
        source_snapshot_root=root.as_posix(),
        required_source_fields=tuple(SEC_COMPANYFACTS_REQUIRED_FIELDS),
        entries=entries,
        supported_count=_count(entries, ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED),
        unsupported_count=_count(entries, ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS),
        missing_snapshot_count=_count(entries, ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT),
        missing_required_source_field_count=_count(
            entries,
            ProfessionalSwingSourceSupportStatus.MISSING_REQUIRED_SOURCE_FIELD,
        ),
        malformed_or_unreadable_count=_count(
            entries,
            ProfessionalSwingSourceSupportStatus.MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT,
        ),
        ambiguous_identity_count=_count(entries, ProfessionalSwingSourceSupportStatus.AMBIGUOUS_IDENTITY),
        manual_review_only_count=_count(entries, ProfessionalSwingSourceSupportStatus.MANUAL_REVIEW_ONLY),
        excluded_count=_count(entries, ProfessionalSwingSourceSupportStatus.EXCLUDED),
        classification_boundary=(
            "Source-support classification only; no provider calls, no source refresh, "
            "no recommendations, and no execution authority."
        ),
    )


@dataclass(frozen=True)
class _SnapshotLoadFailure:
    path: Path
    ticker: str | None
    error_type: str
    error_message: str


@dataclass(frozen=True)
class _ProviderError:
    path: Path
    ticker: str
    cik: str | None
    error_type: str
    error_message: str


@dataclass(frozen=True)
class _ArtifactInventory:
    snapshots_by_ticker: dict[str, tuple[SecCompanyFactsRawSnapshot, ...]]
    failures_by_ticker: dict[str, tuple[_SnapshotLoadFailure, ...]]
    provider_errors_by_ticker: dict[str, tuple[_ProviderError, ...]]


def _discover_sec_companyfacts_artifacts(root: Path) -> _ArtifactInventory:
    snapshots: dict[str, list[SecCompanyFactsRawSnapshot]] = {}
    failures: dict[str, list[_SnapshotLoadFailure]] = {}
    for path in _candidate_snapshot_paths(root):
        try:
            snapshot = load_sec_companyfacts_raw_snapshot(path)
        except SecCompanyFactsSnapshotError as exc:
            ticker = _ticker_from_snapshot_path(path)
            if ticker is not None:
                failures.setdefault(ticker, []).append(
                    _SnapshotLoadFailure(
                        path=path,
                        ticker=ticker,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            continue
        snapshots.setdefault(snapshot.ticker, []).append(snapshot)

    provider_errors: dict[str, list[_ProviderError]] = {}
    for path in _candidate_provider_error_paths(root):
        for error in _read_provider_errors(path):
            provider_errors.setdefault(error.ticker, []).append(error)

    return _ArtifactInventory(
        snapshots_by_ticker={
            ticker: tuple(sorted(values, key=lambda snapshot: _snapshot_sort_key(snapshot)))
            for ticker, values in snapshots.items()
        },
        failures_by_ticker={
            ticker: tuple(sorted(values, key=lambda failure: failure.path.as_posix()))
            for ticker, values in failures.items()
        },
        provider_errors_by_ticker={
            ticker: tuple(sorted(values, key=lambda error: error.path.as_posix()))
            for ticker, values in provider_errors.items()
        },
    )


def _classify_entry(
    *,
    entry: ProfessionalSwingUniverseEntry,
    inventory: _ArtifactInventory,
) -> ProfessionalSwingTickerSourceSupport:
    universe_reference = _universe_entry_reference(entry)
    if not entry.active or entry.universe_status in {"blocked", "rejected"}:
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.EXCLUDED,
            reason="Universe row is inactive, blocked, or rejected.",
            universe_entry_reference=universe_reference,
        )
    if entry.universe_status == "manual_review_only" or entry.source_policy_hint == "manual_review_only":
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.MANUAL_REVIEW_ONLY,
            reason="Universe row is marked manual_review_only.",
            universe_entry_reference=universe_reference,
        )
    if entry.source_policy_hint == "unsupported":
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS,
            reason="Universe row is marked unsupported for cached SEC CompanyFacts source support.",
            universe_entry_reference=universe_reference,
        )

    snapshots = inventory.snapshots_by_ticker.get(entry.ticker, ())
    failures = inventory.failures_by_ticker.get(entry.ticker, ())
    provider_errors = inventory.provider_errors_by_ticker.get(entry.ticker, ())
    provider_error_refs = tuple(_provider_error_reference(error) for error in provider_errors)

    if len(snapshots) > 1:
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.AMBIGUOUS_IDENTITY,
            reason="Multiple local SEC CompanyFacts snapshots match the universe ticker.",
            source_artifacts=tuple(_snapshot_reference(snapshot) for snapshot in snapshots),
            provider_errors=provider_error_refs,
            universe_entry_reference=universe_reference,
        )
    if snapshots:
        return _classify_snapshot_entry(
            entry=entry,
            snapshot=snapshots[0],
            provider_errors=provider_error_refs,
            universe_entry_reference=universe_reference,
        )
    if failures:
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT,
            reason="A matching local SEC CompanyFacts source artifact exists but cannot be read safely.",
            source_artifacts=tuple(_failure_reference(failure) for failure in failures),
            provider_errors=provider_error_refs,
            universe_entry_reference=universe_reference,
        )
    if provider_errors:
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS,
            reason="Local source-refresh provider error records the ticker as unsupported or unavailable.",
            provider_errors=provider_error_refs,
            universe_entry_reference=universe_reference,
        )
    return _ticker_result(
        entry=entry,
        status=ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT,
        reason="No approved local SEC CompanyFacts source snapshot was found for the universe ticker.",
        universe_entry_reference=universe_reference,
    )


def _classify_snapshot_entry(
    *,
    entry: ProfessionalSwingUniverseEntry,
    snapshot: SecCompanyFactsRawSnapshot,
    provider_errors: tuple[SourceArtifactReference, ...],
    universe_entry_reference: dict[str, Any],
) -> ProfessionalSwingTickerSourceSupport:
    mapped_fields = map_sec_companyfacts_fields(snapshot.raw_payload)
    field_support = tuple(
        _field_support(canonical_field, mapped_fields[canonical_field])
        for canonical_field in SEC_COMPANYFACTS_REQUIRED_FIELDS
    )
    missing_fields = tuple(
        field.canonical_field
        for field in field_support
        if not field.present
    )
    zero_present = any(field.present and field.source_value == 0 for field in field_support)
    if missing_fields:
        return _ticker_result(
            entry=entry,
            status=ProfessionalSwingSourceSupportStatus.MISSING_REQUIRED_SOURCE_FIELD,
            reason="Local SEC CompanyFacts snapshot is present but required mapped fields are missing.",
            required_source_fields=field_support,
            missing_required_source_fields=missing_fields,
            source_artifacts=(_snapshot_reference(snapshot),),
            provider_errors=provider_errors,
            numeric_zero_evidence_present=zero_present,
            universe_entry_reference=universe_entry_reference,
        )
    return _ticker_result(
        entry=entry,
        status=ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED,
        reason="Local SEC CompanyFacts snapshot contains all required mapped source fields.",
        required_source_fields=field_support,
        source_artifacts=(_snapshot_reference(snapshot),),
        provider_errors=provider_errors,
        numeric_zero_evidence_present=zero_present,
        universe_entry_reference=universe_entry_reference,
    )


def _ticker_result(
    *,
    entry: ProfessionalSwingUniverseEntry,
    status: ProfessionalSwingSourceSupportStatus,
    reason: str,
    required_source_fields: tuple[RequiredSourceFieldSupport, ...] = (),
    missing_required_source_fields: tuple[str, ...] = (),
    source_artifacts: tuple[SourceArtifactReference, ...] = (),
    provider_errors: tuple[SourceArtifactReference, ...] = (),
    numeric_zero_evidence_present: bool = False,
    universe_entry_reference: dict[str, Any],
) -> ProfessionalSwingTickerSourceSupport:
    return ProfessionalSwingTickerSourceSupport(
        ticker=entry.ticker,
        name=entry.name,
        market=entry.market,
        active=entry.active,
        universe_status=entry.universe_status,
        source_policy_hint=entry.source_policy_hint,
        operator_priority=entry.operator_priority,
        status=status.value,
        reason=reason,
        required_source_fields=required_source_fields,
        missing_required_source_fields=missing_required_source_fields,
        source_artifacts=source_artifacts,
        provider_errors=provider_errors,
        numeric_zero_evidence_present=numeric_zero_evidence_present,
        universe_entry_reference=universe_entry_reference,
    )


def _field_support(
    canonical_field: str,
    mapped_field: SecCompanyFactsMappedField | None,
) -> RequiredSourceFieldSupport:
    if mapped_field is None:
        return RequiredSourceFieldSupport(
            canonical_field=canonical_field,
            present=False,
            source_value=None,
            sec_tag_selected=None,
            taxonomy_namespace=None,
            unit=None,
            filing_form=None,
            fiscal_year=None,
            period_end_date=None,
            accession_number=None,
        )
    return RequiredSourceFieldSupport(
        canonical_field=canonical_field,
        present=True,
        source_value=mapped_field.raw_value,
        sec_tag_selected=mapped_field.sec_tag_selected,
        taxonomy_namespace=mapped_field.taxonomy_namespace,
        unit=mapped_field.unit,
        filing_form=mapped_field.filing_form,
        fiscal_year=mapped_field.fiscal_year,
        period_end_date=mapped_field.period_end_date,
        accession_number=mapped_field.accession_number,
    )


def _candidate_snapshot_paths(root: Path) -> tuple[Path, ...]:
    patterns = ("*/raw/*.json", "sec_companyfacts/*/raw/*.json", "sec_companyfacts/*/raw/*.JSON")
    return tuple(
        sorted(
            {
                path
                for pattern in patterns
                for path in root.glob(pattern)
                if path.is_file()
            },
            key=lambda path: path.as_posix(),
        )
    )


def _candidate_provider_error_paths(root: Path) -> tuple[Path, ...]:
    patterns = ("*/provider_errors.csv", "sec_companyfacts/*/provider_errors.csv")
    return tuple(
        sorted(
            {
                path
                for pattern in patterns
                for path in root.glob(pattern)
                if path.is_file()
            },
            key=lambda path: path.as_posix(),
        )
    )


def _read_provider_errors(path: Path) -> tuple[_ProviderError, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            errors = []
            for row in reader:
                ticker = str(row.get("ticker", "")).strip().upper()
                if not ticker:
                    continue
                errors.append(
                    _ProviderError(
                        path=path,
                        ticker=ticker,
                        cik=str(row.get("cik", "")).strip() or None,
                        error_type=str(row.get("error_type", "")).strip(),
                        error_message=str(row.get("error_message", "")).strip(),
                    )
                )
            return tuple(errors)
    except (OSError, csv.Error):
        return ()


def _snapshot_reference(snapshot: SecCompanyFactsRawSnapshot) -> SourceArtifactReference:
    return SourceArtifactReference(
        source_name=snapshot.source_name,
        source_path=snapshot.path.as_posix() if snapshot.path else "",
        ticker=snapshot.ticker,
        cik=snapshot.cik,
        snapshot_id=snapshot.snapshot_id,
        payload_format_version=snapshot.payload_format_version,
        fetched_at=snapshot.fetched_at,
    )


def _failure_reference(failure: _SnapshotLoadFailure) -> SourceArtifactReference:
    return SourceArtifactReference(
        source_name=SEC_COMPANYFACTS_SOURCE_NAME,
        source_path=failure.path.as_posix(),
        ticker=failure.ticker,
        cik=None,
        snapshot_id=None,
        payload_format_version=None,
        fetched_at=None,
        error_type=failure.error_type,
        error_message=failure.error_message,
    )


def _provider_error_reference(error: _ProviderError) -> SourceArtifactReference:
    return SourceArtifactReference(
        source_name=SEC_COMPANYFACTS_SOURCE_NAME,
        source_path=error.path.as_posix(),
        ticker=error.ticker,
        cik=error.cik,
        snapshot_id=None,
        payload_format_version=None,
        fetched_at=None,
        error_type=error.error_type,
        error_message=error.error_message,
    )


def _universe_entry_reference(entry: ProfessionalSwingUniverseEntry) -> dict[str, Any]:
    return {
        "contract_version": entry.contract_version,
        "source_path": entry.source_path,
        "row_number": entry.row_number,
        "ticker": entry.ticker,
        "market": entry.market,
        "active": entry.active,
        "universe_status": entry.universe_status,
        "source_policy_hint": entry.source_policy_hint,
        "operator_priority": entry.operator_priority,
        "metadata": dict(entry.metadata),
    }


def _ticker_from_snapshot_path(path: Path) -> str | None:
    stem = path.stem
    suffix = "_companyfacts"
    ticker = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    ticker = ticker.strip().upper()
    return ticker or None


def _snapshot_sort_key(snapshot: SecCompanyFactsRawSnapshot) -> tuple[str, str, str]:
    return (
        snapshot.fetched_at,
        snapshot.snapshot_id,
        snapshot.path.as_posix() if snapshot.path else "",
    )


def _count(
    entries: tuple[ProfessionalSwingTickerSourceSupport, ...],
    status: ProfessionalSwingSourceSupportStatus,
) -> int:
    return sum(1 for entry in entries if entry.status == status.value)


def to_plain_dict(result: ProfessionalSwingSourceSupportResult) -> dict[str, Any]:
    return asdict(result)
