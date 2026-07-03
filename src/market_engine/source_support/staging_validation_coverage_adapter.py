from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION,
)
from market_engine.source_support.cached_source_coverage import (
    BlockerCode,
    CachedSourceCoverageInput,
    CoverageStatus,
    SourceFamily,
    SourceFamilyEvidence,
    TargetCapability,
)


class StagingValidationCoverageAdapterError(ValueError):
    """Raised when staging evidence cannot be adapted without guessing."""


_CANONICAL_SOURCE_FAMILY_MAP: dict[str, SourceFamily] = {
    "company_profile": SourceFamily.COMPANY_PROFILE,
    "sec_companyfacts": SourceFamily.FUNDAMENTAL_FACTS,
}

_MANIFEST_ISSUE_MARKERS = (
    "manifest_",
    "local_manifest_",
    "local_payload_",
    "referenced_snapshot_",
)

_PROVENANCE_FIELDS = (
    "ticker",
    "source_name",
    "source_retrieved_at_utc",
    "manifest_path",
)


def adapt_staging_validation_to_cached_source_coverage_input(
    staging_entry: Mapping[str, Any],
    *,
    universe_supported: bool,
    target_capability: TargetCapability,
    source_family_hint: SourceFamily | None = None,
) -> CachedSourceCoverageInput:
    """Adapt one in-memory staging-validation entry into generic coverage input."""
    entry = _validated_entry(staging_entry)
    ticker = _required_text(entry, "ticker")
    market = _optional_text(entry, "market")
    raw_source_family = _optional_text(entry, "source_family")
    mapped_family = _CANONICAL_SOURCE_FAMILY_MAP.get(raw_source_family or "")
    source_family = mapped_family or source_family_hint
    issues = _text_tuple(entry, "issues")
    additional_blockers: list[BlockerCode] = []

    if source_family is None:
        additional_blockers.append(
            BlockerCode.INVALID_MANIFEST
            if _manifest_is_invalid(entry, issues)
            else BlockerCode.UNSUPPORTED_SOURCE_FAMILY
        )
        source_evidence: tuple[SourceFamilyEvidence, ...] = ()
    else:
        source_evidence = (
            SourceFamilyEvidence(
                source_family=source_family,
                support_status=(
                    CoverageStatus.SUPPORTED
                    if mapped_family is not None or raw_source_family is None
                    else CoverageStatus.UNSUPPORTED
                ),
                availability_status=_availability_status(entry),
                manifest_status=_manifest_status(entry, issues),
                provenance_status=_provenance_status(entry, issues),
                freshness_status=_freshness_status(entry),
                consumability_status=_consumability_status(entry),
                completeness_status=_completeness_status(
                    entry=entry,
                    source_family=source_family,
                ),
                evidence_reference=_evidence_reference(entry),
            ),
        )

    return CachedSourceCoverageInput(
        ticker=ticker,
        market=market,
        universe_supported=universe_supported,
        target_capability=target_capability,
        source_evidence=source_evidence,
        additional_blockers=tuple(additional_blockers),
    )


def adapt_staging_validation_batch_to_cached_source_coverage_inputs(
    staging_validation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    universe_supported: bool,
    target_capability: TargetCapability,
    source_family_hint: SourceFamily | None = None,
) -> tuple[CachedSourceCoverageInput, ...]:
    """Adapt a report or ordered entry sequence while preserving entry order."""
    if isinstance(staging_validation, Mapping):
        if (
            staging_validation.get("report_format_version")
            != CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION
        ):
            raise StagingValidationCoverageAdapterError(
                "staging validation report format is unsupported"
            )
        raw_entries = staging_validation.get("entries")
        if not isinstance(raw_entries, (list, tuple)):
            raise StagingValidationCoverageAdapterError(
                "staging validation report entries must be a sequence"
            )
        entries = tuple(raw_entries)
    elif isinstance(staging_validation, Sequence) and not isinstance(
        staging_validation,
        (str, bytes),
    ):
        entries = tuple(staging_validation)
    else:
        raise StagingValidationCoverageAdapterError(
            "staging validation batch must be a report or entry sequence"
        )
    if not entries:
        raise StagingValidationCoverageAdapterError(
            "staging validation batch must not be empty"
        )
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise StagingValidationCoverageAdapterError(
            "staging validation batch entries must be mappings"
        )
    return tuple(
        adapt_staging_validation_to_cached_source_coverage_input(
            entry,
            universe_supported=universe_supported,
            target_capability=target_capability,
            source_family_hint=source_family_hint,
        )
        for entry in entries
    )


def _validated_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise StagingValidationCoverageAdapterError(
            "staging validation entry must be a mapping"
        )
    status = entry.get("staging_validation_status")
    if status not in {
        "accepted",
        "rejected",
        "missing_manifest",
        "malformed_manifest",
        "unknown_format",
    }:
        raise StagingValidationCoverageAdapterError(
            "staging validation status is missing or unsupported"
        )
    for field_name in (
        "accepted_for_cached_source_staging",
        "usable_for_cached_source_dry_run",
    ):
        value = entry.get(field_name)
        if value is not None and not isinstance(value, bool):
            raise StagingValidationCoverageAdapterError(
                f"{field_name} must be a boolean or null"
            )
    _text_tuple(entry, "issues")
    _text_tuple(entry, "validation_errors")
    _text_tuple(entry, "validation_warnings")
    return entry


def _required_text(entry: Mapping[str, Any], field_name: str) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise StagingValidationCoverageAdapterError(
            f"staging validation {field_name} must be non-empty text without padding"
        )
    return value


def _optional_text(entry: Mapping[str, Any], field_name: str) -> str | None:
    value = entry.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise StagingValidationCoverageAdapterError(
            f"staging validation {field_name} must be non-empty text or null"
        )
    return value


def _text_tuple(entry: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    value = entry.get(field_name, ())
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise StagingValidationCoverageAdapterError(
            f"staging validation {field_name} must contain text values"
        )
    return tuple(value)


def _availability_status(entry: Mapping[str, Any]) -> CoverageStatus:
    return (
        CoverageStatus.AVAILABLE
        if _optional_text(entry, "payload_path") is not None
        or _optional_text(entry, "directory_path") is not None
        else CoverageStatus.MISSING_SNAPSHOT
    )


def _manifest_is_invalid(
    entry: Mapping[str, Any],
    issues: tuple[str, ...],
) -> bool:
    return entry["staging_validation_status"] in {
        "missing_manifest",
        "malformed_manifest",
        "unknown_format",
    } or any(
        issue.startswith(_MANIFEST_ISSUE_MARKERS)
        for issue in issues
    )


def _manifest_status(
    entry: Mapping[str, Any],
    issues: tuple[str, ...],
) -> CoverageStatus:
    if _manifest_is_invalid(entry, issues):
        return CoverageStatus.INVALID_MANIFEST
    if _optional_text(entry, "manifest_path") is None:
        return CoverageStatus.INVALID_MANIFEST
    return CoverageStatus.ACCEPTED


def _provenance_status(
    entry: Mapping[str, Any],
    issues: tuple[str, ...],
) -> CoverageStatus:
    if any(
        f"{field_name}_missing" in issues or f"{field_name}_invalid" in issues
        for field_name in _PROVENANCE_FIELDS
    ):
        return CoverageStatus.UNPROVENANCED
    if all(
        _optional_text(entry, field_name) is not None
        for field_name in _PROVENANCE_FIELDS
    ):
        return CoverageStatus.ACCEPTED
    return CoverageStatus.UNPROVENANCED


def _freshness_status(entry: Mapping[str, Any]) -> CoverageStatus:
    return (
        CoverageStatus.ACCEPTED
        if _optional_text(entry, "staleness_status") == "fresh"
        else CoverageStatus.STALE
    )


def _consumability_status(entry: Mapping[str, Any]) -> CoverageStatus:
    accepted = entry.get("accepted_for_cached_source_staging") is True
    usable = entry.get("usable_for_cached_source_dry_run")
    if usable is None:
        usable = accepted
    return (
        CoverageStatus.ACCEPTED
        if accepted
        and usable is True
        and _optional_text(entry, "validation_status") == "passed"
        and not _text_tuple(entry, "validation_errors")
        else CoverageStatus.NOT_CONSUMABLE
    )


def _completeness_status(
    *,
    entry: Mapping[str, Any],
    source_family: SourceFamily,
) -> CoverageStatus:
    if (
        source_family is SourceFamily.COMPANY_PROFILE
        and entry.get("accepted_for_cached_source_staging") is True
    ):
        return CoverageStatus.ACCEPTED
    return CoverageStatus.PARTIAL


def _evidence_reference(entry: Mapping[str, Any]) -> str | None:
    manifest_path = _optional_text(entry, "manifest_path")
    if manifest_path is not None:
        return f"staging-validation://{manifest_path}"
    directory_path = _optional_text(entry, "directory_path")
    if directory_path is not None:
        return f"staging-validation://{directory_path}"
    return None
