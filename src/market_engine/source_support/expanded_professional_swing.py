from __future__ import annotations

import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from market_engine.source_support.professional_swing import (
    DEFAULT_SOURCE_SNAPSHOT_ROOT,
    PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
    ProfessionalSwingSourceSupportError,
    ProfessionalSwingSourceSupportStatus,
    ProfessionalSwingTickerSourceSupport,
    classify_professional_swing_universe_source_support,
)
from market_engine.ticker_universe.professional_swing import (
    REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS,
    ProfessionalSwingUniverseValidationError,
    load_professional_swing_universe,
)
from market_engine.ticker_universe.professional_swing_expansion import (
    PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
    ProfessionalSwingUniverseExpansionResult,
)


EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION = (
    "market-engine-expanded-professional-swing-source-support-v1"
)


@dataclass(frozen=True)
class ExpandedProfessionalSwingTickerSourceSupport:
    ticker: str
    name: str
    market: str
    asset_type: str
    universe_entry_origin: str
    universe_entry_provenance: dict[str, Any]
    source_candidate_id: str | None
    source_candidate_reference: str | None
    status: str
    reason: str
    source_support: ProfessionalSwingTickerSourceSupport


@dataclass(frozen=True)
class ExpandedProfessionalSwingSourceSupportResult:
    format_version: str
    expansion_format_version: str
    source_support_format_version: str
    input_universe_path: str
    input_candidate_classification_path: str
    source_snapshot_root: str
    entries: tuple[ExpandedProfessionalSwingTickerSourceSupport, ...]
    summary_counts: dict[str, int]
    source_support_boundary: str = "Source-support classification only over approved local cached artifacts."


class ExpandedProfessionalSwingSourceSupportError(ProfessionalSwingSourceSupportError):
    """Raised when expanded-universe source support cannot be classified safely."""


def classify_expanded_professional_swing_universe_source_support(
    *,
    expansion_result: ProfessionalSwingUniverseExpansionResult,
    source_snapshot_root: str | Path = DEFAULT_SOURCE_SNAPSHOT_ROOT,
) -> ExpandedProfessionalSwingSourceSupportResult:
    _validate_expansion_result(expansion_result)
    source_root = _validated_path(Path(source_snapshot_root), field_name="source_snapshot_root")
    final_entries = tuple(_validated_universe_row(row) for row in expansion_result.final_universe_entries)
    _validate_unique_ticker_market(final_entries)
    provenance_by_key = _provenance_by_key(expansion_result=expansion_result)
    _attach_final_row_provenance(final_entries=final_entries, provenance_by_key=provenance_by_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        expanded_universe_path = Path(tmpdir) / "expanded_professional_swing_universe.csv"
        _write_universe_csv(expanded_universe_path, final_entries)
        source_support_result = classify_professional_swing_universe_source_support(
            universe_path=expanded_universe_path,
            source_snapshot_root=source_root,
        )

    entries = tuple(
        _expanded_entry(source_entry=source_entry, provenance_by_key=provenance_by_key)
        for source_entry in source_support_result.entries
    )
    return ExpandedProfessionalSwingSourceSupportResult(
        format_version=EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
        expansion_format_version=expansion_result.format_version,
        source_support_format_version=PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
        input_universe_path=expansion_result.input_universe_path,
        input_candidate_classification_path=expansion_result.input_candidate_classification_path,
        source_snapshot_root=source_root.as_posix(),
        entries=entries,
        summary_counts=_summary_counts(entries),
    )


def _validate_expansion_result(expansion_result: ProfessionalSwingUniverseExpansionResult) -> None:
    if not isinstance(expansion_result, ProfessionalSwingUniverseExpansionResult):
        raise ExpandedProfessionalSwingSourceSupportError(
            "Expanded Professional Swing source-support classification requires "
            "market-engine-professional-swing-universe-expansion-v1 input."
        )
    if expansion_result.format_version != PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION:
        raise ExpandedProfessionalSwingSourceSupportError(
            "Unsupported Professional Swing Universe expansion format version: "
            f"{expansion_result.format_version}"
        )
    if not isinstance(expansion_result.final_universe_entries, tuple):
        raise ExpandedProfessionalSwingSourceSupportError(
            "Malformed expansion output: final_universe_entries must be a tuple."
        )


def _validated_path(path: Path, *, field_name: str) -> Path:
    if ".." in path.parts:
        raise ExpandedProfessionalSwingSourceSupportError(f"Unsafe {field_name}: {path}")
    return path


def _validated_universe_row(row: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(row, Mapping):
        raise ExpandedProfessionalSwingSourceSupportError(
            "Malformed expansion output: final_universe_entries must contain mapping rows."
        )
    return {
        column: str(row.get(column, "")).strip()
        for column in REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS
    }


def _validate_unique_ticker_market(rows: tuple[dict[str, str], ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            raise ExpandedProfessionalSwingSourceSupportError(
                f"Duplicate expanded universe ticker/market key: {key[0]} / {key[1]}"
            )
        seen.add(key)


def _write_universe_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _provenance_by_key(
    *,
    expansion_result: ProfessionalSwingUniverseExpansionResult,
) -> dict[tuple[str, str], dict[str, Any]]:
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    universe_path = _validated_path(Path(expansion_result.input_universe_path), field_name="input_universe_path")
    try:
        existing_universe = load_professional_swing_universe(universe_path, include_inactive=True)
    except ProfessionalSwingUniverseValidationError as exc:
        raise ExpandedProfessionalSwingSourceSupportError(
            "Expanded source-support classification failed because the referenced "
            "input Professional Swing Universe is invalid."
        ) from exc

    for entry in existing_universe.entries:
        provenance[(entry.ticker, entry.market)] = {
            "universe_entry_origin": "existing_universe",
            "source_path": entry.source_path,
            "row_number": entry.row_number,
            "contract_version": entry.contract_version,
        }

    for decision in expansion_result.included_candidate_entries:
        proposed_entry = decision.proposed_universe_entry
        if proposed_entry is None:
            raise ExpandedProfessionalSwingSourceSupportError(
                f"Malformed expansion output: included candidate {decision.ticker} has no proposed row."
            )
        row = _validated_universe_row(proposed_entry)
        key = _row_key(row)
        provenance[key] = {
            "universe_entry_origin": "expansion_candidate",
            "source_candidate_id": decision.source_candidate_id,
            "source_candidate_reference": decision.source_candidate_reference,
            "candidate_bucket": decision.candidate_bucket,
            "expansion_decision": decision.inclusion_decision,
            "expansion_reason": decision.reason,
            "candidate_classification_path": expansion_result.input_candidate_classification_path,
        }
    return provenance


def _attach_final_row_provenance(
    *,
    final_entries: tuple[dict[str, str], ...],
    provenance_by_key: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in final_entries:
        key = _row_key(row)
        provenance = provenance_by_key.get(key)
        if provenance is None:
            raise ExpandedProfessionalSwingSourceSupportError(
                f"Expanded source-support classification has no source-row provenance for {key[0]} / {key[1]}."
            )
        provenance["expanded_universe_row"] = dict(row)


def _expanded_entry(
    *,
    source_entry: ProfessionalSwingTickerSourceSupport,
    provenance_by_key: dict[tuple[str, str], dict[str, Any]],
) -> ExpandedProfessionalSwingTickerSourceSupport:
    key = (source_entry.ticker, source_entry.market)
    provenance = provenance_by_key.get(key)
    if provenance is None:
        raise ExpandedProfessionalSwingSourceSupportError(
            f"Expanded source-support classification lost provenance for {source_entry.ticker} / {source_entry.market}."
        )
    expanded_row = provenance.get("expanded_universe_row")
    if not isinstance(expanded_row, Mapping):
        raise ExpandedProfessionalSwingSourceSupportError(
            f"Expanded source-support classification lost expanded row data for {source_entry.ticker} / {source_entry.market}."
        )
    return ExpandedProfessionalSwingTickerSourceSupport(
        ticker=source_entry.ticker,
        name=source_entry.name,
        market=source_entry.market,
        asset_type=str(expanded_row.get("asset_type", "")),
        universe_entry_origin=str(provenance["universe_entry_origin"]),
        universe_entry_provenance=dict(provenance),
        source_candidate_id=provenance.get("source_candidate_id"),
        source_candidate_reference=provenance.get("source_candidate_reference"),
        status=source_entry.status,
        reason=source_entry.reason,
        source_support=source_entry,
    )


def _row_key(row: Mapping[str, str]) -> tuple[str, str]:
    return (str(row.get("ticker", "")).strip().upper(), str(row.get("market", "")).strip().upper())


def _summary_counts(entries: tuple[ExpandedProfessionalSwingTickerSourceSupport, ...]) -> dict[str, int]:
    counts = {
        "total_expanded_universe_entries": len(entries),
        "supported_cached": _count(entries, ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value),
        "missing_snapshot": _count(entries, ProfessionalSwingSourceSupportStatus.MISSING_SNAPSHOT.value),
        "unsupported_sec_companyfacts": _count(
            entries,
            ProfessionalSwingSourceSupportStatus.UNSUPPORTED_SEC_COMPANYFACTS.value,
        ),
        "missing_required_source_field": _count(
            entries,
            ProfessionalSwingSourceSupportStatus.MISSING_REQUIRED_SOURCE_FIELD.value,
        ),
        "malformed_or_unreadable_source_artifact": _count(
            entries,
            ProfessionalSwingSourceSupportStatus.MALFORMED_OR_UNREADABLE_SOURCE_ARTIFACT.value,
        ),
        "ambiguous_identity": _count(entries, ProfessionalSwingSourceSupportStatus.AMBIGUOUS_IDENTITY.value),
        "manual_review_only": _count(entries, ProfessionalSwingSourceSupportStatus.MANUAL_REVIEW_ONLY.value),
        "excluded": _count(entries, ProfessionalSwingSourceSupportStatus.EXCLUDED.value),
    }
    counts["blocked_unsupported_or_manual_review_total"] = (
        counts["total_expanded_universe_entries"] - counts["supported_cached"]
    )
    return counts


def _count(entries: tuple[ExpandedProfessionalSwingTickerSourceSupport, ...], status: str) -> int:
    return sum(1 for entry in entries if entry.status == status)


def to_plain_dict(result: ExpandedProfessionalSwingSourceSupportResult) -> dict[str, Any]:
    return asdict(result)
