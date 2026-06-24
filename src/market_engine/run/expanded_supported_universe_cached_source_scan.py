from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.run.cached_source_batch_execution import (
    MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
    CachedSourceBatchDryRunError,
    build_cached_source_batch_dry_run,
)
from market_engine.source_support import (
    EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
    ExpandedProfessionalSwingSourceSupportError,
    ExpandedProfessionalSwingSourceSupportResult,
    ProfessionalSwingSourceSupportStatus,
    classify_expanded_professional_swing_universe_source_support,
)
from market_engine.ticker_universe.professional_swing import PROFESSIONAL_SWING_UNIVERSE_PATH
from market_engine.ticker_universe.professional_swing_expansion import (
    PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
    ProfessionalSwingUniverseExpansionError,
    build_professional_swing_universe_expansion,
)


EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION = (
    "market-engine-expanded-supported-universe-cached-source-scan-v1"
)
DEFAULT_SOURCE_SNAPSHOT_ROOT = Path("data/market_engine/source_snapshots")
DEFAULT_ARTIFACT_OUTPUT_ROOT = Path("artifacts/market_engine")


class ExpandedSupportedUniverseCachedSourceScanError(ValueError):
    pass


@dataclass(frozen=True)
class ExpandedSupportedUniverseCachedSourceScanResult:
    format_version: str
    generated_at: str | None
    batch_id: str
    input_universe_path: str
    input_candidate_classification_path: str
    source_snapshot_root: str
    source_support_format_version: str
    expansion_format_version: str
    batch_dry_run_format_version: str
    source_support_summary_counts: dict[str, int]
    expanded_universe_count: int
    supported_cached_tickers: tuple[str, ...]
    non_supported_entries: tuple[dict[str, str], ...]
    batch_payload: dict[str, Any] | None
    run_state: str
    blocked_reasons: tuple[str, ...]
    safety_boundary: str = (
        "Expanded supported-universe cached-source scan only; no provider calls, "
        "no live data refresh, no broker access, no portfolio or watchlist mutation, "
        "and no recommendation or order semantics."
    )
    live_provider_call_made: bool = False
    non_production_batch: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_expanded_supported_universe_cached_source_scan(
    *,
    candidate_classification_path: str | Path,
    existing_universe_path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
    source_snapshot_root: str | Path = DEFAULT_SOURCE_SNAPSHOT_ROOT,
    batch_id: str,
    generated_at: str | None = None,
    ticker_limit: int | None = None,
    write_local_artifacts: bool = False,
    artifact_output_root: str | Path = DEFAULT_ARTIFACT_OUTPUT_ROOT,
) -> ExpandedSupportedUniverseCachedSourceScanResult:
    safe_batch_id = _required_text(batch_id, field_name="batch_id")
    candidate_path = _validated_path(Path(candidate_classification_path), field_name="candidate_classification_path")
    universe_path = _validated_path(Path(existing_universe_path), field_name="existing_universe_path")
    source_root = _validated_path(Path(source_snapshot_root), field_name="source_snapshot_root")
    output_root = _validated_path(Path(artifact_output_root), field_name="artifact_output_root")
    if ticker_limit is not None and ticker_limit < 1:
        raise ExpandedSupportedUniverseCachedSourceScanError("ticker_limit must be positive when provided.")

    try:
        expansion = build_professional_swing_universe_expansion(
            existing_universe_path=universe_path,
            candidate_classification_path=candidate_path,
        )
        source_support = classify_expanded_professional_swing_universe_source_support(
            expansion_result=expansion,
            source_snapshot_root=source_root,
        )
    except (
        ProfessionalSwingUniverseExpansionError,
        ExpandedProfessionalSwingSourceSupportError,
    ) as exc:
        raise ExpandedSupportedUniverseCachedSourceScanError(
            "ME-RUN23 expanded supported-universe scan failed closed before cached-source execution."
        ) from exc

    supported_tickers = tuple(
        entry.ticker
        for entry in source_support.entries
        if entry.status == ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    )
    if ticker_limit is not None:
        supported_tickers = supported_tickers[:ticker_limit]
    non_supported_entries = tuple(
        _non_supported_entry_payload(entry)
        for entry in source_support.entries
        if entry.status != ProfessionalSwingSourceSupportStatus.SUPPORTED_CACHED.value
    )

    if not supported_tickers:
        return _blocked_result(
            batch_id=safe_batch_id,
            generated_at=generated_at,
            universe_path=universe_path,
            candidate_path=candidate_path,
            source_root=source_root,
            source_support=source_support,
            non_supported_entries=non_supported_entries,
            reason="No expanded Professional Swing Universe entries are supported_cached.",
        )

    try:
        batch_payload = build_cached_source_batch_dry_run(
            source_snapshot_root=source_root,
            batch_id=safe_batch_id,
            generated_at=generated_at,
            requested_tickers=supported_tickers,
            discover_cached_tickers=False,
            ticker_limit=None,
            write_local_artifacts=write_local_artifacts,
            artifact_output_root=output_root,
            artifact_created_at=generated_at if write_local_artifacts else None,
        )
    except CachedSourceBatchDryRunError as exc:
        raise ExpandedSupportedUniverseCachedSourceScanError(
            "ME-RUN23 cached-source batch dry-run failed closed."
        ) from exc

    return ExpandedSupportedUniverseCachedSourceScanResult(
        format_version=EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION,
        generated_at=generated_at,
        batch_id=safe_batch_id,
        input_universe_path=universe_path.as_posix(),
        input_candidate_classification_path=candidate_path.as_posix(),
        source_snapshot_root=source_root.as_posix(),
        source_support_format_version=EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
        expansion_format_version=PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
        batch_dry_run_format_version=MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
        source_support_summary_counts=dict(source_support.summary_counts),
        expanded_universe_count=len(source_support.entries),
        supported_cached_tickers=supported_tickers,
        non_supported_entries=non_supported_entries,
        batch_payload=batch_payload,
        run_state="completed",
        blocked_reasons=(),
    )


def _blocked_result(
    *,
    batch_id: str,
    generated_at: str | None,
    universe_path: Path,
    candidate_path: Path,
    source_root: Path,
    source_support: ExpandedProfessionalSwingSourceSupportResult,
    non_supported_entries: tuple[dict[str, str], ...],
    reason: str,
) -> ExpandedSupportedUniverseCachedSourceScanResult:
    return ExpandedSupportedUniverseCachedSourceScanResult(
        format_version=EXPANDED_SUPPORTED_UNIVERSE_CACHED_SOURCE_SCAN_FORMAT_VERSION,
        generated_at=generated_at,
        batch_id=batch_id,
        input_universe_path=universe_path.as_posix(),
        input_candidate_classification_path=candidate_path.as_posix(),
        source_snapshot_root=source_root.as_posix(),
        source_support_format_version=EXPANDED_PROFESSIONAL_SWING_SOURCE_SUPPORT_FORMAT_VERSION,
        expansion_format_version=PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
        batch_dry_run_format_version=MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
        source_support_summary_counts=dict(source_support.summary_counts),
        expanded_universe_count=len(source_support.entries),
        supported_cached_tickers=(),
        non_supported_entries=non_supported_entries,
        batch_payload=None,
        run_state="blocked_no_supported_cached_entries",
        blocked_reasons=(reason,),
    )


def _non_supported_entry_payload(entry: Any) -> dict[str, str]:
    return {
        "ticker": entry.ticker,
        "name": entry.name,
        "market": entry.market,
        "asset_type": entry.asset_type,
        "universe_entry_origin": entry.universe_entry_origin,
        "status": entry.status,
        "reason": entry.reason,
        "source_candidate_id": entry.source_candidate_id or "",
        "source_candidate_reference": entry.source_candidate_reference or "",
    }


def _validated_path(path: Path, *, field_name: str) -> Path:
    if ".." in path.parts:
        raise ExpandedSupportedUniverseCachedSourceScanError(f"Unsafe {field_name}: {path}")
    return path


def _required_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ExpandedSupportedUniverseCachedSourceScanError(f"{field_name} is required.")
    return text


def to_plain_dict(result: ExpandedSupportedUniverseCachedSourceScanResult) -> dict[str, Any]:
    return result.to_payload()
