from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from market_engine.ticker_universe import (
    EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
    PROFESSIONAL_SWING_UNIVERSE_PATH,
    ProfessionalSwingUniverseValidationError,
    load_professional_swing_universe,
)


EDITABLE_UNIVERSE_RUNTIME_INPUT_FORMAT_VERSION = (
    "market-engine-editable-universe-runtime-input-v1"
)
PROFESSIONAL_SWING_RUNTIME_INPUT_SELECTION_POLICY = (
    "active_true_and_universe_status_candidate_or_watching_and_"
    "source_policy_hint_cached_source_candidate_or_unknown"
)


class EditableUniverseRuntimeInputError(ValueError):
    pass


@dataclass(frozen=True)
class EditableUniverseRuntimeInput:
    format_version: str
    source_contract_version: str
    source_path: str
    selection_policy: str
    requested_tickers: tuple[str, ...]
    loaded_row_count: int
    selected_row_count: int
    excluded_inactive_tickers: tuple[str, ...]
    excluded_source_mapping_required_tickers: tuple[str, ...]
    excluded_manual_review_only_tickers: tuple[str, ...]
    excluded_unsupported_tickers: tuple[str, ...]
    excluded_other_status_tickers: tuple[str, ...]
    source_policy_hint_authority: str = "operator_hint_not_source_support_authority"
    canonical_promotion_authority: bool = False
    provider_call_authority: bool = False
    runtime_input_authority: str = "local_cached_source_batch_requested_tickers_only"

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "source_contract_version": self.source_contract_version,
            "source_path": self.source_path,
            "selection_policy": self.selection_policy,
            "requested_tickers": self.requested_tickers,
            "loaded_row_count": self.loaded_row_count,
            "selected_row_count": self.selected_row_count,
            "excluded_inactive_tickers": self.excluded_inactive_tickers,
            "excluded_source_mapping_required_tickers": (
                self.excluded_source_mapping_required_tickers
            ),
            "excluded_manual_review_only_tickers": self.excluded_manual_review_only_tickers,
            "excluded_unsupported_tickers": self.excluded_unsupported_tickers,
            "excluded_other_status_tickers": self.excluded_other_status_tickers,
            "source_policy_hint_authority": self.source_policy_hint_authority,
            "canonical_promotion_authority": self.canonical_promotion_authority,
            "provider_call_authority": self.provider_call_authority,
            "runtime_input_authority": self.runtime_input_authority,
        }


def build_professional_swing_runtime_input(
    path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
) -> EditableUniverseRuntimeInput:
    try:
        universe = load_professional_swing_universe(path, include_inactive=True)
    except ProfessionalSwingUniverseValidationError as exc:
        raise EditableUniverseRuntimeInputError(
            f"Unable to load editable Professional Swing Universe runtime input: {exc}"
        ) from exc

    selected_entries = tuple(
        entry
        for entry in universe.entries
        if entry.active
        and entry.universe_status in {"candidate", "watching"}
        and entry.source_policy_hint in {"cached_source_candidate", "unknown"}
    )
    requested_tickers = tuple(entry.ticker for entry in selected_entries)
    if not requested_tickers:
        raise EditableUniverseRuntimeInputError(
            "Editable Professional Swing Universe produced no local runtime input tickers."
        )

    return EditableUniverseRuntimeInput(
        format_version=EDITABLE_UNIVERSE_RUNTIME_INPUT_FORMAT_VERSION,
        source_contract_version=EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
        source_path=universe.source_path,
        selection_policy=PROFESSIONAL_SWING_RUNTIME_INPUT_SELECTION_POLICY,
        requested_tickers=requested_tickers,
        loaded_row_count=universe.loaded_row_count,
        selected_row_count=len(requested_tickers),
        excluded_inactive_tickers=tuple(
            entry.ticker for entry in universe.entries if not entry.active
        ),
        excluded_source_mapping_required_tickers=tuple(
            entry.ticker
            for entry in universe.entries
            if entry.source_policy_hint == "source_mapping_required"
        ),
        excluded_manual_review_only_tickers=tuple(
            entry.ticker
            for entry in universe.entries
            if entry.source_policy_hint == "manual_review_only"
        ),
        excluded_unsupported_tickers=tuple(
            entry.ticker
            for entry in universe.entries
            if entry.source_policy_hint == "unsupported"
        ),
        excluded_other_status_tickers=tuple(
            entry.ticker
            for entry in universe.entries
            if entry.active and entry.universe_status not in {"candidate", "watching"}
        ),
    )


def build_cached_source_batch_argv_from_professional_swing_universe(
    *,
    path: str | Path = PROFESSIONAL_SWING_UNIVERSE_PATH,
    source_snapshot_root: str | Path | None = None,
    portfolio_context: str | Path | None = None,
    batch_id: str | None = None,
    generated_at: str | None = None,
    ticker_limit: int | None = None,
    write_local_artifacts: bool = False,
    artifact_output_root: str | Path | None = None,
    emit_json: bool = False,
) -> tuple[str, ...]:
    runtime_input = build_professional_swing_runtime_input(path)
    argv: list[str] = ["--tickers", ",".join(runtime_input.requested_tickers)]
    if source_snapshot_root is not None:
        argv.extend(["--source-snapshot-root", str(source_snapshot_root)])
    if portfolio_context is not None:
        argv.extend(["--portfolio-context", str(portfolio_context)])
    if batch_id is not None:
        argv.extend(["--batch-id", batch_id])
    if generated_at is not None:
        argv.extend(["--generated-at", generated_at])
    if ticker_limit is not None:
        argv.extend(["--ticker-limit", str(ticker_limit)])
    if write_local_artifacts:
        argv.append("--write-local-artifacts")
        if artifact_output_root is not None:
            argv.extend(["--artifact-output-root", str(artifact_output_root)])
    if emit_json:
        argv.append("--emit-json")
    return tuple(argv)


def selected_tickers_from_runtime_input(
    runtime_input: EditableUniverseRuntimeInput,
    *,
    ticker_limit: int | None = None,
) -> tuple[str, ...]:
    tickers: Sequence[str] = runtime_input.requested_tickers
    if ticker_limit is None:
        return tuple(tickers)
    if ticker_limit < 1:
        raise EditableUniverseRuntimeInputError("ticker_limit must be greater than zero.")
    return tuple(tickers[:ticker_limit])
