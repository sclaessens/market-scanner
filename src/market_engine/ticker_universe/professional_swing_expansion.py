from __future__ import annotations

import csv
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.candidate_classification import (
    ALLOWED_CANDIDATE_BUCKETS,
    MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION,
)
from market_engine.ticker_universe.professional_swing import (
    EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION,
    REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS,
    ProfessionalSwingUniverseEntry,
    ProfessionalSwingUniverseValidationError,
    load_professional_swing_universe,
)


PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION = (
    "market-engine-professional-swing-universe-expansion-v1"
)
ELIGIBLE_CANDIDATE_BUCKETS = {"ready_for_manual_candidate_review"}
_TICKER_RE = re.compile(r"^[A-Z0-9.-]+$")


class ProfessionalSwingUniverseExpansionError(ValueError):
    pass


@dataclass(frozen=True)
class ProfessionalSwingUniverseExpansionDecision:
    ticker: str
    candidate_bucket: str
    inclusion_decision: str
    reason: str
    already_present: bool
    requires_manual_review: bool
    source_candidate_id: str | None
    source_candidate_reference: str | None
    proposed_universe_entry: dict[str, str] | None


@dataclass(frozen=True)
class ProfessionalSwingUniverseExpansionResult:
    format_version: str
    input_universe_path: str
    input_candidate_classification_path: str
    existing_universe_count: int
    candidate_count: int
    included_count: int
    excluded_count: int
    duplicate_count: int
    blocked_or_manual_review_count: int
    resulting_universe_count: int
    summary_counts: dict[str, int]
    included_candidate_entries: tuple[ProfessionalSwingUniverseExpansionDecision, ...]
    excluded_candidate_entries: tuple[ProfessionalSwingUniverseExpansionDecision, ...]
    duplicate_candidate_entries: tuple[ProfessionalSwingUniverseExpansionDecision, ...]
    blocked_or_manual_review_entries: tuple[ProfessionalSwingUniverseExpansionDecision, ...]
    final_universe_entries: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]
    non_actionable_boundary: bool = True

    def to_summary_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_professional_swing_universe_expansion(
    *,
    existing_universe_path: str | Path,
    candidate_classification_path: str | Path,
    operator_approved_tickers: Sequence[str] | None = None,
) -> ProfessionalSwingUniverseExpansionResult:
    universe_path = _validated_path(Path(existing_universe_path))
    candidate_path = _validated_path(Path(candidate_classification_path))
    universe = load_professional_swing_universe(universe_path, include_inactive=True)
    candidate_summary = _read_candidate_summary(candidate_path)
    approved_tickers = (
        None
        if operator_approved_tickers is None
        else frozenset(_normalize_ticker(ticker) for ticker in operator_approved_tickers)
    )
    existing_entries = tuple(universe.entries)
    existing_keys = {(entry.normalized_ticker, entry.market) for entry in existing_entries}
    max_priority = max((entry.operator_priority for entry in existing_entries), default=0)

    decisions: list[ProfessionalSwingUniverseExpansionDecision] = []
    new_entries: list[dict[str, str]] = []
    seen_candidate_keys: set[tuple[str, str]] = set()
    for raw_candidate in _candidate_records(candidate_summary):
        decision, proposed_entry = _decision_for_candidate(
            raw_candidate,
            existing_keys=existing_keys,
            seen_candidate_keys=seen_candidate_keys,
            approved_tickers=approved_tickers,
            next_priority=max_priority + len(new_entries) + 1,
        )
        decisions.append(decision)
        if proposed_entry is not None:
            new_entries.append(proposed_entry)
            existing_keys.add((proposed_entry["ticker"], proposed_entry["market"]))

    final_entries = tuple(
        sorted(
            tuple(_entry_to_csv_dict(entry) for entry in existing_entries) + tuple(new_entries),
            key=_entry_sort_key,
        )
    )
    included = tuple(item for item in decisions if item.inclusion_decision == "included")
    duplicate = tuple(item for item in decisions if item.inclusion_decision == "duplicate")
    blocked_or_manual = tuple(
        item
        for item in decisions
        if item.requires_manual_review or "blocked" in item.reason or "manual_review" in item.reason
    )
    excluded = tuple(item for item in decisions if item.inclusion_decision != "included")
    return ProfessionalSwingUniverseExpansionResult(
        format_version=PROFESSIONAL_SWING_UNIVERSE_EXPANSION_FORMAT_VERSION,
        input_universe_path=universe_path.as_posix(),
        input_candidate_classification_path=candidate_path.as_posix(),
        existing_universe_count=len(existing_entries),
        candidate_count=len(decisions),
        included_count=len(included),
        excluded_count=len(excluded),
        duplicate_count=len(duplicate),
        blocked_or_manual_review_count=len(blocked_or_manual),
        resulting_universe_count=len(final_entries),
        summary_counts={
            "existing_universe_count": len(existing_entries),
            "candidate_count": len(decisions),
            "included_count": len(included),
            "excluded_count": len(excluded),
            "duplicate_count": len(duplicate),
            "blocked_or_manual_review_count": len(blocked_or_manual),
            "resulting_universe_count": len(final_entries),
        },
        included_candidate_entries=included,
        excluded_candidate_entries=excluded,
        duplicate_candidate_entries=duplicate,
        blocked_or_manual_review_entries=blocked_or_manual,
        final_universe_entries=final_entries,
        warnings=(),
    )


def _decision_for_candidate(
    raw_candidate: Mapping[str, Any],
    *,
    existing_keys: set[tuple[str, str]],
    seen_candidate_keys: set[tuple[str, str]],
    approved_tickers: frozenset[str] | None,
    next_priority: int,
) -> tuple[ProfessionalSwingUniverseExpansionDecision, dict[str, str] | None]:
    ticker = _normalize_ticker(_required_text(raw_candidate, "ticker"))
    bucket = _required_text(raw_candidate, "candidate_bucket")
    if bucket not in ALLOWED_CANDIDATE_BUCKETS:
        raise ProfessionalSwingUniverseExpansionError(
            f"Unknown candidate classification bucket for {ticker}: {bucket}"
        )
    source_candidate_id = _optional_text(raw_candidate.get("source_candidate_id"))
    source_reference = _candidate_reference(raw_candidate)
    entry_payload = raw_candidate.get("proposed_universe_entry")
    if not isinstance(entry_payload, Mapping):
        return (
            _decision(
                ticker=ticker,
                bucket=bucket,
                decision="excluded",
                reason="missing_proposed_universe_entry",
                already_present=False,
                manual_review=False,
                source_candidate_id=source_candidate_id,
                source_reference=source_reference,
                proposed_entry=None,
            ),
            None,
        )
    proposed_entry = _proposed_entry(entry_payload, fallback_ticker=ticker, next_priority=next_priority)
    key = (proposed_entry["ticker"], proposed_entry["market"])
    if proposed_entry["ticker"] != ticker:
        raise ProfessionalSwingUniverseExpansionError(
            f"Conflicting candidate ticker identity: {ticker} != {proposed_entry['ticker']}"
        )
    if key in seen_candidate_keys:
        return (
            _decision(
                ticker=ticker,
                bucket=bucket,
                decision="duplicate",
                reason="duplicate_candidate_input",
                already_present=False,
                manual_review=False,
                source_candidate_id=source_candidate_id,
                source_reference=source_reference,
                proposed_entry=proposed_entry,
            ),
            None,
        )
    if approved_tickers is not None and ticker not in approved_tickers:
        return (
            _decision(
                ticker=ticker,
                bucket=bucket,
                decision="excluded",
                reason="not_operator_approved",
                already_present=False,
                manual_review=True,
                source_candidate_id=source_candidate_id,
                source_reference=source_reference,
                proposed_entry=proposed_entry,
            ),
            None,
        )
    if key in existing_keys:
        return (
            _decision(
                ticker=ticker,
                bucket=bucket,
                decision="duplicate",
                reason="already_present_in_professional_swing_universe",
                already_present=True,
                manual_review=False,
                source_candidate_id=source_candidate_id,
                source_reference=source_reference,
                proposed_entry=proposed_entry,
            ),
            None,
        )
    seen_candidate_keys.add(key)
    ineligible_reason = _ineligible_reason(bucket=bucket, proposed_entry=proposed_entry, raw_candidate=raw_candidate)
    if ineligible_reason is not None:
        return (
            _decision(
                ticker=ticker,
                bucket=bucket,
                decision="excluded",
                reason=ineligible_reason,
                already_present=False,
                manual_review=_requires_manual_review(ineligible_reason),
                source_candidate_id=source_candidate_id,
                source_reference=source_reference,
                proposed_entry=proposed_entry,
            ),
            None,
        )
    return (
        _decision(
            ticker=ticker,
            bucket=bucket,
            decision="included",
            reason="eligible_non_actionable_candidate",
            already_present=False,
            manual_review=False,
            source_candidate_id=source_candidate_id,
            source_reference=source_reference,
            proposed_entry=proposed_entry,
        ),
        proposed_entry,
    )


def _ineligible_reason(
    *,
    bucket: str,
    proposed_entry: Mapping[str, str],
    raw_candidate: Mapping[str, Any],
) -> str | None:
    if bucket not in ELIGIBLE_CANDIDATE_BUCKETS:
        return f"ineligible_candidate_bucket:{bucket}"
    universe_status = proposed_entry["universe_status"]
    source_policy_hint = proposed_entry["source_policy_hint"]
    asset_type = proposed_entry["asset_type"]
    blocking_reasons = tuple(str(item) for item in raw_candidate.get("blocking_reasons") or ())
    if universe_status == "manual_review_only":
        return "manual_review_only"
    if universe_status in {"blocked", "rejected"}:
        return f"universe_status_{universe_status}"
    if source_policy_hint in {"manual_review_only", "unsupported", "source_mapping_required"}:
        return f"source_policy_{source_policy_hint}"
    if asset_type != "equity":
        return f"unsupported_asset_type:{asset_type}"
    if any("ambiguous" in reason for reason in blocking_reasons):
        return "ambiguous_identity"
    if any("missing_source" in reason or "source_coverage" in reason for reason in blocking_reasons):
        return "missing_required_source_artifact"
    if raw_candidate.get("safety_flags", {}).get("malformed_input_detected"):
        return "malformed_candidate_input"
    if raw_candidate.get("safety_flags", {}).get("unsupported_input_detected"):
        return "unsupported_candidate_input"
    return None


def _proposed_entry(
    payload: Mapping[str, Any],
    *,
    fallback_ticker: str,
    next_priority: int,
) -> dict[str, str]:
    entry = {column: str(payload.get(column, "")).strip() for column in REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS}
    entry["ticker"] = _normalize_ticker(entry.get("ticker") or fallback_ticker)
    if entry["ticker"] != fallback_ticker:
        raise ProfessionalSwingUniverseExpansionError(
            f"Candidate proposed entry ticker mismatch: {fallback_ticker} != {entry['ticker']}"
        )
    if not entry["name"]:
        raise ProfessionalSwingUniverseExpansionError(
            f"Candidate {fallback_ticker} is missing required company name."
        )
    entry.setdefault("active", "true")
    if not entry["active"]:
        entry["active"] = "true"
    if not entry["universe_status"]:
        entry["universe_status"] = "candidate"
    if not entry["source_policy_hint"]:
        entry["source_policy_hint"] = "cached_source_candidate"
    if not entry["operator_priority"]:
        entry["operator_priority"] = str(next_priority)
    for field in ("swing_profile", "liquidity_profile", "volatility_profile", "market_cap_profile", "theme", "sector"):
        if not entry[field]:
            entry[field] = "unknown"
    _validate_entry_dict(entry)
    return entry


def _validate_entry_dict(entry: Mapping[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "professional_swing_universe.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS)
            writer.writerow(
                [entry[column] for column in REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS]
            )
        try:
            load_professional_swing_universe(path, include_inactive=True)
        except ProfessionalSwingUniverseValidationError as exc:
            raise ProfessionalSwingUniverseExpansionError(
                f"Proposed universe entry is invalid for {entry.get('ticker')}: {exc}"
            ) from exc


def _read_candidate_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfessionalSwingUniverseExpansionError(
            f"Candidate classification summary contains invalid JSON: {path}"
        ) from exc
    except OSError as exc:
        raise ProfessionalSwingUniverseExpansionError(
            f"Unable to read candidate classification summary: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProfessionalSwingUniverseExpansionError(
            "Candidate classification summary must contain a JSON object."
        )
    if payload.get("candidate_classification_format_version") != MARKET_ENGINE_CANDIDATE_CLASSIFICATION_FORMAT_VERSION:
        raise ProfessionalSwingUniverseExpansionError(
            "Candidate classification summary uses an unsupported format version."
        )
    return payload


def _candidate_records(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_records = summary.get("per_ticker_classifications")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ProfessionalSwingUniverseExpansionError(
            "Candidate classification summary is missing per_ticker_classifications."
        )
    records: list[Mapping[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise ProfessionalSwingUniverseExpansionError(
                "Candidate classification record must be a JSON object."
            )
        records.append(item)
    return tuple(sorted(records, key=lambda item: str(item.get("ticker", ""))))


def _validated_path(path: Path) -> Path:
    if any(part == ".." for part in path.parts):
        raise ProfessionalSwingUniverseExpansionError(
            "Input paths may not contain parent traversal."
        )
    resolved = path.resolve()
    if not resolved.exists():
        raise ProfessionalSwingUniverseExpansionError(f"Input path does not exist: {path}")
    if not resolved.is_file():
        raise ProfessionalSwingUniverseExpansionError(f"Input path is not a file: {path}")
    return resolved


def _entry_to_csv_dict(entry: ProfessionalSwingUniverseEntry) -> dict[str, str]:
    return {
        "ticker": entry.ticker,
        "name": entry.name,
        "market": entry.market,
        "asset_type": entry.asset_type,
        "active": "true" if entry.active else "false",
        "universe_status": entry.universe_status,
        "source_policy_hint": entry.source_policy_hint,
        "operator_priority": str(entry.operator_priority),
        "swing_profile": entry.swing_profile,
        "liquidity_profile": entry.liquidity_profile,
        "volatility_profile": entry.volatility_profile,
        "market_cap_profile": entry.market_cap_profile,
        "theme": entry.theme,
        "sector": entry.sector,
        "notes": entry.notes,
    }


def _decision(
    *,
    ticker: str,
    bucket: str,
    decision: str,
    reason: str,
    already_present: bool,
    manual_review: bool,
    source_candidate_id: str | None,
    source_reference: str | None,
    proposed_entry: dict[str, str] | None,
) -> ProfessionalSwingUniverseExpansionDecision:
    return ProfessionalSwingUniverseExpansionDecision(
        ticker=ticker,
        candidate_bucket=bucket,
        inclusion_decision=decision,
        reason=reason,
        already_present=already_present,
        requires_manual_review=manual_review,
        source_candidate_id=source_candidate_id,
        source_candidate_reference=source_reference,
        proposed_universe_entry=proposed_entry,
    )


def _candidate_reference(raw_candidate: Mapping[str, Any]) -> str | None:
    references = raw_candidate.get("evidence_references")
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        for reference in references:
            if isinstance(reference, Mapping) and reference.get("reference"):
                return str(reference["reference"])
    return None


def _requires_manual_review(reason: str) -> bool:
    return any(token in reason for token in ("manual_review", "ambiguous", "unsupported", "missing_required"))


def _normalize_ticker(value: Any) -> str:
    ticker = str(value).strip().upper()
    if not ticker or not _TICKER_RE.fullmatch(ticker):
        raise ProfessionalSwingUniverseExpansionError(
            f"Invalid candidate ticker: {value!r}"
        )
    return ticker


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name, "")).strip()
    if not value:
        raise ProfessionalSwingUniverseExpansionError(
            f"Candidate classification record missing required field: {field_name}"
        )
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _entry_sort_key(entry: Mapping[str, str]) -> tuple[int, str, str]:
    try:
        priority = int(entry["operator_priority"])
    except (KeyError, ValueError):
        priority = 999_999
    return (priority, entry.get("ticker", ""), entry.get("market", ""))
