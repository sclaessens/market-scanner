from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

INPUT_PATH = DATA_DIR / "processed" / "portfolio_intelligence.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "final_decisions.csv"
LOG_PATH = DATA_DIR / "logs" / "decision_engine_log.csv"

DECISION_CONTRACT_VERSION = "SPRINT_6_DECISION_ENGINE_CORE_V1"

REQUIRED_INPUT_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "timing_state",
    "in_portfolio",
    "portfolio_position_state",
    "exposure_state",
    "diversification_state",
    "concentration_state",
    "overlap_state",
    "sector_exposure_state",
    "position_context_state",
    "portfolio_environment",
    "portfolio_metadata_status",
    "portfolio_metadata_reason",
]

OPTIONAL_PASSTHROUGH_COLUMNS = [
    "quality_state",
    "timing_state",
    "in_portfolio",
    "portfolio_position_state",
    "exposure_state",
    "diversification_state",
    "concentration_state",
    "overlap_state",
    "portfolio_metadata_status",
]

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "final_action",
    "allocation_decision",
    "execution_decision",
    "portfolio_decision_state",
    "opportunity_decision_state",
    "arbitration_state",
    "allocation_rationale",
    "execution_rationale",
    "arbitration_reason",
    "conflict_resolution_reason",
    "source_provenance",
    "decision_contract_version",
    "input_row_hash",
]

LOG_COLUMNS = [
    "run_id",
    "generated_at",
    "input_artifact",
    "output_artifact",
    "input_row_count",
    "output_row_count",
    "row_count_preserved",
    "ticker_date_universe_preserved",
    "input_order_preserved",
    "upstream_artifacts_mutated",
    "decision_contract_version",
    "forbidden_authority_leakage_detected",
    "hidden_filtering_detected",
    "silent_suppression_detected",
    "rationale_completeness_status",
    "source_provenance_status",
    "classification_rationale",
]

FINAL_ACTIONS = {
    "BUY",
    "SELL",
    "HOLD",
    "TRIM",
    "WAIT",
    "REMOVE",
    "REVIEW",
    "PREPARE",
    "NO_ACTION",
}

ALLOCATION_DECISIONS = {
    "ALLOCATE",
    "DO_NOT_ALLOCATE",
    "MAINTAIN",
    "REDUCE",
    "EXIT",
    "REVIEW_REQUIRED",
    "NO_ALLOCATION_ACTION",
}

EXECUTION_DECISIONS = {
    "EXECUTE",
    "DO_NOT_EXECUTE",
    "MONITOR",
    "REVIEW_REQUIRED",
    "NO_EXECUTION_ACTION",
}

ARBITRATION_STATES = {
    "NO_CONFLICT",
    "PORTFOLIO_POSITION_CONFLICT",
    "MISSING_METADATA",
    "TIMING_CONFLICT",
    "QUALITY_CONFLICT",
    "REVIEW_REQUIRED",
}

FORBIDDEN_OUTPUT_COLUMNS = {
    "decision_output",
    "conviction_score",
    "ranking_score",
    "portfolio_score",
    "final_score",
    "recommended_trade",
    "recommended_weight",
    "optimal_weight",
    "target_weight",
    "allocation_queue",
    "execution_urgency",
    "urgency",
    "actionable",
    "execution_ready",
}

MISSING_VALUES = {
    "",
    "UNKNOWN",
    "UNAVAILABLE",
    "UNCLASSIFIED",
    "SOURCE_MISSING",
    "MISSING",
    "PARTIAL",
    "SOURCE_PARTIAL",
    "INSUFFICIENT_DATA",
}

READY_TIMING_STATES = {
    "READY",
    "CONFIRMED",
    "PULLBACK",
    "BREAKOUT_READY",
    "BREAKOUT_PENDING",
}

CONFLICT_TIMING_STATES = {
    "EXTENDED",
    "STALE",
    "UNCLASSIFIED",
    "UNKNOWN",
    "UNAVAILABLE",
}

QUALITY_CONFLICT_STATES = {
    "INSUFFICIENT_DATA",
    "UNAVAILABLE",
    "SOURCE_MISSING",
    "UNKNOWN",
}


def _clean_text(value: Any, fallback: str = "UNKNOWN") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _clean_state(value: Any, fallback: str = "UNKNOWN") -> str:
    return _clean_text(value, fallback=fallback).upper()


def _read_required_input(path: Path | None = None) -> pd.DataFrame:
    path = path or INPUT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Required Decision Engine input is missing: {path}")
    return pd.read_csv(path)


def _validate_input_contract(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Decision Engine input missing required columns: {missing_columns}")

    if df[["ticker", "date"]].isna().any().any():
        raise ValueError("Decision Engine input contains missing ticker/date row identity values")

    normalized_keys = (
        df["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + df["date"].astype(str).str.strip()
    )
    if normalized_keys.duplicated().any():
        duplicated = sorted(normalized_keys[normalized_keys.duplicated()].unique().tolist())
        raise ValueError(f"Decision Engine input contains duplicate ticker/date rows: {duplicated}")


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized["date"] = normalized["date"].astype(str).str.strip()
    return normalized


def _hash_input_row(row: pd.Series) -> str:
    payload = {
        column: None if pd.isna(row[column]) else row[column]
        for column in row.index
    }
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _has_missing_metadata(row: pd.Series) -> bool:
    metadata_values = [
        _clean_state(row.get("quality_state")),
        _clean_state(row.get("timing_state")),
        _clean_state(row.get("portfolio_metadata_status")),
    ]
    return any(value in MISSING_VALUES for value in metadata_values)


def _arbitration_state(row: pd.Series) -> str:
    if _has_missing_metadata(row):
        return "MISSING_METADATA"
    if _clean_state(row.get("in_portfolio")) == "PRESENT":
        return "PORTFOLIO_POSITION_CONFLICT"
    if _clean_state(row.get("timing_state")) in CONFLICT_TIMING_STATES:
        return "TIMING_CONFLICT"
    if _clean_state(row.get("quality_state")) in QUALITY_CONFLICT_STATES:
        return "QUALITY_CONFLICT"
    return "NO_CONFLICT"


def _portfolio_decision_state(row: pd.Series, arbitration_state: str) -> str:
    if _clean_state(row.get("in_portfolio")) == "PRESENT":
        if arbitration_state == "MISSING_METADATA":
            return "POSITION_REVIEW_REQUIRED"
        return "POSITION_MAINTAIN_REVIEW"
    if arbitration_state == "MISSING_METADATA":
        return "PORTFOLIO_METADATA_REVIEW_REQUIRED"
    return "NO_PORTFOLIO_POSITION"


def _opportunity_decision_state(row: pd.Series, arbitration_state: str) -> str:
    if arbitration_state == "MISSING_METADATA":
        return "INSUFFICIENT_DECISION_METADATA"
    if arbitration_state == "TIMING_CONFLICT":
        return "TIMING_REVIEW_REQUIRED"
    if arbitration_state == "QUALITY_CONFLICT":
        return "QUALITY_REVIEW_REQUIRED"
    if _clean_state(row.get("timing_state")) in READY_TIMING_STATES:
        return "OPPORTUNITY_READY_FOR_DECISION"
    return "OPPORTUNITY_MONITOR"


def _decision_for_row(row: pd.Series) -> dict[str, Any]:
    arbitration_state = _arbitration_state(row)
    in_portfolio = _clean_state(row.get("in_portfolio"))
    timing_state = _clean_state(row.get("timing_state"))

    if arbitration_state == "MISSING_METADATA":
        final_action = "REVIEW"
        allocation_decision = "REVIEW_REQUIRED"
        execution_decision = "REVIEW_REQUIRED"
    elif in_portfolio == "PRESENT":
        final_action = "HOLD"
        allocation_decision = "MAINTAIN"
        execution_decision = "MONITOR"
    elif arbitration_state in {"TIMING_CONFLICT", "QUALITY_CONFLICT"}:
        final_action = "WAIT"
        allocation_decision = "DO_NOT_ALLOCATE"
        execution_decision = "MONITOR"
    elif timing_state in READY_TIMING_STATES:
        final_action = "PREPARE"
        allocation_decision = "NO_ALLOCATION_ACTION"
        execution_decision = "MONITOR"
    else:
        final_action = "NO_ACTION"
        allocation_decision = "NO_ALLOCATION_ACTION"
        execution_decision = "NO_EXECUTION_ACTION"

    portfolio_decision_state = _portfolio_decision_state(row, arbitration_state)
    opportunity_decision_state = _opportunity_decision_state(row, arbitration_state)
    metadata_reason = _clean_text(row.get("portfolio_metadata_reason"), fallback="metadata evaluated")
    source_provenance = "data/processed/portfolio_intelligence.csv"
    portfolio_source = _clean_text(row.get("portfolio_source_provenance"), fallback="")
    if portfolio_source not in {"", "UNKNOWN"}:
        source_provenance = f"{source_provenance};{portfolio_source}"

    allocation_rationale = (
        f"decision_engine_allocation={allocation_decision}; "
        f"arbitration={arbitration_state}; "
        f"portfolio_state={portfolio_decision_state}"
    )
    execution_rationale = (
        f"decision_engine_execution={execution_decision}; "
        f"final_action={final_action}; "
        f"timing_state={timing_state}"
    )
    arbitration_reason = (
        f"arbitration_state={arbitration_state}; "
        f"metadata_status={_clean_state(row.get('portfolio_metadata_status'))}"
    )
    conflict_resolution_reason = (
        f"conflict_resolution={opportunity_decision_state}; "
        f"portfolio_metadata_reason={metadata_reason}"
    )

    output = {
        "ticker": _clean_state(row.get("ticker")),
        "date": _clean_text(row.get("date")),
        "final_action": final_action,
        "allocation_decision": allocation_decision,
        "execution_decision": execution_decision,
        "portfolio_decision_state": portfolio_decision_state,
        "opportunity_decision_state": opportunity_decision_state,
        "arbitration_state": arbitration_state,
        "allocation_rationale": allocation_rationale,
        "execution_rationale": execution_rationale,
        "arbitration_reason": arbitration_reason,
        "conflict_resolution_reason": conflict_resolution_reason,
        "source_provenance": source_provenance,
        "decision_contract_version": DECISION_CONTRACT_VERSION,
        "input_row_hash": _hash_input_row(row),
    }

    for column in OPTIONAL_PASSTHROUGH_COLUMNS:
        if column in row.index:
            output[column] = row[column]

    return output


def _build_log(input_df: pd.DataFrame, output_df: pd.DataFrame, input_order: list[str]) -> pd.DataFrame:
    output_order = (
        output_df["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + output_df["date"].astype(str).str.strip()
    ).tolist()
    input_universe = set(input_order)
    output_universe = set(output_order)
    rationale_columns = [
        "allocation_rationale",
        "execution_rationale",
        "arbitration_reason",
        "conflict_resolution_reason",
    ]
    rationale_complete = bool(
        not output_df.empty
        and output_df[rationale_columns].notna().all().all()
        and (output_df[rationale_columns].astype(str).apply(lambda col: col.str.strip() != "")).all().all()
    )
    provenance_complete = bool(
        not output_df.empty
        and output_df["source_provenance"].notna().all()
        and (output_df["source_provenance"].astype(str).str.strip() != "").all()
    )
    row_count_preserved = len(input_df) == len(output_df)
    universe_preserved = input_universe == output_universe
    order_preserved = input_order == output_order

    log_row = {
        "run_id": DECISION_CONTRACT_VERSION,
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_artifact": str(INPUT_PATH),
        "output_artifact": str(OUTPUT_PATH),
        "input_row_count": len(input_df),
        "output_row_count": len(output_df),
        "row_count_preserved": row_count_preserved,
        "ticker_date_universe_preserved": universe_preserved,
        "input_order_preserved": order_preserved,
        "upstream_artifacts_mutated": False,
        "decision_contract_version": DECISION_CONTRACT_VERSION,
        "forbidden_authority_leakage_detected": False,
        "hidden_filtering_detected": not row_count_preserved,
        "silent_suppression_detected": not universe_preserved,
        "rationale_completeness_status": "COMPLETE" if rationale_complete else "INCOMPLETE",
        "source_provenance_status": "COMPLETE" if provenance_complete else "INCOMPLETE",
        "classification_rationale": "row-preserving deterministic Decision Engine evaluation from certified Portfolio Intelligence input",
    }
    return pd.DataFrame([log_row], columns=LOG_COLUMNS)


def build_final_decisions() -> pd.DataFrame:
    input_df = _read_required_input()
    _validate_input_contract(input_df)
    normalized_input = _normalize_input(input_df)
    input_order = (
        normalized_input["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + normalized_input["date"].astype(str).str.strip()
    ).tolist()

    output_rows = [_decision_for_row(row) for _, row in normalized_input.iterrows()]
    output_columns = OUTPUT_COLUMNS + [
        column
        for column in OPTIONAL_PASSTHROUGH_COLUMNS
        if column in normalized_input.columns and column not in OUTPUT_COLUMNS
    ]
    output_df = pd.DataFrame(output_rows, columns=output_columns)

    if set(output_df.columns) & FORBIDDEN_OUTPUT_COLUMNS:
        forbidden = sorted(set(output_df.columns) & FORBIDDEN_OUTPUT_COLUMNS)
        raise ValueError(f"Decision Engine generated forbidden output columns: {forbidden}")

    log_df = _build_log(normalized_input, output_df, input_order)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    log_df.to_csv(LOG_PATH, index=False)
    return output_df


def main() -> None:
    df = build_final_decisions()
    print(f"Final decisions written to: {OUTPUT_PATH}")
    print(f"Decision Engine log written to: {LOG_PATH}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker", "date", "final_action", "allocation_decision", "execution_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
