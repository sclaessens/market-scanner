from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

INPUT_PATH = DATA_DIR / "processed" / "final_decisions.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "stability_state.csv"
LOG_PATH = DATA_DIR / "logs" / "stability_layer_log.csv"

STABILITY_CONTRACT_VERSION = "SPRINT_7_STABILITY_PERSISTENCE_V1"

REQUIRED_INPUT_COLUMNS = [
    "ticker",
    "date",
    "final_action",
]

OPTIONAL_CONVICTION_COLUMNS = [
    "conviction",
    "conviction_state",
]

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "stability_state",
    "conviction_persistence",
    "action_persistence",
    "behavioural_stability",
    "transition_frequency",
    "escalation_frequency",
    "stability_reason",
    "persistence_duration",
]

LOG_COLUMNS = [
    "run_id",
    "generated_at",
    "input_artifact",
    "output_artifact",
    "input_status",
    "input_row_count",
    "output_row_count",
    "row_count_preserved",
    "ticker_date_universe_preserved",
    "input_order_preserved",
    "upstream_artifacts_mutated",
    "stability_contract_version",
    "historical_source_status",
    "decision_input_used",
    "stability_state_distribution",
    "conviction_persistence_distribution",
    "action_persistence_distribution",
    "behavioural_stability_distribution",
    "transition_frequency_distribution",
    "escalation_frequency_distribution",
    "persistence_duration_distribution",
    "classification_rationale",
]

FORBIDDEN_OUTPUT_COLUMNS = {
    "suppression_flag",
    "hard_block",
    "cooldown_lock",
    "allocation_gate",
    "execution_gate",
    "hidden_filter",
    "action_override",
    "final_action_override",
    "remove_opportunity",
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


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _load_decision_input(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    path = path or INPUT_PATH
    if not path.exists():
        return pd.DataFrame(), "SOURCE_MISSING"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "SOURCE_EMPTY"
    if df.empty:
        return df, "SOURCE_EMPTY"
    return df, "AVAILABLE"


def _validate_input_contract(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Stability input missing required columns: {missing_columns}")

    if df[["ticker", "date"]].isna().any().any():
        raise ValueError("Stability input contains missing ticker/date row identity values")

    normalized_keys = (
        df["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + df["date"].astype(str).str.strip()
    )
    if normalized_keys.duplicated().any():
        duplicated = sorted(normalized_keys[normalized_keys.duplicated()].unique().tolist())
        raise ValueError(f"Stability input contains duplicate ticker/date rows: {duplicated}")


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized["date"] = normalized["date"].astype(str).str.strip()
    normalized["final_action"] = normalized["final_action"].map(_clean_state)
    normalized["_input_order"] = range(len(normalized))
    normalized["_date_sort"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["_date_sort"] = normalized["_date_sort"].fillna(pd.Timestamp.max)
    return normalized


def _conviction_source_column(df: pd.DataFrame) -> str | None:
    for column in OPTIONAL_CONVICTION_COLUMNS:
        if column in df.columns:
            return column
    return None


def _streak_duration(values: list[str], index: int) -> int:
    current = values[index]
    duration = 1
    position = index - 1
    while position >= 0 and values[position] == current:
        duration += 1
        position -= 1
    return duration


def _prior_transition_count(values: list[str], index: int) -> int:
    if index <= 0:
        return 0
    return sum(1 for previous, current in zip(values[:index], values[1 : index + 1]) if previous != current)


def _escalation_count(values: list[str], index: int) -> int:
    if index <= 0:
        return 0

    observations = 0
    seen_values = {values[0]}
    for current in values[1 : index + 1]:
        if current not in seen_values:
            observations += 1
        seen_values.add(current)
    return observations


def _stability_label(duration: int, transitions: int, escalations: int) -> str:
    if transitions == 0 and duration >= 3:
        return "PERSISTENT"
    if escalations > 0 and transitions <= 2:
        return "ESCALATING"
    if transitions >= 3:
        return "NOISY"
    if transitions > 0:
        return "TRANSITIONING"
    return "STABLE"


def _action_persistence_label(duration: int, transitions: int) -> str:
    if duration >= 3:
        return "PERSISTENT"
    if transitions > 0:
        return "CHANGED"
    return "UNCHANGED"


def _conviction_persistence_label(values: list[str] | None, index: int) -> str:
    if values is None:
        return "SOURCE_UNAVAILABLE"
    duration = _streak_duration(values, index)
    transitions = _prior_transition_count(values, index)
    if duration >= 3:
        return "PERSISTENT"
    if transitions > 0:
        return "CHANGED"
    return "UNCHANGED"


def _build_stability_rows(normalized_input: pd.DataFrame) -> pd.DataFrame:
    source_column = _conviction_source_column(normalized_input)
    sorted_df = normalized_input.sort_values(
        ["ticker", "_date_sort", "date", "_input_order"],
        kind="mergesort",
    )

    rows: list[dict[str, Any]] = []
    for _, group in sorted_df.groupby("ticker", sort=False):
        actions = group["final_action"].map(_clean_state).tolist()
        conviction_values = (
            group[source_column].map(_clean_state).tolist()
            if source_column is not None
            else None
        )

        for local_index, (_, row) in enumerate(group.iterrows()):
            duration = _streak_duration(actions, local_index)
            transitions = _prior_transition_count(actions, local_index)
            escalations = _escalation_count(actions, local_index)
            label = _stability_label(duration, transitions, escalations)
            action_persistence = _action_persistence_label(duration, transitions)
            conviction_persistence = _conviction_persistence_label(conviction_values, local_index)
            reason = (
                f"source=final_decisions.csv; action_duration={duration}; "
                f"action_transitions={transitions}; escalation_observations={escalations}; "
                f"conviction_source={source_column or 'SOURCE_UNAVAILABLE'}"
            )
            rows.append(
                {
                    "_input_order": row["_input_order"],
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "stability_state": label,
                    "conviction_persistence": conviction_persistence,
                    "action_persistence": action_persistence,
                    "behavioural_stability": label,
                    "transition_frequency": transitions,
                    "escalation_frequency": escalations,
                    "stability_reason": reason,
                    "persistence_duration": duration,
                }
            )

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        return _empty_output()
    output_df = output_df.sort_values("_input_order", kind="mergesort")
    return output_df[OUTPUT_COLUMNS].reset_index(drop=True)


def _normalized_keys(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return (
        df["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + df["date"].astype(str).str.strip()
    ).tolist()


def _distribution(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "{}"
    counts = df[column].value_counts(dropna=False).sort_index()
    payload = {str(key): int(value) for key, value in counts.items()}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _numeric_distribution(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "{}"
    values = pd.to_numeric(df[column], errors="coerce")
    payload = {
        "max": None if values.dropna().empty else int(values.max()),
        "mean": None if values.dropna().empty else round(float(values.mean()), 6),
        "min": None if values.dropna().empty else int(values.min()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _build_log(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    input_status: str,
    input_order: list[str],
) -> pd.DataFrame:
    output_order = _normalized_keys(output_df)
    input_universe = set(input_order)
    output_universe = set(output_order)
    row_count_preserved = len(input_df) == len(output_df)
    universe_preserved = input_universe == output_universe
    order_preserved = input_order == output_order
    if input_df.empty or "date" not in input_df.columns:
        historical_status = input_status
    elif len(input_df["date"].unique()) <= 1:
        historical_status = "CURRENT_DECISION_OUTPUT_ONLY"
    else:
        historical_status = "MULTI_DATE_DECISION_OUTPUT"

    log_row = {
        "run_id": STABILITY_CONTRACT_VERSION,
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_artifact": str(INPUT_PATH),
        "output_artifact": str(OUTPUT_PATH),
        "input_status": input_status,
        "input_row_count": len(input_df),
        "output_row_count": len(output_df),
        "row_count_preserved": row_count_preserved,
        "ticker_date_universe_preserved": universe_preserved,
        "input_order_preserved": order_preserved,
        "upstream_artifacts_mutated": False,
        "stability_contract_version": STABILITY_CONTRACT_VERSION,
        "historical_source_status": historical_status if input_status == "AVAILABLE" else input_status,
        "decision_input_used": str(INPUT_PATH) if input_status == "AVAILABLE" else "NONE",
        "stability_state_distribution": _distribution(output_df, "stability_state"),
        "conviction_persistence_distribution": _distribution(output_df, "conviction_persistence"),
        "action_persistence_distribution": _distribution(output_df, "action_persistence"),
        "behavioural_stability_distribution": _distribution(output_df, "behavioural_stability"),
        "transition_frequency_distribution": _numeric_distribution(output_df, "transition_frequency"),
        "escalation_frequency_distribution": _numeric_distribution(output_df, "escalation_frequency"),
        "persistence_duration_distribution": _numeric_distribution(output_df, "persistence_duration"),
        "classification_rationale": "row-preserving stability metadata from existing Decision Engine output",
    }
    return pd.DataFrame([log_row], columns=LOG_COLUMNS)


def build_stability_layer() -> pd.DataFrame:
    input_df, input_status = _load_decision_input()
    if input_status != "AVAILABLE":
        output_df = _empty_output()
        input_order: list[str] = []
    else:
        _validate_input_contract(input_df)
        normalized_input = _normalize_input(input_df)
        input_order = _normalized_keys(normalized_input)
        output_df = _build_stability_rows(normalized_input)

    if set(output_df.columns) & FORBIDDEN_OUTPUT_COLUMNS:
        forbidden = sorted(set(output_df.columns) & FORBIDDEN_OUTPUT_COLUMNS)
        raise ValueError(f"Stability layer generated forbidden output columns: {forbidden}")

    log_df = _build_log(input_df, output_df, input_status, input_order)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    log_df.to_csv(LOG_PATH, index=False)
    return output_df


def main() -> None:
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_stability_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )


if __name__ == "__main__":
    main()
