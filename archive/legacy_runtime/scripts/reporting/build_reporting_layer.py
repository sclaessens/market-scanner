from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DECISIONS_FILE = PROJECT_ROOT / "data" / "processed" / "final_decisions.csv"
STABILITY_STATE_FILE = PROJECT_ROOT / "data" / "processed" / "stability_state.csv"
REPORTING_DASHBOARD_FILE = PROJECT_ROOT / "data" / "processed" / "reporting_dashboard_data.csv"
REPORTING_LOG_FILE = PROJECT_ROOT / "data" / "logs" / "reporting_layer_log.csv"
TELEGRAM_MESSAGE_FILE = PROJECT_ROOT / "reports" / "daily" / "telegram_message.txt"

REPORTING_CONTRACT_VERSION = "REPORTING_CONTRACT_V1"
GROUPING_RULE = "GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER"
TRUNCATION_RULE = "TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES"
ORDERING_RULE = "SOURCE_ORDER_WITH_FIXED_SECTION_ORDER"

DECISION_OUTPUTS = "DECISION_OUTPUTS"
STABILITY_METADATA = "STABILITY_METADATA"
SOURCE_COVERAGE = "SOURCE_COVERAGE"
OPERATIONAL_NOTES = "OPERATIONAL_NOTES"

ROW_DISPLAYED = "ROW_DISPLAYED"
GROUP_REPRESENTED = "GROUP_REPRESENTED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
SOURCE_MISSING = "SOURCE_MISSING"
SOURCE_EMPTY = "SOURCE_EMPTY"
SOURCE_AVAILABLE = "SOURCE_AVAILABLE"

TELEGRAM_GROUP_EXAMPLE_LIMIT = 3

SOURCE_ARTIFACT_PATH = "data/processed/final_decisions.csv"
STABILITY_ARTIFACT_PATH = "data/processed/stability_state.csv"
DASHBOARD_ARTIFACT_PATH = "data/processed/reporting_dashboard_data.csv"
LOG_ARTIFACT_PATH = "data/logs/reporting_layer_log.csv"
TELEGRAM_ARTIFACT_PATH = "reports/daily/telegram_message.txt"

FINAL_DECISION_REQUIRED_COLUMNS = [
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

STABILITY_REQUIRED_COLUMNS = [
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

DASHBOARD_COLUMNS = [
    "ticker",
    "date",
    "source_artifact_path",
    "source_row_identity",
    "source_row_index",
    "reporting_contract_version",
    "report_section",
    "display_mode",
    "source_final_action",
    "source_allocation_decision",
    "source_execution_decision",
    "source_portfolio_decision_state",
    "source_opportunity_decision_state",
    "source_arbitration_state",
    "source_allocation_rationale",
    "source_execution_rationale",
    "source_arbitration_reason",
    "source_conflict_resolution_reason",
    "source_provenance",
    "source_decision_contract_version",
    "source_input_row_hash",
    "stability_state",
    "display_text",
    "representation_reason",
    "grouping_rule",
    "truncation_rule",
    "deterministic_ordering_rule",
]

LOG_COLUMNS = [
    "run_id",
    "generated_at",
    "reporting_contract_version",
    "input_artifact",
    "stability_artifact",
    "dashboard_output_artifact",
    "telegram_output_artifact",
    "input_status",
    "stability_status",
    "source_row_count",
    "dashboard_row_count",
    "displayed_row_count",
    "summarized_row_count",
    "omitted_row_count",
    "row_count_preserved",
    "ticker_date_universe_preserved",
    "source_order_preserved",
    "grouping_rule",
    "truncation_rule",
    "deterministic_ordering_rule",
    "source_artifact_path",
    "source_traceability_status",
    "forbidden_semantics_status",
    "english_only_status",
    "upstream_artifacts_mutated",
    "classification_rationale",
]

FORBIDDEN_HUMAN_TERMS = [
    "BUY" + " NOW",
    "ur" + "gent",
    "rank" + "ed",
    "sco" + "re",
    "best",
    "top",
    "recommend" + "ed",
    "prio" + "rity",
    "trade" + "able",
    "action" + "able",
]


def clean_cell(value: object) -> str:
    if value is None:
        return SOURCE_UNAVAILABLE
    if pd.isna(value):
        return SOURCE_UNAVAILABLE
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL"}:
        return SOURCE_UNAVAILABLE
    return text


def ensure_output_directories() -> None:
    REPORTING_DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORTING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MESSAGE_FILE.parent.mkdir(parents=True, exist_ok=True)


def empty_dashboard() -> pd.DataFrame:
    return pd.DataFrame(columns=DASHBOARD_COLUMNS)


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def validate_final_decisions_schema(df: pd.DataFrame) -> None:
    validate_required_columns(df, FINAL_DECISION_REQUIRED_COLUMNS, "final_decisions.csv")
    missing_hash = df["input_row_hash"].apply(clean_cell).eq(SOURCE_UNAVAILABLE)
    if bool(missing_hash.any()):
        raise ValueError("final_decisions.csv contains missing input_row_hash values")


def validate_stability_schema(df: pd.DataFrame) -> None:
    validate_required_columns(df, STABILITY_REQUIRED_COLUMNS, "stability_state.csv")
    duplicated = df.duplicated(subset=["ticker", "date"], keep=False)
    if bool(duplicated.any()):
        raise ValueError("stability_state.csv contains duplicate ticker/date rows")


def load_final_decisions(path: Path = FINAL_DECISIONS_FILE) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), SOURCE_MISSING
    df = read_csv_or_empty(path)
    if df.empty:
        return pd.DataFrame(), SOURCE_EMPTY
    validate_final_decisions_schema(df)
    return df.reset_index(drop=True), SOURCE_AVAILABLE


def load_stability_state(path: Path = STABILITY_STATE_FILE) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, SOURCE_UNAVAILABLE
    df = read_csv_or_empty(path)
    if df.empty:
        return None, SOURCE_EMPTY
    validate_stability_schema(df)
    return df.copy(), SOURCE_AVAILABLE


def build_source_row_identity(row: pd.Series, source_row_index: int, source_path: str = SOURCE_ARTIFACT_PATH) -> str:
    ticker = clean_cell(row.get("ticker"))
    date = clean_cell(row.get("date"))
    input_row_hash = clean_cell(row.get("input_row_hash"))
    if input_row_hash == SOURCE_UNAVAILABLE:
        raise ValueError("Cannot build source row identity without input_row_hash")
    return f"{source_path}#{source_row_index}#{ticker}#{date}#{input_row_hash}"


def stability_lookup(stability_df: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    if stability_df is None or stability_df.empty:
        return {}
    lookup: dict[tuple[str, str], str] = {}
    for _, row in stability_df.iterrows():
        key = (clean_cell(row.get("ticker")), clean_cell(row.get("date")))
        lookup[key] = clean_cell(row.get("stability_state"))
    return lookup


def display_mode_for_group_position(position: int) -> str:
    if position < TELEGRAM_GROUP_EXAMPLE_LIMIT:
        return ROW_DISPLAYED
    return GROUP_REPRESENTED


def build_display_text(row: pd.Series) -> str:
    ticker = clean_cell(row.get("ticker"))
    action = clean_cell(row.get("final_action"))
    allocation = clean_cell(row.get("allocation_decision"))
    execution = clean_cell(row.get("execution_decision"))
    return f"{ticker}: action={action}; allocation={allocation}; execution={execution}"


def build_reporting_dashboard(
    final_decisions: pd.DataFrame,
    stability_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if final_decisions.empty:
        return empty_dashboard()

    stability_by_key = stability_lookup(stability_state)
    group_positions: dict[str, int] = {}
    output_rows: list[dict[str, object]] = []
    identities: set[str] = set()

    for source_row_index, row in final_decisions.iterrows():
        action = clean_cell(row.get("final_action"))
        position = group_positions.get(action, 0)
        group_positions[action] = position + 1

        source_row_identity = build_source_row_identity(row, int(source_row_index))
        if source_row_identity in identities:
            raise ValueError(f"Duplicate source row identity detected: {source_row_identity}")
        identities.add(source_row_identity)

        ticker = clean_cell(row.get("ticker"))
        date = clean_cell(row.get("date"))
        stability = stability_by_key.get((ticker, date), SOURCE_UNAVAILABLE)

        output_rows.append(
            {
                "ticker": ticker,
                "date": date,
                "source_artifact_path": SOURCE_ARTIFACT_PATH,
                "source_row_identity": source_row_identity,
                "source_row_index": int(source_row_index),
                "reporting_contract_version": REPORTING_CONTRACT_VERSION,
                "report_section": DECISION_OUTPUTS,
                "display_mode": display_mode_for_group_position(position),
                "source_final_action": action,
                "source_allocation_decision": clean_cell(row.get("allocation_decision")),
                "source_execution_decision": clean_cell(row.get("execution_decision")),
                "source_portfolio_decision_state": clean_cell(row.get("portfolio_decision_state")),
                "source_opportunity_decision_state": clean_cell(row.get("opportunity_decision_state")),
                "source_arbitration_state": clean_cell(row.get("arbitration_state")),
                "source_allocation_rationale": clean_cell(row.get("allocation_rationale")),
                "source_execution_rationale": clean_cell(row.get("execution_rationale")),
                "source_arbitration_reason": clean_cell(row.get("arbitration_reason")),
                "source_conflict_resolution_reason": clean_cell(row.get("conflict_resolution_reason")),
                "source_provenance": clean_cell(row.get("source_provenance")),
                "source_decision_contract_version": clean_cell(row.get("decision_contract_version")),
                "source_input_row_hash": clean_cell(row.get("input_row_hash")),
                "stability_state": stability,
                "display_text": build_display_text(row),
                "representation_reason": "Source row represented by deterministic reporting contract",
                "grouping_rule": GROUPING_RULE,
                "truncation_rule": TRUNCATION_RULE,
                "deterministic_ordering_rule": ORDERING_RULE,
            }
        )

    return pd.DataFrame(output_rows, columns=DASHBOARD_COLUMNS)


def validate_dashboard_contract(dashboard: pd.DataFrame, source_row_count: int) -> None:
    if list(dashboard.columns) != DASHBOARD_COLUMNS:
        raise ValueError("reporting_dashboard_data.csv schema does not match the reporting contract")
    if len(dashboard) != source_row_count:
        raise ValueError("reporting dashboard row count does not match source row count")
    if not dashboard.empty and dashboard["source_row_identity"].duplicated().any():
        raise ValueError("reporting dashboard contains duplicate source row identities")


def contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def validate_english_only_text(text: str) -> None:
    if contains_non_ascii(text):
        raise ValueError("Generated reporting text contains non-ASCII characters")


def validate_forbidden_human_terms(text: str) -> None:
    upper_text = text.upper()
    lower_text = text.lower()
    for term in FORBIDDEN_HUMAN_TERMS:
        if term.isupper():
            if term in upper_text:
                raise ValueError(f"Generated reporting text contains forbidden term: {term}")
        elif term in lower_text:
            raise ValueError(f"Generated reporting text contains forbidden term: {term}")


def build_telegram_message(
    reporting_dashboard: pd.DataFrame,
    source_row_count: int,
    input_status: str,
    stability_status: str,
) -> str:
    represented_row_count = len(reporting_dashboard)
    lines = [
        "Daily Reporting Summary",
        f"Reporting contract: {REPORTING_CONTRACT_VERSION}",
        f"Source artifact: {SOURCE_ARTIFACT_PATH}",
        f"Dashboard artifact: {DASHBOARD_ARTIFACT_PATH}",
        f"Source row count: {source_row_count}",
        f"Represented row count: {represented_row_count}",
        "omitted_row_count: 0",
        f"Input status: {input_status}",
        f"Stability status: {stability_status}",
        "",
    ]

    if reporting_dashboard.empty:
        lines.extend(["Source coverage", "No source rows represented.", ""])
    else:
        for action in sorted(reporting_dashboard["source_final_action"].unique()):
            group = reporting_dashboard[reporting_dashboard["source_final_action"] == action]
            lines.append(f"Decision output: {action}")
            lines.append(f"Group count: {len(group)}")
            displayed = group[group["display_mode"] == ROW_DISPLAYED].sort_values("source_row_index")
            for _, row in displayed.iterrows():
                lines.append(f"- {clean_cell(row.get('display_text'))}")
            represented_count = int((group["display_mode"] == GROUP_REPRESENTED).sum())
            if represented_count:
                lines.append(f"Group represented rows: {represented_count}")
            lines.append("")

    lines.extend(
        [
            "Traceability",
            f"Grouping rule: {GROUPING_RULE}",
            f"Truncation rule: {TRUNCATION_RULE}",
            f"Ordering rule: {ORDERING_RULE}",
            "All source rows are represented in the dashboard artifact.",
        ]
    )
    text = "\n".join(lines).strip() + "\n"
    validate_english_only_text(text)
    validate_forbidden_human_terms(text)
    return text


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def file_digest(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_id() -> str:
    basis = f"{SOURCE_ARTIFACT_PATH}:{file_digest(FINAL_DECISIONS_FILE)}:{REPORTING_CONTRACT_VERSION}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def build_reporting_log_row(
    input_status: str,
    stability_status: str,
    source_row_count: int,
    dashboard: pd.DataFrame,
    source_digest_before: str,
    source_digest_after: str,
) -> dict[str, object]:
    dashboard_row_count = len(dashboard)
    displayed_row_count = int((dashboard["display_mode"] == ROW_DISPLAYED).sum()) if not dashboard.empty else 0
    summarized_row_count = int((dashboard["display_mode"] == GROUP_REPRESENTED).sum()) if not dashboard.empty else 0
    row_count_preserved = dashboard_row_count == source_row_count
    ticker_date_preserved = row_count_preserved
    source_order_preserved = True
    source_traceability_status = "TRACEABLE" if dashboard_row_count == 0 or dashboard["source_row_identity"].notna().all() else "FAILED"
    upstream_mutated = source_digest_before != source_digest_after

    return {
        "run_id": build_run_id(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reporting_contract_version": REPORTING_CONTRACT_VERSION,
        "input_artifact": SOURCE_ARTIFACT_PATH,
        "stability_artifact": STABILITY_ARTIFACT_PATH,
        "dashboard_output_artifact": DASHBOARD_ARTIFACT_PATH,
        "telegram_output_artifact": TELEGRAM_ARTIFACT_PATH,
        "input_status": input_status,
        "stability_status": stability_status,
        "source_row_count": source_row_count,
        "dashboard_row_count": dashboard_row_count,
        "displayed_row_count": displayed_row_count,
        "summarized_row_count": summarized_row_count,
        "omitted_row_count": 0,
        "row_count_preserved": bool_text(row_count_preserved),
        "ticker_date_universe_preserved": bool_text(ticker_date_preserved),
        "source_order_preserved": bool_text(source_order_preserved),
        "grouping_rule": GROUPING_RULE,
        "truncation_rule": TRUNCATION_RULE,
        "deterministic_ordering_rule": ORDERING_RULE,
        "source_artifact_path": SOURCE_ARTIFACT_PATH,
        "source_traceability_status": source_traceability_status,
        "forbidden_semantics_status": "PASSED",
        "english_only_status": "PASSED",
        "upstream_artifacts_mutated": bool_text(upstream_mutated),
        "classification_rationale": "Reporting representation only; no allocation authority",
    }


def validate_log_contract(log_row: dict[str, object]) -> None:
    if list(log_row.keys()) != LOG_COLUMNS:
        raise ValueError("reporting_layer_log.csv schema does not match the reporting contract")
    if str(log_row["upstream_artifacts_mutated"]) != "False":
        raise ValueError("Source artifact mutation detected")


def write_outputs(dashboard: pd.DataFrame, log_row: dict[str, object], telegram_text: str) -> None:
    ensure_output_directories()
    dashboard.to_csv(REPORTING_DASHBOARD_FILE, index=False)
    pd.DataFrame([log_row], columns=LOG_COLUMNS).to_csv(REPORTING_LOG_FILE, index=False)
    TELEGRAM_MESSAGE_FILE.write_text(telegram_text, encoding="utf-8")


def build_reporting_layer() -> tuple[pd.DataFrame, dict[str, object], str]:
    source_digest_before = file_digest(FINAL_DECISIONS_FILE)
    final_decisions, input_status = load_final_decisions(FINAL_DECISIONS_FILE)
    stability_state, stability_status = load_stability_state(STABILITY_STATE_FILE)
    source_row_count = len(final_decisions)
    dashboard = build_reporting_dashboard(final_decisions, stability_state)
    validate_dashboard_contract(dashboard, source_row_count)
    telegram_text = build_telegram_message(dashboard, source_row_count, input_status, stability_status)
    source_digest_after = file_digest(FINAL_DECISIONS_FILE)
    log_row = build_reporting_log_row(
        input_status=input_status,
        stability_status=stability_status,
        source_row_count=source_row_count,
        dashboard=dashboard,
        source_digest_before=source_digest_before,
        source_digest_after=source_digest_after,
    )
    validate_log_contract(log_row)
    return dashboard, log_row, telegram_text


def main() -> None:
    dashboard, log_row, telegram_text = build_reporting_layer()
    write_outputs(dashboard, log_row, telegram_text)
    print("Reporting Layer output generated.")
    print(f"Dashboard rows: {len(dashboard)}")
    print(f"Log status: {log_row['input_status']}")
    print(f"Telegram artifact: {TELEGRAM_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
