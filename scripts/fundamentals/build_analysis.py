from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fundamentals.build_metrics import HELPER_COLUMNS, IDENTITY_COLUMNS, METRIC_COLUMNS

REQUIRED_QUALITY_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_metadata_status",
    "source_data_status",
]

REQUIRED_METRIC_COLUMNS = IDENTITY_COLUMNS + METRIC_COLUMNS + HELPER_COLUMNS

ANALYSIS_COLUMNS = [
    "ticker",
    "date",
    "fundamental_analysis_state",
    "fundamental_analysis_reason",
    "fundamental_profile_state",
    "margin_profile_state",
    "growth_profile_state",
    "leverage_profile_state",
    "cash_flow_profile_state",
    "fundamental_review_flag",
    "fundamental_review_reason",
    "analysis_data_status",
    "analysis_input_coverage",
    "analysis_warnings",
]

FORBIDDEN_SEMANTIC_TERMS = {
    "buy",
    "sell",
    "action",
    "final_action",
    "decision",
    "allocation",
    "position_size",
    "urgency",
    "conviction",
    "tradeability",
    "eligible",
    "eligibility",
    "ranking",
    "score",
    "priority",
    "entry",
    "stop",
    "target",
}


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalized(value: Any) -> str:
    return _clean_text(value).upper()


def _parse_float(value: Any) -> float | None:
    text = _clean_text(value)
    if text == "":
        return None
    return float(text)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_no_duplicate_keys(df: pd.DataFrame, key_columns: list[str], label: str) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, key_columns].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate key rows: {duplicates}")


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def _load_quality(path: str | Path) -> pd.DataFrame:
    quality_df = _load_csv(path, "fundamental_quality.csv")
    _validate_columns(quality_df, REQUIRED_QUALITY_COLUMNS, "fundamental_quality.csv")
    _validate_no_duplicate_keys(quality_df, ["ticker", "date"], "fundamental_quality.csv")
    return quality_df


def _load_metrics(path: str | Path) -> pd.DataFrame:
    metrics_df = _load_csv(path, "fundamental_metrics.csv")
    _validate_columns(metrics_df, REQUIRED_METRIC_COLUMNS, "fundamental_metrics.csv")
    _validate_no_duplicate_keys(metrics_df, ["ticker", "fiscal_year", "fiscal_period"], "fundamental_metrics.csv")
    return metrics_df


def _parse_sort_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").fillna(pd.Timestamp.min)


def _latest_metrics_by_ticker(metrics_df: pd.DataFrame) -> dict[str, pd.Series]:
    normalized = metrics_df.copy()
    normalized["_normalized_ticker"] = normalized["ticker"].map(_normalized)
    normalized["_report_date"] = _parse_sort_date(normalized["report_date"])
    normalized["_period_end_date"] = _parse_sort_date(normalized["period_end_date"])
    normalized["_fiscal_year_number"] = pd.to_numeric(normalized["fiscal_year"], errors="coerce").fillna(-1)
    normalized = normalized.sort_values(
        [
            "_normalized_ticker",
            "_report_date",
            "_period_end_date",
            "_fiscal_year_number",
            "fiscal_period",
        ],
        kind="mergesort",
    )

    latest: dict[str, pd.Series] = {}
    for _, row in normalized.iterrows():
        latest[row["_normalized_ticker"]] = row
    return latest


def _is_insufficient_quality(row: pd.Series) -> bool:
    tokens = {
        _normalized(row.get("quality_state")),
        _normalized(row.get("quality_metadata_status")),
        _normalized(row.get("source_data_status")),
    }
    return bool(tokens & {"INSUFFICIENT_DATA", "SOURCE_MISSING", "ROW_MISSING", "INVALID", "INVALID_DATA"})


def _is_partial_quality(row: pd.Series) -> bool:
    tokens = {
        _normalized(row.get("quality_state")),
        _normalized(row.get("quality_metadata_status")),
        _normalized(row.get("source_data_status")),
    }
    return bool(tokens & {"PARTIAL_DATA", "PARTIAL"})


def _is_stale_quality(row: pd.Series) -> bool:
    tokens = {
        _normalized(row.get("quality_state")),
        _normalized(row.get("quality_metadata_status")),
        _normalized(row.get("source_data_status")),
    }
    return bool(tokens & {"STALE_DATA", "STALE"})


def _is_metrics_complete(metrics_row: pd.Series | None) -> bool:
    if metrics_row is None:
        return False
    return _clean_text(metrics_row.get("metric_status")).lower() == "complete"


def _metric(metrics_row: pd.Series | None, column: str) -> float | None:
    if metrics_row is None:
        return None
    return _parse_float(metrics_row.get(column))


def _margin_state(metrics_row: pd.Series | None) -> str:
    operating_margin = _metric(metrics_row, "operating_margin")
    net_margin = _metric(metrics_row, "net_margin")
    if operating_margin is None and net_margin is None:
        return "MARGIN_UNKNOWN"
    if (operating_margin is not None and operating_margin < 0) or (net_margin is not None and net_margin < 0):
        return "MARGIN_NEGATIVE"
    if (operating_margin is not None and operating_margin > 0) or (net_margin is not None and net_margin > 0):
        return "MARGIN_STABLE"
    return "MARGIN_UNKNOWN"


def _growth_state(metrics_row: pd.Series | None) -> str:
    values = [
        value
        for value in [
            _metric(metrics_row, "revenue_yoy_growth"),
            _metric(metrics_row, "eps_yoy_growth"),
        ]
        if value is not None
    ]
    if not values:
        return "GROWTH_UNKNOWN"
    has_positive = any(value > 0 for value in values)
    has_negative = any(value < 0 for value in values)
    if has_positive and has_negative:
        return "GROWTH_MIXED"
    if has_positive:
        return "GROWTH_POSITIVE"
    if has_negative:
        return "GROWTH_NEGATIVE"
    return "GROWTH_UNKNOWN"


def _leverage_state(metrics_row: pd.Series | None) -> str:
    debt_to_equity = _metric(metrics_row, "debt_to_equity")
    if debt_to_equity is None:
        return "LEVERAGE_UNKNOWN"
    if debt_to_equity < 0.5:
        return "LEVERAGE_LOW"
    if debt_to_equity <= 1.5:
        return "LEVERAGE_MODERATE"
    return "LEVERAGE_HIGH"


def _cash_flow_state(metrics_row: pd.Series | None) -> str:
    free_cash_flow_margin = _metric(metrics_row, "free_cash_flow_margin")
    if free_cash_flow_margin is None:
        return "CASH_FLOW_UNKNOWN"
    if free_cash_flow_margin > 0:
        return "CASH_FLOW_POSITIVE"
    if free_cash_flow_margin < 0:
        return "CASH_FLOW_NEGATIVE"
    return "CASH_FLOW_MIXED"


def _fundamental_profile_state(
    margin_state: str,
    growth_state: str,
    cash_flow_state: str,
    metrics_row: pd.Series | None,
) -> str:
    if metrics_row is None:
        return "UNKNOWN_PROFILE"
    if growth_state == "GROWTH_MIXED":
        return "MIXED_PROFILE"
    if (
        margin_state == "MARGIN_NEGATIVE"
        or growth_state == "GROWTH_NEGATIVE"
        or cash_flow_state == "CASH_FLOW_NEGATIVE"
    ):
        return "DETERIORATING_PROFILE"
    if growth_state == "GROWTH_POSITIVE" and margin_state == "MARGIN_STABLE":
        return "IMPROVING_PROFILE"
    if margin_state == "MARGIN_UNKNOWN" and growth_state == "GROWTH_UNKNOWN" and cash_flow_state == "CASH_FLOW_UNKNOWN":
        return "UNKNOWN_PROFILE"
    return "STABLE_PROFILE"


def _analysis_state_and_reason(row: pd.Series, metrics_row: pd.Series | None) -> tuple[str, str]:
    if _is_insufficient_quality(row):
        return "INSUFFICIENT_DATA", "fundamental quality input is insufficient"
    if metrics_row is None:
        return "LIMITED_ANALYSIS", "fundamental metrics input is unavailable"
    if _is_stale_quality(row):
        return "LIMITED_ANALYSIS", "fundamental source data is stale"
    if _is_partial_quality(row) or not _is_metrics_complete(metrics_row):
        return "LIMITED_ANALYSIS", "fundamental metrics or quality input is partial"
    return "ANALYSIS_READY", "fundamental quality and metrics inputs are available"


def _review_context(row: pd.Series, metrics_row: pd.Series | None, growth_state: str) -> tuple[str, str]:
    if _is_stale_quality(row):
        return "REVIEW_STALE_SOURCE", "source data is stale"
    if _is_insufficient_quality(row) or metrics_row is None or _is_partial_quality(row):
        return "REVIEW_DATA_LIMITATION", "analysis input is limited"
    if growth_state == "GROWTH_MIXED":
        return "REVIEW_METRIC_CONFLICT", "growth metrics are mixed"
    return "NO_REVIEW_FLAG", ""


def _input_coverage(row: pd.Series, metrics_row: pd.Series | None) -> str:
    if _is_insufficient_quality(row):
        return "quality_limited"
    if metrics_row is None:
        return "quality_only"
    return "quality_and_metrics"


def _data_status(row: pd.Series, metrics_row: pd.Series | None) -> str:
    if _is_stale_quality(row):
        return "stale_source"
    if _is_insufficient_quality(row):
        return "quality_limited"
    if metrics_row is None:
        return "metrics_missing"
    if not _is_metrics_complete(metrics_row) or _is_partial_quality(row):
        return "metrics_partial"
    return "metrics_available"


def _warnings(row: pd.Series, metrics_row: pd.Series | None, growth_state: str) -> str:
    warnings: list[str] = []
    if _is_stale_quality(row):
        warnings.append("stale_source")
    if _is_insufficient_quality(row):
        warnings.append("quality_limited")
    if metrics_row is None:
        warnings.append("metrics_missing")
    elif not _is_metrics_complete(metrics_row):
        warnings.append("metrics_partial")
        metric_warnings = _clean_text(metrics_row.get("metric_warnings"))
        if metric_warnings:
            warnings.append(f"metric_warnings:{metric_warnings}")
    if growth_state == "GROWTH_MIXED":
        warnings.append("growth_mixed")
    return ";".join(warnings)


def _analysis_row(quality_row: pd.Series, metrics_row: pd.Series | None) -> dict[str, Any]:
    margin_state = _margin_state(metrics_row)
    growth_state = _growth_state(metrics_row)
    leverage_state = _leverage_state(metrics_row)
    cash_flow_state = _cash_flow_state(metrics_row)
    analysis_state, analysis_reason = _analysis_state_and_reason(quality_row, metrics_row)
    review_flag, review_reason = _review_context(quality_row, metrics_row, growth_state)

    return {
        "ticker": quality_row["ticker"],
        "date": quality_row["date"],
        "fundamental_analysis_state": analysis_state,
        "fundamental_analysis_reason": analysis_reason,
        "fundamental_profile_state": _fundamental_profile_state(
            margin_state,
            growth_state,
            cash_flow_state,
            metrics_row,
        ),
        "margin_profile_state": margin_state,
        "growth_profile_state": growth_state,
        "leverage_profile_state": leverage_state,
        "cash_flow_profile_state": cash_flow_state,
        "fundamental_review_flag": review_flag,
        "fundamental_review_reason": review_reason,
        "analysis_data_status": _data_status(quality_row, metrics_row),
        "analysis_input_coverage": _input_coverage(quality_row, metrics_row),
        "analysis_warnings": _warnings(quality_row, metrics_row, growth_state),
    }


def _validate_no_forbidden_output(df: pd.DataFrame) -> None:
    forbidden_columns = sorted(
        column
        for column in df.columns
        if any(term in column.lower() for term in FORBIDDEN_SEMANTIC_TERMS)
    )
    if forbidden_columns:
        raise ValueError(f"fundamental_analysis output contains forbidden columns: {forbidden_columns}")

    forbidden_values: list[str] = []
    for column in df.columns:
        for value in df[column].astype(str):
            normalized_value = value.strip().lower()
            if any(term in normalized_value for term in FORBIDDEN_SEMANTIC_TERMS):
                forbidden_values.append(f"{column}={value}")
    if forbidden_values:
        raise ValueError(f"fundamental_analysis output contains forbidden values: {forbidden_values}")


def analyze_fundamentals(quality_df: pd.DataFrame, metrics_df: pd.DataFrame | None = None) -> pd.DataFrame:
    _validate_columns(quality_df, REQUIRED_QUALITY_COLUMNS, "fundamental_quality.csv")
    _validate_no_duplicate_keys(quality_df, ["ticker", "date"], "fundamental_quality.csv")

    latest_metrics = _latest_metrics_by_ticker(metrics_df) if metrics_df is not None else {}
    rows = [
        _analysis_row(quality_row, latest_metrics.get(_normalized(quality_row["ticker"])))
        for _, quality_row in quality_df.iterrows()
    ]
    analysis_df = pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)
    _validate_no_forbidden_output(analysis_df)
    return analysis_df


def build_fundamental_analysis(
    quality_path: str | Path,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    quality_df = _load_quality(quality_path)
    metrics_df = _load_metrics(metrics_path) if metrics_path is not None else None
    analysis_df = analyze_fundamentals(quality_df, metrics_df)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(output_path, index=False)

    return analysis_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build descriptive Fundamental Analysis from quality and metrics inputs.")
    parser.add_argument("--quality-path", required=True, help="Path to a fundamental_quality.csv input file.")
    parser.add_argument("--metrics-path", help="Optional path to a fundamental_metrics.csv input file.")
    parser.add_argument(
        "--output-path",
        help="Optional path for fundamental_analysis.csv output. No files are written unless supplied.",
    )
    args = parser.parse_args()

    analysis_df = build_fundamental_analysis(args.quality_path, args.metrics_path, args.output_path)
    summary = {
        "status": "VALID",
        "row_count": int(len(analysis_df)),
        "output_path": args.output_path or "",
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
