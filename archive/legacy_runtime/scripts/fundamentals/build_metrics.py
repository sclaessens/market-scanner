from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fundamentals.build_history_intake import validate_fundamentals_history

IDENTITY_COLUMNS = [
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "report_date",
    "currency",
    "source_name",
    "source_reference",
    "source_freshness_date",
    "extraction_date",
]

RAW_NUMERIC_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
]

METRIC_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "free_cash_flow_margin",
    "debt_to_equity",
    "return_on_equity",
    "revenue_yoy_growth",
    "eps_yoy_growth",
    "free_cash_flow_yoy_growth",
]

HELPER_COLUMNS = ["metric_status", "metric_missing_inputs", "metric_warnings"]


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_float(value: Any) -> float | None:
    text = _clean_text(value)
    if text == "":
        return None
    return float(text)


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _yoy_growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def _metric_issue(metric_name: str, inputs: dict[str, float | None], denominator_name: str | None = None) -> str | None:
    missing = [name for name, value in inputs.items() if value is None]
    if missing:
        return f"{metric_name}:missing:{'|'.join(missing)}"
    if denominator_name is not None and inputs[denominator_name] == 0:
        return f"{metric_name}:zero_denominator:{denominator_name}"
    return None


def _prior_lookup_key(row: pd.Series) -> tuple[str, int, str]:
    return (
        _clean_text(row["ticker"]).upper(),
        int(_clean_text(row["fiscal_year"])),
        _clean_text(row["fiscal_period"]).upper(),
    )


def _build_prior_lookup(df: pd.DataFrame) -> dict[tuple[str, int, str], pd.Series]:
    return {_prior_lookup_key(row): row for _, row in df.iterrows()}


def _calculate_row_metrics(row: pd.Series, prior_lookup: dict[tuple[str, int, str], pd.Series]) -> dict[str, Any]:
    values = {column: _parse_float(row[column]) for column in RAW_NUMERIC_COLUMNS}
    metric_values: dict[str, float | None] = {
        "gross_margin": _safe_divide(values["gross_profit"], values["revenue"]),
        "operating_margin": _safe_divide(values["operating_income"], values["revenue"]),
        "net_margin": _safe_divide(values["net_income"], values["revenue"]),
        "free_cash_flow_margin": _safe_divide(values["free_cash_flow"], values["revenue"]),
        "debt_to_equity": _safe_divide(values["total_debt"], values["total_equity"]),
        "return_on_equity": _safe_divide(values["net_income"], values["total_equity"]),
    }

    missing_inputs = [
        issue
        for issue in [
            _metric_issue(
                "gross_margin",
                {"gross_profit": values["gross_profit"], "revenue": values["revenue"]},
                "revenue",
            ),
            _metric_issue(
                "operating_margin",
                {"operating_income": values["operating_income"], "revenue": values["revenue"]},
                "revenue",
            ),
            _metric_issue(
                "net_margin",
                {"net_income": values["net_income"], "revenue": values["revenue"]},
                "revenue",
            ),
            _metric_issue(
                "free_cash_flow_margin",
                {"free_cash_flow": values["free_cash_flow"], "revenue": values["revenue"]},
                "revenue",
            ),
            _metric_issue(
                "debt_to_equity",
                {"total_debt": values["total_debt"], "total_equity": values["total_equity"]},
                "total_equity",
            ),
            _metric_issue(
                "return_on_equity",
                {"net_income": values["net_income"], "total_equity": values["total_equity"]},
                "total_equity",
            ),
        ]
        if issue is not None
    ]

    ticker, fiscal_year, fiscal_period = _prior_lookup_key(row)
    prior_row = prior_lookup.get((ticker, fiscal_year - 1, fiscal_period))
    warnings: list[str] = []

    if prior_row is None:
        metric_values["revenue_yoy_growth"] = None
        metric_values["eps_yoy_growth"] = None
        metric_values["free_cash_flow_yoy_growth"] = None
        warnings.append("yoy_growth:missing_prior_year")
    else:
        prior_values = {column: _parse_float(prior_row[column]) for column in RAW_NUMERIC_COLUMNS}
        metric_values["revenue_yoy_growth"] = _yoy_growth(values["revenue"], prior_values["revenue"])
        metric_values["eps_yoy_growth"] = _yoy_growth(values["diluted_eps"], prior_values["diluted_eps"])
        metric_values["free_cash_flow_yoy_growth"] = _yoy_growth(
            values["free_cash_flow"],
            prior_values["free_cash_flow"],
        )
        for metric_name, current_name, prior_name in [
            ("revenue_yoy_growth", "revenue", "revenue"),
            ("eps_yoy_growth", "diluted_eps", "diluted_eps"),
            ("free_cash_flow_yoy_growth", "free_cash_flow", "free_cash_flow"),
        ]:
            issue = _metric_issue(
                metric_name,
                {
                    current_name: values[current_name],
                    f"prior_{prior_name}": prior_values[prior_name],
                },
                f"prior_{prior_name}",
            )
            if issue is not None:
                warnings.append(issue)

    row_result = {column: row[column] for column in IDENTITY_COLUMNS}
    row_result.update(metric_values)
    row_result["metric_status"] = "complete" if all(value is not None for value in metric_values.values()) else "partial"
    row_result["metric_missing_inputs"] = ";".join(missing_inputs)
    row_result["metric_warnings"] = ";".join(warnings)
    return row_result


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    prior_lookup = _build_prior_lookup(df)
    rows = [_calculate_row_metrics(row, prior_lookup) for _, row in df.iterrows()]
    return pd.DataFrame(rows, columns=IDENTITY_COLUMNS + METRIC_COLUMNS + HELPER_COLUMNS)


def build_fundamental_metrics(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    validation_result = validate_fundamentals_history(input_path)
    if validation_result["status"] != "VALID":
        rendered_result = json.dumps(validation_result, sort_keys=True)
        raise ValueError(f"fundamentals history validation failed: {rendered_result}")

    raw_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    metrics_df = calculate_metrics(raw_df)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)

    return metrics_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic fundamental metrics from raw history.")
    parser.add_argument("--input-path", required=True, help="Path to a validated fundamentals_history.csv input file.")
    parser.add_argument(
        "--output-path",
        help="Optional path for fundamental_metrics.csv output. No files are written unless supplied.",
    )
    args = parser.parse_args()

    metrics_df = build_fundamental_metrics(args.input_path, args.output_path)
    summary = {
        "status": "VALID",
        "row_count": int(len(metrics_df)),
        "output_path": args.output_path or "",
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
