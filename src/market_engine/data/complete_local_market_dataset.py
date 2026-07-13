from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import pandas as pd
import requests
import yfinance as yf

from market_engine.data.canonical_universe_bootstrap import (
    SOURCE_SCHEMA_VERSION,
    run_universe_bootstrap,
)
from market_engine.data.local_market_data_universe import (
    DEFAULT_PRICE_HISTORY_ROOT,
    build_data_run,
    validate_price_history_csv,
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts/market_engine/data_runs")
DEFAULT_UNIVERSE_ARTIFACT_ROOT = Path("artifacts/market_engine/universe_runs")
DEFAULT_CANONICAL_CONFIG = Path("config/market_engine/universes/canonical_universe.json")
DEFAULT_SYMBOL_OVERRIDES = Path("config/market_engine/universes/symbol_overrides.json")
DEFAULT_EVALUATION_ARTIFACT = Path(
    "artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json"
)
DEFAULT_EVALUATION_REFRESH_ROOT = Path("artifacts/market_engine/evaluation_refresh_runs")
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_CUTOFF_DATE = "2026-07-10"

WIKIPEDIA_SOURCES = (
    {
        "universe_id": "sp500",
        "source_name": "Wikipedia S&P 500 constituent table",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_match": "Symbol",
        "symbol_columns": ("Symbol",),
        "name_columns": ("Security",),
        "sector_columns": ("GICS Sector",),
        "exchange": "US",
        "known_limitations": [
            "Wikipedia is a public reproducible table, not the official S&P Dow Jones licensed feed.",
            "Use an official constituent file for production-grade index licensing."
        ],
    },
    {
        "universe_id": "sp400",
        "source_name": "Wikipedia S&P MidCap 400 constituent table",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "table_match": "Symbol",
        "symbol_columns": ("Symbol",),
        "name_columns": ("Security",),
        "sector_columns": ("GICS Sector",),
        "exchange": "US",
        "known_limitations": [
            "Wikipedia is a public reproducible table, not the official S&P Dow Jones licensed feed.",
            "Use an official constituent file for production-grade index licensing."
        ],
    },
)

SUPPLEMENTAL_INSTRUMENTS = (
    {"symbol": "CLS", "name": "Celestica Inc.", "exchange": "NYSE", "country": "CA", "currency": "USD", "sector": "Technology"},
    {"symbol": "CRDO", "name": "Credo Technology Group Holding Ltd", "exchange": "NASDAQ", "country": "US", "currency": "USD", "sector": "Technology"},
    {"symbol": "IREN", "name": "IREN Limited", "exchange": "NASDAQ", "country": "AU", "currency": "USD", "sector": "Technology"},
    {"symbol": "VRT", "name": "Vertiv Holdings Co", "exchange": "NYSE", "country": "US", "currency": "USD", "sector": "Industrials"},
    {"symbol": "ASML", "name": "ASML Holding N.V.", "exchange": "NASDAQ", "country": "NL", "currency": "USD", "sector": "Technology"},
    {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Limited ADR", "exchange": "NYSE", "country": "TW", "currency": "USD", "sector": "Technology"},
)

ETF_INSTRUMENTS = (
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_type": "etf"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "asset_type": "etf"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "asset_type": "etf"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF Trust", "asset_type": "etf"},
    {"symbol": "XLK", "name": "Technology Select Sector SPDR Fund", "asset_type": "etf"},
    {"symbol": "XLF", "name": "Financial Select Sector SPDR Fund", "asset_type": "etf"},
    {"symbol": "XLE", "name": "Energy Select Sector SPDR Fund", "asset_type": "etf"},
    {"symbol": "SMH", "name": "VanEck Semiconductor ETF", "asset_type": "etf"},
    {"symbol": "SOXX", "name": "iShares Semiconductor ETF", "asset_type": "etf"},
)


def run_complete_dataset(
    *,
    run_id: str,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    universe_artifact_root: str | Path = DEFAULT_UNIVERSE_ARTIFACT_ROOT,
    canonical_config: str | Path = DEFAULT_CANONICAL_CONFIG,
    symbol_overrides: str | Path = DEFAULT_SYMBOL_OVERRIDES,
    start_date: str = DEFAULT_START_DATE,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    limit: int | None = None,
) -> tuple[dict[str, Any], Path]:
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_root = run_dir / "membership_sources"
    source_decision = write_membership_sources(source_root, run_id=run_id)
    run_symbol_overrides = write_data04_symbol_overrides(
        run_dir / "symbol_overrides.json",
        base_path=symbol_overrides,
        source_root=source_root,
    )
    universe_run_id = f"{run_id}-universe"
    universe_run, universe_dir = run_universe_bootstrap(
        source_root=source_root,
        base_config_path=canonical_config,
        price_history_root=price_history_root,
        artifact_root=universe_artifact_root,
        run_id=universe_run_id,
        universe_version="me-data04-complete-canonical-local-market-dataset-v1",
        snapshot_date="2026-07-13",
        symbol_overrides_path=run_symbol_overrides,
        canonical_config_output=canonical_config,
    )
    instruments = [
        row
        for row in universe_run["canonical_universe"]["instruments"]
        if row.get("source_mapping_status") == "mapped"
    ]
    if limit:
        instruments = instruments[:limit]
    acquisition = acquire_price_history(
        instruments,
        price_history_root=price_history_root,
        run_id=run_id,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    _write_json(run_dir / "source_decision.json", source_decision)
    _write_json(run_dir / "universe_summary.json", universe_run["canonical_universe"]["summary"])
    _write_json(run_dir / "acquisition_summary.json", acquisition["summary"])
    _write_json(run_dir / "per_ticker_status.json", acquisition["per_ticker_status"])
    _write_json(run_dir / "validation_summary.json", acquisition["validation_summary"])
    _write_json(run_dir / "stale_history_summary.json", acquisition["stale_history_summary"])
    _write_json(run_dir / "unresolved_mapping_summary.json", universe_run["unsupported_symbol_mappings"])

    coverage_run_id = f"{run_id}-coverage-after"
    coverage, coverage_dir = build_data_run(
        universe_path=canonical_config,
        price_history_root=price_history_root,
        artifact_root=artifact_root,
        run_id=coverage_run_id,
        report_only=True,
        required_forward_date=cutoff_date,
    )
    acceptance = build_acceptance_result(
        universe=universe_run["canonical_universe"],
        coverage=coverage["coverage_summary"],
        acquisition=acquisition,
    )
    manifest = {
        "schema_version": "market-engine-data04-complete-local-market-dataset-run-v1",
        "artifact_type": "market-engine-data04-complete-local-market-dataset-run",
        "run_id": run_id,
        "generated_at": _generated_at_from_run_id(run_id),
        "source_decision": {
            "membership_source": "Wikipedia reproducible constituent tables for S&P 500, Nasdaq-100, and S&P MidCap 400 plus explicit project/ETF supplements.",
            "price_history_source": "Yahoo Finance daily OHLCV via existing yfinance dependency.",
            "cutoff_date": cutoff_date,
            "license_note": "Yahoo Finance and Wikipedia terms apply; artifacts record provenance and should be reviewed before redistribution.",
        },
        "paths": {
            "run_dir": run_dir.as_posix(),
            "universe_run_dir": universe_dir.as_posix(),
            "coverage_after_dir": coverage_dir.as_posix(),
            "price_history_root": Path(price_history_root).as_posix(),
        },
        "guardrails": {
            "advice_generation_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "broker_order_execution_performed": False,
            "telegram_delivery_performed": False,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "coverage_after.json", coverage["coverage_summary"])
    _write_json(run_dir / "acceptance_result.json", acceptance)
    (run_dir / "report.md").write_text(render_report(manifest, source_decision, acquisition, coverage["coverage_summary"], acceptance), encoding="utf-8")
    return {"manifest": manifest, "acceptance_result": acceptance, "coverage": coverage, "acquisition": acquisition}, run_dir


def write_membership_sources(source_root: Path, *, run_id: str) -> dict[str, Any]:
    source_root.mkdir(parents=True, exist_ok=True)
    decisions = []
    for source in WIKIPEDIA_SOURCES:
        constituents = _read_wikipedia_constituents(source)
        payload = _source_payload(source, constituents)
        path = source_root / f"{source['universe_id']}.json"
        _write_json(path, payload)
        decisions.append({"universe_id": source["universe_id"], "path": path.as_posix(), "count": len(constituents), "url": source["url"]})
    _write_json(
        source_root / "supplemental.json",
        _source_payload(
            {
                "universe_id": "explicit_supplemental_watch",
                "source_name": "ME-DATA04 explicit project and evaluation supplement",
                "url": "local:ME-DATA04",
                "known_limitations": ["Project supplement, not an index membership source."],
            },
            list(SUPPLEMENTAL_INSTRUMENTS),
        ),
    )
    decisions.append({"universe_id": "explicit_supplemental_watch", "path": (source_root / "supplemental.json").as_posix(), "count": len(SUPPLEMENTAL_INSTRUMENTS), "url": "local:ME-DATA04"})
    _write_json(
        source_root / "etfs.json",
        _source_payload(
            {
                "universe_id": "etf_context",
                "source_name": "ME-DATA04 controlled ETF context",
                "url": "local:ME-DATA04",
                "known_limitations": ["Compact ETF context set, not a full ETF universe."],
                "asset_type": "etf",
            },
            list(ETF_INSTRUMENTS),
        ),
    )
    decisions.append({"universe_id": "etf_context", "path": (source_root / "etfs.json").as_posix(), "count": len(ETF_INSTRUMENTS), "url": "local:ME-DATA04"})
    return {
        "schema_version": "market-engine-data04-source-decision-v1",
        "run_id": run_id,
        "membership_sources": decisions,
        "price_history_source": {
            "provider": "Yahoo Finance",
            "library": "yfinance",
            "granularity": "1d",
            "adjusted_prices": "auto_adjust=False; Adj Close retained when returned",
            "storage": "data/processed/<source_symbol>.csv",
        },
    }


def acquire_price_history(
    instruments: Sequence[Mapping[str, Any]],
    *,
    price_history_root: str | Path,
    run_id: str,
    start_date: str,
    cutoff_date: str,
) -> dict[str, Any]:
    output_root = Path(price_history_root)
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    instrument_rows = [
        {
            "symbol": str(instrument["symbol"]),
            "source_symbol": str(instrument["source_symbol"]),
            "yf_symbol": _to_yfinance_symbol(str(instrument["source_symbol"])),
        }
        for instrument in instruments
    ]
    for batch in _chunks(instrument_rows, 100):
        yf_symbols = [row["yf_symbol"] for row in batch]
        try:
            batch_frame = yf.download(
                yf_symbols,
                start=start_date,
                end="2026-07-14",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            for row in batch:
                destination = output_root / f"{row['source_symbol']}.csv"
                statuses.append(_status(row["symbol"], row["source_symbol"], row["yf_symbol"], "acquisition_failed", f"{type(exc).__name__}: {exc}", destination))
            continue
        for row in batch:
            symbol = row["symbol"]
            source_symbol = row["source_symbol"]
            yf_symbol = row["yf_symbol"]
            destination = output_root / f"{source_symbol}.csv"
            try:
                frame = _extract_batch_frame(batch_frame, yf_symbol, len(batch) == 1)
                frame = _normalize_yfinance_frame(frame, yf_symbol)
                if frame.empty:
                    statuses.append(_status(symbol, source_symbol, yf_symbol, "acquisition_failed", "empty_download", destination))
                    continue
                frame = frame[frame["Date"] <= cutoff_date].copy()
                if frame.empty:
                    statuses.append(_status(symbol, source_symbol, yf_symbol, "acquisition_failed", "no_rows_before_cutoff", destination))
                    continue
                frame.to_csv(destination, index=False)
                validation = validate_price_history_csv(destination)
                if validation["status"] != "valid":
                    statuses.append({**_status(symbol, source_symbol, yf_symbol, "validation_failed", validation["note"], destination), **validation})
                    continue
                end_date = validation["end_date"]
                status = "valid_current_snapshot" if end_date >= cutoff_date else "stale_snapshot"
                statuses.append(
                    {
                        **_status(symbol, source_symbol, yf_symbol, status, "downloaded", destination),
                        "start_date": validation["start_date"],
                        "end_date": end_date,
                        "row_count": validation["row_count"],
                        "checksum": validation["checksum"],
                    }
                )
            except Exception as exc:
                statuses.append(_status(symbol, source_symbol, yf_symbol, "acquisition_failed", f"{type(exc).__name__}: {exc}", destination))
    counts = Counter(row["status"] for row in statuses)
    validation_summary = {
        "schema_version": "market-engine-data04-validation-summary-v1",
        "valid": counts.get("valid_current_snapshot", 0),
        "stale": counts.get("stale_snapshot", 0),
        "invalid": counts.get("validation_failed", 0),
        "failed": counts.get("acquisition_failed", 0),
    }
    stale_rows = [row for row in statuses if row["status"] == "stale_snapshot"]
    return {
        "schema_version": "market-engine-data04-acquisition-run-v1",
        "run_id": run_id,
        "summary": {
            "requested": len(instruments),
            "valid_current_snapshot": counts.get("valid_current_snapshot", 0),
            "stale_snapshot": counts.get("stale_snapshot", 0),
            "validation_failed": counts.get("validation_failed", 0),
            "acquisition_failed": counts.get("acquisition_failed", 0),
            "cutoff_date": cutoff_date,
        },
        "per_ticker_status": {"schema_version": "market-engine-data04-per-ticker-status-v1", "entries": statuses},
        "validation_summary": validation_summary,
        "stale_history_summary": {"schema_version": "market-engine-data04-stale-history-summary-v1", "count": len(stale_rows), "entries": stale_rows},
    }


def _chunks(rows: Sequence[Mapping[str, str]], size: int) -> list[Sequence[Mapping[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _extract_batch_frame(batch_frame: pd.DataFrame, yf_symbol: str, single: bool) -> pd.DataFrame:
    if batch_frame is None or batch_frame.empty:
        return pd.DataFrame()
    if single:
        return batch_frame
    if not isinstance(batch_frame.columns, pd.MultiIndex):
        return pd.DataFrame()
    if yf_symbol in batch_frame.columns.get_level_values(0):
        return batch_frame[yf_symbol]
    return pd.DataFrame()


def build_acceptance_result(*, universe: Mapping[str, Any], coverage: Mapping[str, Any], acquisition: Mapping[str, Any]) -> dict[str, Any]:
    summary = coverage["summary"]
    total = summary["total_canonical_instruments"]
    valid = summary["valid"]
    coverage_pct = round((valid / total) * 100, 2) if total else 0.0
    checks = {
        "canonical_universe_gt_900": total > 900,
        "valid_history_gt_90pct": coverage_pct > 90,
        "current_history_materially_current": acquisition["summary"]["valid_current_snapshot"] > 0 and summary["insufficient"] == 0,
    }
    status = "operational_dataset_complete" if all(checks.values()) else "operational_dataset_partial"
    return {
        "schema_version": "market-engine-data04-acceptance-result-v1",
        "status": status,
        "checks": checks,
        "metrics": {
            "canonical_universe": total,
            "valid_local_history": valid,
            "valid_history_coverage_pct": coverage_pct,
            "insufficient": summary["insufficient"],
            "missing": summary["missing"],
            "invalid": summary["invalid"],
            "unsupported": summary["unsupported"],
        },
    }


def render_report(
    manifest: Mapping[str, Any],
    source_decision: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    coverage: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> str:
    summary = coverage["summary"]
    return "\n".join(
        [
            "# ME-DATA04 Complete Local Market Dataset Run",
            "",
            f"Run ID: {manifest['run_id']}",
            f"Status: {acceptance['status']}",
            f"Cutoff date: {manifest['source_decision']['cutoff_date']}",
            "",
            "## Sources",
            "",
            f"Membership: {manifest['source_decision']['membership_source']}",
            f"Price history: {manifest['source_decision']['price_history_source']}",
            "",
            "## Acquisition",
            "",
            f"Requested: {acquisition['summary']['requested']}",
            f"Valid current snapshots: {acquisition['summary']['valid_current_snapshot']}",
            f"Stale snapshots: {acquisition['summary']['stale_snapshot']}",
            f"Validation failed: {acquisition['summary']['validation_failed']}",
            f"Acquisition failed: {acquisition['summary']['acquisition_failed']}",
            "",
            "## Coverage",
            "",
            f"Canonical instruments: {summary['total_canonical_instruments']}",
            f"Valid: {summary['valid']}",
            f"Missing: {summary['missing']}",
            f"Insufficient: {summary['insufficient']}",
            f"Invalid: {summary['invalid']}",
            f"Unsupported: {summary['unsupported']}",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run, output_dir = run_complete_dataset(
            run_id=args.run_id,
            price_history_root=args.price_history_root,
            artifact_root=args.artifact_root,
            universe_artifact_root=args.universe_artifact_root,
            canonical_config=args.canonical_config,
            symbol_overrides=args.symbol_overrides,
            start_date=args.start_date,
            cutoff_date=args.cutoff_date,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=stderr)
        return 2
    print(json.dumps({"run_id": args.run_id, "output_dir": output_dir.as_posix(), "acceptance": run["acceptance_result"]}, indent=2, sort_keys=True), file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the complete ME-DATA04 local market dataset from reproducible sources.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT.as_posix())
    parser.add_argument("--universe-artifact-root", default=DEFAULT_UNIVERSE_ARTIFACT_ROOT.as_posix())
    parser.add_argument("--canonical-config", default=DEFAULT_CANONICAL_CONFIG.as_posix())
    parser.add_argument("--symbol-overrides", default=DEFAULT_SYMBOL_OVERRIDES.as_posix())
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _read_wikipedia_constituents(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    response = requests.get(
        source["url"],
        headers={"User-Agent": "market-scanner-me-data04-local-dataset/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    table = None
    for candidate in tables:
        if any(str(column) == source["table_match"] for column in candidate.columns):
            table = candidate
            break
    if table is None:
        raise ValueError(f"could not find source table for {source['universe_id']}")
    rows = []
    for _, row in table.iterrows():
        symbol = _first(row, source["symbol_columns"])
        if not symbol:
            continue
        canonical = str(symbol).strip().upper()
        rows.append(
            {
                "symbol": canonical,
                "source_symbol": _to_yfinance_symbol(canonical),
                "name": _first(row, source["name_columns"]) or canonical,
                "exchange": source["exchange"],
                "country": "US",
                "currency": "USD",
                "sector": _first(row, source["sector_columns"]),
            }
        )
    return rows


def write_data04_symbol_overrides(path: Path, *, base_path: str | Path, source_root: Path) -> Path:
    overrides_by_symbol: dict[str, dict[str, Any]] = {}
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    for override in base.get("overrides") or []:
        overrides_by_symbol[str(override["canonical_symbol"]).upper()] = dict(override)
    for source_path in sorted(source_root.glob("*.json")):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        for row in payload.get("constituents") or []:
            symbol = str(row.get("symbol") or "").upper()
            source_symbol = str(row.get("source_symbol") or symbol).upper()
            if symbol and source_symbol and symbol != source_symbol:
                overrides_by_symbol[symbol] = {
                    "canonical_symbol": symbol,
                    "source_symbol": source_symbol,
                    "reason": "ME-DATA04 yfinance acquisition symbol normalization for class-share or provider syntax.",
                }
    _write_json(
        path,
        {
            "schema_version": "market-engine-symbol-overrides-v1",
            "overrides": sorted(overrides_by_symbol.values(), key=lambda row: row["canonical_symbol"]),
        },
    )
    return path


def _source_payload(source: Mapping[str, Any], constituents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "universe_id": source["universe_id"],
        "source_name": source["source_name"],
        "snapshot_date": "2026-07-13",
        "retrieval_date": "2026-07-13",
        "provenance": source["url"],
        "status": "active",
        "asset_type": source.get("asset_type", "equity"),
        "known_limitations": source.get("known_limitations", []),
        "constituents": list(constituents),
    }


def _normalize_yfinance_frame(frame: pd.DataFrame, yf_symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if yf_symbol in out.columns.get_level_values(-1):
            out = out.xs(yf_symbol, axis=1, level=-1)
        else:
            out.columns = out.columns.get_level_values(0)
    out = out.reset_index()
    rename = {str(column): str(column).title() for column in out.columns}
    out = out.rename(columns=rename)
    if "Datetime" in out.columns and "Date" not in out.columns:
        out = out.rename(columns={"Datetime": "Date"})
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(out.columns):
        return pd.DataFrame()
    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]].dropna()


def _to_yfinance_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _first(row: Mapping[str, Any], columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value
    return None


def _status(symbol: str, source_symbol: str, yf_symbol: str, status: str, note: str, path: Path) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "source_symbol": source_symbol,
        "yfinance_symbol": yf_symbol,
        "status": status,
        "note": note,
        "artifactpath": path.as_posix(),
    }


def _generated_at_from_run_id(run_id: str) -> str:
    marker = run_id.rsplit("-", 1)[-1]
    try:
        return datetime.strptime(marker, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
