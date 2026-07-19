from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "market-engine-data09-compact-operator-pilot-evidence-v1"
OUTPUT_NAMES = (
    "manifest",
    "source_document_checksums",
    "source_approval_validation",
    "data08_validation_report",
    "data07_pilot_summary",
    "approved_ticker_metric_evidence",
    "coverage_delta",
    "downstream_run_index",
)


def build_compact_operator_pilot_evidence(
    *,
    operator_input_dir: str | Path,
    data07_run_dir: str | Path,
    data06_run_dir: str | Path,
    run31_evidence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    operator_dir = Path(operator_input_dir)
    data07_dir = Path(data07_run_dir)
    data06_dir = Path(data06_run_dir)
    run31_dir = Path(run31_evidence_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"compact ME-DATA09 evidence directory already exists: {destination}")

    input_path = operator_dir / "fundamental_metric_input.json"
    package_path = operator_dir / "fundamental_metrics.json"
    report_path = operator_dir / "validation_report.json"
    decision_path = operator_dir / "source_approval_decision.json"
    source_input = _read(input_path)
    package = _read(package_path)
    report = _read(report_path)
    decision = _read(decision_path)
    data07_manifest = _read(data07_dir / "manifest.json")
    batch = _read(data07_dir / "batch_execution_summary.json")
    approval = _read(data07_dir / "concrete_source_approval_validation.json")
    normalized = _read(data07_dir / "normalized_metric_evidence.json")
    coverage = _read(data07_dir / "coverage_before_after.json")
    data06_manifest = _read(data06_dir / "manifest.json")
    run31_index = _read(run31_dir / "run_evidence_index.json")

    approved_tickers = decision.get("approved_tickers") or []
    ticker_records = [row for row in normalized.get("records") or [] if row.get("ticker") in approved_tickers]
    if len(approved_tickers) != 1 or len(ticker_records) != 1:
        raise ValueError("compact ME-DATA09 evidence requires exactly one approved normalized ticker")
    current = coverage.get("current_sprint_comparison")
    ticker_delta = coverage.get("ticker_delta")
    if not isinstance(current, Mapping) or not isinstance(ticker_delta, Mapping):
        raise ValueError("current sprint and ticker comparisons are required")

    destination.mkdir(parents=True)
    artifacts = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "run_id": data07_manifest["run_id"],
            "full_run_committed": False,
            "compact_evidence_committed": True,
            "approved_tickers": approved_tickers,
            "run_ids": {
                "data07": data07_manifest["run_id"],
                "data06": data06_manifest["run_id"],
                "run31": run31_index["run_id"],
            },
            "schema_versions": {
                "input": source_input["schema_version"],
                "package": package["schema_version"],
                "data08_report": report["schema_version"],
                "approval_decision": decision["schema_version"],
                "approval_validation": approval["schema_version"],
                "data07": data07_manifest["schema_version"],
            },
            "artifact_checksums": {
                "input_sha256": _sha(input_path),
                "package_sha256": _sha(package_path),
                "validation_report_sha256": _sha(report_path),
                "approval_decision_sha256": _sha(decision_path),
            },
            "full_local_artifact_paths": {
                "data07": data07_dir.as_posix(),
                "data06": data06_dir.as_posix(),
                "run31_compact": run31_dir.as_posix(),
                "raw_snapshot": (data07_manifest.get("raw_snapshot") or {}).get("raw_path"),
            },
            "publication_boundary": "Compact facts, source identities, checksums, validation outcomes, and aggregate deltas are committed. Full run trees and source documents remain local.",
            "guardrails": data07_manifest["guardrails"],
            "outputs": [f"{name}.json" for name in OUTPUT_NAMES] + ["top_level_checksums.json", "report.md"],
        },
        "source_document_checksums": {
            "schema_version": "market-engine-data09-source-document-checksums-v1",
            "documents": decision["source_documents"],
        },
        "source_approval_validation": approval,
        "data08_validation_report": report,
        "data07_pilot_summary": {
            "schema_version": "market-engine-data09-data07-pilot-summary-v1",
            "run_id": data07_manifest["run_id"],
            "counts": {key: batch[key] for key in (
                "selected_count", "imported_count", "normalized_count", "success_count",
                "blocked_count", "failed_count", "pending_count", "not_selected_count",
            )},
            "reconciliation": batch["reconciliation"],
            "raw_snapshot": data07_manifest["raw_snapshot"],
            "provider_calls_performed": batch["provider_calls_performed"],
            "network_access_performed": data07_manifest["guardrails"]["network_access_performed"],
        },
        "approved_ticker_metric_evidence": {
            "schema_version": "market-engine-data09-approved-ticker-metric-evidence-v1",
            "approved_tickers": approved_tickers,
            "record": ticker_records[0],
            "ticker_delta": ticker_delta,
        },
        "coverage_delta": {
            "schema_version": "market-engine-data09-current-sprint-coverage-delta-v1",
            "current_sprint_comparison": current,
            "ticker_delta": ticker_delta,
            "historical_origin_comparison": {
                "attributable_to_current_sprint": False,
                "baseline": (coverage.get("historical_origin_comparison") or {}).get("baseline"),
                "boundary": "Historical DATA06 transitions are not current ME-DATA09 sprint improvements.",
            },
        },
        "downstream_run_index": {
            "schema_version": "market-engine-data09-downstream-run-index-v1",
            "data06": {
                "run_id": data06_manifest["run_id"],
                "manifest_path": (data06_dir / "manifest.json").as_posix(),
                "manifest_sha256": _sha(data06_dir / "manifest.json"),
                "coverage_summary_sha256": _sha(data06_dir / "fundamental_coverage_summary.json"),
                "per_ticker_status_sha256": _sha(data06_dir / "per_ticker_fundamental_status.json"),
            },
            "run31": {
                "run_id": run31_index["run_id"],
                "compact_index_path": (run31_dir / "run_evidence_index.json").as_posix(),
                "compact_index_sha256": _sha(run31_dir / "run_evidence_index.json"),
                "full_artifact": run31_index.get("full_artifact"),
            },
        },
    }
    for name in OUTPUT_NAMES:
        _write(destination / f"{name}.json", artifacts[name])
    (destination / "report.md").write_text(_report(artifacts), encoding="utf-8")
    checksums = {
        **{f"{name}.json": _sha(destination / f"{name}.json") for name in OUTPUT_NAMES},
        "report.md": _sha(destination / "report.md"),
    }
    _write(destination / "top_level_checksums.json", {
        "schema_version": "market-engine-data09-compact-evidence-checksums-v1",
        "checksums": checksums,
    })
    return artifacts


def _report(artifacts: Mapping[str, Any]) -> str:
    manifest = artifacts["manifest"]
    pilot = artifacts["data07_pilot_summary"]
    current = artifacts["coverage_delta"]["current_sprint_comparison"]
    ticker = artifacts["coverage_delta"]["ticker_delta"]
    return "\n".join([
        "# ME-DATA09 Compact Bounded Operator Pilot Evidence",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Approved ticker: `{ticker['ticker']}`",
        f"Ticker status: `{ticker['before_status']}` -> `{ticker['after_status']}`",
        f"Current sprint before: `{json.dumps(current['before'], sort_keys=True)}`",
        f"Current sprint after: `{json.dumps(current['after'], sort_keys=True)}`",
        f"Current sprint absolute delta: `{json.dumps(current['absolute_delta'], sort_keys=True)}`",
        f"Selected/imported/normalized/blocked: `{pilot['counts']['selected_count']}/{pilot['counts']['imported_count']}/{pilot['counts']['normalized_count']}/{pilot['counts']['blocked_count']}`",
        "Historical DATA06 transitions are explicitly not attributable to ME-DATA09.",
        "Full source documents and full run trees remain local and uncommitted.",
        "",
    ])


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact ME-DATA09 operator pilot evidence.")
    parser.add_argument("--operator-input-dir", required=True)
    parser.add_argument("--data07-run-dir", required=True)
    parser.add_argument("--data06-run-dir", required=True)
    parser.add_argument("--run31-evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    build_compact_operator_pilot_evidence(
        operator_input_dir=args.operator_input_dir,
        data07_run_dir=args.data07_run_dir,
        data06_run_dir=args.data06_run_dir,
        run31_evidence_dir=args.run31_evidence_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
