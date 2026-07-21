from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "market-engine-data10-compact-derived-metric-pilot-evidence-v1"
OUTPUT_NAMES = (
    "manifest",
    "source_fact_summary",
    "formula_catalog_snapshot",
    "derivation_validation",
    "derived_metric_evidence",
    "approval_validation",
    "pilot_summary",
    "coverage_delta",
    "downstream_run_index",
)


def build_compact_derivation_pilot_evidence(
    *,
    operator_input_dir: str | Path,
    formula_catalog_path: str | Path,
    data07_run_dir: str | Path,
    data06_run_dir: str | Path,
    run31_evidence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    operator_dir = Path(operator_input_dir)
    formula_path = Path(formula_catalog_path)
    data07_dir = Path(data07_run_dir)
    data06_dir = Path(data06_run_dir)
    run31_dir = Path(run31_evidence_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"compact ME-DATA10 evidence directory already exists: {destination}")

    facts_path = operator_dir / "primary_source_facts.json"
    derived_path = operator_dir / "derived_metrics.json"
    derivation_validation_path = operator_dir / "derivation_validation.json"
    approval_path = operator_dir / "derivation_approval_decision.json"
    governed_path = operator_dir / "governed_fundamental_metrics.json"
    facts = _read(facts_path)
    formula_catalog = _read(formula_path)
    derived = _read(derived_path)
    derivation_validation = _read(derivation_validation_path)
    approval = _read(approval_path)
    governed = _read(governed_path)
    data07_manifest = _read(data07_dir / "manifest.json")
    batch = _read(data07_dir / "batch_execution_summary.json")
    approval_validation = _read(data07_dir / "concrete_source_approval_validation.json")
    normalized = _read(data07_dir / "normalized_metric_evidence.json")
    coverage = _read(data07_dir / "coverage_before_after.json")
    data06_manifest = _read(data06_dir / "manifest.json")
    run31_index = _read(run31_dir / "run_evidence_index.json")

    if approval_validation.get("validation_status") != "approved":
        raise ValueError("compact ME-DATA10 evidence requires approved derivation evidence")
    if normalized.get("record_count") != 1:
        raise ValueError("compact ME-DATA10 evidence requires exactly one normalized ticker")
    current = coverage.get("current_sprint_comparison")
    ticker_delta = coverage.get("ticker_delta")
    if not isinstance(current, Mapping) or not isinstance(ticker_delta, Mapping):
        raise ValueError("current sprint and ticker comparisons are required")
    full_paths = {
        "data07": data07_dir.as_posix(),
        "data06": data06_dir.as_posix(),
        "run31_compact": run31_dir.as_posix(),
        "run31_full": str((run31_index.get("full_artifact") or {}).get("local_path") or ""),
        "raw_snapshot": str((data07_manifest.get("raw_snapshot") or {}).get("raw_path") or ""),
    }
    if any(Path(path).is_absolute() for path in full_paths.values() if path):
        raise ValueError("compact evidence must not publish absolute local paths")

    direct_metrics = []
    derived_metrics = []
    for lineage in normalized["records"][0].get("metric_lineage") or []:
        summary = {
            "canonical_metric": lineage["canonical_metric"],
            "evidence_type": lineage["evidence_type"],
            "normalized_value": lineage["normalized_value"],
            "normalized_unit": lineage["normalized_unit"],
            "reporting_period": lineage["reporting_period"],
        }
        (direct_metrics if lineage["evidence_type"] == "direct" else derived_metrics).append(summary)
    attributed_ticker_delta = dict(ticker_delta)
    attributed_ticker_delta["imported_metric_set"] = list(ticker_delta.get("new_metrics") or [])
    attributed_ticker_delta["new_derived_metrics"] = sorted(row["canonical_metric"] for row in derived_metrics)
    attributed_ticker_delta["retained_direct_metrics"] = sorted(row["canonical_metric"] for row in direct_metrics)
    attributed_ticker_delta["new_metrics"] = list(attributed_ticker_delta["new_derived_metrics"])
    artifacts = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "sprint_id": "ME-DATA10",
            "run_ids": {
                "derivation": facts["package_id"],
                "package_validation": f"me-data10-package-validation-{facts['derivation_timestamp'].replace('-', '').replace(':', '').replace('Z', 'Z')}",
                "approval": approval["decision_id"],
                "data07": data07_manifest["run_id"],
                "data06": data06_manifest["run_id"],
                "run31": run31_index["run_id"],
            },
            "schema_versions": {
                "fact_package": facts["schema_version"],
                "formula_catalog": formula_catalog["schema_version"],
                "derived_package": derived["schema_version"],
                "derivation_validation": derivation_validation["schema_version"],
                "derivation_approval": approval["schema_version"],
                "governed_data07_package": governed["schema_version"],
                "data07_run": data07_manifest["schema_version"],
            },
            "artifact_checksums": {
                "fact_package_sha256": _sha(facts_path),
                "formula_catalog_sha256": _sha(formula_path),
                "derived_package_sha256": _sha(derived_path),
                "derivation_validation_sha256": _sha(derivation_validation_path),
                "approval_decision_sha256": _sha(approval_path),
                "governed_package_sha256": _sha(governed_path),
            },
            "full_run_committed": False,
            "compact_evidence_committed": True,
            "full_local_artifact_paths": full_paths,
            "guardrails": data07_manifest["guardrails"],
            "publication_boundary": "Only compact primary facts, formula definitions, lineage, approvals, checksums, and measured deltas are committed. Full run trees and source documents remain local.",
            "outputs": [f"{name}.json" for name in OUTPUT_NAMES] + ["top_level_checksums.json", "report.md"],
        },
        "source_fact_summary": {
            "schema_version": "market-engine-data10-source-fact-summary-v1",
            "fact_package_id": facts["package_id"],
            "facts": facts["facts"],
            "blocked_fact_requirements": [
                {
                    "canonical_metric": row["canonical_metric"],
                    "component_fact_ids": row.get("component_fact_ids") or [],
                    "denominator_fact_ids": row.get("denominator_fact_ids") or [],
                    "reason_codes": row.get("reason_codes") or [],
                }
                for row in derived["derivations"]
                if row.get("status") == "blocked"
            ],
        },
        "formula_catalog_snapshot": formula_catalog,
        "derivation_validation": derivation_validation,
        "derived_metric_evidence": {
            "schema_version": "market-engine-data10-derived-metric-evidence-summary-v1",
            "direct_metrics": sorted(direct_metrics, key=lambda row: row["canonical_metric"]),
            "derived_metrics": sorted(derived_metrics, key=lambda row: row["canonical_metric"]),
            "derivations": derived["derivations"],
            "boundary": derived["boundary"],
        },
        "approval_validation": approval_validation,
        "pilot_summary": {
            "schema_version": "market-engine-data10-data07-pilot-summary-v1",
            "run_id": data07_manifest["run_id"],
            "counts": {
                key: batch[key]
                for key in (
                    "selected_count",
                    "imported_count",
                    "normalized_count",
                    "success_count",
                    "blocked_count",
                    "failed_count",
                    "pending_count",
                    "not_selected_count",
                )
            },
            "reconciliation": batch["reconciliation"],
            "raw_snapshot": data07_manifest["raw_snapshot"],
            "provider_calls_performed": batch["provider_calls_performed"],
            "network_access_performed": data07_manifest["guardrails"]["network_access_performed"],
        },
        "coverage_delta": {
            "schema_version": "market-engine-data10-current-sprint-coverage-delta-v1",
            "current_sprint_comparison": current,
            "ticker_delta": attributed_ticker_delta,
            "regressions": ((coverage.get("historical_origin_comparison") or {}).get("regression_counts") or {}),
            "historical_origin_comparison": {
                "attributable_to_current_sprint": False,
                "baseline": (coverage.get("historical_origin_comparison") or {}).get("baseline"),
                "boundary": "Historical DATA06 transitions are not current ME-DATA10 sprint improvements.",
            },
        },
        "downstream_run_index": {
            "schema_version": "market-engine-data10-downstream-run-index-v1",
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
    destination.mkdir(parents=True)
    for name in OUTPUT_NAMES:
        _write(destination / f"{name}.json", artifacts[name])
    (destination / "report.md").write_text(_report(artifacts), encoding="utf-8")
    checksums = {
        **{f"{name}.json": _sha(destination / f"{name}.json") for name in OUTPUT_NAMES},
        "report.md": _sha(destination / "report.md"),
    }
    _write(
        destination / "top_level_checksums.json",
        {
            "schema_version": "market-engine-data10-compact-evidence-checksums-v1",
            "checksums": checksums,
        },
    )
    return artifacts


def _report(artifacts: Mapping[str, Any]) -> str:
    manifest = artifacts["manifest"]
    pilot = artifacts["pilot_summary"]
    coverage = artifacts["coverage_delta"]
    ticker = coverage["ticker_delta"]
    return "\n".join(
        [
            "# ME-DATA10 — Implement a generic governed primary-source fundamental metric derivation engine and execute a bounded pilot",
            "",
            f"DATA07 run: `{manifest['run_ids']['data07']}`",
            f"Ticker status: `{ticker['before_status']}` -> `{ticker['after_status']}`",
            f"New governed metrics: `{json.dumps(ticker['new_metrics'], sort_keys=True)}`",
            f"Remaining missing metrics: `{json.dumps(ticker['remaining_missing_metrics'], sort_keys=True)}`",
            f"Current sprint before: `{json.dumps(coverage['current_sprint_comparison']['before'], sort_keys=True)}`",
            f"Current sprint after: `{json.dumps(coverage['current_sprint_comparison']['after'], sort_keys=True)}`",
            f"Current sprint delta: `{json.dumps(coverage['current_sprint_comparison']['absolute_delta'], sort_keys=True)}`",
            f"Selected/imported/normalized/blocked: `{pilot['counts']['selected_count']}/{pilot['counts']['imported_count']}/{pilot['counts']['normalized_count']}/{pilot['counts']['blocked_count']}`",
            "Historical DATA06 transitions are explicitly not attributable to ME-DATA10.",
            "Full source documents and full run trees remain local and uncommitted.",
            "",
        ]
    )


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact ME-DATA10 derived metric pilot evidence.")
    parser.add_argument("--operator-input-dir", required=True)
    parser.add_argument("--formula-catalog", required=True)
    parser.add_argument("--data07-run-dir", required=True)
    parser.add_argument("--data06-run-dir", required=True)
    parser.add_argument("--run31-evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    build_compact_derivation_pilot_evidence(
        operator_input_dir=args.operator_input_dir,
        formula_catalog_path=args.formula_catalog,
        data07_run_dir=args.data07_run_dir,
        data06_run_dir=args.data06_run_dir,
        run31_evidence_dir=args.run31_evidence_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
