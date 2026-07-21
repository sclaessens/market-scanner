from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.data.primary_source_metric_derivation_compact_evidence import (
    OUTPUT_NAMES,
    build_compact_derivation_pilot_evidence,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, absolute_full_path: bool = False) -> dict[str, Path]:
    operator = tmp_path / "operator"
    data07 = tmp_path / "data07"
    data06 = tmp_path / "data06"
    run31 = tmp_path / "run31"
    formula = tmp_path / "formula.json"
    _write(
        operator / "primary_source_facts.json",
        {
            "schema_version": "facts-v1",
            "package_id": "facts-1",
            "derivation_timestamp": "2026-07-19T18:00:00Z",
            "facts": [{"fact_id": "revenue", "canonical_concept": "revenue"}],
        },
    )
    _write(formula, {"schema_version": "formula-v1", "formulas": []})
    derived = {
        "schema_version": "derived-v1",
        "boundary": "derived only",
        "derivations": [
            {"canonical_metric": "gross_margin", "status": "derived", "calculation_result": 0.4},
            {
                "canonical_metric": "debt_to_equity",
                "status": "blocked",
                "component_fact_ids": ["debt"],
                "denominator_fact_ids": ["equity"],
                "reason_codes": ["DEBT_COMPONENT_MISSING"],
            },
        ],
    }
    _write(operator / "derived_metrics.json", derived)
    _write(operator / "derivation_validation.json", {"schema_version": "validation-v1"})
    _write(
        operator / "derivation_approval_decision.json",
        {"schema_version": "approval-v1", "decision_id": "approval-1"},
    )
    _write(
        operator / "governed_fundamental_metrics.json",
        {"schema_version": "governed-v2"},
    )
    _write(
        data07 / "manifest.json",
        {
            "schema_version": "data07-v1",
            "run_id": "data07-1",
            "raw_snapshot": {"raw_path": "data/snapshots/data07-1/operator.json"},
            "guardrails": {"network_access_performed": False},
        },
    )
    _write(
        data07 / "batch_execution_summary.json",
        {
            "selected_count": 12,
            "imported_count": 1,
            "normalized_count": 1,
            "success_count": 1,
            "blocked_count": 11,
            "failed_count": 0,
            "pending_count": 0,
            "not_selected_count": 940,
            "provider_calls_performed": 0,
            "reconciliation": {"reconciled": True},
        },
    )
    _write(
        data07 / "concrete_source_approval_validation.json",
        {"schema_version": "approval-validation-v1", "validation_status": "approved"},
    )
    _write(
        data07 / "normalized_metric_evidence.json",
        {
            "record_count": 1,
            "records": [
                {
                    "metric_lineage": [
                        {
                            "canonical_metric": "revenue_growth_yoy",
                            "evidence_type": "direct",
                            "normalized_value": 0.1,
                            "normalized_unit": "ratio",
                            "reporting_period": "2026-Q2",
                        },
                        {
                            "canonical_metric": "gross_margin",
                            "evidence_type": "derived",
                            "normalized_value": 0.4,
                            "normalized_unit": "ratio",
                            "reporting_period": "2026-Q2",
                        },
                    ]
                }
            ],
        },
    )
    _write(
        data07 / "coverage_before_after.json",
        {
            "current_sprint_comparison": {
                "before": {"fundamental_partial": 39},
                "after": {"fundamental_partial": 39},
                "absolute_delta": {"fundamental_partial": 0},
            },
            "ticker_delta": {
                "ticker": "AAA",
                "before_status": "partial",
                "after_status": "partial",
                "new_metrics": ["gross_margin"],
                "remaining_missing_metrics": ["debt_to_equity"],
            },
            "historical_origin_comparison": {"baseline": {}, "regression_counts": {}},
        },
    )
    _write(data06 / "manifest.json", {"run_id": "data06-1"})
    _write(data06 / "fundamental_coverage_summary.json", {})
    _write(data06 / "per_ticker_fundamental_status.json", {})
    _write(
        run31 / "run_evidence_index.json",
        {
            "run_id": "run31-1",
            "full_artifact": {
                "local_path": str(tmp_path / "absolute") if absolute_full_path else "artifacts/full/run31-1"
            },
        },
    )
    return {
        "operator": operator,
        "formula": formula,
        "data07": data07,
        "data06": data06,
        "run31": run31,
        "output": tmp_path / "compact",
    }


def test_compact_evidence_is_complete_checksum_indexed_and_preserves_evidence_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    artifacts = build_compact_derivation_pilot_evidence(
        operator_input_dir=paths["operator"].name,
        formula_catalog_path=paths["formula"].name,
        data07_run_dir=paths["data07"].name,
        data06_run_dir=paths["data06"].name,
        run31_evidence_dir=paths["run31"].name,
        output_dir=paths["output"].name,
    )

    assert all(paths["output"].joinpath(f"{name}.json").is_file() for name in OUTPUT_NAMES)
    assert paths["output"].joinpath("top_level_checksums.json").is_file()
    assert paths["output"].joinpath("report.md").is_file()
    assert artifacts["derived_metric_evidence"]["direct_metrics"][0]["evidence_type"] == "direct"
    assert artifacts["derived_metric_evidence"]["derived_metrics"][0]["evidence_type"] == "derived"
    assert artifacts["coverage_delta"]["historical_origin_comparison"]["attributable_to_current_sprint"] is False


def test_compact_evidence_rejects_absolute_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, absolute_full_path=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute local paths"):
        build_compact_derivation_pilot_evidence(
            operator_input_dir=paths["operator"].name,
            formula_catalog_path=paths["formula"].name,
            data07_run_dir=paths["data07"].name,
            data06_run_dir=paths["data06"].name,
            run31_evidence_dir=paths["run31"].name,
            output_dir=paths["output"].name,
        )
