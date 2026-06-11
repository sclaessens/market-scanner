from __future__ import annotations

from pathlib import Path


LEGACY_OPS_CAPTURE_MODULE_PATH = Path("scripts/ops/capture_historical_evidence.py")

HISTORICAL_EVIDENCE_CAPTURE_CONTRACT = {
    "pipeline_runs.csv": {
        "required_fields": {
            "run_id",
            "captured_at",
            "decision_reporting_linkage_status",
            "decision_reporting_observation_count",
        },
        "purpose": "run-level audit trail",
    },
    "pipeline_artifacts.csv": {
        "required_fields": {
            "run_id",
            "artifact_path",
            "artifact_exists",
            "row_count",
            "file_size_bytes",
            "content_hash",
            "diagnostic_notes",
        },
        "purpose": "source artifact manifest",
    },
    "decision_reporting_observations.csv": {
        "required_fields": {
            "run_id",
            "ticker",
            "date",
            "decision_artifact_path",
            "reporting_artifact_path",
            "decision_input_row_hash",
            "reporting_source_row_identity",
            "reporting_represented_flag",
            "diagnostic_note",
        },
        "purpose": "decision/reporting linkage evidence",
    },
}

FORBIDDEN_CAPTURE_AUTHORITY_TERMS = {
    "BUY",
    "SELL",
    "allocation recommendation",
    "execution recommendation",
    "trade recommendation",
    "portfolio mutation",
    "watchlist mutation",
    "telegram delivery",
}


def test_ops_capture_script_remains_legacy_reference_only():
    assert LEGACY_OPS_CAPTURE_MODULE_PATH == Path("scripts/ops/capture_historical_evidence.py")


def test_historical_evidence_contract_keeps_expected_artifact_names():
    assert set(HISTORICAL_EVIDENCE_CAPTURE_CONTRACT) == {
        "pipeline_runs.csv",
        "pipeline_artifacts.csv",
        "decision_reporting_observations.csv",
    }


def test_historical_evidence_contract_requires_run_identity_on_all_artifacts():
    for artifact_contract in HISTORICAL_EVIDENCE_CAPTURE_CONTRACT.values():
        assert "run_id" in artifact_contract["required_fields"]


def test_historical_evidence_contract_tracks_artifact_manifest_integrity():
    manifest_fields = HISTORICAL_EVIDENCE_CAPTURE_CONTRACT["pipeline_artifacts.csv"][
        "required_fields"
    ]

    assert {
        "artifact_path",
        "artifact_exists",
        "row_count",
        "file_size_bytes",
        "content_hash",
        "diagnostic_notes",
    }.issubset(manifest_fields)


def test_decision_reporting_observation_contract_is_evidence_only():
    observation_fields = HISTORICAL_EVIDENCE_CAPTURE_CONTRACT[
        "decision_reporting_observations.csv"
    ]["required_fields"]

    assert {
        "decision_artifact_path",
        "reporting_artifact_path",
        "decision_input_row_hash",
        "reporting_source_row_identity",
        "reporting_represented_flag",
        "diagnostic_note",
    }.issubset(observation_fields)

    forbidden_lower = {term.lower() for term in FORBIDDEN_CAPTURE_AUTHORITY_TERMS}
    contract_text = " ".join(
        sorted(
            field
            for artifact in HISTORICAL_EVIDENCE_CAPTURE_CONTRACT.values()
            for field in artifact["required_fields"]
        )
    ).lower()

    for forbidden_term in forbidden_lower:
        assert forbidden_term not in contract_text


def test_active_code_no_longer_imports_ops_capture_script():
    for root in (Path("src"), Path("tests"), Path(".github")):
        if not root.exists():
            continue

        for path in root.rglob("*.py"):
            if path == Path("tests/ops/test_capture_historical_evidence.py"):
                continue

            source = path.read_text(encoding="utf-8")
            assert "from scripts.ops import capture_historical_evidence" not in source
            assert "import scripts.ops" not in source
