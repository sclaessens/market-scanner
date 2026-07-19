from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from market_engine.data import operator_source_approval as approval


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    package_id = "approved-partial-package"
    input_path = _write(tmp_path / "input.json", {
        "schema_version": approval.INPUT_SCHEMA_VERSION,
        "package_id": package_id,
    })
    package_path = _write(tmp_path / "package.json", {
        "schema_version": approval.OPERATOR_IMPORT_SCHEMA_VERSION,
        "package_schema_version": approval.INPUT_SCHEMA_VERSION,
        "package_id": package_id,
        "records": [{"metrics": {"revenue_growth_yoy": {"value": 17.0}, "eps_growth_yoy": {"value": 22.0}}}],
    })
    report_path = _write(tmp_path / "report.json", {
        "schema_version": approval.REPORT_SCHEMA_VERSION,
        "validator_version": approval.VALIDATOR_VERSION,
        "package_id": package_id,
        "status": "accepted",
        "downstream_consumability": "structurally_valid_for_explicit_source_approval_review",
        "input_sha256": _sha(input_path),
    })
    source_path = tmp_path / "source.html"
    source_path.write_text("official source", encoding="utf-8")
    decision = {
        "schema_version": approval.DECISION_SCHEMA_VERSION,
        "decision_id": "decision-1",
        "decision": "approved",
        "scope": approval.APPROVED_SCOPE,
        "reviewer_roles": list(approval.REQUIRED_REVIEWER_ROLES),
        "package_id": package_id,
        "artifact_bindings": {
            "input_path": input_path.as_posix(),
            "input_sha256": _sha(input_path),
            "package_sha256": _sha(package_path),
            "validation_report_path": report_path.as_posix(),
            "validation_report_sha256": _sha(report_path),
        },
        "source_documents": [{"local_path": source_path.as_posix(), "sha256": _sha(source_path)}],
        "reviews": {name: {"status": "approved"} for name in approval.REQUIRED_REVIEW_DIMENSIONS},
        "approved_metrics": ["eps_growth_yoy", "revenue_growth_yoy"],
        "explicitly_missing_metrics": ["debt_to_equity", "gross_margin", "operating_margin"],
    }
    return package_path, source_path, decision


def _validate(tmp_path: Path, package: Path, decision: dict[str, object]) -> dict[str, object]:
    return approval.validate_source_approval_decision(_write(tmp_path / "decision.json", decision), package)


def test_valid_partial_metric_approval_is_accepted_and_deterministic(tmp_path: Path) -> None:
    package, _, decision = _fixture(tmp_path)
    first = _validate(tmp_path, package, decision)
    second = approval.validate_source_approval_decision(tmp_path / "decision.json", package)

    assert first == second
    assert first["validation_status"] == "approved"
    assert first["reason_codes"] == []
    assert "password" not in json.dumps(first).lower()


def test_missing_and_malformed_approval_fail_closed(tmp_path: Path) -> None:
    package, _, _ = _fixture(tmp_path)
    missing = approval.validate_source_approval_decision(None, package)
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    malformed = approval.validate_source_approval_decision(tmp_path / "bad.json", package)

    assert missing["reason_codes"] == ["SOURCE_APPROVAL_DECISION_MISSING"]
    assert malformed["reason_codes"] == ["SOURCE_APPROVAL_MALFORMED"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda d: d.update(decision="blocked"), "SOURCE_APPROVAL_BLOCKED"),
        (lambda d: d.update(decision="rejected"), "SOURCE_APPROVAL_REJECTED"),
        (lambda d: d.update(decision="unexpected"), "UNKNOWN_SOURCE_APPROVAL_DECISION"),
        (lambda d: d.update(package_id="wrong"), "PACKAGE_ID_MISMATCH"),
        (lambda d: d["artifact_bindings"].update(input_sha256="0" * 64), "INPUT_CHECKSUM_MISMATCH"),
        (lambda d: d["artifact_bindings"].update(package_sha256="0" * 64), "PACKAGE_CHECKSUM_MISMATCH"),
        (lambda d: d["artifact_bindings"].update(validation_report_sha256="0" * 64), "REPORT_CHECKSUM_MISMATCH"),
        (lambda d: d["source_documents"][0].update(sha256="0" * 64), "SOURCE_DOCUMENT_CHECKSUM_MISMATCH"),
        (lambda d: d["reviews"].pop("freshness"), "REVIEW_DIMENSION_MISSING"),
        (lambda d: d["reviews"]["freshness"].update(status="blocked"), "REVIEW_DIMENSION_NOT_APPROVED"),
        (lambda d: d.update(approved_metrics=["revenue_growth_yoy"]), "APPROVED_METRIC_SET_MISMATCH"),
        (lambda d: d.update(explicitly_missing_metrics=[]), "EXPLICIT_MISSING_METRIC_SET_MISMATCH"),
    ],
)
def test_decision_mutations_fail_closed(tmp_path: Path, mutation, reason: str) -> None:
    package, _, decision = _fixture(tmp_path)
    changed = copy.deepcopy(decision)
    mutation(changed)
    result = _validate(tmp_path, package, changed)

    assert result["validation_status"] == "blocked"
    assert reason in result["reason_codes"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema_version", "wrong", "DATA08_CONTRACT_MISMATCH"),
        ("validator_version", "wrong", "DATA08_VALIDATOR_VERSION_MISMATCH"),
        ("status", "rejected", "DATA08_REPORT_NOT_ACCEPTED"),
        ("downstream_consumability", "not_consumable", "DATA08_DOWNSTREAM_STATE_MISMATCH"),
    ],
)
def test_data08_report_contract_mutations_fail_closed(tmp_path: Path, field: str, value: str, reason: str) -> None:
    package, _, decision = _fixture(tmp_path)
    report_path = Path(decision["artifact_bindings"]["validation_report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    _write(report_path, report)
    decision["artifact_bindings"]["validation_report_sha256"] = _sha(report_path)

    result = _validate(tmp_path, package, decision)
    assert reason in result["reason_codes"]
