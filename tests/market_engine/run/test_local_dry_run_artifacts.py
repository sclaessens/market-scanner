from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.run.end_to_end_dry_run import (
    MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
    build_market_engine_end_to_end_dry_run,
)
from market_engine.run.end_to_end_dry_run_command import (
    build_synthetic_dry_run_stage_payloads,
    run_market_engine_end_to_end_dry_run_command,
)
from market_engine.run.local_dry_run_artifacts import (
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION,
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY,
    MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION,
    LOCAL_DRY_RUN_PERSISTENCE_MODE,
    LocalDryRunArtifactError,
    persist_market_engine_local_dry_run_artifact,
)


def test_valid_dry_run_artifact_is_persisted_locally(tmp_path: Path) -> None:
    payload = _dry_run_payload()

    result = persist_market_engine_local_dry_run_artifact(
        payload,
        output_root=tmp_path,
        artifact_created_at="2026-06-17T14:00:00Z",
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.run_directory == tmp_path / "dry-run-001"
    assert artifact["artifact_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION
    )
    assert artifact["artifact_persistence_mode"] == LOCAL_DRY_RUN_PERSISTENCE_MODE
    assert artifact["artifact_path_category"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_PATH_CATEGORY
    )
    assert artifact["non_production_artifact"] is True
    assert artifact["source_dry_run_format_version"] == (
        MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION
    )
    assert artifact["source_dry_run_id"] == "dry-run-001"
    assert artifact["payload"] == json.loads(json.dumps(payload))
    assert artifact["payload"]["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.current_quantity"
    ] == 0
    assert manifest["manifest_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION
    )
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["source_dry_run_id"] == "dry-run-001"


def test_persisted_paths_stay_inside_configured_output_root(tmp_path: Path) -> None:
    result = persist_market_engine_local_dry_run_artifact(
        _dry_run_payload(),
        output_root=tmp_path,
        artifact_created_at="2026-06-17T14:00:00Z",
    )

    root = tmp_path.resolve()

    assert result.artifact_path.resolve().relative_to(root)
    assert result.manifest_path.resolve().relative_to(root)


def test_existing_artifact_directory_is_not_overwritten(tmp_path: Path) -> None:
    persist_market_engine_local_dry_run_artifact(
        _dry_run_payload(),
        output_root=tmp_path,
        artifact_created_at="2026-06-17T14:00:00Z",
    )

    with pytest.raises(LocalDryRunArtifactError, match="already exists"):
        persist_market_engine_local_dry_run_artifact(
            _dry_run_payload(),
            output_root=tmp_path,
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_parent_directory_output_root_is_rejected() -> None:
    with pytest.raises(LocalDryRunArtifactError, match="parent traversal"):
        persist_market_engine_local_dry_run_artifact(
            _dry_run_payload(),
            output_root=Path("..") / "outside",
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_parent_traversal_dry_run_id_is_rejected(tmp_path: Path) -> None:
    payload = {
        **_dry_run_payload(),
        "dry_run_id": "../escape",
    }

    with pytest.raises(LocalDryRunArtifactError, match="safe path segment"):
        persist_market_engine_local_dry_run_artifact(
            payload,
            output_root=tmp_path,
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_absolute_path_escape_dry_run_id_is_rejected(tmp_path: Path) -> None:
    payload = {
        **_dry_run_payload(),
        "dry_run_id": "/tmp/escape",
    }

    with pytest.raises(LocalDryRunArtifactError, match="safe path segment"):
        persist_market_engine_local_dry_run_artifact(
            payload,
            output_root=tmp_path,
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_unserializable_payload_fails_clearly(tmp_path: Path) -> None:
    payload = {
        **_dry_run_payload(),
        "audit_metadata": {"unserializable": object()},
    }

    with pytest.raises(LocalDryRunArtifactError, match="not JSON serializable"):
        persist_market_engine_local_dry_run_artifact(
            payload,
            output_root=tmp_path,
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_unsupported_payload_contract_fails_closed(tmp_path: Path) -> None:
    payload = {
        **_dry_run_payload(),
        "dry_run_format_version": "market-engine-end-to-end-dry-run-v0",
    }

    with pytest.raises(LocalDryRunArtifactError, match="unsupported format version"):
        persist_market_engine_local_dry_run_artifact(
            payload,
            output_root=tmp_path,
            artifact_created_at="2026-06-17T14:00:00Z",
        )


def test_command_writes_local_artifact_only_when_explicitly_requested(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--dry-run-id",
            "local-run-001",
            "--generated-at",
            "2026-06-17T14:00:00Z",
            "--artifact-output-root",
            str(tmp_path),
            "--artifact-created-at",
            "2026-06-17T14:01:00Z",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert not (tmp_path / "local-run-001").exists()

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--dry-run-id",
            "local-run-001",
            "--generated-at",
            "2026-06-17T14:00:00Z",
            "--write-local-artifact",
            "--artifact-output-root",
            str(tmp_path),
            "--artifact-created-at",
            "2026-06-17T14:01:00Z",
        ]
    )

    captured = capsys.readouterr()
    artifact_path = (
        tmp_path
        / "local-run-001"
        / "artifacts"
        / "market_engine_dry_run_local-run-001_2026-06-17.json"
    )

    assert exit_code == 0
    assert captured.err == ""
    assert artifact_path.exists()


def test_local_dry_run_artifact_module_has_no_side_effect_dependencies() -> None:
    module_source = Path("src/market_engine/run/local_dry_run_artifacts.py").read_text(
        encoding="utf-8"
    )

    forbidden_terms = (
        "from scripts",
        "import scripts",
        "from market_scanner",
        "import market_scanner",
        "telegram",
        "smtplib",
        "yfinance",
        "urllib",
        "requests",
        "socket",
        "subprocess",
    )

    assert not any(term in module_source for term in forbidden_terms)


def _dry_run_payload() -> dict[str, object]:
    stage_payloads = build_synthetic_dry_run_stage_payloads()
    return build_market_engine_end_to_end_dry_run(
        stage_payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
        generated_at="2026-06-17T14:00:00Z",
    ).to_payload()
