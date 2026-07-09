from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from market_engine.advisory.advisory_response_dry_run import (
    AdvisoryResponseDryRunError,
    run_controlled_advisory_response_dry_run,
    run_controlled_advisory_response_dry_run_command,
)
from tests.market_engine.advisory.test_advisory_artifact import _assemble, _write_json
from tests.market_engine.advisory.test_advisory_response_grounding import (
    _prompt,
    _response,
)


def test_dry_run_persists_grounded_artifacts(tmp_path: Path) -> None:
    source_path, response_path = _write_source_and_response(tmp_path)

    result = run_controlled_advisory_response_dry_run(
        advisory_artifact_path=source_path,
        question="Explain the current state.",
        question_class="current_state_explanation",
        response_fixture_path=response_path,
        run_id="ci08-smoke-grounded",
        artifact_root=tmp_path / "artifacts",
    )

    assert result.summary["dry_run_state"] == "dry_run_completed_grounded"
    assert result.grounding_result.status == "grounded"
    assert result.prompt_package_path.exists()
    assert result.synthetic_response_path.exists()
    assert result.grounding_result_path.exists()
    assert result.dry_run_summary_path.exists()
    assert result.manifest_path.exists()


def test_dry_run_persists_ungrounded_result_without_success_state(tmp_path: Path) -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["assessment"][0]["claim_type"] = "unsupported_materiality_claim"
    source_path, response_path = _write_source_and_response(
        tmp_path,
        source=source,
        response=response,
    )

    result = run_controlled_advisory_response_dry_run(
        advisory_artifact_path=source_path,
        question="Explain the current state.",
        question_class="current_state_explanation",
        response_fixture_path=response_path,
        run_id="ci08-smoke-ungrounded",
        artifact_root=tmp_path / "artifacts",
    )

    assert result.summary["dry_run_state"] == "dry_run_failed_ungrounded"
    assert result.grounding_result.valid is False


def test_dry_run_overwrite_protection(tmp_path: Path) -> None:
    source_path, response_path = _write_source_and_response(tmp_path)
    kwargs = {
        "advisory_artifact_path": source_path,
        "question": "Explain the current state.",
        "question_class": "current_state_explanation",
        "response_fixture_path": response_path,
        "run_id": "ci08-overwrite",
        "artifact_root": tmp_path / "artifacts",
    }
    run_controlled_advisory_response_dry_run(**kwargs)

    with pytest.raises(AdvisoryResponseDryRunError, match="already exists"):
        run_controlled_advisory_response_dry_run(**kwargs)


def test_dry_run_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    source_path, response_path = _write_source_and_response(tmp_path)
    result_a = run_controlled_advisory_response_dry_run(
        advisory_artifact_path=source_path,
        question="Explain the current state.",
        question_class="current_state_explanation",
        response_fixture_path=response_path,
        run_id="ci08-deterministic-a",
        artifact_root=tmp_path / "artifacts",
    )
    result_b = run_controlled_advisory_response_dry_run(
        advisory_artifact_path=source_path,
        question="Explain the current state.",
        question_class="current_state_explanation",
        response_fixture_path=response_path,
        run_id="ci08-deterministic-b",
        artifact_root=tmp_path / "artifacts",
    )

    assert result_a.grounding_result.to_payload() == result_b.grounding_result.to_payload()


def test_cli_grounded_smoke_exits_zero_and_emits_json(tmp_path: Path) -> None:
    source_path, response_path = _write_source_and_response(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_controlled_advisory_response_dry_run_command(
        [
            "--advisory-artifact",
            str(source_path),
            "--question",
            "Explain the current state.",
            "--question-class",
            "current_state_explanation",
            "--response-fixture",
            str(response_path),
            "--run-id",
            "ci08-cli-grounded",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--emit-json",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue())["grounding_status"] == "grounded"


def test_cli_ungrounded_smoke_exits_nonzero(tmp_path: Path) -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_references"][0]["path"] = "$.missing.path"
    source_path, response_path = _write_source_and_response(
        tmp_path,
        source=source,
        response=response,
    )

    exit_code = run_controlled_advisory_response_dry_run_command(
        [
            "--advisory-artifact",
            str(source_path),
            "--question",
            "Explain the current state.",
            "--question-class",
            "current_state_explanation",
            "--response-fixture",
            str(response_path),
            "--run-id",
            "ci08-cli-ungrounded",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 2


def test_cli_authority_violation_smoke_exits_nonzero(tmp_path: Path) -> None:
    source = _assemble()
    prompt = _prompt(source, "sizing_question")
    response = _response(prompt, response_mode="advisory_interpretation")
    response["assessment"][0]["claim_type"] = "unsupported_sizing_claim"
    source_path, response_path = _write_source_and_response(
        tmp_path,
        source=source,
        response=response,
    )

    exit_code = run_controlled_advisory_response_dry_run_command(
        [
            "--advisory-artifact",
            str(source_path),
            "--question",
            "How much should I buy?",
            "--question-class",
            "sizing_question",
            "--response-fixture",
            str(response_path),
            "--run-id",
            "ci08-cli-authority",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 2


def test_invalid_source_artifact_stops_before_response_validation(tmp_path: Path) -> None:
    source = _assemble()
    source["composition_status"]["state"] = "invalid"
    prompt = _prompt(_assemble(), "current_state_explanation")
    response = _response(prompt)
    source_path, response_path = _write_source_and_response(
        tmp_path,
        source=source,
        response=response,
    )

    exit_code = run_controlled_advisory_response_dry_run_command(
        [
            "--advisory-artifact",
            str(source_path),
            "--question",
            "Explain the current state.",
            "--question-class",
            "current_state_explanation",
            "--response-fixture",
            str(response_path),
            "--run-id",
            "ci08-invalid-source",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 2
    assert not (tmp_path / "artifacts" / "ci08-invalid-source").exists()


def _write_source_and_response(
    tmp_path: Path,
    *,
    source: dict[str, object] | None = None,
    response: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    source_payload = source or _assemble()
    if response is None:
        prompt = _prompt(source_payload, "current_state_explanation")
        response_payload = _response(prompt)
    else:
        response_payload = response
    source_path = tmp_path / "chatgpt_ready_advisory.json"
    response_path = tmp_path / "synthetic_response.json"
    _write_json(source_path, source_payload)
    _write_json(response_path, response_payload)
    return source_path, response_path
