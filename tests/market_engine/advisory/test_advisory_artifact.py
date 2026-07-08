from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from market_engine.advisory.advisory_artifact import (
    CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
    CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
    ChatGPTReadyAdvisoryArtifactError,
    assemble_chatgpt_ready_advisory_artifact,
    compose_chatgpt_ready_advisory_artifact_from_directory,
    load_chatgpt_ready_advisory_inputs,
    persist_chatgpt_ready_advisory_artifact,
)
from market_engine.advisory.daily_artifact import (
    run_chatgpt_ready_advisory_artifact_command,
)


GENERATED_AT = "2026-07-08T08:00:00Z"
_DEFAULT = object()


def test_eligible_artifact_embeds_valid_contexts_without_semantic_upgrade() -> None:
    artifact = _assemble()

    assert artifact["contract_identity"]["schema_version"] == (
        CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION
    )
    assert artifact["contract_identity"]["artifact_type"] == (
        CHATGPT_READY_ADVISORY_ARTIFACT_TYPE
    )
    assert artifact["composition_status"]["state"] == "eligible_artifact_produced"
    assert artifact["composition_status"]["semantic_override_performed"] is False
    assert artifact["advisory_eligibility"]["state"] == "eligible"
    assert artifact["advisory_eligibility"]["no_upgrade_from_upstream"] is True
    assert artifact["structured_decision_context"]["payload"]["ticker"] == "NVDA"
    assert artifact["portfolio_intelligence_context"]["payload"]["cash_context"][
        "amount"
    ] is None


def test_descriptive_only_structured_decision_output_cannot_be_upgraded() -> None:
    sdo = _sdo(data_coverage={"coverage_status": "descriptive_only"})
    artifact = _assemble(structured_decision_output=sdo)

    assert artifact["advisory_eligibility"]["state"] == "descriptive_only"
    assert artifact["composition_status"]["state"] == (
        "descriptive_only_artifact_produced"
    )


def test_blocked_upstream_produces_blocked_artifact() -> None:
    sdo = _sdo(
        data_coverage={
            "coverage_status": "blocked",
            "blocked_reason": "required_source_missing",
        },
        decision={"action": "blocked", "is_actionable": False},
    )
    artifact = _assemble(structured_decision_output=sdo)

    assert artifact["advisory_eligibility"]["state"] == "blocked"
    assert artifact["composition_status"]["state"] == "blocked_artifact_produced"
    assert "required_source_missing" in artifact["blockers"]


def test_optional_portfolio_absence_is_unavailable_not_empty_holdings() -> None:
    artifact = _assemble(portfolio_intelligence_context=None)

    portfolio = artifact["portfolio_intelligence_context"]
    assert portfolio["include_mode"] == "absent"
    assert portfolio["availability_state"] == "unavailable"
    assert portfolio["payload"] is None
    assert "holdings" not in portfolio


def test_optional_explainability_absence_keeps_change_rationale_unavailable() -> None:
    artifact = _assemble(explainability_change_rationale_context=None)

    explainability = artifact["explainability_change_rationale_context"]
    assert explainability["availability_state"] == "unavailable"
    assert artifact["advisory_eligibility"]["change_rationale_available"] is False


def test_dispatch_absence_is_optional_and_does_not_block() -> None:
    artifact = _assemble(dispatch_context=None)

    assert artifact["dispatch_context"]["include_mode"] == "absent"
    assert artifact["advisory_eligibility"]["state"] == "eligible"


def test_ticker_conflict_in_optional_context_blocks_without_rewriting_payload() -> None:
    portfolio = _portfolio_context(ticker="AMD")
    artifact = _assemble(portfolio_intelligence_context=portfolio)

    assert artifact["composition_status"]["state"] == "blocked_artifact_produced"
    assert "portfolio_intelligence_context.ticker_conflict" in artifact["blockers"]
    assert artifact["portfolio_intelligence_context"]["payload"]["holdings"][0][
        "ticker"
    ] == "AMD"


def test_run_identity_conflict_in_advisory_context_blocks() -> None:
    advisory = _advisory_context(run_id="other-run")
    artifact = _assemble(chatgpt_advisory_context=advisory)

    assert artifact["advisory_eligibility"]["state"] == "blocked"
    assert "chatgpt_advisory_context.run_identity_conflict" in artifact["blockers"]


def test_unsupported_required_structured_decision_schema_fails_closed() -> None:
    sdo = {**_sdo(), "schema_version": "structured-decision-output-v0"}

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="unsupported"):
        _assemble(structured_decision_output=sdo)


def test_unsupported_optional_schema_blocks_artifact() -> None:
    advisory = {**_advisory_context(), "schema_version": "chatgpt-v0"}
    artifact = _assemble(chatgpt_advisory_context=advisory)

    assert artifact["composition_status"]["state"] == "blocked_artifact_produced"
    assert "chatgpt_advisory_context.unsupported_schema_version" in artifact["blockers"]


def test_instrument_ticker_mismatch_in_required_sdo_fails_closed() -> None:
    sdo = _sdo()
    sdo["instrument"] = {**sdo["instrument"], "ticker": "AMD"}

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="ticker_conflict"):
        _assemble(structured_decision_output=sdo)


def test_partial_holdings_do_not_imply_not_held() -> None:
    portfolio = _portfolio_context(include_target_holding=False)
    artifact = _assemble(portfolio_intelligence_context=portfolio)

    holdings = artifact["portfolio_intelligence_context"]["payload"]["holdings"]
    assert holdings == []
    assert "not_held" not in json.dumps(artifact)


def test_cash_unavailable_is_preserved_as_null_not_zero() -> None:
    artifact = _assemble()
    cash_context = artifact["portfolio_intelligence_context"]["payload"]["cash_context"]

    assert cash_context["state"] == "not_provided"
    assert cash_context["amount"] is None


def test_not_comparable_explainability_is_not_treated_as_unchanged() -> None:
    explainability = _explainability_context(availability_state="not_comparable")
    artifact = _assemble(explainability_change_rationale_context=explainability)

    assert artifact["explainability_change_rationale_context"]["availability_state"] == (
        "not_comparable"
    )
    assert "unchanged" not in json.dumps(
        artifact["explainability_change_rationale_context"]
    )


def test_governor_ticker_conflict_blocks() -> None:
    artifact = _assemble(governor_context=_governor_context(ticker="AMD"))

    assert artifact["composition_status"]["state"] == "blocked_artifact_produced"
    assert "governor_context.ticker_conflict" in artifact["blockers"]


def test_dispatch_cannot_override_structured_decision_output() -> None:
    artifact = _assemble(
        dispatch_context={
            "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
            "subject": {"ticker": "NVDA"},
            "presentation_summary": {"decision": "different"},
        }
    )

    assert artifact["structured_decision_context"]["payload"]["decision"]["action"] == (
        "add_candidate"
    )
    assert artifact["composition_status"]["source_precedence"][0] == (
        "structured_decision_output_over_dispatch_presentation"
    )


def test_mixed_freshness_is_reported_per_family() -> None:
    sdo = _sdo(data_coverage={"freshness_status": "fresh"})
    governor = _governor_context()
    governor["freshness_context"] = {"global_freshness_status": "stale"}
    artifact = _assemble(structured_decision_output=sdo, governor_context=governor)

    assert artifact["freshness_context"]["global_freshness_status"] == "mixed"
    assert {
        entry["family"]: entry["status"]
        for entry in artifact["freshness_context"]["family_freshness"]
    }["governor_context"] == "stale"


def test_source_refs_are_unique_and_sorted() -> None:
    advisory = _advisory_context()
    advisory["source_artifact_refs"] = ["ref:b", "ref:a", "ref:a"]
    artifact = _assemble(chatgpt_advisory_context=advisory)

    assert artifact["source_artifact_references"] == sorted(
        set(artifact["source_artifact_references"])
    )


def test_assembly_is_deterministic_for_reordered_refs() -> None:
    advisory_a = _advisory_context()
    advisory_b = _advisory_context()
    advisory_a["source_artifact_refs"] = ["ref:b", "ref:a"]
    advisory_b["source_artifact_refs"] = ["ref:a", "ref:b"]

    assert _assemble(chatgpt_advisory_context=advisory_a) == _assemble(
        chatgpt_advisory_context=advisory_b
    )


def test_persist_writes_artifact_and_manifest_under_run_and_ticker(tmp_path: Path) -> None:
    result = persist_chatgpt_ready_advisory_artifact(_assemble(), output_root=tmp_path)

    assert result.artifact_path == tmp_path / "run-001" / "NVDA" / (
        "chatgpt_ready_advisory.json"
    )
    assert result.manifest_path == tmp_path / "run-001" / "NVDA" / "manifest.json"
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["contract_identity"]["artifact_type"] == (
        CHATGPT_READY_ADVISORY_ARTIFACT_TYPE
    )
    assert manifest["artifact_count"] == 1


def test_persist_refuses_overwrite_by_default(tmp_path: Path) -> None:
    persist_chatgpt_ready_advisory_artifact(_assemble(), output_root=tmp_path)

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="already exists"):
        persist_chatgpt_ready_advisory_artifact(_assemble(), output_root=tmp_path)


def test_persist_rejects_parent_traversal_output_root() -> None:
    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="parent traversal"):
        persist_chatgpt_ready_advisory_artifact(
            _assemble(),
            output_root=Path("..") / "outside",
        )


def test_persist_rejects_unsafe_run_id(tmp_path: Path) -> None:
    sdo = _sdo(run_id="../escape")

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="safe path segment"):
        persist_chatgpt_ready_advisory_artifact(
            _assemble(structured_decision_output=sdo),
            output_root=tmp_path,
        )


def test_loader_requires_explicit_structured_decision_file(tmp_path: Path) -> None:
    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="Required input"):
        load_chatgpt_ready_advisory_inputs(tmp_path)


def test_loader_reads_optional_files_when_present(tmp_path: Path) -> None:
    _write_json(tmp_path / "structured_decision_output.json", _sdo())
    _write_json(tmp_path / "chatgpt_advisory_context.json", _advisory_context())

    loaded = load_chatgpt_ready_advisory_inputs(tmp_path)

    assert loaded["structured_decision_output"]["ticker"] == "NVDA"
    assert loaded["chatgpt_advisory_context"]["ticker"] == "NVDA"
    assert loaded["portfolio_intelligence_context"] is None


def test_compose_from_directory_persists_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_json(input_dir / "structured_decision_output.json", _sdo())
    _write_json(input_dir / "chatgpt_advisory_context.json", _advisory_context())

    result = compose_chatgpt_ready_advisory_artifact_from_directory(
        input_artifact_dir=input_dir,
        output_root=output_dir,
        generated_at=GENERATED_AT,
    )

    assert result.artifact_path.exists()


def test_command_writes_artifact_and_emits_json_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_json(input_dir / "structured_decision_output.json", _sdo())
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_chatgpt_ready_advisory_artifact_command(
        [
            "--input-artifact-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            GENERATED_AT,
            "--emit-json",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    manifest = json.loads(stdout.getvalue())
    assert manifest["ticker"] == "NVDA"


def test_command_returns_nonzero_for_missing_required_input(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_chatgpt_ready_advisory_artifact_command(
        [
            "--input-artifact-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "Required input" in stderr.getvalue()


def test_module_has_no_side_effect_dependencies() -> None:
    module_source = Path("src/market_engine/advisory/advisory_artifact.py").read_text(
        encoding="utf-8"
    )

    forbidden_terms = (
        "openai",
        "telegram",
        "broker",
        "yfinance",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "portfolio write",
        "watchlist write",
    )

    assert not any(term in module_source.lower() for term in forbidden_terms)


def _assemble(
    *,
    structured_decision_output: dict[str, object] | object = _DEFAULT,
    chatgpt_advisory_context: dict[str, object] | None | object = _DEFAULT,
    portfolio_intelligence_context: dict[str, object] | None | object = _DEFAULT,
    explainability_change_rationale_context: dict[str, object] | None | object = (
        _DEFAULT
    ),
    governor_context: dict[str, object] | None | object = _DEFAULT,
    dispatch_context: dict[str, object] | None | object = _DEFAULT,
) -> dict[str, object]:
    return assemble_chatgpt_ready_advisory_artifact(
        structured_decision_output=(
            _sdo()
            if structured_decision_output is _DEFAULT
            else structured_decision_output
        ),
        generated_at=GENERATED_AT,
        chatgpt_advisory_context=(
            _advisory_context()
            if chatgpt_advisory_context is _DEFAULT
            else chatgpt_advisory_context
        ),
        portfolio_intelligence_context=(
            _portfolio_context()
            if portfolio_intelligence_context is _DEFAULT
            else portfolio_intelligence_context
        ),
        explainability_change_rationale_context=(
            _explainability_context()
            if explainability_change_rationale_context is _DEFAULT
            else explainability_change_rationale_context
        ),
        governor_context=(
            _governor_context()
            if governor_context is _DEFAULT
            else governor_context
        ),
        dispatch_context=(
            _dispatch_context()
            if dispatch_context is _DEFAULT
            else dispatch_context
        ),
    )


def _sdo(
    *,
    run_id: str = "run-001",
    ticker: str = "NVDA",
    data_coverage: dict[str, object] | None = None,
    decision: dict[str, object] | None = None,
) -> dict[str, object]:
    coverage = {
        "coverage_status": "ready",
        "coverage_score": 90,
        "freshness_status": "fresh",
        "missing_families": [],
        "stale_families": [],
        "blocked_reason": None,
    }
    coverage.update(data_coverage or {})
    decision_payload = {
        "action": "add_candidate",
        "action_strength": "medium",
        "time_horizon": "swing",
        "is_actionable": True,
        "actionability_blockers": [],
        "review_required": True,
    }
    decision_payload.update(decision or {})
    return {
        "schema_version": "structured-decision-output-v1",
        "artifact_type": "market-engine-structured-decision-output",
        "generated_at": "2026-07-08T07:00:00Z",
        "run_id": run_id,
        "ticker": ticker,
        "instrument": {
            "ticker": ticker,
            "name": "NVIDIA Corporation",
            "asset_type": "equity",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        "data_coverage": coverage,
        "decision": decision_payload,
        "scores": {
            "confidence": {
                "value": 82,
                "scale": "0_100",
                "status": "available",
                "reason_codes": [],
            }
        },
        "portfolio_context": {
            "position_status": "unknown",
            "current_weight": None,
            "target_weight": None,
            "max_weight": None,
            "exposure_flags": [],
            "concentration_risk": "unknown",
            "cash_dependency": "not_available",
            "position_sizing_available": False,
        },
        "risk": {},
        "levels": {},
        "thesis": {},
        "evidence": {"artifact_refs": ["artifact:sdo:NVDA:run-001"]},
        "explainability": {
            "primary_reason_codes": [],
            "blocking_reasons": [],
            "human_summary_allowed": True,
        },
        "consumer_guidance": {},
        "validation": {
            "contract_status": "valid",
            "required_fields_present": True,
            "semantic_warnings": [],
            "fail_closed_reason": None,
        },
    }


def _advisory_context(run_id: str = "run-001", ticker: str = "NVDA") -> dict[str, object]:
    return {
        "schema_version": "chatgpt-advisory-context-v1",
        "artifact_type": "market-engine-chatgpt-advisory-context",
        "generated_at": "2026-07-08T07:10:00Z",
        "run_id": run_id,
        "ticker": ticker,
        "instrument": {"ticker": ticker, "asset_type": "equity"},
        "source_artifact_refs": ["artifact:sdo:NVDA:run-001"],
        "advisory_eligibility": {
            "state": "eligible",
            "reason_codes": ["structured_decision_output_valid"],
            "allowed_scope": ["explain_decision_state"],
            "required_disclosures": ["human_review_required"],
            "blocking_reasons": [],
        },
        "freshness_context": {
            "global_freshness_status": "fresh",
            "family_freshness": [],
            "stale_markers": [],
            "stale_reasons": [],
            "unknown_freshness": [],
        },
        "uncertainty_context": {
            "confidence": 82,
            "uncertainty_level": "medium",
            "missing_evidence": [],
            "limitations": ["human_review_required"],
        },
    }


def _portfolio_context(
    ticker: str = "NVDA",
    *,
    include_target_holding: bool = True,
) -> dict[str, object]:
    holdings = []
    if include_target_holding:
        holdings.append(
            {
                "ticker": ticker,
                "position_state": "held",
                "quantity": 2,
                "market_value": 1000,
                "portfolio_weight_pct": 5.0,
                "provenance": {"artifact_ref": "portfolio-context:run-001"},
            }
        )
    return {
        "schema_version": "chatgpt-portfolio-intelligence-context-v1",
        "artifact_type": "market-engine-chatgpt-portfolio-intelligence-context",
        "generated_at": "2026-07-08T07:15:00Z",
        "run_id": "portfolio-run-001",
        "portfolio_identity": {"portfolio_id": "synthetic"},
        "portfolio_snapshot_identity": {"snapshot_id": "snapshot-001"},
        "source_artifact_refs": ["artifact:portfolio:run-001"],
        "availability": {"state": "available", "reason_codes": []},
        "holdings": holdings,
        "cash_context": {
            "state": "not_provided",
            "amount": None,
            "currency": None,
            "deployable_cash_state": "unknown",
        },
        "recommendation_to_position_relationship": {"ticker": ticker},
        "freshness": {"portfolio_review_freshness": "fresh"},
    }


def _explainability_context(availability_state: str = "available") -> dict[str, object]:
    return {
        "schema_version": "chatgpt-explainability-change-rationale-context-v1",
        "artifact_type": "market-engine-chatgpt-explainability-change-rationale-context",
        "generated_at": "2026-07-08T07:20:00Z",
        "run_id": "explainability-run-001",
        "instrument": {"ticker": "NVDA", "asset_type": "equity"},
        "current_run_identity": {"run_id": "run-001"},
        "reference_run_identity": None,
        "comparison_window": {"mode": "current_state_only"},
        "source_artifact_refs": ["artifact:explainability:run-001"],
        "availability": {"state": availability_state, "reason_codes": []},
        "current_state_rationale": {"current_state": "eligible"},
        "validation": {"contract_valid": True, "blocked_reasons": []},
    }


def _governor_context(ticker: str = "NVDA") -> dict[str, object]:
    return {
        "schema_version": "market-engine-governor-context-v1",
        "artifact_type": "market-engine-governor-context",
        "run_id": "run-001",
        "ticker": ticker,
        "state": "evaluation_completed_non_actionable",
        "blockers": [],
        "freshness_context": {"global_freshness_status": "fresh"},
    }


def _dispatch_context(ticker: str = "NVDA") -> dict[str, object]:
    return {
        "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
        "subject": {"ticker": ticker},
        "sections": [],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
