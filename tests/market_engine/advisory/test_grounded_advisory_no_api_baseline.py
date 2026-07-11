from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.advisory import grounded_advisory_runtime as runtime
from market_engine.advisory.grounded_advisory_output import generate_grounded_advisory_output


def test_default_grounded_advisory_path_blocks_provider_invocation_even_with_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-that-must-not-be-used")
    monkeypatch.setenv("MARKET_ENGINE_ADVISORY_MODEL", "gpt-4.1-mini")

    def fail_if_network_is_called(*args, **kwargs):
        raise AssertionError("baseline command path must not call provider/network")

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fail_if_network_is_called)

    result = generate_grounded_advisory_output(
        source_artifact_path=_write_source(tmp_path),
        output_root=tmp_path / "outputs",
        run_id="no-api-baseline",
        generated_at="2026-07-11T12:00:00Z",
    )

    structured = _read(result.structured_output_path)
    raw = _read(result.raw_response_path)

    assert structured["invocation_boundary"]["invocation_state"] == "request_blocked"
    assert structured["advisory_status"] == "blocked_invocation_not_configured"
    assert structured["validation_result"]["status"] == "invalid"
    assert "Provider invocation is disabled by default" in raw["error_message"]
    assert raw["provider_name"] == "not_configured"
    assert raw["raw_provider_response"] is None


def _write_source(tmp_path: Path) -> Path:
    path = tmp_path / "dry_run.json"
    path.write_text(json.dumps(_source_payload()), encoding="utf-8")
    return path


def _source_payload() -> dict[str, object]:
    return {
        "artifact_format_version": "market-engine-local-dry-run-artifact-v1",
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": "2026-07-02T11:56:52Z",
        "payload": {
            "dry_run_id": "run-nvda",
            "ticker": "NVDA",
            "generated_at": "2026-07-02T11:56:52Z",
            "input_mode": "cached_source_snapshot",
            "blocked_stage": "portfolio_review",
            "blocked_reasons": ["Stage preserves an upstream blocked state."],
            "missing_data_summary": ["portfolio_context"],
            "analysis_context_readiness": {
                "readiness_level": "partial_analysis",
                "actionable_review_allowed": False,
                "decision_engine_ready": False,
                "context_stale": False,
                "blocked_reasons": ["missing_setup_or_price_context"],
                "evidence_families_missing": ["setup_price_market"],
            },
            "provenance_summary": {
                "fundamental_observations": {
                    "source_refresh_snapshot_id": "NVDA_companyfacts"
                },
                "portfolio_review": {
                    "recommendation_review_provenance": {
                        "review_state": "human_review_required",
                        "review_category": "analysis_mixed_or_conflicted",
                        "input_provenance": {"ticker": "NVDA"},
                    }
                },
            },
        },
    }


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
