from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REFRESH_WORKFLOW = REPO_ROOT / ".github/workflows/canonical-price-refresh.yml"
ANALYSIS_WORKFLOW = REPO_ROOT / ".github/workflows/daily-market-scan.yml"


def _publish_job_block(workflow: str) -> str:
    marker = "  publish-market-data:\n"
    return workflow[workflow.index(marker) :]


def _publish_job_gate(workflow: str) -> str:
    publish_job = _publish_job_block(workflow)
    start = publish_job.index("    if: >-\n")
    end = publish_job.index("    runs-on:", start)
    return publish_job[start:end]


def _publication_gate_allows(
    *,
    job_result: str,
    run_status: str | None,
    trusted_publish: bool = True,
    publication_required: bool = True,
    publication_set_valid: bool = True,
) -> bool:
    return (
        job_result == "success"
        and run_status == "completed"
        and trusted_publish
        and publication_required
        and publication_set_valid
    )


def test_refresh_workflow_has_daily_and_manual_triggers_with_concurrency() -> None:
    workflow = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert 'cron: "30 5 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "timeout-minutes: 60" in workflow
    assert "timeout-minutes: 15" in workflow
    assert 'run_id="me-sr18-canonical-price-refresh-' in workflow


def test_publication_is_privileged_separately_and_only_trusted_main_can_publish() -> None:
    workflow = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert workflow.count("contents: write") == 1
    assert "contents: read" in workflow
    assert 'sys.argv[3] == "refs/heads/main"' in workflow
    assert 'default: false' in workflow
    assert "trusted-publish == 'true'" in workflow
    assert "ref: main" in workflow
    assert "--force" not in workflow
    assert "push --force" not in workflow
    assert "HEAD:main" not in workflow


def test_market_data_branch_is_data_only_and_bootstraps_without_main_history() -> None:
    workflow = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    publish_job = _publish_job_block(workflow)
    assert "switch --orphan market-data" in workflow
    assert "rsync --archive --delete publication-bundle/manifests/ publication/manifests/" in workflow
    assert "add data/processed manifests/canonical_price_freshness_latest.json" in workflow
    assert "push origin HEAD:market-data" in workflow
    assert "No validated data change; no market-data commit created." in workflow
    assert publish_job.count("validate-publication") == 2
    assert "--publication-root publication" in publish_job
    assert "--allow-degraded" not in publish_job


def test_publication_job_requires_successful_completed_refresh() -> None:
    workflow = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    gate = _publish_job_gate(workflow)

    assert "needs.refresh-and-validate.result == 'success'" in gate
    assert (
        "needs.refresh-and-validate.outputs.run-status == 'completed'"
        in gate
    )
    assert (
        "needs.refresh-and-validate.outputs.publication-set-valid == 'true'"
        in gate
    )
    assert (
        "needs.refresh-and-validate.outputs.publication-required == 'true'"
        in gate
    )
    assert "always()" not in gate


def test_publication_job_gate_is_fail_closed_for_every_noncompleted_state() -> None:
    blocked = [
        {"job_result": "failure", "run_status": "completed"},
        {"job_result": "cancelled", "run_status": "completed"},
        {"job_result": "skipped", "run_status": "completed"},
        {"job_result": "success", "run_status": "degraded"},
        {"job_result": "success", "run_status": "failed"},
        {"job_result": "success", "run_status": "unknown"},
        {"job_result": "success", "run_status": ""},
        {"job_result": "success", "run_status": None},
        {
            "job_result": "success",
            "run_status": "completed",
            "trusted_publish": False,
        },
    ]

    assert all(not _publication_gate_allows(**case) for case in blocked)
    assert _publication_gate_allows(
        job_result="success",
        run_status="completed",
    )


def test_freshness_evidence_uploads_even_when_refresh_fails() -> None:
    workflow = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error: true" in workflow
    assert "name: Upload freshness evidence\n        if: always()" in workflow
    assert "Mark degraded or failed refresh visibly" in workflow


def test_daily_analysis_materializes_and_validates_market_data_before_consumption() -> None:
    workflow = ANALYSIS_WORKFLOW.read_text(encoding="utf-8")
    materialize = workflow.index("Materialize published market data")
    consume = workflow.index("consume-analysis")
    dry_run = workflow.index("market_scanner.app --dry-run")
    assert materialize < consume < dry_run
    assert "ref: market-data" in workflow
    assert "workflow_run:" in workflow
    assert 'workflows: ["Canonical Price Refresh"]' in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "me-sr18-validated-daily-analysis-" in workflow
    assert "git add data/" not in workflow


def test_daily_analysis_gate_rejects_red_upstream_and_accepts_success() -> None:
    workflow = ANALYSIS_WORKFLOW.read_text(encoding="utf-8")
    gate = (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.workflow_run.conclusion == 'success'"
    )

    assert gate in workflow
    assert "conclusion == 'failure'" not in workflow
    assert "conclusion != 'cancelled'" not in workflow


def test_workflows_never_execute_code_from_market_data_branch() -> None:
    refresh = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    analysis = ANALYSIS_WORKFLOW.read_text(encoding="utf-8")
    assert "PYTHONPATH=published-market-data" not in analysis
    assert "published-market-data/src" not in analysis
    assert "publication/src" not in refresh
    assert "publication-bundle/src" not in refresh
