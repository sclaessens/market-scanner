# ME-RUN07 - Realistic local fixture dry-run artifact audit

## Status

COMPLETED BY ME-RUN07

## Sprint audited

ME-RUN07 - Execute realistic local fixture dry-run and persist review artifact

## Audit scope

Audited scope:

```text
tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json
tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py
docs/market_engine/run/me_run07_realistic_local_fixture_dry_run_artifact_execution.md
```

The audit checks that ME-RUN07 uses existing Market Engine local dry-run contracts and preserves all side-effect and action-authority boundaries.

## Fixture audit

PASS.

The fixture is stored under a clear non-production test fixture path:

```text
tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json
```

The wrapper uses:

```text
market-engine-local-dry-run-input-fixture-v1
```

The wrapper includes:

* `input_mode: local_snapshot_fixture`;
* `non_production_fixture: true`;
* `stage_payloads` containing only existing approved dry-run stage payloads.

The fixture is realistic enough for local review because it includes a named ticker, CIK, stage run IDs, source-context provenance, financial observation values, derived cash-generation observations, setup/review limitations, portfolio-context zero evidence, Decision Engine handoff references, and Delivery / Reporting blocked state.

## Contract audit

PASS.

ME-RUN07 does not define new runtime contracts.

It uses:

```text
market-engine-local-dry-run-input-fixture-v1
market-engine-end-to-end-dry-run-v1
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

All stage payloads preserve the existing expected format-version fields consumed by the dry-run inspector.

## Dry-run path audit

PASS.

The fixture is executed through the existing command path:

```text
market_engine.run.end_to_end_dry_run_command
```

Input mode is explicitly:

```text
local_snapshot_fixture
```

The fixture path is supplied through:

```text
--stage-payloads-json
```

Local artifact persistence is requested only through:

```text
--write-local-artifact
```

Direct local `python -m market_engine...` execution from the repository root requires `PYTHONPATH=src` unless the package is otherwise installed into the active virtual environment.

## Artifact audit

PASS.

The tests verify that no local artifact directory is written when `--write-local-artifact` is absent.

The tests verify that, when `--write-local-artifact` is present, the command writes:

```text
manifest.json
artifacts/market_engine_dry_run_<dry-run-id>_2026-06-17.json
```

The tests inspect both files and verify:

* manifest format version;
* artifact format version;
* non-production artifact marker;
* source dry-run identity;
* source input mode;
* source blocked run state;
* preserved blocked stage;
* preserved blocked reason;
* preserved missing-data markers;
* preserved stale-data markers;
* preserved numeric-zero evidence;
* preserved provenance.

Local manual execution also produced the expected files:

```text
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/artifacts/market_engine_dry_run_me-run07-realistic-local-fixture-artifact_2026-06-17.json
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/manifest.json
```

The observed manifest confirms:

```text
artifact_count: 1
artifact_persistence_mode: local_dry_run_only
non_production_artifact: true
source_input_mode: local_snapshot_fixture
source_run_state: dry_run_blocked
```

## Numeric-zero audit

PASS.

The fixture intentionally includes numeric-zero portfolio context values:

```text
portfolio_review.portfolio_context_reference.current_quantity = 0
portfolio_review.portfolio_context_reference.current_market_value = 0.0
portfolio_review.portfolio_context_reference.cash_available_for_review = 0
```

The tests assert that these values remain present in `numeric_zero_evidence_summary` and in the persisted artifact payload.

## Missing, stale, blocked, and provenance audit

PASS.

The fixture intentionally preserves:

Missing-data markers:

```text
fundamental_observations.segment_revenue_breakdown
setup_detection.forward_guidance_not_in_fixture
```

Stale-data markers:

```text
sec_companyfacts.snapshot_age.review_required
analysis_review.market_price_snapshot_absent
delivery_reporting.local_fixture_review_timestamp
```

Blocked reason:

```text
ME-RUN07 fixture intentionally blocks before any delivery channel.
```

Provenance references:

```text
source_refresh_snapshot_id
observation_run_id
derived_observation_run_id
setup_detection_run_id
analysis_review_run_id
recommendation_review_run_id
portfolio_review_run_id
handoff_run_id
report_id
```

## Side-effect boundary audit

PASS.

No ME-RUN07 file introduces provider, SEC/EDGAR, yfinance, live market data, broker, Telegram, email, scheduler, UI, portfolio-write, watchlist-write, or production-report behavior.

ME-RUN07 remains local, deterministic, non-production, and inspectable.

## Authority boundary audit

PASS.

ME-RUN07 does not introduce:

* BUY / SELL / HOLD action semantics;
* allocation advice;
* target weights;
* target prices;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

Decision Engine remains the only future action/allocation authority. ME-RUN07 emits integration-review evidence only.

## Validation command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run/test_local_dry_run_inputs.py tests/market_engine/run/test_end_to_end_dry_run_command.py tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py -q
```

Observed local pytest result:

```text
18 passed in 0.04s
```

## Manual dry-run command

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode local_snapshot_fixture --stage-payloads-json tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json --dry-run-id me-run07-realistic-local-fixture-artifact --generated-at 2026-06-17T14:15:00Z --write-local-artifact --artifact-output-root artifacts/market_engine/dry_runs --artifact-created-at 2026-06-17T14:30:00Z
```

Observed local dry-run result:

```text
input_mode: local_snapshot_fixture
run_state: dry_run_blocked
blocked_stage: delivery_reporting
ticker: MSFT
provider_name: sec_companyfacts
```

## Execution-environment limitation

This ChatGPT session could not execute local pytest or the dry-run command from a repository checkout because cloning GitHub from the execution environment failed with DNS/network resolution errors. The sprint changes were created through the GitHub connector, then validated from the user's local checkout.

## Conclusion

ME-RUN07 adds a realistic non-production local snapshot fixture, tests the existing local fixture dry-run path and explicit local artifact persistence, and documents the exact operator command. The implementation preserves numeric-zero evidence, missing-data markers, stale-data markers, blocked state, blocked reasons, provenance, side-effect boundaries, and action-authority boundaries.
