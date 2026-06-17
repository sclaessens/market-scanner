# ME-RUN07 - Local validation follow-up note

## Status

FOLLOW-UP TO MERGED ME-RUN07

## Reason

PR #353 was merged before local validation output confirmed that direct module execution from the repository root requires `PYTHONPATH=src` in the current source-layout setup.

The two documentation updates on this branch correct the operator command and record the observed local validation evidence.

## Local pytest evidence

```text
18 passed in 0.04s
```

## Local dry-run evidence

Observed dry-run identity:

```text
input_mode: local_snapshot_fixture
run_state: dry_run_blocked
blocked_stage: delivery_reporting
ticker: MSFT
provider_name: sec_companyfacts
```

Observed artifact files:

```text
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/artifacts/market_engine_dry_run_me-run07-realistic-local-fixture-artifact_2026-06-17.json
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/manifest.json
```

Observed manifest fields:

```text
artifact_count: 1
artifact_persistence_mode: local_dry_run_only
non_production_artifact: true
source_input_mode: local_snapshot_fixture
source_run_state: dry_run_blocked
```

## Correct manual dry-run command

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode local_snapshot_fixture --stage-payloads-json tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json --dry-run-id me-run07-realistic-local-fixture-artifact --generated-at 2026-06-17T14:15:00Z --write-local-artifact --artifact-output-root artifacts/market_engine/dry_runs --artifact-created-at 2026-06-17T14:30:00Z
```

## Boundary confirmation

This follow-up is docs/audit only. It does not change runtime code, fixture data, tests, provider behavior, delivery behavior, portfolio/watchlist writes, or production artifacts.
