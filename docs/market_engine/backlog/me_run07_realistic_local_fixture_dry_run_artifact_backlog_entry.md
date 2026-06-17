# ME-RUN07 - Realistic local fixture dry-run artifact backlog entry

## Status

COMPLETED BY ME-RUN07

## Owner roles

Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

## Job family

ME-RUN - Run / orchestration jobs

## Goal

Demonstrate that the Market Engine end-to-end dry-run can run locally from a realistic non-production `local_snapshot_fixture` and persist a deterministic review artifact through the existing RUN05 artifact flag.

## Scope

ME-RUN07 authorizes:

* one realistic non-production local snapshot fixture;
* execution through the existing `local_snapshot_fixture` input mode;
* explicit artifact writing only through `--write-local-artifact`;
* tests for fixture acceptance, dry-run execution, deterministic artifact inspectability, numeric-zero preservation, missing/stale markers, blocked state, blocked reasons, and provenance preservation;
* local run documentation;
* sprint audit documentation.

## Implemented fixture

```text
tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json
```

## Implemented tests

```text
tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py
```

## Implemented documentation

```text
docs/market_engine/run/me_run07_realistic_local_fixture_dry_run_artifact_execution.md
docs/market_engine/audits/me_run07_realistic_local_fixture_dry_run_artifact_audit.md
```

## Existing runtime used

```text
src/market_engine/run/local_dry_run_inputs.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/local_dry_run_artifacts.py
```

ME-RUN07 does not introduce new runtime modules or new Market Engine runtime contracts.

## Outcome

ME-RUN07 adds a realistic local fixture for `MSFT` / CIK `0000789019` and proves the existing local dry-run command path can consume the fixture wrapper, construct a `market-engine-end-to-end-dry-run-v1` payload, and persist a `market-engine-local-dry-run-artifact-v1` artifact plus `market-engine-local-dry-run-artifact-manifest-v1` manifest when the explicit artifact flag is present.

The fixture intentionally preserves:

* numeric-zero evidence in Portfolio Review context;
* missing-data markers;
* stale-data markers;
* Delivery / Reporting blocked state;
* blocked reason;
* stage provenance and run IDs;
* side-effect boundary confirmation;
* action-authority boundary confirmation.

## Boundaries preserved

ME-RUN07 does not introduce provider calls, SEC/EDGAR calls, yfinance/live market data calls, broker calls, Telegram delivery, email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, new financial analysis logic, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target weights, target prices, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Next logical step preservation

No mandatory next sprint is inserted by ME-RUN07.

Possible future candidate, not active unless explicitly approved:

```text
ME-RUN08 - Expand local fixture matrix coverage for multiple dry-run states
```

Reason: ME-RUN07 demonstrates one realistic local blocked-review fixture. A future RUN sprint may broaden local fixture coverage across completed, completed-with-limitations, blocked, unsupported-input, and contract-violation states before any broader all-ticker or channel-adapter workflow is approved.
