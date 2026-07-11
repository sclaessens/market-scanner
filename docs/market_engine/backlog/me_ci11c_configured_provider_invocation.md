# ME-CI11C - Configured Provider Invocation

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: BLOCKED BY CODEX COMMAND PROCESS ENVIRONMENT

## Goal

Run the corrected universal CI11 grounded advisory runtime with local API-key configuration and record the first real provider invocation outcome.

## Execution result

ME-CI11C started from `main` SHA `52c0b3729323dec6bfb7c4f45b8c836b715da661` on branch `me-ci11c-run-configured-provider-invocation`.

The Codex command process did not contain a non-empty `OPENAI_API_KEY`. `MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini` was supplied explicitly and appears in the persisted invocation requests, but the runtime correctly blocked before provider invocation because credentials were absent in the process environment.

The existing universal CI11 command path was run against:

* NVDA primary artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json`;
* AMD smoke artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/AMD/dry_run.json`.

Persisted output:

* `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-nvda-real-provider-grounded-advisory-20260711T000000Z/NVDA/`;
* `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-amd-real-provider-grounded-advisory-smoke-20260711T000000Z/AMD/`.

Observed status for both tickers:

```text
advisory_status = blocked_invocation_not_configured
invocation_state = request_blocked
parser_result = invalid / empty_response
validation_status = invalid
grounding_status = null
issues = invocation_not_completed, missing_parsed_response
```

ME-CI11C therefore did not produce a provider response, parsed provider output, CI09 grounding result, or successful grounded `advisory_report.md`.

## Exact blocker

`OPENAI_API_KEY` was not visible as a non-empty value in the Codex command process.

This is a process-environment propagation blocker, not a source artifact blocker, provider API blocker, schema blocker, CI09 blocker, ticker-specific runtime blocker, or Decision Engine issue.

## Required follow-up

The next sprint must resolve only the concrete process-environment blocker:

```text
ME-CI11D - Fix Codex command process provider environment propagation
```

ME-CI11D should first verify that the exact command process reports:

```text
OPENAI_API_KEY nonempty: True
MARKET_ENGINE_ADVISORY_MODEL: gpt-4.1-mini
```

After that verification, it should rerun the same universal CI11 command path for NVDA and AMD. It must not add ticker-specific code, prompt tuning, provider fallback, browsing, source refresh, delivery, broker behavior, portfolio mutation, watchlist mutation, allocation, sizing, or Decision Engine semantic changes.
