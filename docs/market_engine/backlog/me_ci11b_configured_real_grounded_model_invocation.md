# ME-CI11B - Configured Real Grounded Model Invocation

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: BLOCKED BY LOCAL PROVIDER CONFIGURATION

## Goal

Run the ME-CI11 grounded advisory output runtime with approved non-production model configuration and produce the first successful grounded provider response and readable advisory report.

## ME-CI11B execution result

ME-CI11B was executed from `main` SHA `d6195f3fe3c746af91ae29360f80b2dcb1cdaa64` on branch `me-ci11b-first-configured-real-grounded-advisory-output`.

The existing universal CI11 command path was run against:

* NVDA primary artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json`;
* AMD smoke artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/AMD/dry_run.json`.

Both runs failed closed before provider invocation because the local environment did not contain `OPENAI_API_KEY`. `MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini` was supplied for the command attempts, but the runtime correctly reported the invocation boundary as not configured while credentials were absent.

Persisted output:

* `artifacts/market_engine/grounded_advisory_outputs/me-ci11b-nvda-configured-grounded-advisory-blocked-missing-api-key-20260711T000000Z/NVDA/`;
* `artifacts/market_engine/grounded_advisory_outputs/me-ci11b-amd-configured-grounded-advisory-smoke-blocked-missing-api-key-20260711T000000Z/AMD/`.

Observed status for both tickers:

```text
advisory_status = blocked_invocation_not_configured
invocation_state = request_blocked
parser_result = invalid / empty_response
validation_status = invalid
grounding_status = null
issues = invocation_not_completed, missing_parsed_response
```

ME-CI11B therefore did not produce a real provider response or successful grounded `advisory_report.md`. The exact blocker is missing local `OPENAI_API_KEY`.

## Insertion reason

ME-CI11 implemented the local flow and persisted blocked real-run evidence, but the environment did not define the required model invocation configuration:

```text
OPENAI_API_KEY
MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL
```

ME-CI11B confirmed the same concrete blocker with the corrected CI11 runtime and an additional AMD smoke run. The first successful provider response must therefore still be completed before ME-CI12 report-quality or ME-CI12 batch work.

## Scope

ME-CI11B may:

* provide approved local non-production model configuration outside committed repository content;
* run the ME-CI11 command against one real Market Engine artifact;
* persist invocation request, raw response, parser result, validation result, structured output, report, and manifest;
* verify evidence-reference grounding containment;
* document the successful real-run evidence.

## Required follow-up

The next sprint must resolve only the concrete provider-configuration blocker:

```text
ME-CI11C - Run configured provider invocation with local API key
```

ME-CI11C should rerun the same universal CI11 command path with `OPENAI_API_KEY` present outside repository content and `MARKET_ENGINE_ADVISORY_MODEL` explicitly set. It should use NVDA as the primary ticker and AMD as the smoke ticker unless the source artifacts are no longer available. It must not add ticker-specific code, prompt tuning, provider fallback, browsing, source refresh, delivery, broker behavior, portfolio mutation, watchlist mutation, allocation, sizing, or Decision Engine semantic changes.

## Non-goals

ME-CI11B must not add production execution, provider refresh, multi-provider routing, fallback models, streaming, tool use, browsing, delivery, broker integration, portfolio writes, watchlist writes, allocation, sizing, execution, or Decision Engine semantic changes.
