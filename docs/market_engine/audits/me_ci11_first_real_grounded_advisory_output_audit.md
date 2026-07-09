# ME-CI11 - First Real Grounded Advisory Output Audit

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: IMPLEMENTED WITH REAL INVOCATION BLOCKED BY LOCAL CONFIGURATION

## Objective

ME-CI11 implemented the first production-shaped local path that turns one existing Market Engine ticker artifact into a grounded advisory output artifact set and a readable `advisory_report.md`.

The sprint also executed the path against a real cached-source Market Engine artifact for NVDA. The controlled model invocation did not reach the external provider because the local environment did not define `OPENAI_API_KEY` and did not define `MARKET_ENGINE_ADVISORY_MODEL` or `OPENAI_MODEL`. The run failed closed and persisted local evidence of the blocker.

## Source main SHA

```text
81824cacef79d9a1471d93de2e190daffcd85c96
```

## Selected real source artifact

```text
artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json
```

The selected artifact is a real Market Engine dry-run artifact generated from cached source snapshot input for ticker `NVDA`.

Source properties observed by the ME-CI11 flow:

* artifact format: `market-engine-local-dry-run-artifact-v1`
* artifact type: `market_engine_end_to_end_dry_run`
* ticker: `NVDA`
* input mode: `cached_source_snapshot`
* readiness level: `partial_analysis`
* actionable review allowed: `false`
* Decision Engine ready: `false`
* blocked stage: `portfolio_review`
* missing data: `portfolio_context`, `setup_price_market`

## Implemented runtime

ME-CI11 added:

```text
src/market_engine/advisory/grounded_advisory_output.py
src/market_engine/advisory/grounded_advisory_output_command.py
```

The runtime performs:

```text
Market Engine dry-run artifact
  -> source artifact validation
  -> deterministic grounded advisory input package
  -> CI10-shaped invocation request
  -> controlled non-streaming OpenAI Responses boundary
  -> raw response capture
  -> strict JSON parser
  -> allowed-evidence grounding validation
  -> structured output artifact
  -> readable advisory_report.md
  -> local manifest
```

The OpenAI boundary is single-provider, non-streaming, tool-free, browsing-free, local-output-only, and fail-closed. It uses explicit environment configuration and does not persist secrets.

## Invocation configuration boundary

The runtime requires:

```text
OPENAI_API_KEY
MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL
```

`OPENAI_BASE_URL` is optional and defaults to the OpenAI Responses API base URL.

If required configuration is absent, the runtime does not attempt a network call. It writes a blocked invocation result and returns a non-zero command exit code.

## Real run attempt

Command shape:

```text
python -m market_engine.advisory.grounded_advisory_output_command \
  --artifact artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json \
  --output-root artifacts/market_engine/grounded_advisory_outputs \
  --run-id me-ci11-nvda-first-grounded-advisory-20260709T120000Z \
  --generated-at 2026-07-09T12:00:00Z
```

Observed result:

```text
advisory_status: blocked_invocation_not_configured
invocation_state: request_blocked
validation_status: invalid
ticker: NVDA
```

The blocker was:

```text
OPENAI_API_KEY and MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL are required for real invocation.
```

## Persisted local run artifacts

The blocked real run persisted:

```text
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/invocation_request.json
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/raw_model_response.json
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/parser_result.json
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/validation_result.json
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/grounded_advisory_output.json
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/advisory_report.md
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/manifest.json
```

The report is readable without opening JSON internals, but it correctly states that no grounded advisory conclusion was generated because invocation was blocked.

## Grounding behavior

The model response validator accepts only evidence references present in the generated `allowed_evidence_references` handoff. Unknown references fail validation.

For the selected RUN28 dry-run artifact, the existing ME-CI09 validator is not invoked directly because it is coupled to the earlier CI05/CI07 advisory artifact and response-fixture shape. ME-CI11 therefore implements the equivalent required fail-closed grounding boundary for the selected dry-run artifact path while preserving CI10 lineage and explicit evidence-reference containment.

## Fail-closed behavior

The implementation fails closed for:

* unsupported source artifact format or type;
* missing source payload or readiness context;
* missing model invocation configuration;
* provider failures;
* empty model output;
* malformed model JSON;
* schema mismatch;
* missing required response fields;
* unknown evidence references;
* actionable language when the source artifact is not actionable or Decision Engine ready.

## Governance review

ME-CI11 does not add:

* production execution;
* live provider refresh;
* model browsing;
* model tool use;
* delivery;
* Telegram or notification behavior;
* broker integration;
* portfolio writes;
* watchlist writes;
* allocation;
* sizing;
* execution;
* Decision Engine semantic changes.

The first NVDA run preserves `partial_analysis`, `actionable_review_allowed=false`, `decision_engine_ready=false`, and the `portfolio_review` blocked state.

## Tests

ME-CI11 added:

```text
tests/market_engine/advisory/test_grounded_advisory_output.py
```

The test coverage includes:

* successful artifact/report generation using a fake invoker;
* evidence-reference containment in model input;
* rejection of actionable model language for blocked/non-actionable sources;
* stale and missing data preservation;
* malformed model output fail-closed behavior;
* unsupported source artifact fail-closed behavior;
* source and invocation traceability.

## Completion assessment

ME-CI11 completes the local implementation required to produce grounded advisory artifacts and a readable report from a real Market Engine artifact. It also proves that the command fails closed and persists audit evidence when real invocation cannot be configured.

ME-CI11 does not complete the hard real-model-output milestone because no real provider response was obtained in this environment. The next sprint must execute the same path with approved model credentials and model configuration.

## Recommended next sprint

```text
ME-CI11B - Execute configured real grounded advisory model invocation
```

ME-CI11B should use the ME-CI11 runtime without broadening scope. It should provide approved non-production model configuration, execute one real provider call, persist the raw response, parse and validate the grounded response, and produce the first successful grounded `advisory_report.md`.
