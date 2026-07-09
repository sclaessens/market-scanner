# ME-CI11 - First Real Grounded Advisory Output Audit

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: IMPLEMENTED WITH REAL INVOCATION BLOCKED BY LOCAL CONFIGURATION; PR REVIEW REMEDIATION APPLIED

## Objective

ME-CI11 implements a production-shaped local path that turns one existing Market Engine ticker artifact into a controlled advisory artifact set and a readable `advisory_report.md`.

The sprint also executed the path against a real cached-source Market Engine artifact for NVDA. The controlled provider invocation did not occur because the local environment did not define `OPENAI_API_KEY` and did not define `MARKET_ENGINE_ADVISORY_MODEL` or `OPENAI_MODEL`. The run failed closed and persisted local evidence of the blocker.

A subsequent PR review identified that the first CI11 implementation used a parallel evidence-reference containment validator instead of the established ME-CI09 grounding validator. That review finding was accepted as a merge blocker and remediated on the PR branch before merge.

## Source main SHA

```text
81824cacef79d9a1471d93de2e190daffcd85c96
```

## Selected real source artifact

```text
artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json
```

Observed source state:

- ticker: `NVDA`;
- input mode: `cached_source_snapshot`;
- readiness: `partial_analysis`;
- actionable review allowed: `false`;
- Decision Engine ready: `false`;
- blocked stage: `portfolio_review`;
- missing data: `portfolio_context`, `setup_price_market`.

## Runtime architecture after PR review remediation

ME-CI11 now separates compatibility exports, orchestration, runtime helpers and the command entry point:

```text
src/market_engine/advisory/grounded_advisory_output.py
src/market_engine/advisory/grounded_advisory_orchestration.py
src/market_engine/advisory/grounded_advisory_runtime.py
src/market_engine/advisory/grounded_advisory_output_command.py
```

The accepted path is:

```text
Market Engine dry-run artifact
  -> source validation
  -> deterministic evidence catalog with exact source paths
  -> deterministic CI10 pre-invocation request validation
  -> bounded, non-streaming, tool-free, browsing-free provider request
  -> provider structured-output constraint
  -> raw provider response capture
  -> strict response schema parse
  -> structured claim/evidence-reference projection
  -> ME-CI09 validate_advisory_response_grounding(...)
  -> structured advisory output
  -> readable advisory_report.md
  -> local manifest
```

The PR review remediation removed the earlier claim that simple evidence-reference containment was equivalent to CI09 grounding validation. CI09 is now called directly in the accepted model-response path.

## Grounding behavior

The model response contract is structured around:

- explicit claim IDs;
- explicit claim roles;
- approved claim types;
- explicit claim-to-evidence references;
- approved support types;
- exact allowed evidence reference IDs;
- exact original Market Engine source paths;
- deterministic projection paths used by the CI09 validator;
- required disclosures;
- source readiness and actionability ceiling.

The local CI11 validator performs deterministic schema, type, enum, disclosure, claim-reference, allowed-reference and actionability-ceiling checks before the CI09 handoff.

The final grounding authority for accepted model output is:

```text
ME-CI09.validate_advisory_response_grounding
```

A response is not accepted merely because all evidence IDs exist in an allowlist.

## Evidence path remediation

The first implementation incorrectly merged evidence from different source fields while assigning the same source path. This was corrected.

Examples now remain distinct:

```text
$.payload.blocked_reasons[0]
$.payload.analysis_context_readiness.blocked_reasons[0]
$.payload.missing_data_summary[0]
$.payload.analysis_context_readiness.evidence_families_missing[0]
```

The generic recursive setup-message harvesting used by the first implementation was removed from the accepted evidence catalog. CI11 now exposes only explicitly mapped evidence families with deterministic source paths.

## CI10 invocation boundary remediation

The PR review remediation added deterministic checks for:

- approved question class;
- approved permitted use case;
- provider identity;
- model identity;
- strict response schema identity;
- bounded output tokens;
- bounded input size;
- tool use disabled;
- external browsing disabled;
- deterministic idempotency material.

The idempotency key now includes source hash, prompt-package hash, question class, permitted use case, provider, model, configuration profile, response schema and budget profile.

## Provider request shape

The OpenAI Responses request uses provider-side Structured Outputs with a strict JSON Schema and a bounded `max_output_tokens` value.

The provider boundary remains:

- single-provider;
- non-production;
- non-streaming;
- tool-free;
- browsing-free;
- bounded output;
- local persistence only.

## Raw response capture remediation

The persisted raw response artifact now preserves approved non-secret response evidence including:

- invocation state;
- provider name;
- model name;
- provider request ID;
- receive timestamp;
- finish or incomplete reason;
- usage metadata;
- raw provider response payload;
- extracted raw output;
- raw provider response hash;
- raw output hash.

Authentication headers and secrets are not persisted.

## Output path safety

The orchestration layer validates the output root, rejects parent traversal, resolves child paths under the approved root and keeps overwrite behavior explicit.

## Real run attempt

Command shape:

```text
python -m market_engine.advisory.grounded_advisory_output_command \
  --artifact artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json \
  --output-root artifacts/market_engine/grounded_advisory_outputs \
  --run-id me-ci11-nvda-first-grounded-advisory-20260709T120000Z \
  --generated-at 2026-07-09T12:00:00Z
```

Observed pre-remediation run result remains valid audit evidence for the missing local configuration blocker:

```text
advisory_status: blocked_invocation_not_configured
invocation_state: request_blocked
validation_status: invalid
```

Blocker:

```text
OPENAI_API_KEY and MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL are required for real invocation.
```

The persisted blocked run artifacts predate the PR review remediation and must not be interpreted as evidence of a successful provider response or successful CI09-grounded model output.

## Fail-closed behavior

The corrected flow fails closed for:

- unsupported source artifact format or type;
- missing source payload or readiness context;
- invalid invocation request;
- missing provider/model configuration;
- provider failure;
- provider refusal;
- incomplete or truncated response;
- empty response;
- malformed JSON;
- response schema mismatch;
- wrong primitive types;
- unknown fields;
- invalid status enums;
- unknown claim references;
- material claims without evidence references;
- unknown evidence references;
- missing mandatory disclosures;
- actionability ceiling violations;
- CI09 grounding failure.

## Tests

The CI11 test module now covers:

- successful fake-invoker flow through CI09 grounding;
- explicit CI09 validator identity in validation output;
- exact evidence catalog source and projection paths;
- actionability ceiling rejection;
- unknown evidence reference rejection;
- material claim without evidence rejection;
- strict primitive type rejection;
- stale and missing data preservation;
- malformed JSON fail-closed behavior;
- source/invocation/raw-provider traceability;
- model-sensitive idempotency keys;
- provider request structured-output constraint;
- output token budget.

## Governance review

ME-CI11 still does not add:

- production execution;
- live provider refresh;
- model browsing;
- model tool use;
- delivery;
- Telegram or notification behavior;
- broker integration;
- portfolio writes;
- watchlist writes;
- allocation;
- sizing;
- execution;
- Decision Engine semantic changes.

The source readiness ceiling remains authoritative. The selected NVDA source remains `partial_analysis`, non-actionable and blocked at `portfolio_review`.

## Completion assessment

ME-CI11 implements the controlled local path and the corrected CI09 grounding handoff, but it still does not complete the hard real-model-output milestone because no real provider response has been received in the configured environment.

The next active sprint remains:

```text
ME-CI11B - Execute configured real grounded advisory model invocation
```

ME-CI11B must use the corrected CI11 runtime without broadening scope. It must provide approved non-production model configuration, execute one real provider call, persist the provider response, parse the strict structured output, pass ME-CI09 grounding validation, and render the first successful grounded `advisory_report.md`.
