# ME-CI10 - Controlled Model Invocation Boundary Contract Audit

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED DOCS-FIRST CONTRACT

## Objective

ME-CI10 defines the formal contract for a future controlled model invocation
boundary between the existing local advisory pipeline and a future external
model provider.

## Architecture position

The contract sits after ME-CI09 and before any future implementation:

```text
ME-CI05 advisory artifact assembly
  -> ME-CI06 advisory artifact validation
  -> ME-CI07 prompt + response grounding contract
  -> ME-CI08 local synthetic dry-run scaffold
  -> ME-CI09 grounding validator hardening
  -> ME-CI10 controlled model invocation boundary contract
  -> future controlled model invocation implementation
  -> CI09 grounding validator
  -> future grounded-response delivery boundary
```

## Inspected sources

Inspected sources included ME-CI07 prompt and response-grounding contract,
ME-CI08 audit and runtime scaffold, ME-CI09 audit and roadmap entry,
`advisory_prompt_package.py`, `advisory_response_grounding.py`,
`advisory_response_dry_run.py`, advisory tests, CI03 Portfolio Intelligence
contract, CI04 Explainability contract, CI05 advisory artifact audit, CI06
validation audit, Structured Decision Output, Governor and Dispatch context
conventions, central backlog, central roadmap, artifact persistence
conventions, CLI conventions, and manifest conventions.

Source main SHA:

```text
a2d408d1b521ae7e83c792385d67c5cf3d8c3d5f
```

## CI09 baseline

ME-CI09 leaves the local system at:

```text
validated source artifact
  -> prompt package
  -> synthetic response fixture
  -> grounding validation
```

The next gap is not more grounding hardening. The gap is a formal boundary for
any future model invocation before raw responses, parser output, and grounding
results can enter the advisory chain.

## Boundary gap analysis

The repository had explicit no-model, no-provider, and no-delivery boundaries,
but no formal contract for invocation request identity, model/provider identity,
idempotency, timeout, retry, raw response capture, parser handoff, budget
policy, failure taxonomy, or grounding-gated eligibility.

ME-CI10 fills that documentation gap without implementing runtime behavior.

## Invocation eligibility review

The contract requires CI06-valid source artifact, valid prompt package,
approved question class, approved permitted use case, authority boundary,
mandatory disclosure derivation, selected context success, and deterministic
invocation request validation before any provider boundary.

## Input boundary review

The approved input is the validated prompt package, not raw provider payloads,
arbitrary local files, raw portfolio files, secrets, broker credentials,
environment data, or unvalidated user claims as facts.

## Context minimization review

The contract requires question-class-specific minimization while preserving
always-required blockers, missingness, freshness, uncertainty, provenance,
validation evidence, and authority boundary.

## Request envelope review

The contract defines a future machine-checkable invocation request envelope
covering request identity, prompt package identity, source artifact identity,
instrument identity, run identity, invocation identity, attempt identity,
provider/model identity, capability profile, instruction contract, input
context, response schema, budgets, timeout, retry, idempotency, data handling,
grounding handoff, and audit context.

## Identity model

The contract separates:

```text
run_id != invocation_id != attempt_id
```

Retries preserve the same Market Engine run and invocation lineage while using
distinct attempt identities.

## Idempotency review

The contract defines idempotency around source identity, prompt package identity
and hash, question class, permitted use case, provider/model identity,
configuration profile, response schema, and budget policy version. Duplicate
provider calls for one logical invocation require explicit policy.

## Provider/model identity review

The contract remains provider-neutral while requiring provider name, API
family, trace policy, model name, model version or snapshot, model reporting,
capability profile, and configuration profile to be auditable.

## Capability requirements

V1 requires structured JSON support, bounded output size, timeout support,
non-streaming mode, refusal signaling, and usage/token metadata or explicit
limitations. Tool use, browsing, and streaming are unsupported for v1.

## Prompt payload boundary

The prompt payload boundary contains system instructions, application
instructions, user question, selected context, required response schema,
mandatory disclosures, forbidden inferences, authority boundary, and grounding
requirements. Prompt text is not source of truth.

## Budget policy review

The contract defines token budgets and cost budgets conceptually without
hardcoding prices. Budget excess fails closed before invocation or retry.

## Timeout/retry review

Timeouts are technical invocation failures, not investment conclusions. Retry
semantics distinguish retryable technical failures from non-retryable contract,
policy, authority, schema, forbidden-input, and budget failures.

## Raw response capture review

Raw provider responses must be captured separately with invocation, attempt,
provider, model, provider request ID, timestamp, finish reason, usage metadata,
raw output, content type, and hash. Raw output is not parsed, grounded, delivery
eligible, or advisory truth.

## Sensitive data boundary

Secrets, API keys, auth headers, environment variables, and credentials must not
enter prompt payloads or persisted artifacts. ME-CI10 records the future
requirement but does not implement a secret scanner.

## Failure taxonomy

The contract defines pre-invocation contract failure, policy failure, budget
failure, transport failure, timeout failure, provider failure, rate-limit
failure, empty response failure, truncated response failure, malformed response
failure, schema failure, parser failure, grounding ungrounded failure, and
grounding blocked failure.

## Parser boundary

The parser boundary is:

```text
raw response capture
  -> strict parser
  -> structured response candidate
  -> CI09 grounding validator
```

Parser success does not determine grounding.

## Grounding handoff

Every parsed response candidate must go to the CI09 grounding validator.
Model-declared grounding status is not source of truth. Raw output must never
bypass grounding validation.

## Downstream eligibility

Only CI09 `grounded`, `grounded_with_mandatory_caveats`, and
`partially_grounded` results can become future delivery-eligible, and only
under a later delivery contract. `ungrounded` and `blocked` remain not delivery
eligible.

## State machine

The contract defines a success path from `request_validated` through
`grounding_completed` and explicit failure branches for request blocked,
timeout, provider failure, rate limit, empty, truncated, malformed, schema
invalid, parse failed, grounding ungrounded, and grounding blocked.

## Audit evidence requirements

Required evidence includes source main SHA, source artifact identity, prompt
package identity, invocation request ID, invocation ID, attempt ID,
provider/model identity, provider trace ID, timestamps, timeout policy, retry
policy, idempotency key, token usage, cost metadata, finish reason, parser
status, grounding status, and downstream eligibility.

## Test matrix

The contract defines future acceptance scenarios for pre-invocation validation,
transport outcomes, model output states, and retry/idempotency behavior.

## Governance review

ME-CI10 adds no OpenAI API runtime, ChatGPT API runtime, Anthropic runtime,
Gemini runtime, generic model SDK, provider HTTP call, API key, secret loading,
prompt execution, model invocation, retries, timeouts, streaming, token
counting runtime, cost calculation runtime, model fallback runtime, external
web search, provider-side browsing, retrieval augmentation, SEC, EDGAR,
yfinance, live prices, Telegram, email, Messenger, Signal, notification
delivery, broker, orders, portfolio writes, watchlist writes, allocation,
target weights, sizing, execution, scheduler, UI, autonomous loop, Decision
Engine changes, Governor changes, Recommendation Review changes, Portfolio
Review changes, causality engine, or materiality engine.

## Implementation decision

ME-CI10 is docs-only. No runtime, tests, provider adapter, SDK, parser,
validator change, or model call was added.

## Residual gaps

Residual gaps:

* no provider implementation;
* no model invocation;
* no API key management;
* no HTTP runtime;
* no retry runtime;
* no timeout runtime;
* no parser runtime;
* no production response artifact;
* no delivery;
* no full natural-language semantic scanner;
* no external browsing contract;
* no multi-provider routing;
* no fallback model policy.

## Recommended next sprint

```text
ME-CI11 - Implement controlled local model invocation adapter scaffold
```

ME-CI11 should remain non-production, single-provider, single-model-profile,
delivery-free, grounding-mandatory, and fail-closed.
