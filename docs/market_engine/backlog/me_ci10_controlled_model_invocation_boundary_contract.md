# ME-CI10 - Controlled Model Invocation Boundary Contract

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED DOCS-FIRST CONTRACT

## Goal

Define the formal controlled model invocation boundary between the existing
local advisory pipeline and a future external model provider.

## Scope

ME-CI10 defines:

* invocation eligibility;
* pre-invocation validation gates;
* approved and forbidden inputs;
* context minimization;
* invocation request envelope;
* run, invocation, and attempt identity;
* idempotency, timeout, retry, and backoff semantics;
* provider/model identity;
* required model capabilities;
* prompt payload boundary;
* token and cost budget boundaries;
* raw response capture;
* sensitive data handling;
* invocation states and failure taxonomy;
* parser boundary;
* grounding handoff;
* downstream eligibility;
* persistence, auditability, observability, test, and fail-closed matrices.

## Non-goals

ME-CI10 does not implement a provider adapter, model SDK, API call, API key
loading, HTTP client, retry runtime, timeout runtime, streaming, token
estimator, cost calculator, prompt execution, parser runtime, model call,
delivery, broker integration, portfolio/write path, watchlist/write path,
allocation, target weight, sizing, execution, scheduler, UI, autonomous loop,
or Decision Engine semantic change.

## Outcome

The contract is documented at:

```text
docs/market_engine/contracts/me_ci10_controlled_model_invocation_boundary_contract.md
```

ME-CI10 remains docs-only and fail-closed. A future implementation sprint must
use this contract as the source of truth before any model invocation is
allowed.

## Recommended next sprint

```text
ME-CI11 - Implement controlled local model invocation adapter scaffold
```

ME-CI11 should remain non-production, single-provider, single-model-profile,
delivery-free, grounding-mandatory, and fail-closed.
