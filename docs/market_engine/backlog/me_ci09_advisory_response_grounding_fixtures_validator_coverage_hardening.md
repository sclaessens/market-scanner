# ME-CI09 - Advisory Response Grounding Fixtures and Validator Coverage Hardening

Owner roles: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED LOCAL HARDENING SPRINT

## Goal

Harden the ME-CI08 local advisory response grounding scaffold before any future
real model-invocation boundary is introduced.

## Scope

ME-CI09 hardens adversarial response fixtures, claim/reference graph validation,
duplicate evidence reference policy, support-type compatibility checks,
restricted evidence path checks, embedded/referenced/absent context handling,
response-mode consistency, partial-answer completeness, response-declared
grounding summary consistency, portfolio-context misuse detection,
explainability attribution boundaries, family-level freshness disclosure
relevance, blocker preservation, Dispatch versus Structured Decision Output
contradiction handling, lineage mismatch coverage, and deterministic issue
ordering.

## Non-goals

ME-CI09 does not introduce real ChatGPT or OpenAI API integration, model
invocation, prompt execution, LLM runtime behavior, model-provider abstraction,
network calls, live prices, SEC, EDGAR, yfinance, delivery integration, broker
integration, portfolio mutation, watchlist mutation, allocation, target weight,
position sizing, order sizing, execution advice, or Decision Engine semantic
changes.

## Outcome

ME-CI09 extends the local validator and regression fixtures without changing the
advisory runtime architecture. The validator remains deterministic,
model-free, provider-free, delivery-free, local-only, non-production, and
fail-closed.

Runtime change:

```text
src/market_engine/advisory/advisory_response_grounding.py
```

Test change:

```text
tests/market_engine/advisory/test_advisory_response_grounding_hardening.py
```

Audit:

```text
docs/market_engine/audits/me_ci09_advisory_response_grounding_fixtures_validator_coverage_hardening_audit.md
```

## Acceptance criteria

Completed:

* coverage audit exists;
* adversarial fixture matrix exists;
* one-to-many evidence references are tested;
* many-to-one source path reuse is tested;
* duplicate evidence references are rejected;
* orphan evidence references are rejected;
* material claims with unmatched claim IDs are rejected;
* support-type compatibility matrix is enforced;
* broad parent evidence paths are rejected for material claims;
* evidence paths must remain inside their declared context family;
* referenced and absent context cannot be treated as embedded proof;
* subtle portfolio misuse is detected;
* unsupported explainability summary wording is detected;
* irrelevant stale context does not cause a false positive;
* relevant stale context requires disclosure;
* blocker neutralization is detected;
* Dispatch contradiction cherry-picking is blocked;
* separate lineage mismatches are tested;
* declared grounding status mismatch is detected;
* incomplete partial answers are rejected;
* issue ordering is deterministic;
* local dry-run regression smokes cover grounded, caveated, partial,
  ungrounded, blocked-authority, and blocked-contradiction outcomes.

## Recommended next sprint

```text
ME-CI10 - Define controlled model invocation boundary contract
```

ME-CI10 should remain a contract sprint. It should not implement live model
invocation until the invocation boundary, prompt ownership, response capture,
retry policy, artifact persistence, and governance controls are explicitly
approved.
