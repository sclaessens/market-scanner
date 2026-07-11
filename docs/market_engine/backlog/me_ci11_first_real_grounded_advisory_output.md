# ME-CI11 - First Real Grounded Advisory Output

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: IMPLEMENTED WITH REAL INVOCATION BLOCKED BY LOCAL CONFIGURATION; PR REVIEW REMEDIATION APPLIED

## Goal

Generate a local grounded advisory output artifact set and readable report from one real Market Engine ticker artifact.

## Scope

ME-CI11 implements:

- source artifact validation for the selected Market Engine dry-run artifact path;
- deterministic evidence catalog generation with exact source paths;
- deterministic grounded advisory input packaging;
- CI10-shaped invocation request generation and pre-invocation validation;
- a single-provider non-streaming OpenAI Responses invocation boundary;
- provider-side strict structured output;
- bounded output tokens and bounded input size;
- model/provider/configuration-sensitive idempotency;
- raw provider response capture and hashing;
- strict JSON schema/type/enum parsing;
- explicit claim-to-evidence references;
- direct grounding handoff to `ME-CI09.validate_advisory_response_grounding`;
- structured advisory output persistence;
- readable `advisory_report.md` rendering;
- local manifest and traceability metadata;
- fail-closed behavior for missing invocation configuration and invalid or ungrounded model output.

## PR review remediation

The first CI11 implementation used a new evidence-reference containment validator instead of the established CI09 grounding chain. PR review classified that as a merge blocker.

The branch was corrected before merge:

```text
strict response parse
  -> structured claim/evidence projection
  -> ME-CI09 validate_advisory_response_grounding(...)
  -> accepted structured advisory output
```

Simple allowlist containment is no longer treated as equivalent to grounding validation.

The remediation also corrected evidence source paths, removed generic recursive setup-message harvesting, added deterministic invocation validation, added strict provider structured output, added output budgeting, strengthened idempotency material, expanded raw provider response capture, added output path containment checks, and expanded regression tests.

## Outcome

The flow was executed against a real NVDA Market Engine artifact from the ME-RUN28 cached-source batch output.

The recorded run failed closed before provider invocation because the local environment did not define required model configuration:

```text
OPENAI_API_KEY
MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL
```

The blocked run persisted local artifacts under:

```text
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/
```

Those persisted artifacts were regenerated after PR review remediation and now match the corrected CI09-aware runtime shape. They remain blocker evidence only. They are not evidence of a successful provider response or successful CI09-grounded model output because invocation stopped before any provider response existed.

## Non-goals

ME-CI11 does not add production execution, provider refresh, model browsing, model tools, delivery, Telegram or notification behavior, broker integration, portfolio writes, watchlist writes, allocation, sizing, execution, or Decision Engine semantic changes.

## Recommended next sprint

```text
ME-CI11B - Execute configured real grounded advisory model invocation
```

ME-CI11B should use the corrected ME-CI11 runtime, add only approved non-production invocation configuration, and produce the first successful provider response that passes strict parsing and ME-CI09 grounding validation before report rendering.
