# ME-CI11 - First Real Grounded Advisory Output

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: IMPLEMENTED WITH REAL INVOCATION BLOCKED BY LOCAL CONFIGURATION

## Goal

Generate a local grounded advisory output artifact set and readable report from one real Market Engine ticker artifact.

## Scope

ME-CI11 implements:

* source artifact validation for the selected Market Engine dry-run artifact path;
* deterministic grounded advisory input packaging;
* CI10-shaped invocation request generation;
* a single-provider non-streaming OpenAI Responses invocation boundary;
* raw response capture;
* strict JSON parser behavior;
* allowed-evidence grounding validation;
* structured advisory output persistence;
* readable `advisory_report.md` rendering;
* local manifest and traceability metadata;
* fail-closed behavior for missing invocation configuration and invalid model output.

## Outcome

The flow was executed against a real NVDA Market Engine artifact from the ME-RUN28 cached-source batch output.

The run failed closed before provider invocation because the local environment did not define required model configuration:

```text
OPENAI_API_KEY
MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL
```

The blocked run persisted local artifacts under:

```text
artifacts/market_engine/grounded_advisory_outputs/me-ci11-nvda-first-grounded-advisory-20260709T120000Z/NVDA/
```

The generated report is readable and correctly states that no grounded advisory conclusion was generated because invocation was not configured.

## Non-goals

ME-CI11 does not add production execution, provider refresh, model browsing, model tools, delivery, Telegram or notification behavior, broker integration, portfolio writes, watchlist writes, allocation, sizing, execution, or Decision Engine semantic changes.

## Recommended next sprint

```text
ME-CI11B - Execute configured real grounded advisory model invocation
```

ME-CI11B should use the ME-CI11 runtime, add only approved non-production invocation configuration, and produce the first successful grounded model response and report.
