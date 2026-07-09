# ME-CI11B - Configured Real Grounded Model Invocation

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: NEXT ACTIVE ADVISORY SPRINT AFTER ME-CI11

## Goal

Run the ME-CI11 grounded advisory output runtime with approved non-production model configuration and produce the first successful grounded provider response and readable advisory report.

## Insertion reason

ME-CI11 implemented the local flow and persisted blocked real-run evidence, but the environment did not define the required model invocation configuration:

```text
OPENAI_API_KEY
MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL
```

The first successful provider response must therefore be completed before ME-CI12 report-quality work.

## Scope

ME-CI11B may:

* provide approved local non-production model configuration outside committed repository content;
* run the ME-CI11 command against one real Market Engine artifact;
* persist invocation request, raw response, parser result, validation result, structured output, report, and manifest;
* verify evidence-reference grounding containment;
* document the successful real-run evidence.

## Non-goals

ME-CI11B must not add production execution, provider refresh, multi-provider routing, fallback models, streaming, tool use, browsing, delivery, broker integration, portfolio writes, watchlist writes, allocation, sizing, execution, or Decision Engine semantic changes.
