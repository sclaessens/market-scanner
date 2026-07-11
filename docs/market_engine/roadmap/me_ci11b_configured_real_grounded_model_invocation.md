# ME-CI11B - Configured Real Grounded Model Invocation Roadmap Entry

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: NEXT ACTIVE ADVISORY SPRINT AFTER ME-CI11

## Roadmap position

ME-CI11B follows ME-CI11 because ME-CI11 implemented the local grounded advisory output flow and proved fail-closed behavior, but the environment did not provide model credentials or model configuration for a successful provider response.

```text
ME-CI10 controlled model invocation boundary contract
  -> ME-CI11 grounded advisory output runtime and blocked real run evidence
  -> ME-CI11B configured real model invocation
  -> ME-CI12 report quality pass using successful real-run evidence
```

## Goal

Execute the ME-CI11 runtime with approved non-production model configuration and produce the first successful grounded model response and `advisory_report.md` for one real ticker artifact.

## Scope

ME-CI11B may:

* provide approved non-production model invocation configuration outside committed repository content;
* run the existing ME-CI11 command against the selected real Market Engine artifact or a newer approved real artifact;
* persist invocation request, raw response, parser result, validation result, structured output, report, and manifest;
* verify that the accepted report is grounded only in allowed evidence references;
* document the successful run evidence.

## Non-goals

ME-CI11B must not add multi-provider routing, model fallback policy, streaming, tool use, browsing, delivery, broker integration, portfolio writes, watchlist writes, allocation, sizing, execution, or Decision Engine semantic changes.

## Hard completion condition

ME-CI11B is complete only when one provider response is received, parsed, grounded, validated, and rendered into a successful local `advisory_report.md`.

Blocked or unconfigured invocation evidence remains useful audit evidence but is not sufficient to complete ME-CI11B.
