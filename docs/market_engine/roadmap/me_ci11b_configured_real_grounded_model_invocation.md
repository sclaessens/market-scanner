# ME-CI11B - Configured Real Grounded Model Invocation Roadmap Entry

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: BLOCKED BY LOCAL PROVIDER CONFIGURATION

## Roadmap position

ME-CI11B follows ME-CI11 because ME-CI11 implemented the local grounded advisory output flow and proved fail-closed behavior, but the environment did not provide model credentials or model configuration for a successful provider response.

```text
ME-CI10 controlled model invocation boundary contract
  -> ME-CI11 grounded advisory output runtime and blocked real run evidence
  -> ME-CI11B configured real model invocation
  -> ME-CI11C configured provider rerun with local API key
  -> ME-CI12 batch grounded advisory runner for ticker universe
```

## Goal

Execute the ME-CI11 runtime with approved non-production model configuration and produce the first successful grounded model response and `advisory_report.md` for one real ticker artifact.

## Execution result

ME-CI11B executed the existing command path for NVDA and AMD. Both runs failed closed before provider invocation because the local environment did not contain `OPENAI_API_KEY`.

The attempted command supplied `MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini`, but credentials remained absent. The runtime persisted machine-readable blocked artifacts and readable blocked reports under:

```text
artifacts/market_engine/grounded_advisory_outputs/me-ci11b-nvda-configured-grounded-advisory-blocked-missing-api-key-20260711T000000Z/NVDA/
artifacts/market_engine/grounded_advisory_outputs/me-ci11b-amd-configured-grounded-advisory-smoke-blocked-missing-api-key-20260711T000000Z/AMD/
```

Both tickers reported:

```text
advisory_status = blocked_invocation_not_configured
invocation_state = request_blocked
parser_result = invalid / empty_response
validation_status = invalid
grounding_status = null
```

No real provider response was received. No CI09 grounding validation of a parsed provider response occurred. No successful advisory conclusion was generated.

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

## Next step

ME-CI11C should resolve the exact observed blocker by rerunning the same universal CI11 runtime only after `OPENAI_API_KEY` is available locally outside repository content. It should keep `MARKET_ENGINE_ADVISORY_MODEL` explicit, use no provider fallback, add no ticker-specific logic, and repeat the NVDA primary run plus AMD smoke run before any batch sprint begins.
