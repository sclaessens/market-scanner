# ME-CI08 - Controlled Advisory Response Dry Run and Grounding Validator Scaffold

Owner roles: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED LOCAL RUNTIME SCAFFOLD

## Goal

Implement the first local, deterministic, model-free execution layer for the
ME-CI07 prompt and response-grounding contract.

## Scope

ME-CI08 implements:

* deterministic prompt package builder;
* explicit question-class validation;
* deterministic question-relevant context selection;
* prompt package validation;
* response envelope validation;
* claim taxonomy validation;
* evidence reference validation;
* restricted evidence path resolution;
* required disclosure enforcement;
* missingness enforcement;
* portfolio, explainability, Governor, Dispatch, freshness, blocker, and
  authority-boundary checks;
* controlled synthetic response dry-run runner;
* local CLI;
* local non-production dry-run artifacts and manifest;
* deterministic advisory tests.

## Non-goals

ME-CI08 does not introduce:

* OpenAI API or ChatGPT API integration;
* model invocation;
* LLM SDK;
* prompt execution;
* provider calls;
* source refresh;
* live prices;
* Telegram, email, Messenger, Signal, or notification delivery;
* broker integration;
* orders;
* portfolio mutation;
* watchlist mutation;
* allocation;
* target weight;
* position sizing;
* execution advice;
* Decision Engine override;
* Governor, Recommendation Review, or Portfolio Review semantic changes;
* causality or materiality engine.

## Outcome

Runtime modules:

```text
src/market_engine/advisory/advisory_prompt_package.py
src/market_engine/advisory/advisory_response_grounding.py
src/market_engine/advisory/advisory_response_dry_run.py
```

Test modules:

```text
tests/market_engine/advisory/test_advisory_prompt_package.py
tests/market_engine/advisory/test_advisory_response_grounding.py
tests/market_engine/advisory/test_advisory_response_dry_run.py
```

## Acceptance criteria

Completed:

* CI06-valid source artifact is required;
* deterministic prompt package exists;
* question class is explicit and preserved;
* context selection is deterministic;
* missing context remains explicit;
* response envelope validator exists;
* claim IDs are validated;
* evidence references are validated;
* evidence paths are resolved;
* material claims require evidence;
* required disclosures are enforced;
* portfolio misuse is detected;
* unsupported causality is detected;
* unsupported materiality is detected;
* freshness conflicts are detected;
* blocker omission is detected;
* contradictions are detected;
* authority violations are detected;
* grounding status is deterministic;
* grounded, caveated, partial, ungrounded, and blocked outcomes are supported;
* controlled dry-run runner and CLI exist;
* local artifacts distinguish successful and failed states;
* overwrite protection is preserved;
* deterministic tests exist.

## Recommended next sprint

```text
ME-CI09 - Harden advisory response grounding fixtures and validator coverage
```
