# ME-CI07 - ChatGPT Advisory Prompt and Response-Grounding Contract

Owner roles: Product Owner / Technical Architect / Governance Auditor / QA Lead

Status: COMPLETED DOCS-FIRST CONTRACT

## Goal

Define the formal contract between a CI06-validated
`market-engine-chatgpt-ready-advisory-artifact-v1` and a future ChatGPT
advisory prompt and response-grounding layer.

## Scope

Documentation-only contract work:

* prompt input eligibility;
* prompt context boundary;
* instruction hierarchy;
* question taxonomy;
* advisory permission matrix;
* required response envelope;
* response modes;
* claim taxonomy;
* evidence grounding requirements;
* provenance-reference requirements;
* uncertainty and blocker preservation;
* Portfolio Intelligence, Explainability, Governor, Dispatch, and Decision
  Engine boundaries;
* freshness, refusal, inability, contradiction, and fail-closed behavior;
* synthetic JSON examples for future validator design.

## Non-goals

ME-CI07 does not implement:

* Python runtime behavior;
* tests;
* prompt execution;
* model invocation;
* OpenAI API integration;
* response parsing;
* response validation;
* notification delivery;
* broker integration;
* portfolio writes;
* watchlist writes;
* allocation decisions;
* position sizing;
* order or execution guidance.

## Outcome

ME-CI07 defines:

```text
contract_name: chatgpt_advisory_prompt_response_grounding
contract_version: v1
schema_version: chatgpt-advisory-prompt-response-grounding-v1
artifact_type: market-engine-chatgpt-advisory-prompt-response-grounding-contract
```

It establishes six approved response modes:

```text
advisory_interpretation
descriptive_only
partial_answer
unable_to_determine
refused_outside_authority
blocked_invalid_context
```

and five future grounding statuses:

```text
grounded
grounded_with_mandatory_caveats
partially_grounded
ungrounded
blocked
```

## Acceptance criteria

Completed:

* CI06-validated artifact input boundary documented;
* prompt authority boundary documented;
* instruction hierarchy documented;
* question taxonomy documented;
* permission matrix documented;
* response envelope documented;
* claim taxonomy and evidence-reference rules documented;
* refusal, inability, contradiction, and fail-closed semantics documented;
* synthetic examples added under `docs/market_engine/contracts/examples/`;
* central backlog and roadmap updated.

## Recommended next sprint

```text
ME-CI08 - Controlled advisory response dry run and grounding validator scaffold
```
