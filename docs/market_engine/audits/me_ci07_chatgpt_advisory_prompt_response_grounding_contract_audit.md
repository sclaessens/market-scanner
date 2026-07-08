# ME-CI07 - ChatGPT Advisory Prompt and Response-Grounding Contract Audit

Owner roles: Governance Auditor / Product Owner / Technical Architect / QA Lead

Status: COMPLETED DOCS-FIRST CONTRACT AUDIT

## Scope audited

ME-CI07 defines the documentation-only contract between a CI06-validated
`market-engine-chatgpt-ready-advisory-artifact-v1` artifact and a future
ChatGPT advisory prompt and response-grounding layer.

The audit reviewed that ME-CI07:

* starts after ME-CI06 validation and contract enforcement;
* introduces no runtime code, tests, API calls, model invocation, prompt
  execution, delivery adapter, broker integration, portfolio mutation,
  watchlist mutation, allocation logic, sizing logic, or execution authority;
* preserves the ChatGPT Advisory Layer as interpretation over deterministic
  Market Engine artifacts;
* defines input eligibility, prompt boundary, instruction hierarchy, question
  taxonomy, response modes, claim taxonomy, response envelope, evidence
  grounding, refusal and inability behavior, contradiction handling, and future
  validator requirements;
* supplies synthetic JSON examples for the response-grounding contract states.

## Source context inspected

The sprint reviewed the current CI chain and adjacent contracts:

* ME-RM06 ChatGPT Advisory Layer repositioning;
* ME-CI01 Structured Decision Output;
* ME-CI02 ChatGPT Advisory Context;
* ME-CI03 Portfolio Intelligence Context;
* ME-CI04 Explainability / Change-Rationale Context;
* ME-CI05 ChatGPT-ready advisory artifact composition;
* ME-CI06 advisory artifact schema validation and contract enforcement;
* Governor and Dispatch contract documentation relevant to advisory
  interpretation boundaries.

## Contract identity

Approved contract identity:

```text
contract_name: chatgpt_advisory_prompt_response_grounding
contract_version: v1
schema_version: chatgpt-advisory-prompt-response-grounding-v1
artifact_type: market-engine-chatgpt-advisory-prompt-response-grounding-contract
```

This identity belongs to the contract document. It does not create a runtime
artifact writer or validator.

## Architecture finding

ME-CI07 keeps the approved sequence:

```text
Market Engine deterministic jobs
  -> Structured Decision Output
  -> advisory context contracts
  -> deterministic advisory artifact assembly
  -> deterministic advisory artifact validation
  -> prompt and response-grounding contract
  -> future controlled dry run
  -> future grounding validator
  -> future delivery
```

The audit finds no authority expansion. The ChatGPT Advisory Layer remains a
bounded explanatory interface and does not become Analyzer, Recommendation
Review, Portfolio Review, Governor, Dispatch Station, Decision Engine, broker,
allocation, sizing, or execution authority.

## Input eligibility audit

ME-CI07 requires a future prompt runtime to accept only a CI06-validated
primary input:

```text
schema_version: market-engine-chatgpt-ready-advisory-artifact-v1
artifact_type: market-engine-chatgpt-ready-advisory-artifact
validation_status: valid
validation_issue_count: 0
```

The contract explicitly separates file existence, artifact assembly, schema
validation, and advisory permission for a particular user question.

## Prompt boundary audit

The contract separates always-required context, question-relevant context,
conditionally required context, and unauthorized context. It preserves missing
portfolio context, absent explainability context, unknown freshness, and
unsupported user claims as unavailable rather than inferred.

## Instruction hierarchy audit

ME-CI07 defines a future instruction hierarchy:

* system-level boundaries over authority, evidence, missingness, uncertainty,
  blockers, disclosures, and refusal;
* application/developer-level context packaging, classification, envelope, and
  grounding-reference responsibilities;
* user-question limitations that cannot override contract authority.

## Response mode audit

The approved response modes are:

```text
advisory_interpretation
descriptive_only
partial_answer
unable_to_determine
refused_outside_authority
blocked_invalid_context
```

The audit confirms that descriptive-only, partial, refusal, and blocked states
cannot be converted into recommendation, allocation, urgency, sizing, or
execution semantics.

## Claim and grounding audit

ME-CI07 defines allowed and forbidden claim types and requires each material
claim to have a stable claim identifier and machine grounding reference unless
the claim is explicitly an inability, missingness, uncertainty, or authority
boundary statement.

The future grounding status model is:

```text
grounded
grounded_with_mandatory_caveats
partially_grounded
ungrounded
blocked
```

This status model is documentation-only in ME-CI07. A later validator must
implement deterministic checks before the status is runtime-authoritative.

## Boundary audit

ME-CI07 preserves:

* ME-CI03 as the source of truth for portfolio advisory context;
* ME-CI04 as the source of truth for explainability and change rationale;
* Governor context as explanatory input only;
* Dispatch Station as presentation/reference context only;
* Structured Decision Output as canonical when Dispatch wording conflicts;
* Decision Engine authority as the only allocation authority.

## Synthetic example audit

ME-CI07 adds six synthetic JSON examples:

* grounded advisory interpretation;
* descriptive-only response;
* partial answer;
* unable-to-determine response;
* refused outside authority response;
* blocked contradiction response.

The examples are non-production, synthetic, and do not provide real investment
advice, real portfolio instructions, broker actions, target prices, sizing, or
execution guidance.

## Governance result

PASS.

ME-CI07 is docs-first and contract-only. It introduces no runtime behavior and
does not expand advisory, recommendation, allocation, broker, portfolio,
watchlist, delivery, or execution authority.

## Residual gaps

ME-CI07 intentionally leaves these gaps for future sprints:

* no prompt template implementation;
* no model call or API integration;
* no response parser;
* no deterministic grounding validator;
* no persisted advisory-response artifact writer;
* no delivery or notification adapter;
* no production advisory answer workflow.

## Recommended next sprint

```text
ME-CI08 - Controlled advisory response dry run and grounding validator scaffold
```

ME-CI08 should remain local, deterministic, model-free, provider-free,
delivery-free, broker-free, portfolio-write-free, watchlist-write-free, and
outside allocation authority.
