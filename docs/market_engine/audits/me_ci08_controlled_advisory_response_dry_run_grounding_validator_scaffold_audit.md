# ME-CI08 - Controlled Advisory Response Dry Run and Grounding Validator Scaffold Audit

Owner roles: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED LOCAL RUNTIME SCAFFOLD

## Objective

ME-CI08 implements the first local, deterministic, model-free execution layer
for the ME-CI07 prompt and response-grounding contract.

The implemented chain is:

```text
CI06-valid advisory artifact
  -> deterministic prompt package
  -> explicit synthetic response fixture
  -> response envelope validation
  -> claim and evidence grounding validation
  -> authority-boundary validation
  -> deterministic grounding result
  -> local non-production dry-run artifact
```

## Architecture position

ME-CI08 follows ME-CI05, ME-CI06, and ME-CI07:

```text
Market Engine deterministic upstream jobs
  -> ME-CI05 advisory artifact assembly
  -> ME-CI06 artifact contract validation
  -> ME-CI07 prompt and response-grounding contract
  -> ME-CI08 local deterministic response dry run
  -> ME-CI08 grounding validator scaffold
  -> future model-invocation boundary
  -> future grounded-response delivery
```

ME-CI08 is not an investment reasoning engine. It does not generate market
interpretations, recommendations, sizing, allocation, execution advice, buy
zones, target prices, causality, materiality, or portfolio fit.

## Inspected sources

The implementation reviewed:

* ME-CI05 advisory artifact assembler and tests;
* ME-CI06 advisory artifact validator and tests;
* ME-CI06 audit;
* ME-CI07 prompt and response-grounding contract;
* ME-CI07 audit;
* ME-CI07 synthetic response-grounding examples;
* central backlog and roadmap;
* advisory package exports and CLI conventions;
* local artifact persistence and manifest conventions;
* Structured Decision Output, Portfolio Intelligence, Explainability, Governor,
  and Dispatch context structures available through the CI05 advisory artifact.

## CI07 contract mapping

ME-CI08 implements the CI07 scaffold requirements for:

* response envelope required fields;
* approved response modes;
* approved grounding statuses;
* approved claim types and forbidden claim types;
* evidence reference shape;
* restricted deterministic path resolution;
* required disclosures;
* missingness enforcement;
* portfolio, explainability, Governor, Dispatch, freshness, blocker, and
  authority boundaries;
* deterministic issue taxonomy.

## Prompt package architecture

`src/market_engine/advisory/advisory_prompt_package.py` builds a controlled
machine-readable prompt package. It is not a natural-language prompt executor.

The package contains:

* prompt package identity;
* source artifact identity;
* instrument identity;
* explicit user question;
* explicit question classification;
* permitted use case;
* selected context;
* mandatory disclosures;
* forbidden inferences;
* required response contract;
* grounding requirements;
* authority boundary.

The builder requires a CI06-valid advisory artifact and fails before response
dry-run execution when the source artifact is invalid.

## Question classification handling

ME-CI08 uses explicit question classification input. It does not implement NLP,
keyword matching, or intent inference.

Supported question classes are exactly the ME-CI07 class set, including current
state, recommendation interpretation, portfolio context, change rationale,
risk, freshness, missing evidence, buy-zone explanation, position-management
explanation, comparison, sizing, allocation, execution, and unsupported
questions.

## Context selection rules

The prompt package always preserves contract identity, artifact identity,
validation evidence, run identity, advisory eligibility, Structured Decision
Output context, blockers, missing context, freshness, uncertainty, and
provenance context.

Question-relevant context is selected deterministically:

* Portfolio Intelligence for portfolio, position, sizing, and allocation
  questions;
* Explainability context for change-rationale and comparative questions;
* Governor context for recommendation, buy-zone, and position-management
  questions;
* Dispatch only as presentation/reference context.

Absent context remains explicit absent context. It is not converted into an
empty object, zero holdings, zero cash, no position, or no change.

## Response envelope validation

`src/market_engine/advisory/advisory_response_grounding.py` validates the
CI07-required response envelope fields:

```text
response_identity
source_artifact_identity
instrument_identity
question_classification
response_mode
summary
assessment
evidence_supporting
evidence_opposing
blockers
uncertainty
freshness_caveats
portfolio_context
change_rationale
required_disclosures
unable_to_determine
evidence_references
grounding_summary
authority_boundary
```

It validates required fields, primitive shapes, list/object semantics, approved
enum values, source identity, instrument identity, question class, response
mode, grounding summary, and authority boundary fields.

## Claim taxonomy enforcement

The validator accepts only CI07 allowed claim types and rejects forbidden claim
types such as invented facts, unsupported causal claims, unsupported
materiality claims, unsupported sizing, unsupported allocation, unsupported
execution, and authority override claims.

Material claims require stable claim IDs, uniqueness, approved claim types, and
evidence references. Inability, missingness, uncertainty, and authority-boundary
statements may exist without evidence references when they are explicit
boundary statements.

## Evidence reference validation

Evidence references are validated for:

* known claim ID;
* claim type match;
* approved context family;
* approved support type;
* source artifact lineage;
* run identity;
* deterministic path existence;
* null or absent context misuse;
* `associated_only` misuse for causal/reason claims.

## Path resolution

ME-CI08 implements a restricted deterministic path resolver for simple
CI07-style paths such as:

```text
$.structured_decision_context.payload.decision.action
```

It does not add a JSONPath dependency, fuzzy matching, auto-correction, or
nearest-key logic. Missing paths fail closed.

## Grounding status mapping

The deterministic status mapping is:

* `grounded` for schema-valid, fully grounded responses with no mandatory
  caveats;
* `grounded_with_mandatory_caveats` for grounded refusal, inability, or
  disclosure-bearing responses;
* `partially_grounded` for valid partial answers;
* `ungrounded` for missing evidence, invalid evidence paths, missing
  disclosures, freshness conflicts, blocker omissions, missingness misuse, or
  unsupported non-blocking claims;
* `blocked` for source identity mismatch, instrument mismatch, question
  mismatch, authority violations, forbidden claim types, semantic override,
  recommendation remapping, or undisclosed Dispatch/Structured Decision Output
  contradiction.

## Disclosure enforcement

ME-CI08 enforces context-driven disclosures:

* descriptive-only responses require `descriptive_only_disclosure`;
* portfolio questions with absent portfolio context require
  `missing_portfolio_disclosure`;
* stale or unknown relevant freshness requires freshness disclosures;
* associated-only support for reason-like claims requires
  `causality_disclosure`;
* sizing, allocation, or execution questions require `authority_disclosure`.

No blanket disclaimer requirement is introduced.

## Missingness enforcement

The validator detects:

* absent portfolio context used as a holdings, cash, weight, exposure, or
  portfolio-fit fact;
* absent explainability context used as change-cause support;
* null/absent evidence paths used for material claims;
* current-state wording against unknown freshness without disclosure.

## Portfolio boundary

ME-CI08 detects portfolio-context misuse and authority violations. It does not
infer holdings, cash, exposure, target weight, position size, rebalancing,
selling, or portfolio fit from absent context.

## Explainability boundary

ME-CI08 rejects unsupported causal claims, associated-only causal support,
unsupported materiality claims, root-cause-like overrides, and semantic
recommendation remapping.

## Governor boundary

ME-CI08 does not compute Governor scores, overall scores, ranks, buy zones,
targets, stops, or add/reduce/exit decisions. It validates only response claims
against supplied paths and authority boundaries.

## Dispatch contradiction handling

Dispatch remains presentation context. If Dispatch presentation wording
conflicts with Structured Decision Output canonical state, ME-CI08 requires a
contradiction disclosure or blocks grounding.

## Freshness handling

Freshness is treated as context family state. Artifact generation time is not
treated as upstream evidence freshness. Unknown or stale relevant freshness
requires disclosure before current-state wording can be grounded.

## Blocker preservation

Source blockers present in the CI06-valid advisory artifact must be preserved
in response blockers. ME-CI08 does not rank blockers and does not convert
blockers into negative investment conclusions.

## Authority enforcement

ME-CI08 fails closed for sizing, allocation, execution, broker action, order
quantity, target weight, position size, semantic override, recommendation
remapping, hidden ranking, hidden composite score, invented conviction,
invented urgency, and invented tradeability claims.

## Dry-run architecture

`src/market_engine/advisory/advisory_response_dry_run.py` implements a
controlled local runner:

1. load CI06-valid advisory artifact;
2. build deterministic prompt package;
3. load explicit synthetic response fixture;
4. validate response grounding;
5. persist local dry-run artifacts.

The runner does not generate response text and does not call a model.

## CLI behavior

The CLI accepts:

```text
--advisory-artifact
--question
--question-class
--response-fixture
--run-id
--artifact-root
```

Grounded, caveated, and partial responses exit `0`. Ungrounded, blocked, and
invalid source artifacts exit non-zero.

## Artifact persistence

The local dry-run writes:

```text
prompt_package.json
synthetic_response.json
grounding_result.json
dry_run_summary.json
manifest.json
```

Successful and failed states are separated:

```text
dry_run_completed_grounded
dry_run_completed_with_caveats
dry_run_completed_partial
dry_run_failed_ungrounded
dry_run_blocked
```

Overwrite protection is enabled by default.

## Determinism

Validation issue ordering, grounding status, response mode handling, disclosure
requirements, evidence path resolution, and dry-run summaries are deterministic
for the same inputs. Run IDs are explicit inputs.

## Test evidence

Implemented advisory tests cover:

* prompt package source validation;
* explicit question classes;
* context selection;
* missing context preservation;
* mandatory disclosures;
* response envelope validation;
* identity mismatches;
* response modes;
* duplicate claim IDs;
* missing evidence references;
* unknown claim references;
* evidence path failures;
* invalid context families;
* disclosure omissions;
* sizing, allocation, and execution violations;
* associated-only causality misuse;
* unsupported materiality;
* freshness conflicts;
* portfolio missingness misuse;
* blocker omission;
* Dispatch contradiction handling;
* semantic override and recommendation remapping;
* dry-run persistence;
* overwrite protection;
* CLI grounded, ungrounded, and authority-violation smokes.

## Governance boundary

ME-CI08 remains local-only, deterministic, model-free, provider-free,
delivery-free, non-production, and fail-closed.

It adds no OpenAI API, ChatGPT API, model invocation, LLM SDK, prompt execution,
model selection, temperature setting, token limit, streaming, provider retry,
external network call, source provider, live price, Telegram, email, Messenger,
Signal, notification delivery, broker integration, order generation, portfolio
mutation, watchlist mutation, allocation, target weight, position sizing,
order sizing, autonomous loop, scheduler, UI, Decision Engine override,
Governor semantic change, Recommendation Review semantic change, Portfolio
Review semantic change, causality engine, or materiality engine.

## Residual gaps

ME-CI08 intentionally leaves these gaps:

* no real model invocation boundary;
* no prompt template execution;
* no production response artifact contract;
* no external delivery;
* no comprehensive natural-language semantic scanner;
* no full JSONPath support;
* no advanced causal/materiality reasoning.

## Recommended next sprint

Recommended next sprint:

```text
ME-CI09 - Harden advisory response grounding fixtures and validator coverage
```

Rationale: CI08 proves the local scaffold. The safer next step is to harden
fixtures, path coverage, issue taxonomy, and validator edge cases before any
real model-invocation boundary is introduced.
