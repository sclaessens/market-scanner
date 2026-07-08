# ME-CI07 - ChatGPT Advisory Prompt and Response-Grounding Contract Roadmap Entry

Owner roles: Product Owner / Technical Architect / Governance Auditor / QA Lead

Status: COMPLETED DOCS-FIRST CONTRACT

## Roadmap position

ME-CI07 follows ME-CI06, which made deterministic ChatGPT-ready advisory
artifacts fail-closed and schema-validated.

The roadmap sequence is:

```text
ME-CI01 - Structured Decision Output
  -> ME-CI02 - ChatGPT Advisory Context
  -> ME-CI03 - Portfolio Intelligence Context
  -> ME-CI04 - Explainability / Change-Rationale Context
  -> ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-CI06 - Advisory artifact schema validation and contract enforcement
  -> ME-CI07 - ChatGPT advisory prompt and response-grounding contract
  -> ME-CI08 - Controlled advisory response dry run and grounding validator scaffold
```

## Purpose

ME-CI07 defines how a future ChatGPT advisory prompt may consume a validated
advisory artifact and how a future response must remain grounded, bounded,
machine-checkable, and fail-closed.

## Outcome

ME-CI07 adds a contract for:

* input eligibility;
* prompt context selection;
* instruction hierarchy;
* question classification;
* response modes;
* response envelope;
* evidence grounding;
* uncertainty and blocker preservation;
* refusal and inability behavior;
* prohibited inference;
* contradiction handling;
* future response-grounding validator requirements.

## Governance boundary

ME-CI07 does not approve prompt execution, model invocation, response
generation, notification delivery, portfolio mutation, watchlist mutation,
broker integration, allocation, sizing, or execution authority.

## Next

```text
ME-CI08 - Controlled advisory response dry run and grounding validator scaffold
```

ME-CI08 should implement only local deterministic scaffolding unless a later
approved sprint explicitly expands scope.
