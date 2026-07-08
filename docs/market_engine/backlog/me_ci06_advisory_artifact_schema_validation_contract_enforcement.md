# ME-CI06 - Advisory Artifact Schema Validation and Contract Enforcement Backlog Entry

## Status

COMPLETED RUNTIME CONTRACT ENFORCEMENT

## Goal

Implement deterministic, fail-closed validation for
`market-engine-chatgpt-ready-advisory-artifact-v1` so that a CI05 advisory
artifact is not considered contract-valid, ChatGPT-ready, or downstream
consumable until its top-level structure, embedded or referenced context
families, validation metadata, missingness, freshness, and cross-context
identity are validated.

## Scope

* advisory artifact validator API;
* typed validation result and issue model;
* explicit machine-readable issue taxonomy;
* top-level schema enforcement;
* Structured Decision Output validation;
* optional Portfolio Intelligence, Explainability, Governor, and Dispatch
  context validation;
* embedded/reference/absent context enforcement;
* cross-context ticker and run identity checks;
* contextual forbidden authority field enforcement;
* persistence gate before artifact writing;
* validation evidence in persisted artifact and manifest;
* CLI validation failure behavior;
* dedicated tests;
* audit and roadmap/backlog updates.

## Non-goals

* no new advisory assembler;
* no analysis engine;
* no Recommendation Review, Portfolio Review, Governor, Dispatch Station, or
  Decision Engine redesign;
* no prompt execution;
* no ChatGPT or OpenAI API integration;
* no LLM output;
* no notification delivery;
* no provider, SEC, EDGAR, yfinance, or live-price access;
* no broker integration;
* no portfolio or watchlist writes;
* no allocation, target weight, position sizing, order sizing, execution
  advice, invented targets, invented stops, invented probabilities, invented
  conviction, invented urgency, invented tradeability, causality, or materiality
  inference.

## Acceptance criteria

* Valid CI05 artifacts produce deterministic `valid` results.
* Invalid artifacts produce deterministic machine-readable issues.
* Required top-level fields are enforced.
* Contract identity, artifact type, and schema version are enforced.
* Embedded context payloads are validated.
* Referenced context references are validated without claiming full embedded
  validation.
* Allowed absent context remains explicit and does not fabricate holdings,
  cash, unchanged state, or freshness.
* Cross-context ticker and run identity conflicts fail closed.
* Invalid artifacts are not persisted as valid advisory artifacts.
* Manifest and artifact validation evidence is written for valid artifacts.
* CLI returns non-zero for validation failures.
* Existing overwrite protection remains active.
* Full advisory, Market Engine, and repository test suites pass.

## Dependencies

* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence Context.
* ME-CI04 - Explainability / Change-Rationale Contract.
* ME-CI05 - Daily ChatGPT-ready advisory artifact.

## Follow-ups

* ME-CI07 - Define ChatGPT advisory prompt and response-grounding contract.
* future controlled advisory dry run after prompt and response grounding are
  contractually defined.
