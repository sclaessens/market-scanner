# ME-CI05 - Daily ChatGPT-ready Advisory Artifact Backlog Entry

## Status

COMPLETED RUNTIME COMPOSITION LAYER

## Goal

Produce a deterministic local ChatGPT-ready advisory artifact from existing
Market Engine contracts and runtime artifacts without introducing advisory
prose, ChatGPT calls, prompt execution, provider access, portfolio mutation,
delivery behavior, broker behavior, or Decision Engine semantic changes.

## Scope

* pure assembler for
  `market-engine-chatgpt-ready-advisory-artifact-v1`;
* explicit input directory loader;
* local JSON persistence with manifest;
* thin CLI wrapper;
* fail-closed identity and schema validation;
* blocked, descriptive-only, and eligible composition states;
* source precedence preservation;
* missing context preservation;
* synthetic test coverage;
* implementation audit and roadmap update.

## Non-goals

* no ChatGPT API integration;
* no prompt template;
* no advisory prose generation;
* no market-data, SEC, EDGAR, yfinance, or provider calls;
* no broker integration;
* no Telegram or email delivery;
* no portfolio or watchlist writes;
* no allocation, target weight, position sizing, or rebalancing;
* no Recommendation Review, Portfolio Review, Governor, Dispatch Station, or
  Decision Engine redesign;
* no temporal diff engine;
* no causality or materiality model.

## Runtime identity

```text
schema_version: market-engine-chatgpt-ready-advisory-artifact-v1
artifact_type: market-engine-chatgpt-ready-advisory-artifact
```

## Acceptance criteria

* Structured Decision Output is required and validated.
* Optional ME-CI02, ME-CI03, ME-CI04, Governor, and Dispatch inputs validate
  identity before inclusion.
* Optional context conflicts block composition instead of being ignored.
* Portfolio absence is not interpreted as no holdings.
* Cash absence is not interpreted as zero cash.
* Explainability absence leaves change rationale unavailable.
* Dispatch context is presentation-only and cannot override canonical state.
* Generated artifacts are deterministic JSON.
* Persistence is local-only, non-production, and overwrite-protected.
* CLI returns a controlled non-zero exit for input or persistence failures.
* Synthetic tests cover happy, blocked, descriptive, missing, conflict,
  determinism, persistence, and CLI behavior.

## Dependencies

* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence Context.
* ME-CI04 - Explainability / Change-Rationale Contract.
* ME-RUN04 / ME-RUN05 - local dry-run artifact persistence conventions.

## Follow-ups

* typed schema and contract enforcement for ME-CI05 artifacts;
* integration into a controlled local daily dry-run path;
* prompt contract that consumes this artifact without adding authority;
* controlled advisory dry run with no provider, broker, delivery, or production
  side effects.
