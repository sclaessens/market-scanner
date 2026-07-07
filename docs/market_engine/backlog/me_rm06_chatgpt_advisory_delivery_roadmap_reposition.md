# ME-RM06 - Reposition Delivery Layer around ChatGPT Advisory Integration

Sprint ID: ME-RM06

Status: COMPLETED DOCS-ONLY ROADMAP REPOSITION

Job family: ME-RM / Roadmap Governance

Date: 2026-07-07

## Summary

ME-RM06 repositions the delivery roadmap around ChatGPT Advisory Layer as the
primary interactive user interface above reproducible Market Engine artifacts.

## Context

Earlier delivery planning implicitly pointed toward Telegram-first delivery and
long-form channel reports. The current Market Engine architecture now has
structured artifacts, Recommendation Review, Portfolio Review, The Governor,
Dispatch Station, and Decision Engine boundaries. Those artifacts are a better
source for interactive advisory explanation through ChatGPT.

## Decision

Primary path:

```text
Market Engine
-> Structured Artifacts / Structured Decision Output
-> ChatGPT Advisory Layer
-> user
```

Compact signal path:

```text
Market Engine
-> Notification Layer
-> Messenger / Signal / Telegram / email / later adapters
```

## Scope

ME-RM06 updates roadmap, backlog, and audit documentation only.

## Non-goals

ME-RM06 does not:

* change runtime code;
* change tests;
* add scripts or CLI behavior;
* add provider or live data behavior;
* implement ChatGPT API integration;
* implement Messenger, Signal, Telegram, email, or other adapters;
* change portfolio/watchlist data;
* change Decision Engine semantics.

## Acceptance criteria

* Roadmap names ChatGPT Advisory Layer as the primary interactive user
  interface.
* Telegram, Messenger, Signal, and email are notification channels only.
* Structured Decision Output precedes notification delivery.
* Conviction, Position Sizing, and Portfolio Intelligence precede channel
  integrations.
* Backlog contains concrete follow-up items.
* Existing delivery/reporting documents stay historically intact.
* No runtime code is changed.

## Follow-up backlog

* ME-CI01 - Define Structured Decision Output contract for ChatGPT consumption.
* ME-CI02 - Define ChatGPT Advisory Context Contract.
* ME-CI03 - Add ChatGPT-readable Portfolio Intelligence context.
* ME-CI04 - Define explainability/change-rationale contract.
* ME-CI05 - Produce daily ChatGPT-ready advisory artifact.
* ME-PI01 - Define Portfolio Intelligence exposure contract.
* ME-PS01 - Define Position Sizing decision contract.
* ME-NL01 - Reframe notification layer as channel-neutral compact summary.
* ME-NL02 - Define daily notification payload contract.
* ME-NL03 - Select first notification adapter after structured outputs stabilize.

## Validation

Required validation:

```text
git diff --check
rg validation for ChatGPT Advisory Layer / Structured Decision Output / Notification Layer
```

Runtime tests are not required because this sprint is docs-only.
