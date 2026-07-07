# ME-RM06 - ChatGPT Advisory Delivery Roadmap Reposition

Sprint ID: ME-RM06

Status: COMPLETED DOCS-ONLY ROADMAP REPOSITION

Job family: ME-RM / Roadmap Governance

Date: 2026-07-07

## Roadmap position

ME-RM06 follows the Governor and Dispatch Station contract sequence and
repositions future delivery work before channel-specific implementation.

Previous delivery direction:

```text
Market Engine
-> delivery/reporting
-> Telegram-style long-form channel output
```

New delivery direction:

```text
Market Engine
-> Structured Artifacts / Structured Decision Output
-> ChatGPT Advisory Layer
-> user
```

Compact attention signaling moves to:

```text
Market Engine
-> Notification Layer
-> Messenger / Signal / Telegram / email / later adapters
```

## Roadmap decision

ChatGPT Advisory Layer is the primary interactive user interface above Market
Engine artifacts. GitHub and artifacts remain the source of truth. ChatGPT reads
structured outputs and provides explanation, comparison, scenario analysis, and
Decision Engine output interpretation. ChatGPT is not the calculation engine.

Telegram, Messenger, Signal, email, and future channels are Notification Layer
adapters. They are interchangeable delivery adapters for compact daily signals,
not the main long-form analysis channel.

## Required sequence

The future sequence is:

```text
ME-CI01 - Define Structured Decision Output contract for ChatGPT consumption
  -> ME-CI02 - Define ChatGPT Advisory Context Contract
  -> ME-CI03 - Add ChatGPT-readable Portfolio Intelligence context
  -> ME-CI04 - Define explainability/change-rationale contract
  -> ME-CI05 - Produce daily ChatGPT-ready advisory artifact
  -> ME-PI01 - Define Portfolio Intelligence exposure contract
  -> ME-PS01 - Define Position Sizing decision contract
  -> ME-NL01 - Reframe notification layer as channel-neutral compact summary
  -> ME-NL02 - Define daily notification payload contract
  -> ME-NL03 - Select first notification adapter after structured outputs stabilize
```

## Priority rules

Structured Decision Output and ChatGPT Advisory Context come before any
notification adapter.

Conviction, Position Sizing, and Portfolio Intelligence come before Messenger,
Signal, Telegram, email, or other notification implementation.

Notification Layer must stay compact, channel-neutral, and downstream of stable
structured artifacts.

## Non-goals

ME-RM06 does not implement runtime behavior, tests, ChatGPT API calls,
Messenger, Signal, Telegram, email, providers, production integrations,
portfolio/watchlist writes, scripts, CLI behavior, or Decision Engine semantic
changes.

## Acceptance criteria

* ChatGPT Advisory Layer is named as the primary interactive interface.
* Structured Decision Output is placed before notification delivery.
* Notification Layer is channel-neutral and compact.
* Telegram, Messenger, Signal, and email are adapters, not the primary report
  channel.
* Conviction, Position Sizing, and Portfolio Intelligence are prioritized before
  notification adapters.
* Backlog contains concrete follow-up items.
* Existing historical delivery/reporting documentation remains intact.
* No runtime code is changed.
