# ME-RM02 - Real-world run and Telegram-style terminal preview roadmap lock

Owner roles: Product Owner / Operator / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap / Governance

Status: PLANNED ROADMAP LOCK AFTER ME-SR11

## Purpose

This roadmap lock prevents the Market Engine from drifting back into abstract infrastructure-only work after ME-SR12.

The project has completed the cached-source preparation chain through ME-SR11:

```text
ME-SR07 acquisition plan
ME-SR08 acquisition manifest contract
ME-SR09 cached-source snapshot inventory command
ME-SR10 manual cached-source snapshot staging validator
ME-SR11 cached-source snapshot acquisition dry-run command
```

ME-SR12 remains the next source-refresh implementation sprint because it is the bridge from planned acquisition intent to real local operator-supplied files.

After ME-SR12, the roadmap must move quickly into real-world validation and visible operator-facing output.

## Locked near-term sequence

The near-term sequence after ME-SR12 is now explicitly locked as:

```text
ME-SR12 - Implement operator-supplied cached-source snapshot import command
ME-SR13 - Run real-world operator-supplied cached-source sample import for NVDA, AMD, ASML
ME-SR14 - Run first real cached-source Market Engine analysis for accepted sample tickers
ME-SR15 - Render Telegram-style terminal preview from real cached-source analysis output
ME-SR16 - Real-world findings correction sprint
```

## ME-SR12 role

ME-SR12 must implement a local operator-supplied import path for source payloads.

It must enable a human/operator to point Market Engine at a local file and create a controlled cached-source snapshot staging package with payload and manifest metadata that can be validated by ME-SR10.

ME-SR12 must not become another abstract source-refresh planning sprint.

Required ME-SR12 outcome:

* local operator-supplied payload import command;
* controlled staging layout;
* generated or verified ME-SR08-compatible manifest metadata;
* no provider calls;
* no network calls;
* no broker calls;
* no Telegram sends;
* no portfolio/watchlist mutation;
* no Decision Engine or recommendation authority;
* explicit documentation that the next sprint is ME-SR13 real-world sample import.

## ME-SR13 role

ME-SR13 must be a real-world sample run, not a new abstraction layer.

Target sample:

```text
NVDA
AMD
ASML
```

The purpose is to import real locally supplied source files, run ME-SR10 staging validation, and document what succeeds, what fails, and where the pipeline breaks.

ME-SR13 may expose blockers. If it does, follow-up work must be based on those concrete blockers, not speculative infrastructure.

ASML is intentionally included because it is a useful non-US stress case. It may fail or require source-family/source-mapping work; that outcome must be made visible rather than hidden behind planning.

## ME-SR14 role

ME-SR14 must take the accepted real sample snapshots and run them through the existing Market Engine cached-source analysis path.

The purpose is to inspect actual Market Engine analysis output from real local cached-source data.

ME-SR14 should answer:

* can accepted cached-source snapshots drive the existing run chain;
* which stages complete;
* where missing-data, stale-data, provenance, or source-family gaps appear;
* whether the analysis output is useful enough to render for an operator;
* which concrete failures need correction.

ME-SR14 must remain non-actionable unless an already approved downstream review boundary explicitly permits a field. It must not introduce BUY / SELL / HOLD, allocation, position sizing, execution, tradeability, urgency, conviction, hidden scoring, or ranking authority.

## ME-SR15 role

ME-SR15 must render a Telegram-style preview message in the terminal from real cached-source analysis output.

This is a terminal preview only.

It must not send anything to Telegram.

Required behavior:

* render the exact operator-facing message body that a later delivery layer might send;
* print it visibly in the terminal/stdout;
* optionally write a local preview artifact such as Markdown or JSON;
* make missing-data, blocked stages, weak evidence, and source limitations visible;
* preserve all non-actionable and no-trade-authority boundaries.

Forbidden behavior:

* no Telegram bot token;
* no Telegram API call;
* no network call;
* no production notification;
* no broker call;
* no portfolio/watchlist mutation;
* no Decision Engine decision;
* no BUY / SELL / HOLD authority;
* no allocation, sizing, order, execution, urgency, conviction, score, rank, or tradeability semantics.

The Telegram-style terminal preview is a diagnostic tool. It lets the operator inspect whether the generated output is useful, whether the logic has gaps, whether the language is too vague, and whether the system is actually producing something understandable.

## ME-SR16 role

ME-SR16 must be a correction sprint based only on concrete findings from ME-SR13, ME-SR14, and ME-SR15.

Examples of acceptable ME-SR16 inputs:

* imported source files failed validation for documented reasons;
* real cached-source analysis exposed missing field mapping;
* ASML/non-US source-family assumptions failed;
* terminal preview output was too vague or structurally unusable;
* missing-data or blocked-stage communication was unclear;
* an existing run-chain integration broke on real sample data.

ME-SR16 must not be a speculative infrastructure sprint.

## Roadmap guardrail

After ME-SR12, do not insert new abstract infrastructure, governance, QA, provider, delivery, portfolio, or reporting polish sprints ahead of ME-SR13, ME-SR14, and ME-SR15 unless a concrete blocker is discovered and documented.

A concrete blocker must name:

* the sprint/run that exposed it;
* the exact failure mode;
* the files, command, or output that prove it;
* why the locked sequence cannot proceed without resolving it.

## Rationale

The operator needs to see Market Engine working on real local data soon. The fastest useful validation path is:

```text
operator-supplied source file
-> ME-SR12 import
-> ME-SR13 real sample validation
-> ME-SR14 real cached-source analysis
-> ME-SR15 Telegram-style terminal preview
-> ME-SR16 correction from observed failures
```

This sequence balances source-refresh safety with visible real-world validation.
