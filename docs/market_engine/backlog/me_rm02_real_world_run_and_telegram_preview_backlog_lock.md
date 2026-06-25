# ME-RM02 - Real-world run and Telegram-style terminal preview backlog lock

Owner roles: Product Owner / Operator / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap / Governance

Status: PLANNED ROADMAP/BACKLOG LOCK AFTER ME-SR11

## Purpose

This backlog lock preserves the next concrete execution path after ME-SR12 so Market Engine does not return to repeated abstract infrastructure sprints before real-world validation.

The locked intent is to move from operator-supplied local source files to a real cached-source analysis run and then to a Telegram-style terminal preview.

## Locked backlog sequence

The next sequence is:

```text
ME-SR12 - Implement operator-supplied cached-source snapshot import command
ME-SR13 - Run real-world operator-supplied cached-source sample import for NVDA, AMD, ASML
ME-SR14 - Run first real cached-source Market Engine analysis for accepted sample tickers
ME-SR15 - Render Telegram-style terminal preview from real cached-source analysis output
ME-SR16 - Real-world findings correction sprint
```

## ME-SR12 - Operator-supplied cached-source snapshot import command

Status: NEXT ACTIVE IMPLEMENTATION SPRINT

Goal: implement a local import command that takes operator-supplied payload files and places or copies them into a controlled cached-source snapshot staging layout with ME-SR08-compatible manifest metadata.

Acceptance direction:

* local import command exists;
* local payload file is copied or registered into a controlled staging layout;
* manifest metadata is generated or verified;
* imported snapshots can be passed to the ME-SR10 staging validator;
* no provider calls;
* no network calls;
* no real external acquisition;
* no broker, Telegram, portfolio, watchlist, Decision Engine, recommendation, ranking, scoring, allocation, order, execution, urgency, conviction, or tradeability authority;
* documentation explicitly names ME-SR13 as the next sprint.

## ME-SR13 - Real-world operator-supplied cached-source sample import

Status: LOCKED NEXT AFTER ME-SR12 UNLESS CONCRETE BLOCKER IS FOUND

Goal: use ME-SR12 to import real local operator-supplied source files for a small sample and run ME-SR10 validation.

Initial target sample:

```text
NVDA
AMD
ASML
```

Outcome expected:

* real local sample files are imported or rejected with explicit reasons;
* staging validation is run;
* accepted/rejected states are documented;
* ASML/non-US source-family issues are made visible;
* concrete blockers are captured before analysis expansion.

This sprint must not become another planning-only sprint. Its purpose is to touch real local data.

## ME-SR14 - First real cached-source Market Engine analysis

Status: LOCKED AFTER ME-SR13 UNLESS CONCRETE BLOCKER IS FOUND

Goal: run the accepted ME-SR13 sample snapshots through the existing Market Engine cached-source analysis path and inspect real analysis output.

Outcome expected:

* accepted sample tickers are run through the existing cached-source dry-run/analysis path;
* completed, blocked, stale, missing-data, and limitation states are visible;
* generated analysis artifacts or terminal output are inspected;
* gaps in the analysis logic are captured as concrete findings;
* no new investment authority is introduced.

## ME-SR15 - Telegram-style terminal preview from real analysis output

Status: LOCKED AFTER ME-SR14 UNLESS CONCRETE BLOCKER IS FOUND

Goal: render an operator-facing Telegram-style message from real cached-source analysis output and print it in the terminal.

This is preview-only. It must not send to Telegram.

Required output behavior:

* render the message body that would later be sent;
* print the message visibly in stdout/terminal;
* optionally write a local preview artifact;
* expose missing-data, blocked-stage, stale-data, source-limitation, and weak-evidence notes clearly;
* make output quality inspectable by the operator.

Forbidden behavior:

* no Telegram send;
* no Telegram bot token;
* no Telegram API call;
* no network call;
* no production notification;
* no broker call;
* no portfolio/watchlist mutation;
* no Decision Engine decision;
* no BUY / SELL / HOLD authority;
* no allocation, sizing, order, execution, urgency, conviction, score, rank, or tradeability semantics.

## ME-SR16 - Real-world findings correction sprint

Status: LOCKED AFTER ME-SR15

Goal: correct only the concrete issues discovered by ME-SR13, ME-SR14, and ME-SR15.

Allowed correction inputs:

* real import failed;
* staging validation rejected real files for documented reasons;
* accepted snapshots could not drive analysis;
* missing-data or stale-data handling was unclear;
* ASML/non-US handling exposed a real blocker;
* Telegram-style terminal preview was vague, misleading, too technical, or missing required operator context.

ME-SR16 must not become speculative infrastructure. It must cite the real-world evidence that justifies each fix.

## Insertion rule

Do not insert any new abstract infrastructure sprint between ME-SR12 and ME-SR15 unless the project discovers a concrete blocker during ME-SR12, ME-SR13, or ME-SR14.

If a sprint is inserted, the insertion must document:

* exact blocker;
* command or artifact that revealed it;
* why the locked sequence cannot continue;
* why the new sprint is the minimum necessary correction.

## Operator rationale

The operator needs visible real-world feedback soon. Terminal-rendered Telegram-style output is treated as an early diagnostic surface, not as production delivery.

The purpose is to reveal whether Market Engine's output is useful, whether the logic has holes, whether missing-data warnings are understandable, and whether the system is moving toward practical analysis rather than only abstract contracts.
