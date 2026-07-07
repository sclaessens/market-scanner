# ME-RM06 - ChatGPT Advisory Delivery Roadmap Reposition Audit

## Context

The original delivery direction in the Market Engine roadmap was strongly
oriented around Telegram and delivery reporting. That direction was appropriate
while the system was still proving local dry-run, report, and preview
boundaries, but it is no longer the best primary user-facing architecture.

Market Engine has evolved toward structured artifacts, Recommendation Review,
Portfolio Review, The Governor, Dispatch Station, and Decision Engine boundary
contracts. That evolution creates a better architecture:

```text
Market Engine
-> Structured Artifacts / Structured Decision Output
-> ChatGPT Advisory Layer
-> user
```

The user still wants short daily signals. Those signals should be compact
attention messages, not the primary analysis surface. Deep analysis should
happen through ChatGPT using GitHub and reproducible artifacts as context.

## Decision

Market Engine remains the analysis and artifact engine.

ChatGPT becomes the primary interactive advisory layer for:

* portfolio analysis;
* recommendation explanation;
* score-change explanation;
* scenario comparison;
* user questions such as whether a position should be expanded, held, or
  reduced, when such interpretation is grounded in approved artifacts;
* Decision Engine output interpretation.

Notification Layer becomes channel-neutral. Candidate channels include:

* Messenger;
* Signal;
* Telegram;
* email;
* other future adapters.

Notifications are compact and may include:

* portfolio status;
* notable changes;
* new opportunities;
* warnings;
* a prompt to open ChatGPT for detailed analysis.

Notifications must not become the long-form primary report channel.

## Roadmap Impact

The original Telegram-first delivery direction is deprioritized.

ChatGPT Advisory Layer becomes a new roadmap epic and the primary interactive
user interface above Market Engine artifacts.

Structured Decision Output must come before channel notification delivery so
ChatGPT and future adapters consume a stable machine-readable contract instead
of destination-specific report text.

Conviction, Position Sizing, and Portfolio Intelligence must move higher in the
roadmap than notification adapters because they define the decision context that
notifications would summarize.

Notification Layer comes only after stable structured outputs, ChatGPT context,
Portfolio Intelligence, and Position Sizing contracts.

## Non-goals

ME-RM06 does not:

* implement ChatGPT API integration;
* implement Telegram, Messenger, Signal, email, or other notification channels;
* change runtime code;
* change scripts or CLI behavior;
* change tests;
* change providers, yfinance, SEC, EDGAR, source acquisition, or live data
  behavior;
* provide financial buy or sell advice in documentation;
* change existing Decision Engine semantics;
* change portfolio or watchlist state;
* change existing contract semantics except by creating future roadmap items.

## Proposed New Roadmap Epics

### ChatGPT Advisory Layer

Goal: ChatGPT uses reproducible Market Engine artifacts as context and provides
natural interactive explanation to the user.

Future user questions may include:

* Analyze my portfolio.
* Why was AMD recommended?
* Compare AMD with NVDA.
* Should I add to, hold, or reduce IREN according to approved artifacts?
* Which position should be reviewed first for expansion?
* Which recommendation changed since yesterday?

ChatGPT Advisory Layer must remain artifact-grounded and must not become the
calculation engine, order generator, broker layer, portfolio mutation authority,
or Decision Engine replacement.

### Structured Decision Output

Goal: Decision Engine output becomes machine-readable and stable enough for
ChatGPT consumption.

Candidate fields include:

* ticker;
* action;
* conviction;
* confidence;
* risk;
* data coverage;
* portfolio fit;
* buy zone;
* add zone;
* trim zone;
* invalidation level;
* thesis changes;
* explanation references.

### Portfolio Intelligence

Goal: the engine can include approved portfolio context:

* existing positions;
* concentration risk;
* sector and theme exposure;
* AI, datacenter, and semiconductor overlap;
* cash;
* position size;
* correlation risk.

### Position Sizing

Goal: future downstream output can express more than a simple directional
state, including:

* do not add;
* cautious add;
* normal add;
* aggressive add;
* trim;
* exit.

Position sizing must remain downstream and must not bypass Decision Engine
authority.

### Notification Layer

Goal: compact, channel-neutral daily signaling.

Examples include:

* Portfolio stable.
* IREN risk increased.
* AMD entered watchlist.
* NVDA add-zone confirmed.
* Open ChatGPT for detailed analysis.

## Acceptance Criteria

* Roadmap names ChatGPT Advisory Layer as the primary interactive user
  interface above Market Engine artifacts.
* Telegram, Messenger, Signal, and email are repositioned as notification
  channels, not the main report channel.
* Structured Decision Output is planned before notification delivery.
* Conviction, Position Sizing, and Portfolio Intelligence are planned before
  channel integrations.
* Backlog contains concrete follow-up items.
* Existing delivery and reporting documentation remains historically
  consistent.
* No runtime code is changed.

## Prefix Decision

Repository search found no existing `ME-CI`, `ME-PI`, `ME-PS`, or `ME-NL`
roadmap/backlog IDs. ME-RM06 therefore introduces:

* `ME-CI` for ChatGPT Advisory Integration;
* `ME-PI` for Portfolio Intelligence;
* `ME-PS` for Position Sizing;
* `ME-NL` for Notification Layer.

## Validation

Validation results:

```text
PASS - git diff --check
PASS - rg -n "ChatGPT Advisory Layer|Structured Decision Output|Notification Layer|Messenger|Signal|Telegram|Position Sizing|Portfolio Intelligence|Conviction" docs/market_engine
PASS - focused rg validation across the changed central and ME-RM06 files
PASS - .venv/bin/python -m pytest (1394 passed)
REVIEWED - grep -R "BUY" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "SELL" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

The requested `rg ... | tee /dev/tty` form could not write to `/dev/tty` in
this execution environment and returned `tee: /dev/tty: Operation not
permitted`. The same `rg` validation was rerun without `tee` and passed.

The script grep hits were pre-existing legacy portfolio and `__pycache__`
matches. ME-RM06 changed documentation only and introduced no runtime code,
scripts, tests, provider behavior, notification adapter, portfolio/watchlist
write, or Decision Engine semantic change.
