# V2 Canonical Runtime Architecture

Status: ACTIVE
Reset stage: RESET-10L-BL28

## Purpose

This document defines the canonical v2 runtime architecture for the
`market-scanner` application.

The goal is to stop the project from carrying multiple competing runners,
scanners, analysis flows, reporting flows, Telegram builders, and legacy runtime
shortcuts. Future cleanup and migration sprints must use this document as the
reference for deciding which logic belongs in the clean v2 application and which
legacy files must be decoupled, migrated, archived, or deleted.

This is a documentation/governance artifact only. It does not modify code, move
files, delete files, change tests, run provider calls, write data, generate
reports, send Telegram messages, or change runtime behavior.

## Policies Applied

This architecture applies:

```text
docs/active/v2_python_file_creation_policy.md
docs/active/v2_legacy_python_decoupling_policy.md
docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md
```

Core principles:

- update existing Python files before creating new files;
- legacy Python files that are still used are not automatically approved for
  long-term retention;
- useful legacy logic must be migrated into canonical v2 owners;
- duplicate runners, scanners, analysis flows, report builders, and Telegram
  builders must be decoupled;
- no new analysis logic should be added to files likely to be archived.

## Canonical Runtime Shape

The v2 runtime must converge toward this single flow:

```text
canonical application entrypoint
-> scanner / universe selection
-> provider / source access
-> fundamentals normalization and evidence
-> analysis
-> decision / review boundary
-> message composition
-> report generation where approved
-> delivery / Telegram where approved
```

Each responsibility may call supporting modules, but there must be one canonical
owner for each responsibility.

## Canonical Ownership Map

### Application Entrypoint

Canonical owner to define or migrate toward:

```text
src/market_scanner/app.py
```

Purpose:

- the only official application entrypoint;
- orchestrates the canonical v2 runtime flow;
- delegates all domain responsibilities to canonical modules;
- does not contain scanning, analysis, message formatting, or delivery logic
  inline.

Current legacy pressure to decouple:

```text
scripts/run_scan.py
scripts/run_full_pipeline.py
```

These legacy runners must not remain permanent runtime authorities.

### Scanner / Universe Selection

Canonical owner to define or migrate toward:

```text
src/market_scanner/scanner/
```

Purpose:

- owns universe selection and scan candidate construction;
- does not perform final analysis decisions;
- does not generate reports or Telegram messages;
- hands structured candidates to the fundamentals and analysis layers.

Legacy scanner logic must be identified and migrated into this canonical
responsibility before old scanner files are archived or deleted.

### Provider / Source Access

Canonical owner:

```text
src/market_scanner/fundamentals/
```

Current canonical files include:

```text
src/market_scanner/fundamentals/fundamentals_provider_contracts.py
src/market_scanner/fundamentals/fundamentals_provider_adapter.py
src/market_scanner/fundamentals/fundamentals_real_source_smoke.py
src/market_scanner/fundamentals/fundamentals_persistence.py
```

Purpose:

- source/provider contracts;
- source-shaped normalization;
- governed derived FreeCashFlow;
- governed prior-year growth evidence;
- explicit missingness and fail-closed behavior;
- controlled persistence boundaries.

This layer must not generate final recommendations, reports, or Telegram output.

### Fundamentals Normalization and Evidence

Canonical owner:

```text
src/market_scanner/fundamentals/
```

Purpose:

- normalized fundamentals;
- raw-evidence separation;
- provenance;
- readiness;
- missingness;
- derived metric status;
- growth evidence status.

Any older fundamentals logic outside this boundary must be reviewed for
migration or archive.

### Analysis

Canonical owner to define or migrate toward:

```text
src/market_scanner/analysis/
```

Purpose:

- transforms governed fundamentals evidence into review-oriented analysis
  profiles;
- may consume cash-flow, growth, quality, valuation, and context evidence when
  governed;
- must preserve review limitations and evidence gaps;
- must not directly send Telegram messages or write reports;
- must not bypass the decision/review boundary.

If an existing analysis module already acts as the closest v2 owner, it should
be identified in the next migration sprint. If no suitable owner exists, a new
module may only be created after satisfying the Python file creation policy.

### Decision / Review Boundary

Canonical owner to define or migrate toward:

```text
src/market_scanner/decision/
```

Purpose:

- separates analysis evidence from decision/review outcomes;
- ensures incomplete evidence remains review-limited;
- prevents investment semantics from appearing before explicit approval;
- owns any future decision-state or review-state logic.

The current project must not add BUY, SELL, HOLD, allocation, conviction,
urgency, scoring, target-price, tradeability, or recommendation behavior without
a separate approved sprint.

### Message Composition

Canonical owner to define or migrate toward:

```text
src/market_scanner/messaging/
```

Purpose:

- transforms approved review/report data into human-readable messages;
- remains separate from delivery;
- does not call Telegram APIs;
- does not compute analysis or decisions.

Legacy message builders must be decoupled from delivery shortcuts.

### Report Generation

Canonical owner to define or migrate toward:

```text
src/market_scanner/reporting/
```

Purpose:

- owns report artifact generation when explicitly approved;
- does not send Telegram messages;
- does not perform scanning, provider access, or analysis decisions.

Report generation is not automatically approved by this architecture document.

### Delivery / Telegram

Canonical owner to define or migrate toward:

```text
src/market_scanner/delivery/
```

Purpose:

- owns delivery mechanisms such as Telegram;
- receives already-composed messages or approved artifacts;
- does not compute analysis;
- does not build report content;
- does not trigger the full pipeline.

Telegram delivery must remain separate from message composition.

### Configuration

Canonical owner to define or migrate toward:

```text
src/market_scanner/config/
```

Purpose:

- configuration loading;
- environment validation;
- path policy;
- credential presence checks without exposing secrets.

Configuration code must not perform provider calls or runtime orchestration.

### Shared Utilities

Canonical owner to define or migrate toward:

```text
src/market_scanner/utils/
```

Purpose:

- small, dependency-light shared utilities;
- no domain orchestration;
- no scanning, analysis, reporting, delivery, or provider behavior.

Utility modules must not become dumping grounds for domain logic.

## Legacy Runtime Authority Rules

The following files are not approved as permanent canonical runtime authorities:

```text
scripts/run_scan.py
scripts/run_full_pipeline.py
```

They may remain temporarily only as migration references or legacy bridges until
logic is migrated into canonical v2 owners and controlled validation proves they
can be archived or removed.

A legacy runner that still works is not automatically retained.

## Duplicate Responsibility Rules

Future cleanup sprints must resolve duplicate responsibility groups identified
by BL27.

Priority duplicate groups:

- application runners / full-pipeline entrypoints;
- scanner / universe selection;
- fundamentals logic outside the canonical fundamentals boundary;
- Decision Engine / review boundary overlap;
- report generation;
- Telegram message composition and delivery;
- portfolio/watchlist runtime side effects.

For each group, the migration sprint must identify:

- canonical owner;
- legacy files to decouple;
- useful logic to migrate;
- tests or controlled runs required;
- archive/delete criteria.

## Migration Order

Recommended cleanup order:

1. define and certify the canonical runtime ownership map;
2. migrate or wrap the single official application entrypoint;
3. decouple legacy runners from canonical runtime responsibility;
4. migrate scanner/universe logic into the canonical scanner boundary;
5. migrate analysis logic into the canonical analysis boundary;
6. separate message composition from Telegram delivery;
7. archive confirmed obsolete Python files;
8. delete files only after confirmation and tests.

## Required Next Step

Proceed to:

```text
RESET-10L-BL29 — Migrate Legacy Runtime Entrypoint Logic
```

BL29 should be a narrow implementation sprint that starts by migrating runtime
entrypoint authority away from legacy runners and toward the canonical v2
application entrypoint. If implementation risk is too high, BL29 may instead
produce an executable migration plan with import and test gates.

## Guardrails for Future Migration Sprints

Future migration sprints must:

- update existing Python files first;
- avoid new Python files unless formally justified;
- avoid one-off migration helpers committed to the repo;
- preserve tests;
- preserve runtime behavior unless explicitly changing a canonical boundary;
- avoid provider calls unless explicitly approved;
- avoid production data writes;
- avoid report generation;
- avoid Telegram delivery;
- avoid portfolio/watchlist updates;
- avoid Decision Engine investment behavior;
- avoid BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price,
  tradeability, or recommendation behavior.

## Non-Goals

RESET-10L-BL28 does not:

- create `src/market_scanner/app.py`;
- create new packages;
- modify Python code;
- move files;
- delete files;
- modify tests;
- run provider calls;
- write production data;
- generate reports;
- send Telegram messages;
- change runtime behavior.

## Conclusion

The canonical v2 architecture is now defined as a single official runtime flow
with clear ownership boundaries. Legacy runners and duplicated Python files must
be treated as migration targets, not permanent authorities.

The next step is to start decoupling runtime entrypoint authority from legacy
scripts and migrate required logic into canonical v2 ownership without adding new
analysis features or preserving old files by default.
