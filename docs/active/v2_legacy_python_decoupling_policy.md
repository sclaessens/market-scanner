# V2 Legacy Python Decoupling Policy

Status: ACTIVE
Reset stage: RESET-10M

## Purpose

This document defines a binding policy for removing legacy Python ballast from
the v2 market-scanner application.

The project goal is not to keep old Python files alive because they still work or
because they are still referenced. The goal is to migrate necessary logic into a
clean canonical v2 architecture and then archive or remove obsolete legacy files
through controlled, tested cleanup sprints.

This policy complements:

```text
docs/active/v2_python_file_creation_policy.md
```

The Python file creation policy prevents uncontrolled new Python files. This
policy prevents uncontrolled long-term retention of old Python files.

## Core Rule

A legacy Python file that is still used is not automatically approved for
long-term retention.

If a legacy Python file is still used, the project must review whether the used
logic should be:

- migrated into a canonical v2 module;
- replaced by an existing canonical v2 module;
- wrapped temporarily only as a certified migration bridge;
- archived after migration;
- deleted after confirmation;
- left untouched only when risk or uncertainty is explicitly documented.

The presence of an import, script call, or runtime dependency is evidence that
the file needs review, not evidence that it should remain permanently.

## Target Architecture Principle

The v2 application should move toward one canonical runtime architecture.

The desired end state is:

- one official application entrypoint;
- one official scanner/data-collection flow;
- one official fundamentals/provider/normalization layer;
- one official analysis layer;
- one official decision/review boundary;
- one official message/report composition layer;
- one official delivery layer for Telegram or other delivery mechanisms.

A canonical file may call other modules, but there should not be multiple
competing files that appear to run the whole application, scan the same universe,
perform the same analysis, compose the same Telegram message, or duplicate the
same orchestration responsibility.

## Forbidden Long-Term Patterns

The following patterns are not acceptable as a long-term v2 architecture:

- multiple application entrypoints with unclear authority;
- multiple files that can run the full program;
- multiple scanner files with overlapping universe-selection or scan logic;
- multiple analysis files that compute similar outputs independently;
- multiple reporting or Telegram message builders with overlapping behavior;
- legacy scripts that bypass the v2 provider, fundamentals, analysis, review, or
  delivery boundaries;
- old files kept permanently because removing them is inconvenient;
- new v2 logic attached to old files in a way that gives the old flow new
  long-term authority;
- duplicate helpers that exist because the canonical responsibility was never
  chosen.

## File Classification Statuses

Every Python file reviewed during cleanup must receive one of these statuses.

### CANONICAL_V2

The file is part of the intended v2 architecture and has a clear responsibility.

Requirements:

- responsibility is explicit;
- tests or controlled runs prove active use;
- it does not duplicate another canonical module;
- it does not bypass governance boundaries.

### LEGACY_DEPENDENCY

The file is legacy but still used by current behavior.

This status is temporary. It means the file must be reviewed for decoupling,
migration, or replacement.

A `LEGACY_DEPENDENCY` file must not receive new v2 responsibilities unless a
separate migration plan explicitly authorizes it.

### MIGRATE_LOGIC

The file contains useful logic that should survive, but the file itself should
not remain as the long-term owner.

The required next step is to identify the canonical v2 module that should own the
logic and migrate it there with tests.

### ARCHIVE_AFTER_MIGRATION

The file may be archived after its required logic has been migrated and the new
path is validated.

### DELETE_AFTER_CONFIRMATION

The file appears obsolete and may be deleted after confirmation through tests,
search, and controlled runs.

### DO_NOT_TOUCH_YET

The file is unclear, risky, or insufficiently understood.

This status must include the reason why the file cannot yet be classified more
strongly.

## Required Review Questions

For every Python file in a cleanup review, answer:

- What responsibility does this file currently have?
- Is this responsibility still needed in v2?
- Is there another file with the same or overlapping responsibility?
- Is this file an entrypoint, runner, scanner, analyzer, reporter, delivery file,
  helper, or legacy utility?
- Is it imported by canonical v2 code?
- Does it import or call legacy code?
- Does it write data, generate reports, send Telegram messages, call providers,
  or trigger pipeline behavior?
- Can its useful logic be migrated into an existing canonical v2 module?
- What tests or controlled runs are required before archive or deletion?

## Decoupling Rule

If a legacy file is still used, the next action should normally be decoupling,
not permanent retention.

Decoupling means:

1. identify the exact used logic;
2. move or recreate that logic in the canonical v2 owner;
3. add or update tests for the canonical owner;
4. update callers to use the canonical owner;
5. verify the old file is no longer used;
6. archive or delete the old file in a separate approved cleanup sprint.

## Canonical Ownership Expectations

The project should converge toward a small set of clear ownership boundaries.

Expected ownership categories:

```text
application entrypoint
scanner / universe selection
provider / source access
fundamentals normalization and evidence
analysis
decision / review boundary
message composition
report generation
delivery / Telegram
configuration
shared utilities
```

A cleanup review may recommend exact filenames or packages, but it must avoid
creating new Python files unless the Python file creation policy is satisfied.

## Entry Point Rule

The project should have one official application entrypoint.

Other runnable Python files may exist only if they are explicitly classified as:

- tests;
- development-only utilities;
- migration-only scripts;
- archived legacy artifacts;
- non-canonical historical files pending removal.

A file that runs the whole app must not coexist indefinitely with another file
that also runs the whole app.

## Scanner and Analysis Rule

The project should not keep multiple competing scanner or analysis flows.

If multiple files scan stocks, evaluate fundamentals, build analysis, or produce
similar outputs, the cleanup review must identify:

- the canonical v2 owner;
- duplicate responsibilities;
- logic to migrate;
- files to archive or delete after validation.

## Reporting and Telegram Rule

The project should not keep multiple competing Telegram/report/message builders.

There should be one canonical message composition path and one canonical delivery
path. Message composition and delivery should remain separate responsibilities.

Legacy files that generate reports or send Telegram messages must not be used as
shortcuts around the canonical v2 flow.

## Cleanup Sprint Safety

This policy does not authorize immediate deletion.

File removal must happen only after a cleanup sprint proves:

- no canonical imports depend on the file;
- tests pass without the file;
- controlled runtime checks still pass;
- any necessary logic was migrated;
- archive or delete decision is documented.

## Required BL27 Behavior

The next cleanup review sprint should be:

```text
RESET-10L-BL27 — Python Architecture Cleanup and Legacy Decoupling Review
```

BL27 should be review-only.

BL27 must:

- inventory Python files;
- identify entrypoints and runners;
- identify scanner files;
- identify analysis files;
- identify report, message, and Telegram files;
- identify duplicate responsibilities;
- identify legacy dependencies still used by the v2 path;
- classify files using this policy;
- recommend migration, archive, or deletion stages;
- propose a canonical v2 runtime architecture;
- update the backlog with the next cleanup implementation step.

BL27 must not:

- delete files;
- move files;
- create new Python files;
- change runtime behavior;
- run provider calls;
- write production data;
- generate reports;
- send Telegram messages;
- change Decision Engine investment behavior.

## Relationship to Future Analysis Work

Further analysis features, such as EPS YoY growth, should not be prioritized
until the project understands which Python files are canonical and which are
legacy dependencies.

The project should avoid adding new analysis logic to old files that are likely
to be archived.

## Guardrails

This policy does not authorize:

- code changes;
- test changes;
- file deletion;
- file moves;
- runtime behavior changes;
- production data writes;
- report generation;
- Telegram delivery;
- provider calls;
- portfolio/watchlist updates;
- BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price,
  tradeability, or recommendation behavior.

## Status

This policy is binding for BL27 and all future cleanup, migration, and
implementation sprints unless superseded by a later approved governance document.
