# V2 Manual Real-Source Smoke Execution Review

Status: ACTIVE
Reset stage: RESET-10L-BL7

## Purpose

This document defines the manual review process for controlled v2 real-source
smoke execution.

It is not production execution. It is not investment analysis. It does not
approve automated provider execution, report generation, Telegram delivery,
Decision Engine behavior, scoring, recommendations, or BUY, SELL, HOLD,
allocation, conviction, urgency, or tradeability logic.

## Current Allowed Execution Boundary

Manual smoke execution may only:

- explicitly invoke the controlled smoke harness;
- use an injected source client or an explicit provider/source response;
- return an in-memory result;
- inspect raw evidence, normalized fundamentals, missing values, provenance,
  and neutral source-data readiness.

Manual smoke execution may not:

- write data files;
- generate reports;
- send Telegram messages;
- run the production pipeline;
- modify portfolio or watchlist data;
- commit live provider/source output;
- commit credentials or API keys;
- alter Decision Engine behavior;
- produce BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability
  logic.

## Pre-Run Checklist

Before a manual local review, confirm:

- the branch is clean;
- the branch is based on the latest `main`;
- the virtual environment is active;
- there are no uncommitted changes;
- targeted smoke/provider tests, or the full suite, are green;
- no credentials are stored in the repository;
- no output files are staged;
- the smoke target is limited to one ticker and one source;
- the run is manual-only;
- output will be terminal-only or local scratch-only;
- no files under `data/` or `reports/` will be written.

## Safe Smoke Target

The first allowed target pattern is:

```text
one ticker
one provider/source response
one reported period
one manual invocation
in-memory result only
```

ASML is the preferred first review target because the existing dry-run fixture
history already uses ASML-shaped official/regulatory source metadata. This is a
review preference only and must not be hard-coded into production logic.

## Review Checklist

The reviewer must check:

- provider/source name;
- provider category;
- source reference;
- ticker;
- source timestamp;
- retrieval timestamp;
- reported period;
- fiscal year and fiscal quarter, where available;
- currency;
- unit;
- raw field preservation;
- normalized field mapping;
- missing fields remain explicit;
- missing values are not converted to zero;
- readiness status is neutral;
- provenance is traceable;
- no investment conclusions are present;
- no side effects occurred.

## Allowed Local-Only Output

Terminal output may be reviewed locally. Live provider/source output must not be
committed.

If a summary is later committed, it must be manually written and must exclude:

- credentials;
- secrets;
- API keys;
- raw live payloads;
- private data;
- large provider output;
- investment conclusions.

## Forbidden Outcomes

Manual review must not result in:

- committed live source payloads;
- committed credentials;
- committed API keys;
- generated data files;
- report files;
- Telegram message files;
- automated provider runs;
- production pipeline runs;
- Decision Engine investment behavior;
- BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic;
- missing values converted to zero.

## Pass/Fail Criteria

The review passes only if:

- manual invocation returns an in-memory smoke result;
- raw evidence exists and preserves provenance;
- normalized fundamentals remain program-ready input only;
- readiness is neutral;
- missing values remain explicit;
- no side effects occur;
- no live output is committed;
- the working tree remains clean after local review.

The review fails if:

- missing values are converted to zero;
- provenance is absent;
- readiness implies investment quality;
- any report, data, or Telegram file is generated;
- the production pipeline is touched;
- credentials or live output appear in the git diff;
- Decision Engine behavior is touched.

## Post-Run Checklist

After a local manual review, run:

```bash
git status
git diff --stat
git diff --check
```

Optional manual inspection may use a reference timestamp:

```bash
find reports -type f -newer <reference>
```

That inspection is optional and must not create or update files.

## Next Step

The next candidate step is
`RESET-10L-BL8 — Manual Real-Source Smoke Execution`.

That future step must remain manual-only and local-only. It should review one
ticker and one source, commit no credentials or live output, write no data files
unless separately approved, run no production pipeline, generate no reports,
send no Telegram messages, add no Decision Engine investment logic, and add no
BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability behavior.
