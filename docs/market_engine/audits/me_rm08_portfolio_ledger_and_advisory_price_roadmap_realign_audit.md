# ME-RM08 — Portfolio ledger and advisory price roadmap realignment audit

Sprint ID: ME-RM08

Status: COMPLETED DOCS-ONLY

Date: 2026-08-12

Branch: `me-rm08-portfolio-ledger-advisory-price-roadmap-realignment`

Base commit: `8e71af5935db5e4bc0cd5261035497115df0573d`

Base verification: merge commit for PR #475, ME-SR24 blocker closeout.

## Findings

The roadmap sources were inconsistent. The active baseline stopped at
ME-SR19/ME-DATA11, the main roadmap header still referenced ME-RUN26, and the
ChatGPT advisory chain still presented ME-CI11D as active next work. ME-SR20
through ME-SR24 and ME-RUN32 existed in later narrative but were not reconciled
into the active sequence.

The repository already contains:

- ME-PR01 portfolio review contract;
- ME-PR02 portfolio review runtime using explicitly supplied context;
- portfolio source-shape metadata;
- legacy manual transaction scripts;
- synthetic portfolio contract tests.

It does not contain an approved authoritative transaction-ledger runtime.
Legacy scripts remain reference-only.

## Decision recorded

- ME-PR03 is the next implementation story.
- ME-SR25 follows directly as advisory-only scheduled price refresh.
- ME-PI01 follows after both inputs are stable.
- ME-DATA11, ME-CI11D, canonical provider activation, and notifications remain
  planned but deferred.
- Real portfolio data must remain outside the public Git repository.
- Canonical `market-data` publication remains blocked by ME-SR24.
- Advisory prices may not be represented as canonical evidence.

## Files

This sprint changes documentation only:

- active current-state, governance, and repository-structure pointers;
- active baseline direction;
- main roadmap;
- main backlog;
- ME-PR03 backlog story;
- ME-SR25 backlog story;
- ME-RM08 roadmap decision;
- this audit.

## Boundary verification

- No Python code changed.
- No tests changed.
- No workflow changed.
- No data or CSV file changed.
- No provider call was introduced.
- No canary was dispatched.
- No `market-data` change was made.
- No portfolio transaction was recorded.
- No broker or order behavior was introduced.
- No recommendation, allocation, sizing, or Decision Engine authority changed.

## Validation

Documentation validation consists of:

- exact story-ID collision checks for ME-RM08, ME-PR03, and ME-SR25;
- cross-document active-sequence reconciliation;
- English-only review;
- Markdown structure review;
- final branch diff inspection.

Draft PR: pending creation at the time of this initial audit commit.
