# Knowledge Extraction Policy

Owner role: Governance Auditor

Status: ACTIVE MARKET ENGINE POLICY

## Purpose

This policy defines how existing repository material is inspected and converted into Market Engine specifications.

The existing repository contains useful institutional knowledge, but Market Engine must not inherit old script-era implementation by default. Extraction is a governance process, not a migration shortcut.

## Sources To Inspect

Future extraction work may inspect:

- active documentation;
- archived and legacy documentation;
- old backlog and sprint records;
- audits and provider smoke reports;
- runtime code;
- tests and fixtures;
- data contracts and CSV outputs;
- provider and source-readiness findings;
- reporting and Telegram boundaries;
- portfolio and watchlist behavior;
- runtime entrypoints and operational notes.

Inspection does not authorize modification.

## Extraction Method

For each reference source, the owner role must:

1. Identify the source and repository path.
2. Summarize observed logic, assumptions, lessons, and risks.
3. Decide whether Market Engine keeps, rejects, or defers the lesson.
4. Record implementation implications.
5. Record testing implications.
6. Record data and source implications where relevant.
7. Capture open questions without blocking progress unless the question changes the specification boundary.

## Keep, Reject, Defer

`Keep` means the lesson is useful for Market Engine specifications.

`Reject` means the old behavior is not suitable for Market Engine and must not be carried forward.

`Defer` means the lesson may be relevant, but the current sprint does not need to resolve it.

Every keep, reject, or defer decision must be tied to a reason.

## Prohibited Extraction Outcomes

Extraction must not:

- blindly copy old script-era code;
- treat old quick scripts as canonical runtime;
- continue old cleanup work as the active implementation path;
- introduce hidden provider calls;
- introduce production writes;
- mutate portfolio or watchlist data;
- trigger reports or Telegram delivery;
- move allocation authority outside the Decision Engine;
- convert missing data to zero;
- create implementation work that a later sprint has not authorized.

## Evidence Preservation

Old code, documentation, tests, audits, backlog items, smoke findings, and runtime notes remain evidence. Market Engine extraction must preserve old lessons even when the old behavior is rejected.

Rejected material should be explained, not erased.

## Good Enough Documentation

Market Engine documentation is good enough when it can steer implementation and tests without requiring the reader to rediscover the same source material.

Good enough documentation should:

- state the owner role;
- describe the intended behavior or boundary;
- identify implementation implications;
- identify testing implications;
- preserve important risks and rejected assumptions;
- leave narrow open questions for later sprints.

Good enough documentation does not require exhaustive historical coverage, perfect wording, or repeated restatement of settled governance.

## Anti-Iteration Control

Documentation work must not loop indefinitely. Each extraction sprint must inspect, extract, decide, write implications, and move forward.

When a source is informative but not decisive, mark the decision as defer and continue to the next sprint.

