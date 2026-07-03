# ME-RM04A - Runtime Architecture Alignment Pipeline Update

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: PLANNED ROADMAP UPDATE

## Purpose

This docs-only update adds a future runtime-architecture alignment sprint to the Market Engine roadmap after the first working Governor / Dispatch Station report artifact exists.

The product architecture remains:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

ME-RM04 introduced this architecture as the product direction. ME-RM04A adds the explicit future refactor step that will align runtime packages and module paths with that architecture once the shape of The Governor and Dispatch Station is proven.

## Decision

Add the following future sprint to the pipeline:

```text
ME-ARCH01 - Align runtime architecture with Boiler / Refinery / Analyzer / The Governor / Dispatch Station
```

ME-ARCH01 is intentionally scheduled after the first local non-production Governor report artifact, not before Governor implementation.

## Revised target sequence

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
  -> ME-RM04 - Align steampunk Market Engine architecture and roadmap insertion
  -> ME-RUN29 - Run expanded generic coverage classification from staging-validation evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
  -> ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
  -> ME-GV04 - Implement factor scoring from approved analysis evidence
  -> ME-GV05 - Implement recommendation-state mapping under approved boundary
  -> ME-GV06 - Implement buy-zone and position-management explanation contract
  -> ME-DS01 - Define Dispatch Station output contract for Governor reports
  -> ME-DS02 - Implement local non-production Governor report artifact
  -> ME-ARCH01 - Align runtime architecture with Boiler / Refinery / Analyzer / The Governor / Dispatch Station
```

## Why ME-ARCH01 comes after ME-DS02

Runtime package restructuring should not happen while the Governor and Dispatch Station shapes are still uncertain. Moving packages too early would create churn, large import diffs, and likely repeated follow-up refactors.

By placing ME-ARCH01 after ME-DS02, the project first proves:

* which Governor modules are real;
* which Dispatch Station modules are real;
* which Analyzer modules remain stable;
* which Refinery modules are already mature;
* which Boiler modules still need source-acquisition expansion;
* which public contracts must remain backward-compatible.

ME-ARCH01 can then be a single controlled architecture refactor instead of several partial renames.

## ME-ARCH01 scope

ME-ARCH01 should align runtime package structure, imports, tests, and documentation with the final product architecture.

Expected future shape may look like:

```text
src/market_engine/
  boiler/
  refinery/
  analyzer/
  governor/
  dispatch_station/
```

Nested modules may retain precise domain names where useful. For example, `analyzer/` may still contain dedicated modules for Source Context, Fundamental Observations, Derived Observations, Setup Detection, Analysis Review, and Recommendation Review rather than flattening those concepts into generic names.

## ME-ARCH01 non-goals

ME-ARCH01 must not introduce new behavior. It is an architecture alignment / refactor sprint only.

Explicit non-goals:

* no new provider behavior;
* no live data;
* no broker behavior;
* no Telegram sending;
* no production writes;
* no portfolio mutation;
* no watchlist mutation;
* no scheduler behavior;
* no UI behavior;
* no changed scoring algorithms;
* no changed recommendation semantics;
* no changed buy-zone calculations;
* no changed Decision Engine authority;
* no changed contracts unless a backward-compatible alias or migration is explicitly approved;
* no hidden behavior changes through import moves.

## Guardrails

* Contract names should remain stable unless a later versioned contract explicitly authorizes a migration.
* Existing persisted artifact formats should remain readable.
* Imports should be updated through one controlled sprint, not piecemeal across unrelated feature work.
* Tests must prove behavior parity before and after the refactor.
* Any backwards-compatibility aliases or migration shims must be explicit and documented.

## Acceptance criteria

ME-ARCH01 will be considered ready to start only after:

* ME-DS02 has produced a local non-production Governor report artifact;
* the Governor package shape is known;
* the Dispatch Station package shape is known;
* active source/readiness work has no unresolved package boundary blockers;
* a no-functional-change refactor plan can be reviewed before runtime changes begin.

ME-RM04A does not implement ME-ARCH01. It only reserves the sprint in the official roadmap pipeline.
