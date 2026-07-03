# ME-RM04A - Runtime Architecture Alignment Backlog Entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: PLANNED

## Goal

Reserve a future no-functional-change runtime refactor sprint so the codebase can eventually align with the product architecture:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

## New future sprint

```text
ME-ARCH01 - Align runtime architecture with Boiler / Refinery / Analyzer / The Governor / Dispatch Station
```

## Placement

ME-ARCH01 must come after the first local non-production Governor report artifact is implemented.

Planned sequence:

```text
ME-GV01
  -> ME-GV02
  -> ME-GV03
  -> ME-GV04
  -> ME-GV05
  -> ME-GV06
  -> ME-DS01
  -> ME-DS02
  -> ME-ARCH01
```

## Rationale

The product names should eventually be reflected in runtime package structure, but doing this before The Governor and Dispatch Station exist would create premature churn.

ME-ARCH01 should happen once the real module boundaries are known. It should be one deliberate refactor instead of many small renaming changes hidden inside feature sprints.

## ME-ARCH01 expected scope

* package restructuring;
* module moves;
* import updates;
* test import updates;
* documentation synchronization;
* compatibility aliases or migration notes where explicitly needed;
* behavior-parity validation.

Potential target structure:

```text
src/market_engine/
  boiler/
  refinery/
  analyzer/
  governor/
  dispatch_station/
```

Precise lower-level modules may keep descriptive domain names. For example, Analyzer may still contain `source_context`, `fundamental_observations`, `derived_observations`, `setup_detection`, `analysis_review`, and `recommendation_review` as explicit subdomains.

## Explicit non-scope

ME-ARCH01 must not change behavior.

No new functionality, changed algorithms, changed scoring, changed recommendation semantics, changed buy-zone logic, changed contracts without explicit versioning, provider calls, live data, external action integrations, delivery behavior, production writes, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or Decision Engine authority changes.

## Acceptance criteria for this backlog update

* ME-ARCH01 is documented as a planned future sprint.
* ME-ARCH01 is placed after ME-DS02, not before Governor work.
* The no-functional-change boundary is explicit.
* Current ME-SA / coverage / Governor planning remains unchanged.
* No runtime files are changed by ME-RM04A.
