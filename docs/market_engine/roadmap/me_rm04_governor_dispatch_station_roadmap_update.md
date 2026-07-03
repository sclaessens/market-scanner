# ME-RM04 - The Governor and Dispatch Station Roadmap Update

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: PLANNED ROADMAP UPDATE

## Purpose

This roadmap update records the product direction for turning the current Market Engine dry-run and readiness pipeline into a user-facing investment evaluation flow without dropping active in-flight work.

The intended long-term architecture language is:

```text
Market Engine

Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

## Architecture mapping

| Product name | Current / planned responsibility | Boundary |
| --- | --- | --- |
| Boiler | Source acquisition and raw cached-source package creation | Data collection only |
| Refinery | Staging validation, coverage classification, source readiness, provenance, freshness, consumability | Data trust / quality only |
| Analyzer | Source Context, Fundamental Observations, Derived Observations, Setup Detection, Analysis Review, Recommendation Review boundary | Analysis and evidence packaging |
| The Governor | Evidence-weighted investment evaluation, factor scores, recommendation state, buy-zone / position-management explanation | Future governed evaluation authority only |
| Dispatch Station | Operator reports, Telegram-style previews, dashboard/API/PDF/email dispatch artifacts | Output generation / delivery only |

## Roadmap insertion decision

The active ME-SA generic cached-source coverage chain must not be abandoned. ME-SA12 and ME-SA13 already established the generic coverage contract and pure classifier, and the next planned sprint remains:

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
```

ME-SA14 must stay first because it closes an already-started coverage/readiness thread. Removing or bypassing it would leave the generic coverage classifier disconnected from staging validation evidence.

After ME-SA14, the roadmap should add one short governance sprint before more execution work:

```text
ME-RM04 - Align Market Engine product architecture around Boiler / Refinery / Analyzer / The Governor / Dispatch Station
```

ME-RM04 should be docs-only. It should update the active roadmap/backlog indexes and define where The Governor and Dispatch Station fit without creating scoring, recommendation, allocation, order, broker, Telegram-send, portfolio-mutation, or production-write behavior.

## Revised near-term sequence

```text
ME-SA12 - Generic supported-universe cached-source coverage contract
  -> ME-SA13 - Implement generic cached-source coverage classification model
  -> ME-SA14 - Adapt cached-source staging validation into generic coverage input
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
```

## The Governor target output

The Governor should eventually produce an `investment_evaluation` style report with fields such as:

```text
Ticker
Overall Score
Recommendation State
Confidence
Fundamentals
Growth
Valuation
Trend
Momentum
Risk
Technical Setup
Portfolio Fit
Buy Zone
Exceptional Buy Zone
Position Management Guidance
Evidence Summary
Data Confidence
Boundary / Non-actionable markers where required
```

The first Governor implementation must be non-production, local-only, source-grounded, testable, and fail-closed. It must not invent missing factors. If approved evidence is missing, the output must downgrade to descriptive, partial, blocked, or human-review states instead of manufacturing a BUY / SELL / HOLD answer.

## Required sequencing guardrails

* Do not remove ME-SA14.
* Do not bypass staging validation evidence.
* Do not bypass generic coverage classification.
* Do not use ticker-specific branches or shortcuts for expanded coverage.
* Do not introduce scoring before the Governor evidence contract exists.
* Do not introduce buy zones before price / market / setup evidence contracts exist.
* Do not introduce portfolio-fit scoring before approved portfolio context is available.
* Do not introduce allocation, position sizing, broker instructions, order generation, or portfolio mutation inside The Governor.
* Dispatch Station may generate preview/report artifacts only until a later sprint explicitly approves delivery behavior.

## Deferred candidates to preserve

The following candidates remain valid and must not be silently dropped:

* ME-DL03 - Non-production delivery preview.
* ME-PR03 - Approved portfolio context source/persistence contract.
* ME-DE03 - Decision Engine handoff review hardening.
* ME-CANDIDATE03 - Candidate classification QA/review contract.
* ME-OUT03 - Operator report readability/polish improvements.

They should be repositioned under the new architecture as follows:

* ME-DL03 and ME-OUT03 become Dispatch Station candidates.
* ME-PR03 remains a Portfolio / Governor input dependency.
* ME-DE03 remains downstream of The Governor / Decision Engine authority boundaries.
* ME-CANDIDATE03 remains a Refinery / Analyzer quality candidate unless Governor evidence readiness exposes a more urgent blocker.

## Non-goals

ME-RM04 and this roadmap update do not implement runtime behavior, provider calls, live market data, broker calls, Telegram sending, portfolio writes, watchlist writes, scheduler behavior, UI behavior, production reports, scoring, ranking, conviction, BUY / SELL / HOLD action semantics, target prices, target weights, position sizing, order generation, allocation, trade execution, or Decision Engine decisions.
