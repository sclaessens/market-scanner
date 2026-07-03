# ME-RM04 - The Governor and Dispatch Station Backlog Entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: PLANNED

## Goal

Embed the new Market Engine product direction into the backlog without abandoning active coverage/readiness work or leaving unfinished stories dangling.

The product architecture direction is:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

## Rationale

Recent portfolio-position examples showed that the most useful end-user output is not raw dry-run JSON, but a structured investment evaluation explaining whether a position should be held, accumulated, reduced, or reviewed, and why.

The existing pipeline already provides the foundation: source acquisition, staging validation, cached-source dry-runs, source/readiness metadata, analysis context, Recommendation Review boundaries, Portfolio Review foundations, Decision Engine handoff foundations, and operator reporting. The next product direction should preserve that foundation and add The Governor as the governed investment-evaluation layer, followed by Dispatch Station as the report/preview/output layer.

## Backlog insertion rule

Do not replace the active ME-SA14 work. ME-SA14 remains the next logical sprint because it connects cached-source staging validation evidence into the generic coverage classifier from ME-SA13.

Insert ME-RM04 immediately after ME-SA14, before ME-RUN29 / Governor work, to update the active roadmap and backlog indexes deliberately.

## Planned sequence

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
  -> ME-RM04 - Align Boiler / Refinery / Analyzer / The Governor / Dispatch Station roadmap
  -> ME-RUN29 - Run expanded generic coverage classification from staging-validation evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
  -> ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
  -> ME-GV04 - Implement Governor factor scoring from approved evidence
  -> ME-GV05 - Implement governed recommendation-state mapping
  -> ME-GV06 - Implement buy-zone and position-management explanation contract
  -> ME-DS01 - Define Dispatch Station output contract for Governor reports
  -> ME-DS02 - Implement local non-production Governor report artifact
```

## Proposed sprint definitions

### ME-GV01 - Define The Governor investment evaluation contract

Scope: docs-only contract for the governed evaluation layer.

Must define:

* approved upstream inputs;
* evidence gates;
* allowed readiness states;
* blocked / partial / human-review outcomes;
* output format;
* factor families;
* data-confidence requirements;
* non-actionable versus actionable boundaries;
* relationship with Recommendation Review, Portfolio Review, and Decision Engine handoff.

Non-goals: no implementation, no scoring, no BUY / SELL / HOLD action semantics, no allocation, no position sizing, no target prices, no buy zones, no order generation.

### ME-GV02 - Define Governor factor taxonomy and evidence requirements

Scope: docs-only taxonomy.

Initial factor families:

* Fundamentals;
* Growth;
* Valuation;
* Trend;
* Momentum;
* Risk;
* Technical Setup;
* Portfolio Fit;
* Data Confidence.

Each factor must declare required evidence, optional evidence, missing-evidence downgrade behavior, provenance requirements, and whether it is currently reachable.

### ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold

Scope: local deterministic scaffold only.

The scaffold may emit blocked / descriptive / partial evaluation payloads, but must not emit live scores or recommendation states unless ME-GV01 and ME-GV02 authorize the required evidence.

### ME-GV04 - Implement Governor factor scoring from approved evidence

Scope: implement only factor scores whose evidence contracts are approved and reachable.

Scores must be explicit, source-grounded, provenance-backed, deterministic, and fail-closed. Missing evidence must downgrade confidence or block the factor, never be inferred silently.

### ME-GV05 - Implement governed recommendation-state mapping

Scope: map approved Governor evaluation outcomes to governed states.

Possible future labels may include:

```text
STRONG_BUY
BUY
ACCUMULATE
HOLD
REDUCE
SELL
HUMAN_REVIEW_REQUIRED
BLOCKED_INSUFFICIENT_EVIDENCE
```

Labels must remain report/evaluation states until a separate Decision Engine authority sprint approves any action semantics.

### ME-GV06 - Implement buy-zone and position-management explanation contract

Scope: define and later implement buy-zone / position-management explanation only when price, market, trend, setup, valuation, risk, and portfolio evidence contracts are available.

Must not invent limit prices from incomplete evidence.

### ME-DS01 - Define Dispatch Station output contract for Governor reports

Scope: docs-only contract for report artifacts and previews.

Destinations may include local Markdown/JSON, dashboard payload, Telegram-style preview, PDF, email, or API, but actual delivery/sending remains separate and must be explicitly authorized.

### ME-DS02 - Implement local non-production Governor report artifact

Scope: local artifact generation only.

No Telegram send, email send, production publish, portfolio mutation, watchlist mutation, broker action, scheduler behavior, or UI behavior.

## Acceptance criteria for ME-RM04

* The active ME-SA14 story remains intact.
* The revised sequence is documented.
* The Governor and Dispatch Station are named and positioned.
* Existing deferred stories are repositioned, not dropped.
* No runtime behavior is implemented.
* No scoring or action authority is introduced.
* The backlog and roadmap remain synchronized through dedicated entry files.

## Explicit non-scope

No Python implementation, tests, provider calls, live data, broker calls, Telegram sending, production writes, portfolio writes, watchlist writes, scheduler behavior, UI behavior, scoring, ranking, conviction, BUY / SELL / HOLD action semantics, target prices, buy-zone generation, allocation, position sizing, order generation, trade execution, or Decision Engine decisions.
