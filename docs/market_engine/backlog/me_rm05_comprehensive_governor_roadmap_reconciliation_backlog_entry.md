# ME-RM05 - Comprehensive Governor Roadmap Reconciliation Backlog Entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: COMPLETED DOCS-ONLY RECONCILIATION

## Goal

Reconcile the Market Engine backlog after introducing the Boiler / Refinery / Analyzer / The Governor / Dispatch Station architecture.

ME-RM05 converts the RM04 and RM04A planning direction into one canonical backlog sequence, maps deferred candidates into the new architecture, and preserves authority boundaries before any Governor implementation sprint begins.

## Scope

Documentation and backlog governance only:

* reconcile RM04 and RM04A into a single canonical sequence;
* preserve ME-SA14 and the generic coverage/readiness path;
* define where ME-RUN29 belongs before Governor contract work;
* define Governor sprint order from contract to taxonomy to scaffold to scoring to recommendation-state mapping to buy-zone explanation;
* define Dispatch Station sprint order from output contract to local non-production artifact;
* keep ME-ARCH01 after ME-DS02;
* remap deferred candidates under the new product architecture;
* restate non-goals and authority boundaries.

## Non-goals

ME-RM05 does not implement or authorize:

* runtime code changes;
* package moves;
* import rewrites;
* tests;
* provider calls;
* live market data calls;
* source acquisition expansion;
* staging validator changes;
* generic coverage classifier changes;
* Analysis Review, Recommendation Review, Portfolio Review, or Decision Engine behavior changes;
* scoring, ranking, urgency, conviction, BUY / SELL / HOLD action semantics, target prices, target weights, allocation, position sizing, order generation, or execution advice;
* Telegram/email sending;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* broker behavior.

## Reconciled backlog sequence

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
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

## Backlog decisions

### Decision 1 - ME-SA14 remains first

ME-SA14 closes the already-started ME-SA12 / ME-SA13 generic coverage thread by adapting staging-validation evidence into generic coverage input. Governor work may not bypass this.

### Decision 2 - ME-RUN29 bridges Refinery evidence into Governor planning

ME-RUN29 should inspect or execute expanded generic coverage classification from staging-validation evidence. Its output should inform the first Governor contract without creating scoring, recommendations, delivery, portfolio mutation, or Decision Engine-ready states.

### Decision 3 - The Governor starts with contracts

Governor work must start with ME-GV01 and ME-GV02. Implementation begins only with a non-actionable scaffold in ME-GV03.

Scoring, recommendation-state mapping, and buy-zone explanation are sequenced later and depend on approved evidence requirements.

### Decision 4 - Dispatch Station starts local and non-production

Dispatch Station begins with an output contract and then a local non-production Governor report artifact. Delivery-preview work remains separate and later.

### Decision 5 - Runtime architecture alignment waits

ME-ARCH01 remains after ME-DS02. It is a future no-functional-change refactor only after Governor and Dispatch Station module shapes are known.

## Deferred candidate mapping

```text
ME-DL03      -> Dispatch Station delivery-preview follow-up after ME-DS01/ME-DS02
ME-OUT03     -> Dispatch Station report-readability follow-up
ME-PR03      -> Governor portfolio-fit input dependency
ME-DE03      -> downstream of Governor recommendation-state / Decision Engine boundary definition
ME-CANDIDATE03 -> Refinery / Analyzer quality follow-up
ME-QAxx / ME-GOVxx -> evidence-triggered governance only
```

## Acceptance criteria

* The reconciled sequence is documented.
* RM04 and RM04A are preserved as completed planning inputs.
* ME-SA14 is not dropped or bypassed.
* ME-RUN29 is placed before ME-GV01.
* ME-GV01 / ME-GV02 precede Governor implementation.
* ME-DS01 precedes ME-DS02 and both precede delivery behavior.
* ME-ARCH01 remains after ME-DS02.
* Deferred candidates are mapped without silently dropping them.
* No runtime or behavioral authority is introduced.

## Outcome

ME-RM05 completes the roadmap reconciliation. The backlog should proceed through ME-SA14 and ME-RUN29 before starting Governor contract work.
