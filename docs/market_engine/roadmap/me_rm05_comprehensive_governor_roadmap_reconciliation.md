# ME-RM05 - Comprehensive Governor Roadmap Reconciliation

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: COMPLETED DOCS-ONLY RECONCILIATION

## Purpose

ME-RM05 reconciles the roadmap after the introduction of the Market Engine product architecture:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

The goal is to turn the RM04 and RM04A planning direction into a single canonical sprint sequence with explicit component ownership, deferred-work mapping, evidence gates, and authority boundaries.

This reconciliation does not implement runtime behavior and does not change contracts, source acquisition, analysis semantics, recommendation semantics, output generation, delivery, portfolio behavior, watchlist behavior, broker behavior, scheduler behavior, UI behavior, or Decision Engine authority.

## Inputs reconciled

ME-RM05 reconciles the following completed planning artifacts:

```text
docs/market_engine/roadmap/me_rm04_governor_dispatch_station_roadmap_update.md
docs/market_engine/backlog/me_rm04_governor_dispatch_station_backlog_entry.md
docs/market_engine/audits/me_rm04_governor_dispatch_station_roadmap_audit.md
docs/market_engine/roadmap/me_rm04a_runtime_architecture_alignment_pipeline_update.md
docs/market_engine/backlog/me_rm04a_runtime_architecture_alignment_backlog_entry.md
docs/market_engine/audits/me_rm04a_runtime_architecture_alignment_pipeline_audit.md
```

It preserves the ME-SA12 -> ME-SA13 -> ME-SA14 generic coverage sequence and keeps ME-ARCH01 after the first local non-production Governor / Dispatch Station report artifact.

## Reconciled architecture ownership

| Product layer | Canonical responsibility | Existing or planned job families | Authority boundary |
| --- | --- | --- | --- |
| Boiler | Source acquisition, source-family adapters, cached-source package creation | ME-SA / ME-SR source acquisition and source refresh work | Data collection and package creation only |
| Refinery | Staging validation, source support, generic coverage classification, provenance, freshness, consumability, completeness | ME-SA12, ME-SA13, ME-SA14, ME-RUN29-style evidence runs | Data trust / readiness only |
| Analyzer | Source Context, Fundamental Observations, Derived Observations, Setup Detection, Analysis Review, Recommendation Review boundary | ME-SC, ME-FO, ME-DO, ME-SD, ME-AR, ME-RR | Evidence packaging and non-actionable review until explicitly approved |
| The Governor | Future governed investment evaluation, factor taxonomy, recommendation-state mapping, buy-zone / position-management explanation | ME-GV01 through ME-GV06 | Future evaluation authority only; no broker/order/allocation mutation |
| Dispatch Station | Operator-readable Governor reports and later delivery-preview artifacts | ME-DS01, ME-DS02, later ME-DL03 / ME-OUT03 mapping | Output artifact generation only until delivery is explicitly approved |
| Runtime Architecture | Package/module alignment after product behavior has proven stable | ME-ARCH01 | No-functional-change refactor only |

## Canonical reconciled sequence

The reconciled near-to-mid-term sequence is:

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

ME-RM04 and ME-RM04A are now considered completed planning inputs, not continuing active sprint slots. Their decisions are preserved through this reconciled sequence.

## Why ME-SA14 still comes first

ME-SA14 remains ahead of Governor work because ME-SA13 created a generic coverage classifier that must be connected to staging-validation evidence before downstream evidence readiness can be trusted.

Starting The Governor before ME-SA14 and ME-RUN29 would create a scoring/report layer over incomplete or disconnected source-readiness evidence. That would violate the source-readiness versus investment-quality boundary.

## Why ME-RUN29 comes before ME-GV01

ME-RUN29 should execute or inspect the generic coverage classification output from staging-validation evidence before Governor contracts are finalized.

The Governor contract should be grounded in actual coverage/readiness evidence, including:

* supported versus unsupported source families;
* valid versus invalid manifests;
* stale versus fresh evidence;
* consumable versus non-consumable snapshots;
* complete versus partial analysis context;
* blocked readiness reasons;
* explicit non-actionable and non-Decision-Engine-ready states.

ME-RUN29 must remain local, evidence-only, non-actionable, mutation-free, and delivery-free.

## Governor gate decisions

The Governor may not start with scoring implementation. It must start with a contract sprint.

Required order:

1. ME-GV01 defines the investment evaluation contract, allowed inputs, output shape, failure states, evidence requirements, and authority boundary.
2. ME-GV02 defines factor taxonomy and minimum evidence requirements.
3. ME-GV03 implements a non-actionable scaffold that can fail closed without producing a score.
4. ME-GV04 adds factor scoring only where evidence is approved and sufficient.
5. ME-GV05 maps score/factor evidence into governed recommendation states only under an approved boundary.
6. ME-GV06 defines buy-zone and position-management explanation only after price, setup, market, valuation, and portfolio evidence requirements are explicit.

## Dispatch Station gate decisions

Dispatch Station must not send Telegram/email messages or create production reports as part of its first sprint.

Required order:

1. ME-DS01 defines the output contract for Governor reports.
2. ME-DS02 implements a local non-production report artifact only.
3. ME-DL03 or a later delivery sprint may define non-production delivery preview behavior, but only after the Dispatch Station contract exists.

## Runtime architecture gate decision

ME-ARCH01 stays after ME-DS02.

Rationale:

* Governor module shape is unknown until ME-GV03 through ME-GV06 are implemented or contractually constrained.
* Dispatch Station module shape is unknown until ME-DS01 / ME-DS02.
* Package moves before those shapes are proven would create churn and repeated import rewrites.
* ME-ARCH01 must be a no-functional-change refactor with behavior-parity tests.

Expected future shape remains a candidate, not an implementation authorization:

```text
src/market_engine/
  boiler/
  refinery/
  analyzer/
  governor/
  dispatch_station/
```

## Deferred-work reconciliation

| Deferred candidate | Reconciled placement | Reason |
| --- | --- | --- |
| ME-DL03 - Non-production delivery preview | After ME-DS01 / ME-DS02 or as a DS follow-up | Delivery preview requires a Dispatch Station output contract first |
| ME-OUT03 - Operator report readability/polish improvements | Dispatch Station follow-up | Readability polish belongs to output/report artifacts, not source readiness |
| ME-PR03 - Approved portfolio context source/persistence contract | Governor input dependency before portfolio-fit scoring | Portfolio fit cannot be scored from absent or unapproved portfolio evidence |
| ME-DE03 - Decision Engine handoff review hardening | After Governor recommendation-state boundary is explicit | Decision Engine handoff must not be hardened around undefined Governor semantics |
| ME-CANDIDATE03 - Candidate classification QA/review contract | Refinery / Analyzer QA candidate | Candidate classification remains upstream quality review unless Governor evidence exposes a higher-priority blocker |
| ME-QAxx / ME-GOVxx | Evidence-triggered only | Additional governance should follow concrete run evidence, not refinement loops |

## Authority boundary reconciliation

The following remain forbidden unless a later explicit sprint authorizes them:

* provider calls outside approved source acquisition jobs;
* live market data calls;
* broker behavior;
* Telegram/email sending;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* hidden ranking or urgency semantics;
* BUY / SELL / HOLD action semantics outside an approved recommendation/evaluation boundary;
* target prices or target weights;
* position sizing;
* allocation advice;
* order generation;
* execution instructions;
* Decision Engine decisions.

## Acceptance criteria for this reconciliation

ME-RM05 is complete when:

* RM04 and RM04A are preserved as planning inputs;
* a single canonical sequence exists from ME-SA14 through ME-ARCH01;
* deferred candidates are mapped to the new architecture;
* The Governor starts with contract and taxonomy work, not scoring implementation;
* Dispatch Station starts with output contract and local artifact work, not delivery;
* ME-ARCH01 remains after ME-DS02;
* no runtime, test, provider, delivery, portfolio, watchlist, scheduler, UI, broker, or Decision Engine behavior is changed.

## Result

PASS. The roadmap is reconciled around the Boiler -> Refinery -> Analyzer -> The Governor -> Dispatch Station architecture while preserving the current generic coverage/readiness sequence and deferring runtime package alignment until after a local non-production Governor report artifact exists.
