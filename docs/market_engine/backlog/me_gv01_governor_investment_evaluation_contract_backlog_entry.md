# ME-GV01 - The Governor Investment Evaluation Contract Backlog Entry

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

## Goal

Define the first contract for The Governor investment-evaluation layer before any Governor runtime, scoring, recommendation-state mapping, buy-zone explanation, or Dispatch Station output work begins.

## Scope

ME-GV01 is documentation-only and defines:

* contract identity: `market-engine-governor-investment-evaluation-v1`;
* approved future input families;
* required top-level output shape;
* evaluation states;
* reserved future authority states;
* evidence readiness gates;
* fail-closed behavior;
* factor output reservation without taxonomy or scoring;
* recommendation-state boundary;
* buy-zone and position-management boundary;
* portfolio-fit boundary;
* authority non-goals;
* next sprint: ME-GV02.

## Non-goals

ME-GV01 does not implement or authorize:

* runtime code;
* tests;
* provider calls;
* live market data calls;
* source acquisition;
* snapshot import;
* staging validator changes;
* generic coverage classifier changes;
* Analyzer semantic changes;
* Recommendation Review semantic changes;
* Portfolio Review semantic changes;
* Dispatch Station behavior;
* Telegram/email sending;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* broker behavior;
* BUY / SELL / HOLD action semantics;
* scoring;
* ranking;
* urgency;
* conviction;
* tradeability;
* target prices;
* target weights;
* allocation;
* position sizing;
* order generation;
* execution instructions;
* Decision Engine decisions.

## Roadmap position

```text
ME-SA12 - Generic supported-universe cached-source coverage contract
  -> ME-SA13 - Generic coverage classifier
  -> ME-SA14 - Staging-validation adapter
  -> ME-RUN29 - Expanded generic coverage classification evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
```

## Contract decisions

### Decision 1 - Evidence readiness precedes investment evaluation

The Governor must first prove data readiness, source trust, provenance, freshness, consumability, completeness, and Analyzer integrity before evaluating investment quality.

### Decision 2 - Scoring is not part of ME-GV01

ME-GV01 may reserve factor and overall-evaluation sections, but it does not define numeric scoring, weighting, ranking, urgency, conviction, or tradeability.

### Decision 3 - Recommendation-state mapping is blocked by default

The contract reserves a future `recommendation_state` section, but all recommendation-state output remains `blocked_not_authorized` until ME-GV05 or a later explicitly approved sprint.

### Decision 4 - Buy-zone and position-management output is blocked by default

Buy-zone and position-management explanation remain blocked until ME-GV06 or a later explicitly approved sprint.

### Decision 5 - Portfolio fit requires approved portfolio context

Portfolio-fit evaluation remains blocked until an approved portfolio-context contract exists, likely ME-PR03 or equivalent later work.

### Decision 6 - Decision Engine readiness remains false

The Governor contract may expose boundary metadata, but it does not make anything Decision Engine-ready. `decision_engine_ready=false` remains mandatory in blocked and non-actionable states.

## Acceptance criteria

ME-GV01 is complete when:

* the Governor contract document exists;
* contract version is explicit;
* input evidence families are listed;
* output shape is defined;
* allowed and reserved states are defined;
* mandatory fail-closed cases are documented;
* authority boundaries are explicit;
* ME-GV02 is listed as next sprint;
* no runtime or behavioral change is introduced.

## Outcome

ME-GV01 defines The Governor investment evaluation contract without implementing it. The next sprint is ME-GV02, which must define the factor taxonomy and evidence requirements before any scaffold or scoring work begins.
