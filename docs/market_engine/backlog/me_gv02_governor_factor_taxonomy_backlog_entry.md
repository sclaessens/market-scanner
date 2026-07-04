# ME-GV02 - Governor Factor Taxonomy Backlog Entry

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

## Goal

Define the canonical Governor factor taxonomy and factor-level evidence requirements before any Governor scaffold or scoring implementation.

## Scope

ME-GV02 defines:

* nine canonical factor families;
* canonical factor states;
* factor output target shape;
* global evidence gates;
* minimum evidence expectations per factor;
* factor-specific downgrade rules;
* cross-factor fail-closed rules;
* conflicting-evidence handling;
* factor completeness semantics;
* overall-evaluation prerequisites;
* scoring semantics reserved for ME-GV04;
* recommendation boundary reserved for ME-GV05;
* buy-zone boundary reserved for ME-GV06;
* ME-GV03 acceptance criteria.

## Canonical factor families

```text
fundamentals
growth
valuation
trend
momentum
risk
technical_setup
portfolio_fit
data_confidence
```

## Canonical factor states

```text
not_started
blocked
unavailable
insufficient_evidence
partial
qualitative_only
evaluable
```

`evaluable` indicates evidence sufficiency only. It does not imply quality, score, recommendation, or actionability.

## Key sequencing decisions

### ME-GV03 next

ME-GV03 may implement a deterministic non-actionable scaffold that:

* consumes approved evidence;
* emits factor states;
* preserves evidence references;
* applies fail-closed downgrade rules;
* leaves score and weight fields null or absent.

### ME-GV04 reserved for scoring

ME-GV02 does not define numeric scales, factor weights, normalization, weighted aggregation, score bands, ranking, or missing-factor imputation.

### ME-GV05 reserved for recommendation-state mapping

Factor states and future factor scores do not imply BUY / SELL / HOLD or any other recommendation state.

### ME-GV06 reserved for buy-zone and position-management explanation

No factor taxonomy item authorizes entry levels, buy-under levels, breakout triggers, stops, target prices, or position-management instructions.

## Non-goals

ME-GV02 does not implement or authorize:

* runtime code;
* tests;
* provider calls;
* live market data;
* source acquisition;
* staging/classifier semantic changes;
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
* numeric scoring;
* weighting;
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

## Acceptance criteria

ME-GV02 is complete when:

* factor families are explicit;
* factor states are explicit;
* minimum evidence requirements are defined per factor;
* missing/stale/unprovenanced/non-consumable/malformed evidence behavior is defined;
* company-profile-only evidence cannot become a numeric factor evaluation;
* portfolio fit remains blocked without approved portfolio context;
* scoring remains deferred to ME-GV04;
* recommendation mapping remains deferred to ME-GV05;
* buy-zone and position-management explanation remain deferred to ME-GV06;
* ME-GV03 is documented as the next sprint.

## Outcome

ME-GV02 completes the semantic factor contract required before Governor runtime implementation. The next sprint is ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold.
