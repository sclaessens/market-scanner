# ME-GV02 - Governor Factor Taxonomy Roadmap Entry

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

## Purpose

ME-GV02 defines the semantic factor layer between the ME-GV01 investment-evaluation contract and the future ME-GV03 runtime scaffold.

## Roadmap sequence

```text
ME-RUN29 - Expanded generic coverage classification evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
  -> ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
  -> ME-GV04 - Implement factor scoring from approved analysis evidence
  -> ME-GV05 - Implement recommendation-state mapping under approved boundary
  -> ME-GV06 - Implement buy-zone and position-management explanation contract
```

## Taxonomy delivered

ME-GV02 defines nine factor families:

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

It also defines canonical factor states:

```text
not_started
blocked
unavailable
insufficient_evidence
partial
qualitative_only
evaluable
```

The taxonomy explicitly separates evidence presence, evidence sufficiency, evaluability, scoring eligibility, recommendation mapping, and actionability.

## Sequencing rule after ME-GV02

ME-GV03 is next.

ME-GV03 may implement factor-state evaluation and evidence packaging only. It must not introduce numeric scores, factor weights, weighted aggregation, ranking, recommendation-state mapping, buy zones, position sizing, allocation, or Decision Engine authority.

ME-GV04 remains the first planned scoring sprint.

## Dependencies preserved

* `portfolio_fit` remains blocked without approved portfolio context.
* recommendation-state mapping remains deferred to ME-GV05.
* buy-zone and position-management explanation remain deferred to ME-GV06.
* Dispatch Station work remains downstream of Governor contract and implementation work.

## Non-goals preserved

ME-GV02 adds no runtime behavior, tests, provider calls, live market data, source acquisition, staging/classifier changes, Analyzer changes, recommendation semantics, delivery behavior, portfolio/watchlist mutation, broker behavior, numeric scoring, weighting, ranking, BUY / SELL / HOLD semantics, target prices, allocation, position sizing, order generation, execution advice, or Decision Engine authority.

## Next sprint

```text
ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
```
