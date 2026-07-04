# ME-GV01 - The Governor Investment Evaluation Contract Roadmap Entry

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

## Purpose

ME-GV01 opens the Governor sequence by defining the contract for future investment-evaluation output.

It follows the completed Refinery evidence chain:

```text
ME-SA12 - Generic supported-universe cached-source coverage contract
  -> ME-SA13 - Generic coverage classifier
  -> ME-SA14 - Staging-validation adapter
  -> ME-RUN29 - Expanded generic coverage classification evidence
```

ME-RUN29 proved that coverage/readiness evidence can be classified reproducibly while reserved actionable and Decision Engine-ready states remain zero. ME-GV01 uses that as the planning boundary for The Governor, but it does not implement Governor runtime behavior.

## Roadmap sequence

```text
ME-RUN29 - Expanded generic coverage classification evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
  -> ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
  -> ME-GV04 - Implement factor scoring from approved analysis evidence
  -> ME-GV05 - Implement recommendation-state mapping under approved boundary
  -> ME-GV06 - Implement buy-zone and position-management explanation contract
  -> ME-DS01 - Define Dispatch Station output contract for Governor reports
```

## Contract delivered

ME-GV01 defines:

```text
market-engine-governor-investment-evaluation-v1
```

The contract covers:

* approved future input families;
* top-level output shape;
* evaluation states;
* reserved future authority states;
* evidence readiness gates;
* fail-closed behavior;
* factor section reservation without taxonomy;
* recommendation-state boundary;
* buy-zone and position-management boundary;
* portfolio-fit boundary;
* authority non-goals.

## Sequencing rule after ME-GV01

ME-GV02 must come next.

Reason: ME-GV01 reserves factor output but intentionally does not define factor taxonomy, factor states, scales, weights, or evidence thresholds. A Governor scaffold or scoring implementation before ME-GV02 would create undefined factor semantics.

## Non-goals preserved

ME-GV01 adds no runtime behavior, tests, provider calls, live market data calls, source acquisition, staging validator changes, classifier changes, Analyzer changes, Recommendation Review changes, Portfolio Review changes, Dispatch Station behavior, delivery behavior, portfolio/watchlist mutation, broker behavior, scoring, ranking, BUY / SELL / HOLD semantics, allocation, target prices, position sizing, order generation, execution advice, or Decision Engine authority.

## Next sprint

```text
ME-GV02 - Define Governor factor taxonomy and evidence requirements
```
