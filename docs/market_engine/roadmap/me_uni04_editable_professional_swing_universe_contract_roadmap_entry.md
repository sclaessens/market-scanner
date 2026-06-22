# ME-UNI04 Roadmap Entry - Editable Professional Swing Universe contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI04

## Placement

ME-UNI04 follows ME-SR04.

ME-SR04 resolved the remaining HO blocker for the current canonical SEC CompanyFacts cached-source universe by moving HO to `manual_review_only`. ME-UNI04 starts the next universe-governance phase before output, reporting, delivery, or candidate-classification work.

## Contract defined

```text
market-engine-editable-professional-swing-universe-v1
```

## Roadmap decision

The Professional Swing Universe is introduced as an editable candidate universe, not as a replacement for the canonical execution universe.

It supports future import, normalization, source-support classification, clean cached-source scanning, readable operator output, and non-actionable candidate classification.

It does not grant source refresh, provider, delivery, portfolio, watchlist, Decision Engine, action, allocation, ranking, scoring, target-price, position-sizing, tradeability or execution authority.

## Files introduced

```text
docs/market_engine/ticker_universe/me_uni04_editable_professional_swing_universe_contract.md
docs/market_engine/audits/me_uni04_editable_professional_swing_universe_contract_audit.md
docs/market_engine/backlog/me_uni04_editable_professional_swing_universe_contract_backlog_entry.md
docs/market_engine/roadmap/me_uni04_editable_professional_swing_universe_contract_roadmap_entry.md
```

## Next sprint

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
```

ME-UNI05 may create the first editable seed CSV under:

```text
data/market_engine/ticker_universe/professional_swing_universe/
```

It must remain documentation/data-import only unless explicitly re-scoped.

## Planned sequence

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```
