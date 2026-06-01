# Strategy, Logic, Calculation, and Ticker-Category Rationalization

Status: ACTIVE ANALYSIS
Backlog driver: BL-0022

## 1. Purpose

This document rationalizes the market-scanner project's trading logic, calculation placement, ticker-category direction, and future-proof review process before further implementation work.

The goal is to prevent new logic from being forced into old files, prevent useful existing logic from being removed accidentally, and make future implementation simpler, clearer, and easier to maintain.

This document does not authorize implementation, code changes, tests, data changes, generated artifact updates, provider/API usage, scraping, pipeline execution, allocation changes, or Decision Engine changes.

## 2. Core Finding

The project should not treat the current runtime as untouchable. The previous governance constraints were useful to prevent hidden allocation semantics, but they must not become a reason to preserve unnecessary complexity.

The correct next project phase is:

```text
preserve certified authority boundaries
+ review strategy and calculations deliberately
+ simplify logic placement
+ replace outdated structures where needed
+ implement only after scoped approval and tests
```

The project may change logic and code in future implementation sprints, but only after documenting:

- what logic is being changed;
- why the old logic is insufficient;
- where the new logic belongs;
- which calculations are affected;
- which artifacts change;
- which tests protect the change;
- how Decision Engine authority remains intact.

## 3. Strategic Direction

The market-scanner should evolve from a generic linear pipeline into a category-aware decision-support platform.

The certified pipeline remains valid:

```text
scanner
-> validation
-> context
-> fundamental quality / metrics / analysis
-> timing state
-> portfolio intelligence
-> decision engine
-> reporting
```

But the logic inside that pipeline should become more explicit about business type, sector sensitivity, calculation relevance, and evidence quality.

The key improvement is not more complexity. The key improvement is better placement of complexity.

## 4. What Should Stay Stable

The following principles should remain stable:

- upstream layers classify and enrich;
- Decision Engine is the only allocation and final-action authority;
- Reporting communicates only;
- generated artifacts are evidence, not doctrine;
- source-supported data matters;
- row identity and deterministic behavior matter;
- useful logic should be preserved before files are moved or deleted;
- formulas and calculations should be documented before implementation.

## 5. What Should Be Reconsidered

The following areas need rationalization before broad implementation:

| Area | Current concern | Direction |
|---|---|---|
| Generic stock treatment | Different sectors and business types react differently to signals. | Add a descriptive ticker-category model. |
| Fundamental logic | Current MVP mixes source presence, sufficiency, metadata, and metrics too early. | Separate raw history, metrics, quality, and analysis. |
| Calculation placement | Some calculations or helper logic may be spread across files. | Use the calculation registry as the placement authority. |
| Runtime code surface | Some files are active, some are wrappers, some are legacy candidates. | Migrate useful logic before cleanup. |
| Reporting and Telegram | Communication must stay neutral but operator visibility is important. | Add pass-through only after governed contract design. |
| Research and feedback loops | Learning from predictions is valuable but risky. | Keep observational until governed into Decision Engine consumption. |

## 6. Ticker-Category Logic Model

Ticker-category logic should be descriptive upstream. It may determine which calculations are relevant, which review notes matter, or how analysis should be interpreted. It must not create allocation decisions outside the Decision Engine.

### 6.1 Candidate category dimensions

A ticker may later receive one or more descriptive category dimensions:

| Dimension | Examples | Purpose |
|---|---|---|
| Sector | Technology, Energy, Financials, Healthcare, Industrials, Retail. | Broad economic and reporting context. |
| Business model | Software, semiconductors, commodity producer, bank, retailer, biotech. | Determines which metrics are meaningful. |
| Cycle sensitivity | Cyclical, defensive, rate-sensitive, commodity-sensitive. | Helps interpret timing and context. |
| Growth profile | Early growth, durable growth, mature compounder, turnaround. | Helps interpret growth and margin metrics. |
| Financial maturity | Profitable, loss-making, cash-flow positive, cash-burn stage. | Helps prevent wrong metric comparisons. |
| Portfolio role | Core holding, tactical position, watchlist candidate, research-only. | Descriptive portfolio context only. |

### 6.2 Candidate category examples

| Category | Useful logic focus | Caution |
|---|---|---|
| Semiconductors | Revenue growth, gross margin, operating margin, sector leadership, cycle sensitivity. | Cyclical swings can make short-term growth misleading. |
| Software | Revenue durability, margin expansion, free cash flow, recurring revenue if source-supported. | Recurring revenue requires explicit source-data contract. |
| Retail | Gross margin, operating margin, inventory/consumer context if source-supported. | Margin levels differ materially from software or semiconductors. |
| Energy | Free cash flow, debt, commodity sensitivity, cycle context. | Commodity price exposure must be treated as context, not hidden signal. |
| Financials | Balance-sheet structure, rate sensitivity, credit-cycle context. | Standard industrial debt/equity logic may not apply directly. |
| Biotech / healthcare innovation | Cash runway, catalysts, pipeline stage, revenue maturity. | Requires a separate source-data model before use. |
| Defensive compounders | Stability, drawdown behavior, cash-flow consistency, margin durability. | Defensive quality does not equal automatic buy authority. |
| Cyclical growth | Revenue trend, margin trend, sector cycle, relative strength. | Timing and context may matter more than static quality. |

### 6.3 Required future controls

Before implementation, ticker-category logic requires:

- source-supported category assignment rules;
- a category registry or schema;
- calculation relevance mapping;
- tests for category assignment;
- tests that category assignment does not create hidden allocation;
- Decision Engine boundary review if category affects final decisions.

## 7. Calculation Placement Rationalization

The calculation registry is the starting point for all calculation placement decisions.

Use this placement rule:

| Logic type | Correct layer |
|---|---|
| Price/volume transformations | Scanner or market data utility. |
| Setup/structure classification | Validation Layer. |
| Entry quality diagnostics | Entry Quality / Validation. |
| Market leadership and relative strength | Context Layer. |
| Raw reported financial facts | Raw Fundamentals History. |
| Deterministic financial formulas | Fundamental Metrics. |
| Data completeness/readiness | Fundamental Quality. |
| Business interpretation | Fundamental Analysis. |
| Timing condition | Timing State. |
| Portfolio presence/exposure | Portfolio Intelligence. |
| Final action/allocation | Decision Engine only. |
| Grouping, truncation, communication | Reporting only. |

## 8. Current Logic Assessment

| Logic family | Current assessment | Recommendation |
|---|---|---|
| Validation / structure classification | Useful and aligned if descriptive only. | Keep, but ensure diagnostics and entry-quality helpers are well placed. |
| Entry quality metrics | Useful, but should remain descriptive and documented. | Keep and document calculation details before expansion. |
| Context / relative strength | Strategically useful. | Keep; consider category-aware interpretation later. |
| Fundamental quality MVP | Useful as compatibility surface, but overloaded. | Preserve wrapper; implement raw history, metrics, quality, analysis separation behind/alongside it. |
| Timing state | Useful and should remain descriptive. | Keep protected until approved scope. |
| Portfolio intelligence | Useful but portfolio-only holdings remain unresolved. | Keep; revisit after portfolio contract review. |
| Decision Engine | Correct authority location. | Keep protected; future logic changes require explicit Decision Engine spec. |
| Reporting / Telegram | Improved after reporting sprint, still has wrappers. | Keep communication-only; future pass-through fields require contract. |
| Watchlist legacy | Likely historical or legacy. | Do not delete until useful logic and references are checked. |
| Source-data prefill helpers | Potentially useful but risky. | Keep out of active runtime until source-data governance is approved. |

## 9. Logic Simplification Rules

Future implementation should follow these rules:

1. Prefer smaller focused builders over overloaded layers.
2. Keep compatibility wrappers only when they protect downstream contracts.
3. Move useful logic before deleting files.
4. Do not preserve old files just because they exist.
5. Do not add new logic to a file if the file is already overloaded.
6. Do not duplicate formulas across files.
7. Document calculations before implementing them.
8. Keep research outputs observational until promoted through governance.
9. Keep operator visibility separate from decision authority.
10. Keep sector/category logic descriptive until Decision Engine consumption is explicitly designed.

## 10. Future-Proof Review Cycle

The project should regularly review logic and code placement.

A review is required:

- before every implementation sprint that changes logic;
- after every implementation sprint closeout;
- before deleting or moving files with analytical logic;
- before adding sector/category-specific behavior;
- before introducing a new calculation family;
- after misleading or failed signal behavior is observed;
- when backlog pressure repeats around the same layer.

The review should classify each logic element as:

| Classification | Meaning |
|---|---|
| `KEEP` | Logic remains valid and correctly placed. |
| `KEEP_BUT_DOCUMENT` | Logic remains useful but needs better documentation/tests. |
| `MOVE` | Logic is useful but belongs elsewhere. |
| `SPLIT` | Logic is too broad and should become smaller units. |
| `REPLACE` | Purpose is valid, implementation is too complex or flawed. |
| `REMOVE_AFTER_MIGRATION` | Obsolete after useful logic is preserved elsewhere. |
| `REQUIRES_REVIEW` | More evidence is needed. |

## 11. Implementation Implications

This analysis changes the recommended sequencing.

Do not start broad Sprint E implementation until at least one of these paths is explicitly accepted:

### Path A — Logic-first path

1. Use this document as the logic baseline.
2. Update calculation registry entries as implementation needs become concrete.
3. Write narrow developer specs for selected logic areas.
4. Implement only the selected scoped area.

### Path B — Controlled fundamentals implementation path

1. Start Sprint E1 for raw history validation only.
2. Keep all category-aware logic out of E1.
3. Keep `build_fundamental_layer.py` as compatibility surface.
4. Do not touch Decision Engine, Reporting, Timing, Portfolio Intelligence, or Telegram.
5. Return to strategy/category logic before expanding metrics into decision use.

### Path C — Cleanup-first path

1. Execute only a tiny cleanup batch from BL-0023.
2. Do not delete analytical logic.
3. Convert wrappers or move obvious diagnostics only after reference checks.
4. Return to fundamentals implementation after cleanup risk is reduced.

Recommended next implementation-adjacent path:

```text
Path A first, then a narrow Sprint E1 or BL-0023 cleanup depending on Product Owner priority.
```

## 12. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0022 already covers this rationalization work. BL-0023 already covers narrow Python cleanup scope. BL-0015 already covers controlled fundamentals implementation.

## 13. Validation

Documentation-only validation for this change should confirm:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run unless explicitly needed.

## 14. Sprint P3 / C.5 Closeout Conclusion

Strategy, logic, calculation placement, and ticker-category rationalization are now documented at active-analysis level.

Recommended next decision:

```text
Choose between:
1. a narrow Sprint E1 raw-history implementation;
2. a narrow BL-0023 cleanup implementation scope;
3. a more detailed ticker-category model specification before implementation.
```

Do not combine these into one sprint. Each should be separately scoped and capacity-checked.