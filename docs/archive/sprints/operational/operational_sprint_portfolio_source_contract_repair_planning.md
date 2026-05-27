# Operational Sprint Planning — Portfolio Source Contract Repair

## 1. Status and Scope

This document is a documentation-only PM / Functional Analyst / Technical Analyst / Governance planning artifact for the portfolio source contract repair workstream.

This is a ChatGPT-owned planning step before any Codex implementation work.

This document prepares the scope, governance boundaries, contract decision, implementation handoff, and backlog reconciliation for a future portfolio source contract repair sprint.

This document does not implement:

- code;
- tests;
- CSV files;
- generated artifacts;
- reports;
- workflows;
- provider integration;
- provider/API calls;
- scraping;
- credentials or secrets;
- runtime orchestration;
- daily ingestion;
- backtesting code;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- scanner changes;
- Fundamental Layer changes;
- portfolio file edits;
- watchlist file edits;
- runtime behavior changes.

No sprint is closed or certified complete by this document.

No portfolio repair is authorized by this document.

No CSV edit is authorized by this document.

## 2. Background

During the Approved Fundamental Data Source MVP implementation validation, the focused Fundamental Layer tests passed, but the full test suite failed on an existing portfolio source contract test.

Failing test:

```text
 tests/portfolio/test_portfolio_source_contract.py::test_active_positions_are_rebuildable_from_transaction_source
```

Failure class:

```text
AssertionError: DataFrame.iloc[:, 3] (column name="last_price") are different
left:  [nan, nan, nan, nan]
right: [1010.215, 163.13, 100.29, 58.27]
```

The failure reproduced on clean `main`, so it was classified as pre-existing and out-of-scope for the Fundamental MVP implementation.

The relevant backlog item is:

- `BL-0011 — Define and repair authoritative active portfolio source`.

## 3. Objective

The objective of the next governed workstream is to repair the portfolio source contract failure without mixing it with Fundamental Layer work.

The workstream must decide whether `last_price` in active positions is:

1. rebuildable from transaction source data;
2. derived from a separate approved market price source;
3. an optional nullable descriptive field;
4. a field that should be excluded from strict rebuild equivalence;
5. or a field whose expected values belong to a separate portfolio valuation artifact.

The repair must preserve deterministic portfolio state and must not introduce allocation authority outside the Decision Engine.

## 4. Governance Boundaries

Portfolio source contract repair must not create or imply:

- new allocation authority;
- buy/sell advice;
- tradeability;
- urgency;
- conviction;
- ranking authority;
- scoring authority;
- hidden filtering;
- Decision Engine bypass;
- Reporting recommendation semantics;
- Telegram recommendation semantics.

Portfolio files and portfolio-derived artifacts may describe portfolio state, cost basis, quantities, and source-supported metadata only.

Any action decision remains under Decision Engine authority.

Reporting and Telegram remain communication-only.

## 5. Contract Problem Statement

The failing test expects active positions to be rebuildable from transaction source data.

However, the observed mismatch is in `last_price`:

- actual committed active positions contain `last_price = NaN` values;
- rebuilt active positions contain numeric `last_price` values;
- the test compares the full active positions output and therefore fails.

This indicates that the current source contract is ambiguous about whether `last_price` is part of the authoritative transaction-derived portfolio state or a volatile market-price enrichment field.

## 6. Recommended Contract Decision

Recommended decision: separate authoritative position state from volatile market-price enrichment.

`last_price` should not be treated as a strict transaction-rebuild source-of-truth field unless the transaction source explicitly contains authoritative latest price data.

Preferred contract:

| Field category | Examples | Contract treatment |
|---|---|---|
| Authoritative position state | ticker, quantity, average cost, transactions-derived position values | Must be rebuildable from transaction source. |
| Descriptive market enrichment | last_price, market value, unrealized gain/loss, valuation timestamp | Must come from an approved price or valuation source, or remain nullable. |
| Decision authority | portfolio action, allocation decision, buy/sell decision | Must remain Decision Engine-only. |

Under this contract, `last_price` should either:

- be excluded from strict transaction-source rebuild equality; or
- be rebuilt only if the transaction source explicitly defines it as a source-supported field; or
- be validated through a separate approved valuation source contract.

## 7. Candidate Repair Options

| Option | Description | Pros | Cons | Recommendation |
|---|---|---|---|---|
| Exclude `last_price` from strict rebuild equality | Test authoritative transaction-derived fields separately from valuation fields. | Aligns source truth with transaction semantics; avoids forcing volatile price data into transaction source. | Requires test update and clear contract documentation. | Recommended. |
| Require committed active positions to contain rebuilt `last_price` | Edit source CSV to match rebuilt numeric values. | Makes current test pass. | Risks treating stale price values as authoritative portfolio source; may create data drift. | Not recommended without price-source governance. |
| Remove `last_price` from active positions entirely | Keep active positions purely transaction-derived. | Very clear boundary. | May break existing downstream descriptive displays if they expect the column. | Consider only with broader contract review. |
| Add approved price source contract | Treat `last_price` as valuation enrichment from a separate source. | Long-term correct for valuation. | Requires additional source governance and implementation scope. | Future work, not immediate repair. |
| Make `last_price` nullable and compare null-safe | Accept nullable committed values when no price source exists. | Minimal contract disruption. | Still requires test semantics to distinguish authoritative and optional fields. | Acceptable if documented. |

## 8. Recommended Sprint Shape

Recommended sprint name:

`Operational Sprint — Portfolio Source Contract Repair`

Recommended scope:

1. Inspect the current portfolio source contract test.
2. Inspect active portfolio source CSVs and transaction source CSVs.
3. Identify authoritative transaction-derived fields.
4. Identify optional valuation/enrichment fields.
5. Update test expectations so transaction rebuild equivalence applies only to authoritative transaction-derived fields.
6. Preserve `last_price` as optional nullable enrichment unless a price source is explicitly approved.
7. Add or update contract comments/docstrings if needed.
8. Confirm no Decision Engine, Reporting, Telegram, or portfolio action semantics are changed.
9. Run focused portfolio tests.
10. Run full test suite.

## 9. Files Codex Should Inspect Before Implementation

Codex should inspect, but not assume changes are required in all files:

- `tests/portfolio/test_portfolio_source_contract.py`
- portfolio source and transaction CSV files under `data/portfolio/`
- any portfolio source builder or validation utilities under `scripts/` if present
- `docs/sprints/project_backlog.md`
- `docs/sprints/operational_sprint_portfolio_source_contract_repair_planning.md`
- `.gitignore`
- any active portfolio contract documentation if present

## 10. Future Implementation Scope

Likely allowed in a later Codex implementation prompt:

- focused portfolio source contract tests;
- portfolio contract documentation comments if needed;
- minimal portfolio validation helper adjustment if the current logic incorrectly treats valuation enrichment as transaction-derived authority.

Likely not allowed unless explicitly approved:

- portfolio CSV edits;
- transaction CSV edits;
- Decision Engine logic;
- Reporting logic;
- Telegram logic;
- scanner logic;
- Fundamental Layer logic;
- provider/API integration;
- price source integration;
- generated processed CSV commits;
- workflow changes.

## 11. Required Tests for Future Implementation

Future implementation should include or preserve tests for:

- active positions rebuild from transaction-derived fields;
- exclusion or separate handling of `last_price` from strict transaction-source equality;
- nullable `last_price` when no approved price source exists;
- no row loss in active positions;
- no duplicate ticker ambiguity;
- no portfolio action semantics outside Decision Engine;
- no Reporting or Telegram decision semantics;
- full portfolio source contract validation.

## 12. Acceptance Criteria

A future implementation can be accepted only if:

- the failing portfolio source contract test is repaired without CSV guesswork;
- `last_price` semantics are explicitly documented or tested;
- authoritative transaction-derived fields remain rebuildable;
- optional valuation/enrichment fields do not block transaction-source rebuild validation;
- no portfolio CSVs are edited unless explicitly approved;
- no Decision Engine, Reporting, Telegram, scanner, or Fundamental Layer logic is changed;
- focused portfolio tests pass;
- the full test suite passes, unless a separate clean-main failure is formally reproduced and documented;
- no generated artifacts are improperly committed.

## 13. Backlog Impact Assessment

Existing backlog item `BL-0011` remains sufficient.

`BL-0011` already captures the need to define and repair the authoritative active portfolio source and repair stale or incomplete portfolio CSV state.

This document prepares that backlog item for future sprint planning but does not identify additional deferred work.

Backlog impact assessment:
- No new backlog items identified.

## 14. Recommended Next Step

The recommended next step is a Codex developer-spec preparation prompt for the Portfolio Source Contract Repair sprint.

That prompt should ask Codex to create an implementation plan and developer specification only, not to implement code yet.

After that, a separate Codex implementation sprint can be launched if the developer specification is reviewed and approved.
