# Operational Sprint Developer Specification - Portfolio Source Contract Repair

## 1. Status and Scope

This document is a documentation-only developer specification for the Operational Sprint - Portfolio Source Contract Repair.

This document does not implement:

- code changes
- tests
- data changes
- CSV files
- generated artifacts
- provider integration
- API calls
- scraping
- credentials or secrets
- runtime behavior changes
- Decision Engine logic
- Reporting logic
- Telegram logic
- scanner logic
- Fundamental Layer logic
- portfolio CSV edits

No sprint is closed or certified complete by this document.

Implementation requires a later explicit Codex implementation prompt after this developer specification is reviewed and approved.

## 2. Background

This developer specification follows `docs/sprints/operational_sprint_portfolio_source_contract_repair_planning.md`.

Relevant backlog item:

- `BL-0011 - Define and repair authoritative active portfolio source`

During validation of the Approved Fundamental Data Source MVP, focused Fundamental Layer tests passed, but the full test suite failed on an existing portfolio source contract test:

```text
tests/portfolio/test_portfolio_source_contract.py::test_active_positions_are_rebuildable_from_transaction_source
```

Failure class:

```text
AssertionError: DataFrame.iloc[:, 3] (column name="last_price") are different
left:  [nan, nan, nan, nan]
right: [1010.215, 163.13, 100.29, 58.27]
```

The same failure reproduced on clean `main`, so it was classified as a pre-existing and out-of-scope portfolio source contract failure, not a regression caused by the Fundamental MVP.

This must be handled as a separate portfolio source contract repair because the failure concerns the boundary between transaction-derived active position state and descriptive market enrichment. That boundary belongs to the portfolio source contract and must not be repaired through Fundamental Layer, Decision Engine, Reporting, Telegram, scanner, or generated data changes.

## 3. Governance Boundaries

Portfolio source contract repair must not create or imply:

- buy advice
- sell advice
- allocation authority
- tradeability
- urgency
- conviction
- ranking authority
- scoring authority
- hidden filtering
- Decision Engine bypass
- Reporting decision semantics
- Telegram decision semantics

Portfolio source files and portfolio-derived artifacts may describe transaction-supported portfolio state, quantities, cost basis, descriptive metadata, and explicitly governed enrichment only.

Any portfolio action remains controlled by `scripts/core/decision_engine.py`.

Reporting and Telegram remain communication-only.

## 4. Contract Decision To Implement Later

The intended contract decision for the later implementation is:

- authoritative transaction-derived position fields must be rebuildable from the transaction source;
- descriptive market enrichment fields are not necessarily rebuildable from the transaction source;
- `last_price` belongs to descriptive market enrichment unless a separately approved price source makes it authoritative;
- `last_price` may be nullable when no approved price source exists;
- `last_price` must not block transaction-source rebuild validation unless it is explicitly classified as authoritative source data.

The later implementation must not make stale numeric `last_price` values authoritative by test expectation alone.

The later implementation must not edit portfolio CSV files simply to make numeric `last_price` values match.

## 5. Field Classification

The later implementation must classify portfolio source fields by source authority before changing test expectations.

| Field category | Candidate fields | Contract treatment |
|---|---|---|
| Transaction-derived / authoritative | `ticker`, `quantity`, average cost fields, cost basis fields, transaction-derived position value fields, realized transaction-derived fields | Must be rebuildable from the authoritative transaction source. These fields define the strict transaction rebuild contract. |
| Descriptive enrichment | `last_price`, market value, unrealized gain/loss, valuation timestamp, price source, price freshness metadata | Optional unless an approved enrichment source exists. These fields may be nullable and should be validated separately from strict transaction-derived equality. |
| Decision-authority fields | portfolio action, allocation decision, buy decision, sell decision, trim recommendation, add recommendation | Must remain outside portfolio source contract repair unless routed through Decision Engine governance. These fields must not be created or authorized by portfolio source tests. |

The table is a developer-spec classification guide. The later implementation must inspect the actual repository columns before modifying tests or helpers.

## 6. Expected Test Repair Strategy

The recommended repair strategy is to update the rebuild equivalence test so that it compares only authoritative transaction-derived fields.

Expected approach:

1. Identify the authoritative transaction-derived columns in `portfolio_positions.csv` and the rebuilt transaction output.
2. Compare those fields for strict rebuild equivalence.
3. Validate `last_price` separately as optional descriptive enrichment.
4. Allow `last_price` to be null when no approved price source exists.
5. Treat non-null `last_price` as enrichment evidence, not transaction authority.
6. Preserve row count and ticker identity checks.
7. Preserve duplicate ticker ambiguity checks.

The later implementation must not simply overwrite active portfolio CSVs with rebuilt numeric `last_price` values.

The later implementation must not treat `last_price` as transaction-derived source-of-truth unless a future approved price-source contract explicitly authorizes that meaning.

## 7. Files Codex Is Expected To Inspect Before Implementation

Before implementing, Codex should inspect:

- `tests/portfolio/test_portfolio_source_contract.py`
- portfolio source CSVs under `data/portfolio/`
- transaction source CSVs under `data/portfolio/`
- any portfolio source validation or rebuild utility under `scripts/`
- any portfolio intelligence builder if relevant
- `docs/sprints/operational_sprint_portfolio_source_contract_repair_planning.md`
- `docs/sprints/project_backlog.md`
- `.gitignore`

This developer specification describes expected touchpoints only. It does not authorize runtime file changes.

## 8. Future Implementation Scope

Likely allowed in a later implementation prompt:

- `tests/portfolio/test_portfolio_source_contract.py`
- minimal portfolio source contract helper logic if needed
- targeted portfolio contract comments or documentation comments if needed

Not allowed unless explicitly approved:

- portfolio CSV edits
- transaction CSV edits
- Decision Engine logic
- Reporting logic
- Telegram logic
- scanner logic
- Fundamental Layer logic
- provider/API integration
- price source integration
- generated processed CSV commits
- workflow changes

If implementation pressure suggests editing portfolio CSV data or adding a price source, Codex must stop and report the governance blocker instead of proceeding.

## 9. Required Tests For Future Implementation

The later implementation must add or update tests for:

- active positions rebuild from transaction-derived fields;
- `last_price` excluded from strict transaction-derived equality;
- nullable `last_price` accepted when no approved price source exists;
- non-null `last_price` treated as enrichment, not transaction authority;
- no row loss;
- no duplicate ticker ambiguity;
- no portfolio action semantics outside Decision Engine;
- no Reporting decision semantics;
- no Telegram decision semantics;
- full portfolio source contract test passes.

Tests must not introduce buy advice, sell advice, allocation, tradeability, urgency, conviction, eligibility, hidden filtering, or Decision Engine bypass semantics.

## 10. Validation Plan For Future Implementation

Future implementation should run:

```bash
git diff --check
PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio/test_portfolio_source_contract.py
PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio
PYTHONPATH=. .venv/bin/python -m pytest
git status --short
```

Generated artifacts must not be committed unless explicitly approved.

If the full test suite fails on an unrelated issue, the implementation must reproduce the failure on clean `main` before treating it as an out-of-scope validation exception.

## 11. Acceptance Criteria

The later implementation is acceptable only if:

- the failing portfolio source contract test is repaired without CSV guesswork;
- transaction-derived fields remain rebuildable from the transaction source;
- `last_price` semantics are explicitly tested or documented;
- nullable `last_price` is accepted when no approved price source exists;
- no portfolio CSVs are edited unless explicitly approved;
- no Decision Engine logic is changed;
- no Reporting logic is changed;
- no Telegram logic is changed;
- no scanner logic is changed;
- no Fundamental Layer logic is changed;
- focused portfolio tests pass;
- the full test suite passes, unless another clean-main failure is formally reproduced and documented;
- no generated artifacts are improperly committed.

## 12. Implementation Prompt Skeleton

Future use only. Do not execute this prompt until this developer specification is reviewed and approved.

```text
You are operating inside the institutional `market-scanner` repository.

Task:
Implement the Operational Sprint - Portfolio Source Contract Repair.

Follow the approved developer specification:

- `docs/sprints/operational_sprint_portfolio_source_contract_repair_developer_spec.md`

This is a governed implementation task.

Do NOT modify:
- portfolio CSV files;
- transaction CSV files;
- Decision Engine logic;
- Reporting logic;
- Telegram logic;
- scanner logic;
- Fundamental Layer logic;
- provider/API integration;
- price source integration;
- generated processed CSV files;
- GitHub workflows.

Goal:
Repair the portfolio source contract test so transaction-derived active position fields are rebuildable from the transaction source, while `last_price` remains optional descriptive market enrichment unless an approved price source makes it authoritative.

Required behavior:
- compare strict rebuild equality only for authoritative transaction-derived fields;
- validate `last_price` separately as optional enrichment;
- allow nullable `last_price` when no approved price source exists;
- do not edit CSV files to force a match;
- do not create allocation, tradeability, urgency, conviction, ranking, scoring, hidden filtering, or Decision Engine bypass semantics.

Before editing:
- run `git status`;
- inspect `tests/portfolio/test_portfolio_source_contract.py`;
- inspect relevant portfolio source and transaction CSV files under `data/portfolio/`;
- inspect relevant portfolio rebuild utilities under `scripts/`;
- inspect `.gitignore`.

After implementation, run:
- `git diff --check`;
- `PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio/test_portfolio_source_contract.py`;
- `PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio`;
- `PYTHONPATH=. .venv/bin/python -m pytest`;
- `git status --short`.

Do not commit, push, merge, or open a PR until human review approves the diff.
```

## 13. Backlog Impact Assessment

Existing backlog item `BL-0011` remains sufficient.

`BL-0011` already captures the need to define and repair the authoritative active portfolio source and repair stale or incomplete active portfolio CSV state.

Backlog impact assessment:
- No new backlog items identified.

## 14. Recommended Next Step

Review and merge this developer specification.

After review approval, launch a separate Codex implementation sprint for the Portfolio Source Contract Repair.
