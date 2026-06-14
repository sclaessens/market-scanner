# RESET-4 — V2 Data Contracts and Fixtures Closeout

## 1. Purpose

RESET-4 defines the minimal v2 data contract and fixture baseline needed before RESET-5 can begin pipeline-core work.

This reset does not implement runtime pipeline behavior. It establishes approved fixture classifications, schema expectations, and tests that validate the fixture surface.

## 2. Files Created or Changed

Created:

- `docs/active/data_contracts.md`
- `data/fixtures/v2/universe_candidates.csv`
- `data/fixtures/v2/portfolio_transactions.csv`
- `data/fixtures/v2/source_data_readiness.csv`
- `src/market_scanner/shared/data_contracts.py`
- `tests/contract/test_v2_data_contracts.py`
- `tests/fixtures/test_v2_fixture_contracts.py`
- `docs/resets/reset_4_v2_data_contracts_and_fixtures_closeout.md`

Changed:

- None.

## 3. Fixture Files Created

RESET-4 created three small synthetic fixtures:

- `data/fixtures/v2/universe_candidates.csv`
- `data/fixtures/v2/portfolio_transactions.csv`
- `data/fixtures/v2/source_data_readiness.csv`

These fixtures are synthetic, deterministic, tracked intentionally, and independent from generated runtime outputs.

## 4. Tests Added

Added contract and fixture tests for:

- approved fixture contract registration;
- v2 fixture path boundaries;
- fixture existence;
- required fixture columns;
- non-empty fixture files;
- absence of final-decision terms in fixture values;
- explicit source-data missing-value handling;
- no filesystem side effects while reading fixtures.

## 5. Tests and Validation

Requested validation:

```bash
.venv/bin/python -m pytest
git diff --check
git status
grep -R "BUY" src tests data/fixtures docs/active docs/resets || true
grep -R "SELL" src tests data/fixtures docs/active docs/resets || true
grep -R "tradeable" src tests data/fixtures docs/active docs/resets || true
```

Results are recorded in the RESET-4 pull request summary.

Results:

- `.venv/bin/python -m pytest` passed: 401 passed.
- `git diff --check` passed.
- `git status` showed only RESET-4 files as untracked before staging.
- Governance grep checks returned matches in existing legacy tests, active doctrine documents, and reset validation command text.
- Governance grep checks returned no matches from RESET-4 fixture CSV values.
- Governance grep checks returned no implementation of allocation, execution, arbitration, final-action, or tradeability behavior outside the Decision Engine.

## 6. Scope Confirmation

No old Python files were modified.

No old Python files were copied into the v2 implementation.

No old generated CSV or data files were reused as v2 source-of-truth.

No existing legacy data files were modified.

No reports, workflows, generated business outputs, or old runtime modules were changed.

No SEC diagnostics, live provider calls, Telegram behavior, or production pipeline runs were performed.

No scanner, validation, fundamentals, timing, portfolio, Decision Engine, or reporting behavior was implemented.

## 7. Governance Validation Notes

Any governance grep matches in canonical documentation are doctrine or forbidden-behavior descriptions, not implementation semantics.

RESET-4 introduces no implementation of allocation, execution, arbitration, final-action, or tradeability behavior outside the Decision Engine.

## 8. Backlog Impact Assessment

Backlog impact assessment:

- RESET-4 satisfies the v2 data contract and fixture baseline step needed before RESET-5.
- No business logic backlog items were implemented.
- Future backlog work should define minimal pipeline-core validation using only approved contracts and fixtures.

## 9. Recommended Next Action

Recommended next action: execute RESET-5 to build minimal pipeline-core scaffolding and contract validation without implementing business-layer decision behavior.
