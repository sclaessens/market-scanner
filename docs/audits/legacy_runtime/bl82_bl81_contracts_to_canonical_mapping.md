# BL82 — Map extracted BL81 contracts to existing canonical fundamentals modules

Status: COMPLETED

## Purpose

BL82 maps the contracts extracted in BL81 to the existing canonical `src/market_scanner/` modules.

This is a documentation-only governance sprint.

BL82 does not implement code, add tests, archive files, delete files, move files, refactor runtime behavior, execute script-era Python files, call providers, or write production data.

The goal is to determine whether the BL81 history and metrics contracts already have canonical ownership, partial ownership, or no canonical owner yet.

## Background

Recent cleanup sequence:

* BL74 removed active runtime imports from `scripts.fundamentals`.
* BL75 archived `scripts/fundamentals/__init__.py`.
* BL76 classified the remaining script-era Python files.
* BL77 archived low-risk non-fundamentals helper files.
* BL78 reviewed fundamentals migration/archive readiness.
* BL79 archived `scripts/core/build_fundamental_*` compatibility wrappers.
* BL80 classified the remaining `scripts/fundamentals/*.py` blockers.
* BL81 extracted the reusable history and metrics contracts from:

  * `scripts/fundamentals/build_history_intake.py`
  * `scripts/fundamentals/build_metrics.py`

BL82 maps those extracted contracts to current canonical modules.

## Scope

Mapped BL81 contract groups:

1. Fundamentals history schema contract
2. History key and duplicate policy
3. Fiscal-period policy
4. Fiscal-year policy
5. Date validation policy
6. Numeric validation policy
7. Required-value policy
8. Forbidden investment-semantics policy
9. Validation result shape
10. Metrics identity columns
11. Derived metric columns
12. Ratio formulas
13. YoY growth formulas
14. Metric missing-input policy
15. Metric status policy
16. Input-validation-before-metrics policy
17. Output/write separation policy

Canonical areas reviewed:

* `src/market_scanner/fundamentals/`
* `src/market_scanner/analysis/`

Out of scope:

* Implementing canonical tests
* Migrating script-era logic
* Editing runtime modules
* Archiving remaining `scripts/fundamentals/*.py`
* Executing any script-era fundamentals module
* Live SEC/EDGAR calls
* Production data writes

## Canonical module inventory

Command:

```bash
find src/market_scanner/fundamentals src/market_scanner/analysis -type f -name "*.py" | sort
```

Observed canonical modules:

```text
PASTE_LOCAL_RESULT_HERE
```

Expected relevant modules include:

* `src/market_scanner/analysis/analysis_boundary.py`
* `src/market_scanner/analysis/analysis_contracts.py`
* `src/market_scanner/fundamentals/fundamental_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_normalization_adapter.py`
* `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_persistence.py`
* `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
* `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
* `src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py`
* `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
* `src/market_scanner/fundamentals/source_data_readiness.py`
* `src/market_scanner/fundamentals/source_data_records.py`

## Mapping status definitions

| Status                            | Meaning                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| `CANONICAL_OWNER_EXISTS`          | A canonical module appears to own the contract already.                                         |
| `PARTIAL_CANONICAL_OWNER_EXISTS`  | A canonical module owns part of the contract, but parity is not proven.                         |
| `NO_CANONICAL_OWNER_IDENTIFIED`   | No clear canonical owner exists yet.                                                            |
| `TEST_COVERAGE_REQUIRED`          | Contract may have an owner, but canonical tests are required before retiring script-era source. |
| `DO_NOT_MIGRATE_RUNTIME_BEHAVIOR` | Script-era CLI/write behavior should not be copied into canonical runtime without approval.     |

## Contract mapping table

| BL81 contract                                 | Canonical mapping status         | Candidate canonical owner                                                                       | Notes                                                                                                                           |
| --------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Fundamentals history required-column contract | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamental_contracts.py`, `fundamentals_normalization_contracts.py`, `source_data_records.py` | Canonical fundamentals modules exist, but full script-era history schema parity is not yet proven.                              |
| History key and duplicate policy              | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical history contract tests                                                         | No clear canonical owner is confirmed for `ticker + fiscal_year + fiscal_period` duplicate policy.                              |
| Fiscal-period policy                          | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamental_contracts.py`, `fundamentals_normalization_contracts.py`                           | Period concepts likely belong in canonical contracts, but accepted values and validation need explicit tests.                   |
| Fiscal-year policy                            | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamental_contracts.py`, `fundamentals_normalization_contracts.py`                           | Numeric year validation range requires canonical test coverage.                                                                 |
| Date validation policy                        | `PARTIAL_CANONICAL_OWNER_EXISTS` | `source_data_records.py`, `fundamentals_normalization_contracts.py`                             | Canonical source records may own this, but date-column parity is not proven.                                                    |
| Numeric validation policy                     | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamentals_normalization_contracts.py`                                                       | Numeric parsing and empty-value tolerance require explicit canonical tests.                                                     |
| Required-value policy                         | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamental_contracts.py`, `fundamentals_normalization_contracts.py`                           | Required value rules should be canonicalized before script-era retirement.                                                      |
| Forbidden investment-semantics policy         | `PARTIAL_CANONICAL_OWNER_EXISTS` | `analysis_contracts.py`, `analysis_boundary.py`, future doctrine tests                          | No-allocation/no-recommendation doctrine exists conceptually, but BL81 forbidden column list needs explicit canonical coverage. |
| Validation result shape                       | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future fundamentals validation contract                                                         | The script-era validation result structure should not be assumed canonical without a designed owner.                            |
| Metrics identity columns                      | `PARTIAL_CANONICAL_OWNER_EXISTS` | `analysis_contracts.py`, `fundamental_contracts.py`                                             | Identity/provenance passthrough belongs canonically, but exact output schema parity is not proven.                              |
| Derived metric columns                        | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical metrics contract                                                               | Gross margin, operating margin, net margin, FCF margin, debt/equity, ROE, and YoY columns need explicit canonical ownership.    |
| Ratio formulas                                | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical metrics contract or `analysis_contracts.py`                                    | Formula parity must be implemented or explicitly rejected before archiving script-era metrics.                                  |
| YoY growth formulas                           | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical metrics contract or `analysis_contracts.py`                                    | Prior-year lookup and denominator policy require canonical tests.                                                               |
| Metric missing-input policy                   | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical metrics contract                                                               | Missing-input string format and zero-denominator warnings require explicit canonical decision.                                  |
| Metric status policy                          | `NO_CANONICAL_OWNER_IDENTIFIED`  | Future canonical metrics contract                                                               | `complete`/`partial` behavior is not proven in canonical modules.                                                               |
| Input-validation-before-metrics policy        | `PARTIAL_CANONICAL_OWNER_EXISTS` | `fundamentals_normalization_contracts.py`, future canonical metrics tests                       | Dependency between valid history input and metrics generation should be explicit.                                               |
| Output/write separation policy                | `CANONICAL_OWNER_EXISTS`         | `fundamentals_persistence.py`, existing source/persistence separation doctrine                  | Canonical persistence boundaries exist; script-era optional CLI writes should not be copied blindly.                            |

## Findings

1. Canonical fundamentals and analysis modules already exist under `src/market_scanner/`.

2. Existing canonical modules appear to cover broad source-data, normalization, provider, readiness, analysis-boundary, and persistence responsibilities.

3. BL81 contracts are more specific than the currently proven canonical ownership.

4. The most important missing canonical owner is a dedicated fundamentals metrics contract covering:

   * ratio formulas;
   * YoY formulas;
   * prior-year lookup;
   * missing-input policy;
   * zero-denominator policy;
   * metric status.

5. History schema validation also requires clearer canonical ownership:

   * required columns;
   * required values;
   * duplicate keys;
   * supported periods;
   * fiscal-year range;
   * date parsing;
   * numeric parsing.

6. The forbidden investment-semantics policy is aligned with the project doctrine, but the BL81 forbidden-column list should be converted into canonical tests before relying on it for retirement.

7. BL82 does not make `build_history_intake.py` or `build_metrics.py` archive-ready.

8. The grep result shows canonical modules for contracts, provider ingestion, normalization, source-data readiness, persistence, and analysis boundary, but no explicit canonical metrics contract for BL81 formulas such as `gross_margin`, `operating_margin`, `net_margin`, `free_cash_flow_margin`, `debt_to_equity`, `return_on_equity`, `revenue_yoy_growth`, `eps_yoy_growth`, and `free_cash_flow_yoy_growth`.

## Decision

The BL81 contracts are not yet fully covered by existing canonical modules.

BL82 classifies the current state as:

```text
PARTIAL_CANONICAL_COVERAGE_WITH_TEST_GAPS
```

This means:

* canonical modules exist;
* responsibility likely belongs under `src/market_scanner/fundamentals/` and `src/market_scanner/analysis/`;
* explicit canonical tests are required before script-era `build_history_intake.py` and `build_metrics.py` can be archived.

## Recommended canonical ownership model

Recommended target ownership:

| Contract group                 | Recommended owner                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| History required columns       | `src/market_scanner/fundamentals/fundamental_contracts.py`                                    |
| History validation rules       | `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`                     |
| Source/provenance fields       | `src/market_scanner/fundamentals/source_data_records.py`                                      |
| Metrics identity columns       | `src/market_scanner/analysis/analysis_contracts.py` or a future fundamentals metrics contract |
| Ratio formulas                 | Future canonical fundamentals metrics contract                                                |
| YoY formulas                   | Future canonical fundamentals metrics contract                                                |
| Missing-input policy           | Future canonical fundamentals metrics contract                                                |
| Forbidden investment semantics | `src/market_scanner/analysis/analysis_contracts.py` plus governance tests                     |
| Persistence/write separation   | `src/market_scanner/fundamentals/fundamentals_persistence.py`                                 |

## Recommended next sprint

Recommended next sprint:

```text
BL83 — Add canonical fundamentals history and metrics contract tests
```

This should be a code-aware Codex sprint, because it should add or update canonical tests.

BL83 should not migrate runtime behavior yet. It should first add tests that make the intended canonical contracts explicit.

Recommended BL83 test themes:

1. history required columns;
2. duplicate key policy;
3. fiscal-period validation;
4. fiscal-year validation;
5. date and numeric parsing;
6. forbidden investment-semantics columns;
7. ratio formulas;
8. YoY growth formulas;
9. missing-input and zero-denominator behavior;
10. output/write separation.

Alternative documentation-only sprint:

```text
BL83 — Write canonical fundamentals metrics contract specification
```

## Archive readiness decision

`build_history_intake.py` and `build_metrics.py` remain not archive-ready.

| File                                           | Status after BL82   | Reason                                                                                                      |
| ---------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py` | `NOT_ARCHIVE_READY` | Extracted contracts are not yet covered by explicit canonical implementation/tests.                         |
| `scripts/fundamentals/build_metrics.py`        | `NOT_ARCHIVE_READY` | Metrics formulas and missing-input behavior are not yet covered by explicit canonical implementation/tests. |

## Validation

No script-era runtime files were executed.

Full test suite should be run for repository safety.

Command:

```bash
pytest -q
```

Result: 522 passed in 0.56s

```text
src/market_scanner/analysis/__init__.py
src/market_scanner/analysis/analysis_boundary.py
src/market_scanner/analysis/analysis_contracts.py
src/market_scanner/fundamentals/__init__.py
src/market_scanner/fundamentals/fundamental_contracts.py
src/market_scanner/fundamentals/fundamentals_normalization_adapter.py
src/market_scanner/fundamentals/fundamentals_normalization_contracts.py
src/market_scanner/fundamentals/fundamentals_persistence.py
src/market_scanner/fundamentals/fundamentals_provider_adapter.py
src/market_scanner/fundamentals/fundamentals_provider_contracts.py
src/market_scanner/fundamentals/fundamentals_real_source_smoke.py
src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py
src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py
src/market_scanner/fundamentals/source_data_readiness.py
src/market_scanner/fundamentals/source_data_records.py
```

## Guardrails

* No live SEC/EDGAR calls were run.
* No yfinance calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist production state was modified.
* No Decision Engine authority was changed.
* No script-era Python runtime files were executed.

## Final status

BL82 completed the mapping of BL81 extracted contracts to existing canonical fundamentals and analysis modules.

The result is partial canonical coverage with explicit test gaps.

No script-era fundamentals file is archived or declared archive-ready by BL82.
