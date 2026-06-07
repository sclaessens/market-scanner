# NVDA SEC CompanyFacts Smoke Boundary Validation

## Status

Completed by RESET-10L-BL49.

## Reset stage

RESET-10L-BL49 - Validate NVDA SEC CompanyFacts Smoke Boundary Against Redacted Source-Shaped Evidence.

## Purpose

Validate the canonical SEC CompanyFacts smoke boundary created in BL48 against more realistic, redacted, source-shaped NVDA SEC CompanyFacts evidence.

BL49 does not perform live SEC/EDGAR calls and does not implement a live SEC provider client.

The validation uses injected redacted/source-shaped NVDA SEC CompanyFacts evidence only.

No raw SEC payload, cache file, production data write, report, Telegram artifact, workflow execution, scanner-triggered execution, portfolio/watchlist integration, or recommendation behavior was added.

## Policies applied

- `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`
- `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`
- `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_free_cash_flow_derivation_policy.md`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/v2_nvda_real_source_persistence_smoke.md`
- `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- `docs/active/v2_real_analysis_output_defect_review.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Files inspected

Governance and implementation records:

- `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`
- `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`
- `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/backlog.md`
- fundamentals governance and evidence records from BL19 through BL26.

Canonical fundamentals files and tests:

- `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `src/market_scanner/fundamentals/__init__.py`
- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

## Files changed

- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
- `docs/active/v2_nvda_sec_companyfacts_smoke_boundary_validation.md`
- `docs/active/backlog.md`

No production Python files, script-era files, archived scripts, workflows, data files, reports, raw payload files, cache files, portfolio/watchlist files, or production artifacts were changed.

## Fixture policy

The redacted NVDA fixture remains inside `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`.

No JSON fixture, raw SEC payload, downloaded SEC response, large CompanyFacts payload, cache file, `data/` fixture, or report fixture was created.

The fixture uses synthetic numeric values and redacted accession/source references only. It models a source-shaped SEC CompanyFacts structure without preserving or committing raw live values.

## Redacted NVDA evidence shape

The injected validation fixture includes:

- ticker: `NVDA`;
- CIK: `0001045810`;
- entity name: NVIDIA Corporation;
- source family: `SEC EDGAR / SEC CompanyFacts`;
- provider: `SEC CompanyFacts`;
- redacted current accession/report identifier;
- redacted prior accession/report identifier;
- annual fiscal period;
- current fiscal year: `2025`;
- prior fiscal year: `2024`;
- current period end date: `2025-01-26`;
- prior period end date: `2024-01-28`;
- SEC-like concepts:
  - `Revenues`;
  - `NetIncomeLoss`;
  - `OperatingIncomeLoss`;
  - `NetCashProvidedByUsedInOperatingActivities`;
  - `PaymentsToAcquirePropertyPlantAndEquipment`;
- annual and quarterly revenue candidates to prove annual selection is deterministic;
- per-fact ticker, CIK, unit, currency, accession, source reference, source timestamp, fiscal year, fiscal period, and period-end provenance.

Direct `FreeCashFlow` remains absent.

## Positive validation result

The redacted/source-shaped NVDA evidence is accepted by the canonical SEC CompanyFacts smoke boundary.

Validation confirms:

- source family is preserved as `SEC EDGAR / SEC CompanyFacts`;
- provider is preserved as `SEC CompanyFacts`;
- ticker is `NVDA`;
- CIK provenance is preserved;
- entity identity is preserved;
- annual fiscal context is preserved;
- current period end date is preserved in provenance metadata;
- readiness is `available`;
- reported direct values remain reported values;
- direct `FreeCashFlow` missingness remains visible in raw evidence while resolved by governed derivation.

## Fail-closed validation result

The validation suite now proves fail-closed behavior for:

- wrong ticker;
- missing CIK;
- CIK mismatch;
- wrong source family;
- multi-ticker input;
- missing fiscal context;
- ambiguous annual facts;
- ambiguous current-year facts;
- unit mismatch;
- currency mismatch;
- period mismatch;
- missing provenance;
- non-numeric fact value;
- attempted live mode;
- attempted production persistence.

No existing fail-closed tests were weakened.

## Fact-selection validation result

The redacted fixture includes both annual and quarterly `Revenues` candidates.

The deterministic selector chooses the governed annual `FY` fact for the configured annual smoke input and ignores the non-annual candidate. When two annual candidates are supplied for the same concept and fiscal context, the boundary fails closed with an ambiguity issue rather than guessing.

## FreeCashFlow validation result

Direct `FreeCashFlow` is absent in the redacted NVDA fixture.

The boundary produces source-derived FreeCashFlow only because:

- operating cash flow is numeric;
- capital expenditures is numeric;
- both inputs use the same ticker;
- both inputs use the same CIK;
- both inputs use the same fiscal year;
- both inputs use the same fiscal period;
- both inputs use the same unit and currency context;
- both inputs include provenance;
- no ambiguity exists.

The derived value uses:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

The normalized record preserves both source field names and the derivation formula.

## Growth evidence validation result

Comparable prior-year redacted/source-shaped facts produce governed growth evidence for:

- revenue;
- free cash flow;
- net income;
- operating income.

Growth evidence preserves current and prior period references, source references, source record identities, source field names, unit, currency, formula, growth status, and validation warnings.

Growth evidence remains evidence only. It does not create allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

## Readiness result

The positive redacted/source-shaped NVDA validation returns:

- smoke status: `passed`;
- readiness state: `available`;
- source data status: `available`;
- missing fundamentals count: `0`;
- `FreeCashFlow` raw missingness remains visible but is resolved by governed source-derived FreeCashFlow.

Failure cases return `review_required` and no ingestion result.

## Script-era decoupling confirmation

The canonical implementation does not import script-era files.

Static guardrail tests continue to check against imports of:

- `scripts.fundamentals.sec_companyfacts_bulk_intake`
- `scripts.fundamentals.sec_companyfacts_transform`
- `scripts.data_sources.common`
- `scripts.data_sources.prefill_fundamentals`
- `scripts.fundamentals.build_quality`

No script-era files were modified, imported, or executed.

## Side-effect guarantees

- No live SEC/EDGAR calls.
- No yfinance calls.
- No provider calls.
- No network calls.
- No credential reads.
- No environment variable reads.
- No production data writes.
- No raw payload commits.
- No raw SEC payload commits.
- No cache writes.
- No report generation.
- No Telegram artifacts.
- No Telegram delivery.
- No workflow execution.
- No scanner-triggered execution.
- No portfolio/watchlist mutation.
- No Decision Engine final behavior.
- No BUY/SELL/HOLD behavior.
- No allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.
- No missing values converted to zero.
- No script-era imports or execution.
- No archived script execution.

## Guardrails confirmation

- No script-era production files changed.
- No archived scripts modified.
- No archived scripts executed.
- No script-era files executed.
- No script-era files imported by canonical implementation.
- No workflows changed.
- No credentials committed.
- No credentials read.
- No environment variables read.
- No raw live payloads committed.
- No raw SEC payloads committed.
- No network calls performed.
- No SEC/EDGAR calls made.
- No yfinance/provider calls made.
- No production data writes.
- No raw payload/cache files written.
- No reports generated.
- No report files written.
- No reports/daily/telegram_message.txt created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No Telegram API calls made.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No replacement runtime scripts created.

## Known limitations

BL49 validates the existing smoke boundary against redacted/source-shaped evidence only.

It does not implement a live SEC provider client, SEC network access, SEC User-Agent handling, production persistence, raw-payload retention, cache governance, workflow execution, scanner integration, portfolio/watchlist integration, multi-ticker capture, or complete SEC CompanyFacts field mapping.

The fixture remains minimal and redacted. It is not a raw SEC CompanyFacts payload.

## Next recommended step

Proceed to:

```text
RESET-10L-BL50 — Govern Controlled Live SEC CompanyFacts One-Ticker Smoke
```
