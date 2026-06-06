# Canonical SEC CompanyFacts Smoke Boundary Implementation

## Status

Completed by RESET-10L-BL48.

## Reset stage

RESET-10L-BL48 - Implement Canonical Fundamentals SEC CompanyFacts Smoke Boundary.

## Purpose

Implement a canonical v2 fundamentals SEC CompanyFacts smoke boundary that consumes injected SEC CompanyFacts-shaped input for one ticker and proves deterministic fact selection, provenance preservation, explicit missingness, governed FreeCashFlow derivation, governed prior-year growth evidence, and fail-closed behavior without live provider execution.

BL48 does not implement a live SEC/EDGAR provider client.

BL48 does not approve production persistence, raw payload commits, cache commits, workflow execution, scanner-triggered execution, or multi-ticker capture.

The SEC CompanyFacts smoke boundary consumes injected SEC-shaped input only.

## Policies applied

- `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`
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
- `docs/active/backlog.md`

Canonical fundamentals files and tests:

- `src/market_scanner/fundamentals/`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `src/market_scanner/fundamentals/__init__.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

## Files changed

- `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
- `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`
- `docs/active/backlog.md`

No script-era files, archived scripts, workflows, data files, report files, raw payload files, cache files, portfolio/watchlist files, or production artifacts were changed.

## Python file creation justification if applicable

`src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py` was created because BL47 approved a distinct canonical SEC CompanyFacts smoke boundary and the existing fundamentals provider adapter is intentionally generic provider-ingestion logic. The new file owns only SEC CompanyFacts-shaped smoke input validation, deterministic fact selection, and conversion into the existing provider adapter boundary. It does not implement a live client, fetcher, downloader, cache, production persistence path, or script-era bridge.

`tests/unit/test_v2_sec_companyfacts_smoke_boundary.py` was created because no existing test file owned the canonical SEC CompanyFacts smoke-boundary behavior. The test file uses fake/redacted SEC-shaped fixtures only and does not import or execute script-era files.

## Canonical SEC CompanyFacts smoke boundary result

The new canonical boundary exposes:

- `SecCompanyFactsFact`
- `SecCompanyFactsSmokeInput`
- `SecFactSelectionResult`
- `SecCompanyFactsSmokeResult`
- `build_sec_companyfacts_smoke_result`

The boundary is deterministic, in-memory, side-effect-free by default, and provider-call-free. It accepts injected SEC-shaped facts, validates BL47 policy constraints, selects governed facts, builds a canonical `ProviderSourceResponse`, and delegates normalization/readiness/growth handling to the existing canonical fundamentals adapter.

## Source family and ticker scope

Approved source family:

```text
SEC EDGAR / SEC CompanyFacts
```

Implemented scope:

- one ticker only;
- default expected ticker: `NVDA`;
- deterministic CIK preservation;
- no portfolio-wide capture;
- no watchlist-wide capture;
- no scanner-produced universe capture;
- no multi-ticker capture.

## Injected fixture policy

Tests use fake/redacted SEC-shaped fixtures only.

The fixture values are synthetic and minimal, such as:

- revenue: `1200`;
- net income: `250`;
- operating income: `300`;
- operating cash flow: `900`;
- capital expenditures: `100`;
- prior-year comparable values.

No raw unredacted SEC payloads were committed, downloaded, cached, or written.

## Fact-selection behavior

The smoke boundary implements a small governed mapping:

| canonical field | SEC concept |
|---|---|
| `revenue` | `Revenues` |
| `net_income` | `NetIncomeLoss` |
| `operating_income` | `OperatingIncomeLoss` |
| `operating_cash_flow` | `NetCashProvidedByUsedInOperatingActivities` |
| `capital_expenditures` | `PaymentsToAcquirePropertyPlantAndEquipment` |
| `free_cash_flow` | `FreeCashFlow` |

Selection is deterministic and fails closed on ambiguous candidates, wrong ticker, missing CIK, ambiguous CIK, wrong source family, missing fiscal context, unit mismatch, currency mismatch, period mismatch, missing provenance, non-numeric values, attempted live mode, or attempted production persistence.

## FreeCashFlow handling

Direct `FreeCashFlow` may be absent.

When direct `FreeCashFlow` is absent and operating cash flow plus capital expenditures are valid and comparable, the boundary delegates to the existing governed derivation:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

The derived record preserves both input source fields, the derivation formula, currency, unit, fiscal context, and validation warnings.

Missing, invalid, ambiguous, mismatched, or provenance-gap inputs fail closed and do not convert missing values to zero.

## Growth evidence handling

When comparable prior-year SEC-shaped facts are supplied, the smoke boundary builds prior-year growth evidence through the existing canonical provider adapter.

The implemented smoke test proves governed growth evidence is available for:

- revenue;
- free cash flow;
- net income;
- operating income.

Growth evidence remains source-data evidence only. It does not produce investment recommendations, allocation behavior, scoring, target-price behavior, tradeability, urgency, or conviction.

## Missingness and failure behavior

The boundary returns `review_required` with explicit issues or warnings when smoke input fails policy checks or fact selection cannot be trusted.

Covered fail-closed cases include:

- wrong ticker;
- missing CIK;
- ambiguous CIK;
- multi-ticker input;
- ambiguous fact candidates;
- unit mismatch;
- currency mismatch;
- fiscal-period mismatch;
- missing provenance;
- non-numeric facts;
- wrong source family;
- attempted live/network mode;
- attempted production persistence.

Missing values remain explicit and are never converted to zero.

## Provenance behavior

The smoke boundary preserves:

- source family;
- provider name;
- ticker;
- CIK;
- company identity;
- fiscal year;
- fiscal period;
- period end date;
- accession/report identifier;
- source concept;
- unit;
- currency;
- reported value;
- source timestamp;
- retrieval timestamp;
- source reference;
- source record identity;
- missingness evidence;
- derivation formula and source field names when derived.

## Persistence behavior

BL48 adds no production persistence.

The smoke boundary performs no file writes. Persistence remains limited to existing canonical tmp-path-safe tests in the fundamentals persistence boundary.

No writes under `data/`, no raw payload writes, no cache writes, no report writes, and no `reports/daily/telegram_message.txt` writes were added.

## Script-era decoupling confirmation

The canonical implementation does not import script-era files.

Static tests guard against imports of:

- `scripts.fundamentals.sec_companyfacts_bulk_intake`
- `scripts.fundamentals.sec_companyfacts_transform`
- `scripts.data_sources.common`
- `scripts.data_sources.prefill_fundamentals`
- `scripts.fundamentals.build_quality`

Script-era fundamentals files remain migration candidates only.

## Tests added or updated

Added:

- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`

The tests prove:

- one-ticker NVDA-shaped injected input passes;
- source family, ticker, CIK, and company provenance are preserved;
- SEC facts are selected deterministically;
- FreeCashFlow is source-derived when direct `FreeCashFlow` is absent;
- prior-year growth evidence is available for comparable facts;
- missingness remains explicit;
- wrong ticker, missing CIK, ambiguous CIK, multi-ticker input, ambiguous facts, unit mismatch, currency mismatch, period mismatch, missing provenance, non-numeric values, wrong source family, attempted live mode, and attempted production persistence fail closed;
- imports and boundary execution create no files;
- implementation source contains no network, credential, provider-client, yfinance, or script-era imports.

## Side-effect guarantees

- No live SEC/EDGAR calls.
- No yfinance calls.
- No provider network calls.
- No credential reads.
- No production data writes.
- No raw payload writes.
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
- No raw live payloads committed.
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
- No one-off temporary helper files created.
- No replacement runtime scripts created.

## Known limitations

BL48 does not implement a live SEC/EDGAR provider client.

It does not implement SEC network access, SEC User-Agent handling, CompanyFacts bulk intake, cache governance, production persistence, raw-payload retention, workflow integration, scanner integration, multi-ticker capture, or complete SEC fact-selection parity.

The mapping is intentionally narrow and covers only the first governed smoke fields.

## Next recommended step

Proceed to:

```text
RESET-10L-BL49 — Validate NVDA SEC CompanyFacts Smoke Boundary Against Redacted Source-Shaped Evidence
```
