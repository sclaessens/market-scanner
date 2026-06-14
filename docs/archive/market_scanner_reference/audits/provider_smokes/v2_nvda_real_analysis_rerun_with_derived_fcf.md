# NVDA Real Analysis Re-run with Derived FreeCashFlow

Status: COMPLETED
Reset stage: RESET-10L-BL23
Ticker: NVDA
Company: NVIDIA Corporation

## Input Basis

This review used the BL19 and BL20 redacted NVDA source-shaped findings and the BL22 governed FreeCashFlow derivation behavior.

The input basis remained controlled and governance-safe:

- source family: SEC EDGAR Form 10-K;
- reference summary: accession `0001045810-25-000023`, redacted source-shaped review only;
- direct `FreeCashFlow`: absent in the source-shaped input;
- operating cash flow: present and valid in redacted source-shaped input;
- capital expenditures: present and valid in redacted source-shaped input;
- no new live provider capture was performed;
- no raw SEC payload or unredacted live source body was committed.

## Execution Mode

The re-run used controlled local in-memory execution only.

Code paths executed:

- `market_scanner.fundamentals.fundamentals_real_source_smoke.review_injected_source_response`;
- `market_scanner.fundamentals.fundamentals_provider_adapter.ingest_provider_fundamentals`;
- governed provider normalization and readiness behavior in `market_scanner.fundamentals.fundamentals_provider_adapter`;
- `scripts.fundamentals.build_analysis.analyze_fundamentals`.

The public CSV-writing analysis wrapper was not used. No production pipeline, report, Telegram, portfolio, watchlist, or Decision Engine path was executed.

## Code Paths Inspected

Existing Python files inspected:

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`;
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`;
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`;
- `src/market_scanner/fundamentals/fundamentals_persistence.py`;
- `scripts/fundamentals/build_analysis.py`;
- `scripts/fundamentals/build_metrics.py`;
- `src/market_scanner/decisions/decision_engine.py`;
- `tests/core/test_build_fundamental_analysis.py`;
- `tests/core/test_fundamentals_operational_validation.py`.

No Python files were changed.

## Analysis Result

The analysis could run in memory.

Provider/source readiness result:

- smoke status: `passed`;
- readiness state: `available`;
- source data status: `available`;
- missing fundamentals count: `0`;
- readiness warnings: `free_cash_flow:source_derived`.

Analysis output result:

- fundamental analysis state: `LIMITED_ANALYSIS`;
- fundamental analysis reason: `fundamental metrics or quality input is partial`;
- fundamental profile state: `STABLE_PROFILE`;
- margin profile state: `MARGIN_STABLE`;
- growth profile state: `GROWTH_UNKNOWN`;
- leverage profile state: `LEVERAGE_MODERATE`;
- cash flow profile state: `CASH_FLOW_POSITIVE`;
- analysis data status: `metrics_partial`;
- analysis input coverage: `quality_and_metrics`;
- analysis warnings: `metrics_partial;metric_warnings:yoy_growth:missing_prior_year;free_cash_flow:source_derived`.

No final investment recommendation was produced.

## FreeCashFlow Result

FreeCashFlow status: `source_derived`

The direct source-reported `FreeCashFlow` field remained absent. Governed derivation produced a normalized `free_cash_flow` value from:

- `NetCashProvidedByUsedInOperatingActivities`;
- `PaymentsToAcquirePropertyPlantAndEquipment`.

The derived record exposed:

- derivation formula: `free_cash_flow = operating_cash_flow - capital_expenditures`;
- derivation status: `source_derived`;
- provenance links to both input fields;
- validation warning: `free_cash_flow:source_derived`.

No missing value was converted to zero. Analysis consumed FreeCashFlow as derived, not source-reported.

## Comparison with BL20

BL20 result:

- FreeCashFlow stayed explicitly missing;
- cash flow profile state was blocked as `CASH_FLOW_UNKNOWN`;
- analysis remained `LIMITED_ANALYSIS`;
- primary blockers were `CASH_FLOW_UNKNOWN` and `REVIEW_DATA_LIMITATION`.

BL23 result:

- governed FreeCashFlow derivation produced `source_derived`;
- cash flow profile moved to `CASH_FLOW_POSITIVE`;
- provider/source readiness improved to `available`;
- `CASH_FLOW_UNKNOWN` was resolved;
- analysis still remained `LIMITED_ANALYSIS`.

The remaining limitation is no longer missing FreeCashFlow. The next blocker is incomplete analysis metric coverage, especially missing prior-year growth evidence in the controlled BL19/BL20 source-shaped review input.

## Remaining Blockers

- The current narrow analysis review requires complete metric coverage before it can return `ANALYSIS_READY`.
- The controlled source-shaped input used for this re-run did not include governed prior-year comparative metric evidence.
- Growth remained `GROWTH_UNKNOWN`.
- The real-source path is not yet a production analysis ingestion path. This sprint intentionally used in-memory handoff only.

## Defects Discovered

No new runtime defect was fixed in this sprint.

The main real-data limitation exposed is that derived FreeCashFlow resolves the cash-flow blocker, but the analysis layer still needs a governed way to consume or construct complete real metric history before broader analysis readiness can be reviewed.

## Candidate Backlog Items

Recommended next candidate:

- `RESET-10L-BL24 — Real Analysis Output Defect Review`

Candidate details for that review:

- identify which real metrics, prior-year fields, and source-history handoffs are required for `ANALYSIS_READY`;
- decide whether the current analysis output needs clearer review flags when metric status is partial;
- keep analysis descriptive and non-recommendation;
- avoid production pipeline execution unless separately approved.

## Python File Creation Policy Result

No new Python files were created.

No Python files were changed.

The sprint used existing inspected modules and did not create a ticker-specific helper, temporary committed script, or parallel analysis flow.

## Side-Effect Check

No final BUY/SELL/HOLD recommendation produced.

No portfolio/watchlist update performed.

No reports generated.

No Telegram artifacts generated.

No unsafe production pipeline executed.

No production data writes committed.

No credentials or raw live payloads committed.

No missing values converted to zero.

No one-off ticker-specific Python files created.

## Conclusion

The controlled NVDA one-ticker analysis re-run succeeded as a review. Governed derived FreeCashFlow resolved the previous `CASH_FLOW_UNKNOWN` blocker, and the analysis became more informative while remaining non-recommendation.

The analysis still stayed `LIMITED_ANALYSIS` because the controlled input lacked complete prior-year growth evidence. The next step should review real analysis output limitations and define the governed path for complete real metric history.

## Next Recommended Step

Proceed to `RESET-10L-BL24 — Real Analysis Output Defect Review`.
