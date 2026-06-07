# NVDA Real Analysis Re-run with Governed Growth Evidence

Status: COMPLETED
Reset stage: RESET-10L-BL26
Ticker: NVDA
Company: NVIDIA Corporation

## Input Basis

This review used the BL19, BL20, and BL23 redacted NVDA source-shaped findings plus the governed BL22 and BL25 behavior:

- direct `FreeCashFlow` remained absent;
- `free_cash_flow` was derived from operating cash flow and capital expenditures;
- prior-year growth evidence was built from current/prior comparable normalized fundamentals;
- no new live provider capture was performed;
- no raw SEC payload or unredacted source body was committed.

The review used synthetic/redacted NVDA-shaped current and prior fiscal-year values only. The values were used to exercise the approved v2 boundaries and are not committed raw live payload values.

## Execution Mode

The re-run used controlled local in-memory execution only.

Code paths executed:

- `market_scanner.fundamentals.fundamentals_real_source_smoke.review_injected_source_response`;
- `market_scanner.fundamentals.fundamentals_provider_adapter.ingest_provider_fundamentals`;
- `market_scanner.fundamentals.fundamentals_provider_adapter.build_prior_year_growth_evidence`;
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
- `tests/unit/test_v2_fundamentals_provider_adapter.py`;
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`;
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`;
- `tests/core/test_build_fundamental_analysis.py`;
- `tests/core/test_fundamentals_operational_validation.py`.

No Python files were changed.

## Analysis Result

The analysis could run in memory.

Provider/source readiness result:

- current smoke status: `passed`;
- current readiness state: `available`;
- current source data status: `available`;
- current missing fundamentals count: `0`;
- current readiness warnings: `free_cash_flow:source_derived`.

The prior source-shaped record was sufficient for governed growth evidence, but the prior smoke wrapper returned `review_required` because the prior input intentionally did not include a governed EPS field. This did not block revenue, FreeCashFlow, net income, or operating income growth evidence.

Analysis output result:

- fundamental analysis state: `LIMITED_ANALYSIS`;
- fundamental analysis reason: `fundamental metrics or quality input is partial`;
- fundamental profile state: `IMPROVING_PROFILE`;
- margin profile state: `MARGIN_STABLE`;
- growth profile state: `GROWTH_POSITIVE`;
- leverage profile state: `LEVERAGE_MODERATE`;
- cash flow profile state: `CASH_FLOW_POSITIVE`;
- analysis data status: `metrics_partial`;
- analysis input coverage: `quality_and_metrics`;
- analysis warnings: `metrics_partial;metric_warnings:free_cash_flow:source_derived;growth_evidence:revenue:growth_available;growth_evidence:free_cash_flow:growth_available;growth_evidence:net_income:growth_available;eps_yoy_growth:missing_governed_prior_year_growth_evidence`.

No final investment recommendation was produced.

## FreeCashFlow Result

FreeCashFlow status: `source_derived`

The direct source-reported `FreeCashFlow` field remained absent. Governed derivation produced `free_cash_flow` from:

- `NetCashProvidedByUsedInOperatingActivities`;
- `PaymentsToAcquirePropertyPlantAndEquipment`.

The analysis path consumed FreeCashFlow as derived, not source-reported.

No missing value was converted to zero.

## Growth Evidence Result

Governed growth evidence was produced from comparable current/prior normalized fundamentals.

Growth evidence states observed:

- `revenue`: `growth_available`, growth rate `0.2`;
- `free_cash_flow`: `growth_available`, growth rate `1`;
- `net_income`: `growth_available`, growth rate `0.25`;
- `operating_income`: `growth_available`, growth rate `0.2`.

The FreeCashFlow growth evidence preserved current and prior provenance to:

- `NetCashProvidedByUsedInOperatingActivities`;
- `PaymentsToAcquirePropertyPlantAndEquipment`.

Growth evidence remained evidence only. It did not create recommendations, allocation behavior, portfolio action, urgency, conviction, target price, tradeability, or Decision Engine behavior.

## Comparison with BL23

BL23 result:

- FreeCashFlow was `source_derived`;
- cash-flow profile was `CASH_FLOW_POSITIVE`;
- readiness was `available`;
- growth remained `GROWTH_UNKNOWN`;
- fundamental profile was `STABLE_PROFILE`;
- analysis remained `LIMITED_ANALYSIS` due missing governed prior-year growth evidence.

BL26 result:

- FreeCashFlow remains `source_derived`;
- cash-flow profile remains `CASH_FLOW_POSITIVE`;
- readiness remains `available`;
- governed growth evidence is available for revenue, FreeCashFlow, net income, and operating income;
- growth profile improved to `GROWTH_POSITIVE`;
- fundamental profile improved to `IMPROVING_PROFILE`;
- analysis still remains `LIMITED_ANALYSIS`.

## LIMITED_ANALYSIS Status

`LIMITED_ANALYSIS` improved but did not resolve.

The previous blocker, missing governed prior-year growth evidence for revenue and FreeCashFlow, is resolved.

The new blocker is:

```text
EPS_YOY_GROWTH_NOT_GOVERNED
```

The existing analysis schema includes `eps_yoy_growth` and treats the metric row as partial when it is not available. BL25 did not implement governed EPS growth evidence, and this BL26 review did not silently infer EPS growth from ungoverned data.

## Remaining Blockers

- The existing descriptive analysis table still expects `eps_yoy_growth` for complete metric status.
- There is not yet a governed EPS prior-year growth evidence rule.
- The v2 governed growth evidence records are not yet integrated as a first-class analysis input schema; this sprint used a controlled in-memory bridge into the existing metrics shape.

## Defects Discovered

No runtime defect was fixed in this sprint.

The real-analysis path exposed a remaining evidence-policy gap: analysis completeness depends on EPS growth evidence, but current governed growth evidence covers revenue, FreeCashFlow, net income, and operating income.

## Candidate Backlog Items

Recommended next candidate:

- `RESET-10L-BL27 — Real Analysis Remaining Evidence Gap Review`

Candidate details for that review:

- decide whether EPS YoY should receive a governed prior-year growth evidence policy;
- decide whether analysis completeness should require EPS YoY when governed revenue, FreeCashFlow, net income, and operating income growth evidence are available;
- review whether governed growth evidence records should become first-class analysis inputs instead of being bridged into the legacy metrics table;
- keep all output descriptive and non-recommendation.

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

The controlled NVDA real-analysis re-run succeeded as a review. Governed prior-year growth evidence is now available and made the analysis output more useful: growth moved from unknown to positive, and the fundamental profile moved from stable to improving.

The analysis still remains `LIMITED_ANALYSIS` because EPS YoY growth is not yet governed. The next step should review this remaining evidence gap before adding more implementation.

## Next Recommended Step

Proceed to `RESET-10L-BL27 — Real Analysis Remaining Evidence Gap Review`.
