# Controlled Live SEC CompanyFacts One-Ticker Smoke Policy

## Status

Completed by RESET-10L-BL50.

## Reset stage

RESET-10L-BL50 - Govern Controlled Live SEC CompanyFacts One-Ticker Smoke.

## Purpose

Define the exact pre-flight rules required before the project may execute its first controlled live SEC CompanyFacts one-ticker smoke in a later sprint.

BL50 does not execute a live SEC/EDGAR call.

BL50 does not implement a live SEC provider client.

BL50 only governs the conditions under which BL51 may execute a controlled one-ticker live SEC CompanyFacts smoke.

This sprint is governance-only and documentation-only. It does not modify Python files, tests, workflows, data files, report files, archived scripts, script-era files, raw payloads, cache files, portfolio/watchlist files, or production artifacts.

## Policies applied

- `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`
- `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`
- `docs/active/v2_nvda_sec_companyfacts_smoke_boundary_validation.md`
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

## Documents inspected

Governance and cleanup records:

- `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`
- `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`
- `docs/active/v2_nvda_sec_companyfacts_smoke_boundary_validation.md`
- `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/backlog.md`

Fundamentals records:

- `docs/active/v2_free_cash_flow_derivation_policy.md`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/v2_nvda_real_source_persistence_smoke.md`
- `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- `docs/active/v2_real_analysis_output_defect_review.md`

Canonical fundamentals context:

- `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

## Current canonical readiness

The canonical fundamentals boundary already supports injected provider-shaped responses, provenance, explicit missingness, deterministic SEC CompanyFacts smoke-boundary fact selection, governed source-derived FreeCashFlow, prior-year growth evidence, readiness outcomes, and tmp-path-safe persistence contracts.

BL48 implemented `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py` for injected SEC CompanyFacts-shaped input only.

BL49 validated the boundary against redacted/source-shaped NVDA evidence for:

- one ticker: `NVDA`;
- CIK: `0001045810`;
- company: NVIDIA Corporation;
- source family: SEC EDGAR / SEC CompanyFacts;
- annual FY2025/FY2024 context;
- deterministic SEC-like fact selection;
- source-derived FreeCashFlow;
- FreeCashFlow provenance and formula;
- prior-year growth evidence;
- readiness state `available` for complete evidence;
- `review_required` for invalid evidence;
- fail-closed behavior for ambiguity, mismatch, missing provenance, non-numeric values, attempted live mode, and attempted production persistence.

Canonical readiness is sufficient to govern the first live-smoke pre-flight. It is not sufficient to approve live execution without a separate implementation/execution sprint.

## Live smoke objective

The first controlled live smoke, if approved in BL51, may only aim to:

- confirm that one live SEC CompanyFacts response for one ticker can be retrieved;
- adapt the retrieved response into the already validated canonical smoke boundary;
- produce a redacted validation summary;
- identify source-shape gaps between the live response and canonical smoke-boundary expectations;
- confirm fail-closed behavior when live response shape, provenance, units, periods, or facts are incomplete or ambiguous.

The live smoke must not be treated as:

- production data ingestion;
- scanner input;
- portfolio input;
- watchlist input;
- investment analysis;
- reporting feed;
- Telegram feed;
- Decision Engine input.

## Approved source and ticker

The only approved first live-smoke target is NVDA / CIK 0001045810.

Approved source and identity:

| field | approved value |
|---|---|
| source family | SEC EDGAR / SEC CompanyFacts |
| provider | SEC CompanyFacts |
| ticker | NVDA |
| CIK | 0001045810 |
| company | NVIDIA Corporation |

Forbidden without later approval:

- multi-ticker smoke;
- portfolio-wide smoke;
- watchlist-wide smoke;
- scanner-selected smoke;
- ticker expansion;
- CIK expansion;
- yfinance fallback;
- alternative provider fallback;
- broker/API-provider fallback.

## Invocation policy

The first live smoke, if implemented later, may only be:

- explicitly invoked by a local operator;
- manual;
- limited to a single command or test target;
- disabled by default;
- absent from default canonical app dry-run;
- absent from scheduled automation;
- absent from workflows;
- absent from scanner execution;
- absent from portfolio/watchlist logic;
- absent from reporting and Telegram behavior.

The default canonical app remains dry-run-only.

No workflow, scanner, portfolio, watchlist, reporting, Telegram, or default app path may trigger the live smoke.

## SEC User-Agent policy

SEC User-Agent handling must be explicit and governed before any live SEC request is executed.

Policy:

- no SEC User-Agent value is added by BL50;
- no environment variable name is added to code by BL50;
- no User-Agent value may be hardcoded with private personal data;
- no User-Agent may be committed if it contains private contact details;
- no credentials or API keys are required for SEC CompanyFacts;
- no secrets may be committed;
- no environment variable may be read unless a later sprint explicitly approves it;
- if an environment variable is later approved, a missing, empty, malformed, or private-data-bearing value must fail closed before network access;
- any documentation-only placeholder must remain generic and must not contain private contact details.

BL51 must define how the operator supplies the User-Agent and how the implementation validates it before any network request occurs.

## Network policy

No network calls are approved or performed in BL50.

A future BL51 live smoke may perform exactly one governed SEC CompanyFacts request for NVDA / CIK `0001045810` only.

BL51 must forbid:

- retry loops unless separately governed;
- background network jobs;
- bulk downloads;
- SEC cache population;
- yfinance fallback;
- alternative API fallback;
- parallel calls;
- portfolio/watchlist/scanner expansion;
- scheduled or workflow network execution.

Network failure, timeout, HTTP failure, malformed response, or unexpected response shape must fail closed as `review_required` or `smoke_failed` and must not trigger fallback providers.

## Raw payload policy

Raw SEC payload retention is not approved by BL50.

Policy:

- raw SEC payload must not be committed;
- raw SEC payload must not be stored under `data/`;
- raw SEC payload must not be stored under `reports/`;
- raw SEC payload must not be added to tests as a fixture;
- raw SEC payload retention is not approved;
- no full SEC JSON response may be committed;
- no unredacted source dump may be committed;
- if a raw payload is temporarily held during a future live smoke, it must remain in memory or in an explicitly ignored local temp path;
- any temporary raw payload must be removed before commit;
- only redacted/source-shaped summaries may be documented.

## Cache policy

No cache writes are approved in BL50.

Policy:

- no production cache path is approved;
- no committed cache files are approved;
- no broad CompanyFacts cache is approved;
- no CIK/ticker cache is approved;
- no cache hydration in workflow is approved;
- no local cache population is approved as part of the first smoke unless a later sprint governs it;
- future cache governance requires a separate sprint.

## Persistence policy

No production persistence path is approved by BL50.

No production persistence path, raw payload retention, cache path, workflow execution, scanner-triggered execution, portfolio/watchlist integration, report generation, Telegram delivery, or recommendation behavior is approved by BL50.

Policy:

- no writes under `data/`;
- no processed CSV updates;
- no raw payload writes;
- no cache writes;
- no report writes;
- no Telegram artifacts;
- no portfolio/watchlist updates;
- no production data writes;
- no generated data may be committed;
- a future live smoke may use `tmp_path` or an explicitly ignored local temp path only.

Any future production persistence requires a separate governed path-policy sprint.

## Redacted output policy

A future BL51 live smoke may document only a redacted validation summary.

Allowed summary fields:

- ticker;
- CIK;
- company identity;
- source family;
- provider;
- retrieval status;
- redacted report/accession identifier if safe;
- fiscal context summary;
- canonical fields found or missing;
- readiness state;
- missingness reasons;
- fail-closed reason if failure;
- provenance summary.

Forbidden summary content:

- raw payload;
- full SEC JSON;
- unredacted source dump;
- request headers;
- private contact details;
- credentials;
- cache contents;
- generated production artifacts;
- report-ready financial tables.

## Boundary handoff policy

The future live smoke must hand off any adapted live response only through:

```text
src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py
```

and the existing canonical fundamentals adapter/contracts.

The future live smoke must not hand off through script-era files, including:

- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/data_sources/common.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/fundamentals/build_quality.py`

Script-era fundamentals files remain migration/retirement candidates and are not approved live-smoke authorities.

## Failure and missingness policy

Failure must be explicit and fail closed.

Required behavior:

- missing facts stay explicit;
- missing values never become zero;
- network failure fails closed;
- HTTP failure fails closed;
- timeout fails closed;
- invalid JSON fails closed;
- unexpected SEC shape fails closed;
- ticker mismatch fails closed;
- CIK mismatch fails closed;
- ambiguous facts fail closed;
- unit mismatch fails closed;
- currency mismatch fails closed;
- period mismatch fails closed;
- missing provenance fails closed;
- missing or invalid User-Agent policy state fails closed if applicable;
- smoke failure must not trigger retry loops or fallback providers unless later governed.

Failure may produce `review_required`, `smoke_failed`, or an equivalent explicit source-data state. It must not produce investment behavior.

## Test policy for later implementation

Before or during BL51, tests must prove:

- live smoke code is disabled by default;
- one-ticker NVDA enforcement;
- wrong ticker fails closed;
- multi-ticker input fails closed;
- missing User-Agent policy fails closed if applicable;
- network function is injectable or mocked in tests;
- no real network calls occur in unit tests;
- no `data/` writes occur;
- no raw payload fixture is committed;
- no cache fixture is committed;
- no script-era imports are used;
- no archived scripts are executed;
- no workflow integration exists;
- no scanner-triggered execution exists;
- no portfolio/watchlist integration exists;
- no report or Telegram artifacts are generated;
- no Decision Engine or investment semantics are produced.

If BL51 cannot satisfy these tests, it must fail closed and document the blocker rather than executing a live smoke.

## Forbidden behavior

BL50 forbids and does not approve:

- production ingestion;
- automatic pipeline execution;
- workflow execution;
- scanner-triggered execution;
- multi-ticker capture;
- portfolio-wide capture;
- watchlist-wide capture;
- raw payload commits;
- cache commits;
- writes under `data/`;
- report generation;
- Telegram delivery;
- Decision Engine final behavior;
- BUY, SELL, or HOLD behavior;
- allocation;
- conviction;
- urgency;
- scoring;
- target-price behavior;
- tradeability;
- recommendation behavior;
- missing-to-zero conversion;
- script-era imports;
- script-era execution;
- archived script execution.

## Implementation readiness conclusion

BL50 approves policy only, not implementation or execution.

The repository is governance-ready for a future BL51 controlled live SEC CompanyFacts one-ticker smoke only if BL51 obeys this document and remains limited to explicit local invocation, NVDA / CIK `0001045810`, one governed SEC CompanyFacts request, no raw payload retention, no cache, no production persistence, no workflow/scanner/portfolio/watchlist/report/Telegram integration, and fail-closed source-data behavior.

## Required BL51 guardrails

BL51 must explicitly confirm:

- one ticker only: `NVDA`;
- one CIK only: `0001045810`;
- one source family only: SEC EDGAR / SEC CompanyFacts;
- explicit local operator invocation only;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio/watchlist integration;
- no report generation;
- no Telegram delivery;
- no production data writes;
- no raw payload commits;
- no cache commits;
- no raw payload fixture;
- no script-era imports or execution;
- no archived script execution;
- no yfinance/provider fallback;
- no Decision Engine final behavior;
- no BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior;
- no missing-to-zero conversion.

## Recommended next sprint

Proceed to:

```text
RESET-10L-BL51 - Execute Controlled Live SEC CompanyFacts One-Ticker Smoke
```

If BL51 discovers that User-Agent governance, network containment, raw payload handling, or test isolation cannot be satisfied, it must stop and document:

```text
RESET-10L-BL51 - Resolve Controlled Live SEC Smoke Governance Blocker
```

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
- No script-era files executed.
- No archived scripts executed.
- No SEC/EDGAR calls made.
- No yfinance/provider calls made.
- No network calls.
- No credentials read.
- No environment variables read.
- No production data writes.
- No raw payloads written.
- No cache files written.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery.
- No portfolio/watchlist updates.
- No Decision Engine behavior changed.
- No BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior added.

## Known limitations

BL50 does not validate live SEC connectivity, SEC User-Agent mechanics, SEC rate-limit behavior, live response shape, ticker-to-CIK lookup, accession selection, cache policy implementation, raw payload cleanup mechanics, production persistence, or future operator command design.

Those topics must be handled by BL51 or later governed sprints without expanding beyond the one-ticker NVDA smoke scope.
