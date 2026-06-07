# Live NVDA SEC CompanyFacts One-Ticker Smoke Result

## Status

Completed by RESET-10L-BL51 with a fail-closed pre-network result.

## Reset stage

RESET-10L-BL51 - Execute Controlled Live SEC CompanyFacts One-Ticker Smoke.

## Purpose

Execute, or fail closed before execution, the first controlled live SEC CompanyFacts one-ticker smoke under BL50 guardrails.

Only one live SEC CompanyFacts request for NVDA / CIK 0001045810 was permitted by BL51.

The request was not executed. BL51 failed closed before network execution because the local operator-supplied `SEC_USER_AGENT` value was missing.

No raw SEC payload was committed.

No production data, cache, report, Telegram artifact, portfolio/watchlist update, workflow integration, scanner-triggered execution, or recommendation behavior was added.

## Policies applied

- `docs/active/v2_controlled_live_sec_companyfacts_one_ticker_smoke_policy.md`
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
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Files inspected

Governance and cleanup records:

- `docs/active/v2_controlled_live_sec_companyfacts_one_ticker_smoke_policy.md`
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

Canonical fundamentals files and tests:

- `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

## Files changed

- `src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py`
- `tests/unit/test_v2_sec_companyfacts_live_smoke.py`
- `docs/active/v2_live_nvda_sec_companyfacts_one_ticker_smoke_result.md`
- `docs/active/backlog.md`

No script-era production files, archived scripts, workflows, data files, reports, raw payload files, cache files, portfolio/watchlist files, credential files, or production artifacts were changed.

## Pre-flight result

Pre-flight checks completed before any live request:

| check | result |
|---|---|
| working tree clean before smoke attempt | passed |
| branch created from up-to-date `main` | passed |
| target ticker exactly `NVDA` | passed |
| target CIK exactly `0001045810` | passed |
| source family exactly SEC EDGAR / SEC CompanyFacts | passed |
| one ticker only | passed |
| no workflow execution | passed |
| no scanner-triggered execution | passed |
| no production data path | passed |
| no raw payload commit | passed |
| no cache write | passed |
| no report write | passed |
| no Telegram | passed |
| no portfolio/watchlist update | passed |
| no recommendation behavior | passed |
| local `SEC_USER_AGENT` present | failed |

Because `SEC_USER_AGENT` was missing, BL51 failed closed before network execution.

## Invocation method

The smoke was invoked locally through the canonical fundamentals live-smoke function with:

- ticker: `NVDA`;
- CIK: `0001045810`;
- source family: SEC EDGAR / SEC CompanyFacts;
- approved endpoint: `https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json`;
- `execute_live=True`;
- local User-Agent read through `SEC_USER_AGENT`;
- no workflow;
- no scanner;
- no portfolio/watchlist;
- no reporting/Telegram;
- no production pipeline.

The function returned a fail-closed result before invoking its network fetcher.

## User-Agent handling

The local operator-supplied `SEC_USER_AGENT` value was checked without printing or committing its value.

Result:

```text
SEC_USER_AGENT_MISSING
```

The canonical smoke result was:

```text
status=smoke_failed
failure_category=user_agent_missing
request_executed=False
request_count=0
readiness_state=review_required
issues=user_agent_missing
```

No private User-Agent value was committed.

## Network scope

Allowed network scope:

- exactly one SEC CompanyFacts request;
- endpoint: `https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json`;
- ticker: `NVDA`;
- CIK: `0001045810`.

Actual network result:

- request executed: no;
- request count: `0`;
- endpoint used: none;
- yfinance calls: none;
- fallback provider calls: none;
- retries: none;
- bulk download: none;
- cache population: none.

## Raw payload handling

No raw SEC payload was downloaded, written, cached, or committed.

The canonical live-smoke implementation keeps raw response text in memory only when a request is actually executed. Because the User-Agent pre-flight failed, no raw response existed.

## Cache handling

No cache path was used.

No SEC cache file was written.

No broad CompanyFacts cache or CIK/ticker cache was created.

## Persistence behavior

No persistence write occurred.

No files were written under:

- `data/`;
- `reports/`;
- `tests/`;
- `docs/` except this redacted result document;
- `archive/`.

No production data, processed CSV, raw payload, cache, report, Telegram artifact, portfolio/watchlist, or generated production artifact was created.

## Redacted live result summary

Because the smoke failed closed before network execution, there is no live SEC response summary.

Redacted summary:

| field | result |
|---|---|
| ticker | `NVDA` |
| CIK | `0001045810` |
| company | NVIDIA Corporation |
| source family | SEC EDGAR / SEC CompanyFacts |
| provider | SEC CompanyFacts |
| request status | not executed |
| retrieval timestamp | generated by local smoke result |
| HTTP status category | not applicable |
| canonical fields found | none; not evaluated |
| canonical fields missing | none; not evaluated |
| readiness state | `review_required` |
| failure category | `user_agent_missing` |
| raw payload | none |

## Canonical boundary handoff result

No live response was handed off to `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py` because the User-Agent pre-flight failed before network execution.

The canonical live-smoke module still enforces that any future successful live response must hand off only to the canonical SEC CompanyFacts smoke boundary and existing canonical fundamentals contracts/adapter.

## Fact-selection result

Not evaluated.

No SEC CompanyFacts payload was retrieved, so no live facts were selected.

## FreeCashFlow result

Not evaluated.

No live SEC CompanyFacts payload was retrieved. FreeCashFlow was neither direct, derived, nor missing from live evidence.

## Growth evidence result

Not evaluated.

No live current/prior facts were retrieved, so no growth evidence was produced.

## Readiness/missingness result

Readiness result:

```text
review_required
```

Missingness/failure reason:

```text
user_agent_missing
```

Missing facts remain explicit. No missing value was converted to zero.

## Failure result if applicable

BL51 failed closed before network execution.

Failure category:

```text
user_agent_missing
```

Fail-closed properties:

- no retry loop;
- no fallback provider;
- no yfinance call;
- no raw payload;
- no cache;
- no production write;
- no report;
- no Telegram;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio/watchlist update;
- no recommendation behavior.

## Side-effect guarantees

- No script-era files were imported or executed by the canonical implementation.
- No archived scripts were executed.
- No SEC/EDGAR request was executed because pre-flight failed.
- No yfinance or alternative provider call occurred.
- No network call occurred.
- No credentials, API keys, tokens, or private User-Agent values were committed.
- No raw live payload or raw SEC payload was committed.
- No SEC cache file was committed.
- No production data write occurred.
- No report file was generated or written.
- No `reports/daily/telegram_message.txt` file was created or modified.
- No Telegram artifact was generated.
- No Telegram API call was made.
- No production pipeline was executed.
- No portfolio/watchlist file was updated.
- No final BUY/SELL/HOLD recommendation was produced.
- No allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.
- No missing value was converted to zero.

## Guardrails confirmation

- No script-era production files changed.
- No archived scripts modified.
- No archived scripts executed.
- No script-era files executed.
- No script-era files imported by canonical implementation.
- No workflows changed.
- No credentials committed.
- No API keys committed.
- No tokens committed.
- No private User-Agent value committed.
- No raw live payloads committed.
- No raw SEC payloads committed.
- No SEC cache files committed.
- No more than one SEC CompanyFacts request performed.
- No endpoint other than NVDA / CIK0001045810 CompanyFacts used.
- No yfinance/provider fallback calls.
- No production data writes.
- No raw payload/cache files written to the repository.
- No reports generated.
- No report files written.
- No reports/daily/telegram_message.txt created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No Telegram API calls made.
- No production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No replacement runtime scripts created.

## Known limitations

BL51 did not validate live SEC connectivity, HTTP status handling against a real SEC response, live response shape, SEC rate-limit behavior, live fact-selection behavior, live FreeCashFlow handling, live growth evidence, or live canonical boundary handoff because the local SEC User-Agent pre-flight was missing.

The new canonical live-smoke module is tested with mocked SEC-shaped responses only.

## Next recommended step

Proceed to:

```text
RESET-10L-BL52 - Resolve Live SEC CompanyFacts Smoke Failure
```

The next sprint should provide a governed local User-Agent invocation method, still without committing private contact details, raw payloads, caches, production data, reports, Telegram artifacts, workflow integration, scanner-triggered execution, portfolio/watchlist updates, or recommendation behavior.
