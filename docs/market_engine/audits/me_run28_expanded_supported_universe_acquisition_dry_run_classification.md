# ME-RUN28 - Expanded Supported-Universe Acquisition and Dry-Run Classification

Sprint ID: ME-RUN28
Status: COMPLETED WITH BLOCKED OUTCOME BY ME-RUN28
Job family: ME-RUN / Run and orchestration
Date: 2026-07-02
Branch: `me-run28-expanded-supported-universe-acquisition-dry-run-classification`

## Purpose and Scope

ME-RUN28 executes a controlled local classification over 16 active
Professional Swing Universe tickers. It separates:

1. current automated cached-source acquisition support;
2. staging validation of newly acquired packages;
3. existing local cached-source inventory coverage;
4. `cached_source_snapshot` batch dry-run behavior;
5. persisted analysis-context readiness;
6. Recommendation Review, actionable-review, and Decision Engine readiness.

The sprint is operational and docs-only. It does not change runtime, tests,
source contracts, validation rules, portfolio behavior, Decision Engine
behavior, or delivery behavior.

## Source Basis

Source main commit:

```text
a91619f Merge pull request #416 from sclaessens/me-run28a-nvda-amd-asml-readiness-boundary-audit
```

The run uses the existing entry points:

```text
market_engine.source_acquisition.automated_cached_source_acquisition.run_automated_cached_source_acquisition
market_engine.source_refresh.cached_source_snapshot_staging_validator_command
market_engine.run.cached_source_batch_dry_run_command
```

No new command, provider adapter, source fallback, or runtime path was added.

## Run Identity

```text
run_id: me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z
generated_at: 2026-07-02T11:56:52Z
artifact_root: artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z
```

Local artifacts:

```text
acquisition/acquisition_result.json
staging_validation.json
acquisition_package_dry_runs/me-run28-<ticker>-company-profile-dry-run/artifacts/market_engine_dry_run_<run-id>_2026-07-02.json
acquisition_package_dry_runs/me-run28-<ticker>-company-profile-dry-run/manifest.json
cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/batch_manifest.json
cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/<TICKER>/dry_run.json
cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/<TICKER>/manifest.json
me_run28_classification_summary.json
me_run28_classification_summary.md
```

Generated artifacts remain local and are not committed.

## Ticker Selection

The requested selection is:

```text
NVDA
AMD
ASML
AVGO
CLS
VRT
COST
META
MSFT
AAPL
GOOGL
AMZN
TSM
MU
CRDO
IREN
```

All 16 tickers are active entries in:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

No ticker required an ME-RUN28 `not_in_supported_universe` classification.
The selection covers all 12 tickers with existing tracked SEC CompanyFacts
cache plus four active-universe tickers without such a snapshot.

## Commands Executed

### Automated Acquisition

The existing ME-SA02 function entry point was invoked from a local one-off
Python command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<construct the request below and call run_automated_cached_source_acquisition(request)>'
```

Exact request inputs:

```text
request_format: market-engine-automated-cached-source-acquisition-request-v1
request_id: me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z
requested_at: 2026-07-02T11:56:52Z
generated_at: 2026-07-02T11:56:52Z
run_mode: dry_run
ticker_source.mode: explicit_list
ticker_source.source_id: me_run28_expanded_supported_universe
tickers: NVDA, AMD, ASML, AVGO, CLS, VRT, COST, META, MSFT, AAPL, GOOGL, AMZN, TSM, MU, CRDO, IREN
source_families: company_profile
destination_root: artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition
approved_adapter: fake_company_profile_adapter/test-v1
allow_hidden_fallback: false
allow_silent_substitution: false
allow_provider_calls: false
allow_network: false
allow_production_writes: false
allow_telegram_send: false
allow_portfolio_writes: false
allow_watchlist_writes: false
allow_broker_actions: false
```

### Staging Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_staging_validator_command --staging-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition --output-json artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/staging_validation.json --validated-at 2026-07-02T11:56:52Z --human
```

### Existing Cached-Source Batch Dry-Run

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command --source-snapshot-root data/market_engine/source_snapshots --tickers NVDA,AMD,ASML,AVGO,CLS,VRT,COST,META,MSFT,AAPL,GOOGL,AMZN,TSM,MU,CRDO,IREN --batch-id me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z --generated-at 2026-07-02T11:56:52Z --write-local-artifacts --artifact-output-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch --emit-json
```

No portfolio-context flag was supplied. The run therefore exercised the
explicit absent-context boundary without reading or mutating portfolio state.

### Direct Dry-Run of Newly Acquired Packages

The three acquired and accepted company-profile packages were also passed
directly into the existing local dry-run CLI:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition/NVDA/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition --dry-run-id me-run28-nvda-company-profile-dry-run --generated-at 2026-07-02T11:56:52Z --artifact-output-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition_package_dry_runs --artifact-created-at 2026-07-02T11:56:52Z --write-local-artifact --compact

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition/AMD/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition --dry-run-id me-run28-amd-company-profile-dry-run --generated-at 2026-07-02T11:56:52Z --artifact-output-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition_package_dry_runs --artifact-created-at 2026-07-02T11:56:52Z --write-local-artifact --compact

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition/ASML/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition --dry-run-id me-run28-asml-company-profile-dry-run --generated-at 2026-07-02T11:56:52Z --artifact-output-root artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/acquisition_package_dry_runs --artifact-created-at 2026-07-02T11:56:52Z --write-local-artifact --compact
```

All three direct acquisition-package dry-runs report:

```text
run_state: dry_run_blocked
blocked_stage: recommendation_review
readiness_level: descriptive_only
blocked_reasons:
  - stale_or_unprovenanced_analysis_context
  - company_profile_only_context_non_actionable
actionable_review_allowed: false
decision_engine_ready: false
```

### Central Classification Summary

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /private/tmp/me_run28_summarize.py
```

The local helper only read run artifacts and the tracked universe CSV, asserted
the non-actionable readiness boundaries, and wrote the two local ME-RUN28
summary artifacts.

## Source Separation

The automated acquisition job produces `company_profile` packages for its
three currently supported tickers. Those packages were validated and directly
dry-run as `descriptive_only`.

The broader batch dry-run uses separately existing tracked SEC CompanyFacts
snapshots under:

```text
data/market_engine/source_snapshots
```

ME-RUN28 does not treat the existing SEC cache as output from the current
acquisition job. Acquisition and dry-run source origin remain separate in the
machine-readable summary. No silent fallback, source substitution, or manual
data correction occurred.

The primary matrix below reports the broader SEC cached-source batch outcome.
The direct acquisition-package path is recorded separately because it carries
different, company-profile-only evidence.

## Result Matrix

Common dry-run artifact base:

```text
artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z
```

Audit labels `not_run_no_acquisition_artifact` and
`not_available_missing_cached_source` describe ME-RUN28 orchestration state;
they do not add runtime contract states.

| Ticker | Acquisition | Validation | Dry-run | Readiness | Blockers | Recommendation Review | Actionable | DE-ready | Artifact path |
|---|---|---|---|---|---|---|---|---|---|
| NVDA | `completed` | `accepted` | `blocked_downstream_contract_failure` | `partial_analysis` | `missing_setup_or_price_context`; Portfolio Review blocked by absent portfolio context | `completed` | no | no | `NVDA/dry_run.json` |
| AMD | `completed` | `accepted` | `blocked_downstream_contract_failure` | `partial_analysis` | `missing_setup_or_price_context`; Portfolio Review blocked by absent portfolio context | `completed` | no | no | `AMD/dry_run.json` |
| ASML | `completed` | `accepted` | `blocked_downstream_contract_failure` | `partial_analysis` | `missing_setup_or_price_context`; Portfolio Review blocked by absent portfolio context | `completed` | no | no | `ASML/dry_run.json` |
| AVGO | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `AVGO/dry_run.json` |
| CLS | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `CLS/dry_run.json` |
| VRT | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `VRT/dry_run.json` |
| COST | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `COST/dry_run.json` |
| META | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `META/dry_run.json` |
| MSFT | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `MSFT/dry_run.json` |
| AAPL | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_missing_cached_source` | `not_available_missing_cached_source` | acquisition coverage; `missing_cached_source_snapshot` | not run | no | no | none |
| GOOGL | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_missing_cached_source` | `not_available_missing_cached_source` | acquisition coverage; `missing_cached_source_snapshot` | not run | no | no | none |
| AMZN | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_missing_cached_source` | `not_available_missing_cached_source` | acquisition coverage; `missing_cached_source_snapshot` | not run | no | no | none |
| TSM | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `TSM/dry_run.json` |
| MU | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_missing_cached_source` | `not_available_missing_cached_source` | acquisition coverage; `missing_cached_source_snapshot` | not run | no | no | none |
| CRDO | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `CRDO/dry_run.json` |
| IREN | `unsupported` (`unsupported_ticker`) | `not_run_no_acquisition_artifact` | `blocked_downstream_contract_failure` from existing cache | `partial_analysis` | acquisition coverage; `missing_setup_or_price_context`; absent portfolio context | `completed` | no | no | `IREN/dry_run.json` |

## Aggregate Results

```text
active supported-universe entries selected: 16
automated acquisition completed: 3
automated acquisition unsupported_ticker: 13
new acquisition packages accepted by staging: 3
direct acquisition-package dry-runs: 3
direct acquisition-package descriptive_only: 3
existing SEC cached source found: 12
missing cached source snapshot: 4
cached-source dry-runs executed: 12
partial_analysis: 12
readiness unavailable because cached source is missing: 4
Recommendation Review completed: 12
actionable_review_allowed: 0
decision_engine_ready: 0
```

The cached-source batch state is:

```text
completed_with_ticker_failures
```

All 16 requested tickers are blocked: 12 at a downstream contract boundary and
four at source discovery.

## Blocker Classification

### Data Acquisition Problems

The automated acquisition implementation has:

```text
SUPPORTED_TICKERS = NVDA, AMD, ASML
SUPPORTED_SOURCE_FAMILIES = company_profile
```

Thirteen active-universe tickers therefore return `unsupported_ticker`.
This is the primary universe-scale acquisition blocker.

### Staging and Import Validation Problems

No staging defect was observed. All three generated packages were accepted
with no manifest, hash, size, staleness, or usability issue.

The other 13 acquisition entries created no package, so staging validation was
not applicable to them.

The three accepted packages were directly consumable by the existing local
dry-run path. Each produced descriptive company-profile context and stopped at
Recommendation Review with `company_profile_only_context_non_actionable`.

### Cached-Source Coverage Problems

Existing SEC CompanyFacts snapshots were found for 12 tickers. No matching
cached source exists for:

```text
AAPL
GOOGL
AMZN
MU
```

These four are fail-closed with `blocked_missing_cached_source`.

### Dry-Run Pipeline Problems

No parsing, provenance, artifact-persistence, or ticker-specific runtime defect
was observed for the 12 executed tickers. Their SEC snapshots were consumed,
and every ticker persisted readiness metadata.

All 12 dry-runs stop at Portfolio Review because portfolio context was
intentionally absent. The common batch execution state is:

```text
blocked_downstream_contract_failure
```

### Source-Consumption and Readiness Gaps

All 12 executed artifacts report:

```text
readiness_level: partial_analysis
evidence_families_present:
  - fundamentals
  - provenance_manifest_staleness
evidence_families_missing:
  - setup_price_market
blocked_reasons:
  - missing_setup_or_price_context
recommendation_review_eligible: false
actionable_review_allowed: false
decision_engine_ready: false
```

The current SEC-only cached-source input does not satisfy approved setup,
price, and market evidence. No missing family was inferred from setup labels
or other descriptive output.

### Actionable and Decision Engine Readiness Gaps

No ticker is actionable. No ticker is Decision Engine-ready.

Recommendation Review has a `completed` stage result for the 12 executed
tickers under current pipeline semantics. This is a non-actionable review
result, not permission to act. Persisted readiness independently confirms that
Recommendation Review eligibility is false and that actionable-review and
Decision Engine readiness remain false.

`actionable_review` and `decision_ready` remain reserved and unreachable.
The Decision Engine remains the only allocation authority.

## Safety

Acquisition safety:

```text
provider_calls_performed: false
network_used: false
telegram_sent: false
portfolio_written: false
watchlist_written: false
broker_action_performed: false
production_write_performed: false
```

The batch command confirms no live provider, market-data, broker, delivery,
scheduler, UI, portfolio, watchlist, production-report, or execution side
effect.

ME-RUN28 performed no live SEC/EDGAR or yfinance call, no Telegram send, no
portfolio/watchlist read or mutation, no production write, and no Decision
Engine change.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
546 passed in 2.33s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1213 passed in 3.49s
```

The local summary helper also asserted:

```text
16 of 16 tickers are active universe entries
3 direct acquisition-package dry-runs are descriptive_only
3 direct acquisition-package actionable_review_allowed values are false
3 direct acquisition-package decision_engine_ready values are false
12 persisted readiness levels are not actionable_review or decision_ready
12 actionable_review_allowed values are false
12 decision_engine_ready values are false
```

Repository checks:

```text
git diff --check
PASS

grep -R "BUY" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "SELL" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "tradeable" scripts/ | grep -v decision_engine.py
Ignored bytecode matches only.
```

No file under `scripts/` changed in ME-RUN28.

The complete staged-diff governance grep returned documentation-only matches:
run-result descriptions, explicit false safety flags, absent portfolio-context
explanations, legacy-grep interpretation, and no-side-effect confirmations.
Inspection confirmed that no provider/network, Telegram, broker/order,
portfolio/watchlist, or production-write behavior was added.

## Current State Toward Real Market Engine Analysis

The pipeline can consume valid existing SEC CompanyFacts cache for 12 of the
selected 16 tickers, persist provenance-aware fundamental readiness, and reach
a non-actionable Recommendation Review result deterministically.

It can also acquire, validate, and directly consume company-profile packages
for the bounded three-ticker allowlist, but that path remains descriptive only.
It cannot yet acquire analytical cached-source coverage across the active
universe through the automated acquisition job. It also lacks approved
setup/price/market evidence and, in this run, portfolio context. The current
state is therefore useful partial analysis, not actionable analysis and not a
Decision Engine handoff.

## Recommended Next Sprint

```text
ME-SA12 - Expanded supported-universe cached-source acquisition coverage contract
```

ME-SA12 should define how the automated acquisition path obtains approved
analytical source-family coverage for active Professional Swing Universe
tickers without a hard-coded three-ticker boundary. It must preserve explicit
source identity, provenance, staging validation, no hidden fallback,
fail-closed unsupported classifications, and no action or allocation
authority.

A later implementation sprint may expand the acquisition runtime only after
that contract is approved. Setup/price/market evidence and portfolio-context
readiness remain separate follow-ups and must not be conflated with acquisition
coverage.

## Final Status

```text
PASS WITH STRUCTURAL BLOCKERS
```
