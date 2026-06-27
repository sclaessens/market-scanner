# ME-RUN27 - Company Profile Cross-Ticker Dry-Run Audit

Sprint ID: ME-RUN27
Status: COMPLETED WITH CONTROLLED STOP BY ME-RUN27
Job family: ME-RUN / Run and orchestration
Date: 2026-06-27
Branch: `me-run27-company-profile-cross-ticker-dry-run-audit`

## Goal

ME-RUN27 executes the existing deterministic `company_profile` path for the
bounded validation set `NVDA`, `AMD`, and `ASML` through:

```text
acquisition
staging validation
compatibility gate
Source Context
Fundamental Observations
controlled downstream stop
```

The sprint records cross-ticker evidence. It does not alter runtime source,
observation, review, or decision semantics.

## Source Basis

The run uses the merged contracts and implementations from:

* ME-SA02 automated cached-source acquisition;
* ME-SA03 company-profile consumption compatibility contract;
* ME-SA04 fail-closed compatibility gate;
* ME-SA05 Company Profile Source Context;
* ME-SA06 Company Profile Fundamental Observations;
* ME-RM04 fast full-output sprint ordering and ticker-agnostic guardrails.

## Run Identity

```text
run_id: me-run27-company-profile-cross-ticker-20260627T150000Z
generated_at: 2026-06-27T15:00:00Z
artifact_root: artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z
bounded_tickers: NVDA, AMD, ASML
source_family: company_profile
```

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/market_engine/me_run27_company_profile_cross_ticker_dry_run.py --run-id me-run27-company-profile-cross-ticker-20260627T150000Z --generated-at 2026-06-27T15:00:00Z --artifact-root artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z
```

The runner uses the deterministic fake company-profile adapter and the existing
acquisition, staging, compatibility, Source Context, Fundamental Observations,
dry-run, and local artifact functions.

## Overall Result

```text
completed_with_controlled_stop
```

Acquisition completed for three of three entries. Staging accepted three of
three entries. Every compatibility gate allowed consumption. Every Source
Context was consumed. Every ticker completed Fundamental Observations and then
stopped at Derived Observations.

## Cross-Ticker Results

| Ticker | Acquisition | Staging validation | Gate | Source Context | Fundamental Observations | Completed stages | Stop stage | Observations |
|---|---|---|---|---|---|---|---|---|
| NVDA | completed | accepted | allowed | consumed | completed | source_context, fundamental_observations | derived_observations | produced |
| AMD | completed | accepted | allowed | consumed | completed | source_context, fundamental_observations | derived_observations | produced |
| ASML | completed | accepted | allowed | consumed | completed | source_context, fundamental_observations | derived_observations | produced |

Compatibility reason for every ticker:

```text
company_profile_consumption_allowed
```

Controlled stop reason for every ticker:

```text
company_profile_fundamental_observations_do_not_provide_derived_financial_evidence
```

This is the ME-SA06 contract boundary, not an unexpected failure.

## Per-Ticker Artifact Paths

| Ticker | Acquisition package | Dry-run artifact |
|---|---|---|
| NVDA | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/acquisition/NVDA/company_profile` | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/dry_runs/me-run27-company-profile-cross-ticker-20260627T150000Z-nvda/artifacts/market_engine_dry_run_me-run27-company-profile-cross-ticker-20260627T150000Z-nvda_2026-06-27.json` |
| AMD | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/acquisition/AMD/company_profile` | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/dry_runs/me-run27-company-profile-cross-ticker-20260627T150000Z-amd/artifacts/market_engine_dry_run_me-run27-company-profile-cross-ticker-20260627T150000Z-amd_2026-06-27.json` |
| ASML | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/acquisition/ASML/company_profile` | `artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/dry_runs/me-run27-company-profile-cross-ticker-20260627T150000Z-asml/artifacts/market_engine_dry_run_me-run27-company-profile-cross-ticker-20260627T150000Z-asml_2026-06-27.json` |

Machine-readable summary:

```text
artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/me_run27_summary.json
```

Human-readable local summary:

```text
artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z/me_run27_summary.md
```

## Observation Evidence

Each ticker produced the same applicable observation-code set:

```text
company_profile_identity_observed
company_profile_symbol_observed
company_profile_exchange_observed
company_profile_country_observed
company_profile_description_available
company_profile_provenance_retained
company_profile_as_of_retained
```

Optional sector, industry, currency, and website observations were not fabricated
because those fields are absent from the deterministic acquisition payload.

## Ticker-Agnostic Evidence

The runner contains one bounded ticker tuple used only as explicit run input and
one loop that applies identical functions and assertions to every ticker.

There are:

* no ticker-specific branches;
* no symbol-specific field mappings;
* no exchange-specific workarounds;
* no US-only runtime assumptions;
* no per-ticker blocker overrides.

All three tickers produced the same stage progression and controlled stop
reason.

## ASML Governance Note

ASML was processed through the same path as NVDA and AMD. Its consumed Source
Context retained:

```text
ticker: ASML
entity_name: ASML Holding N.V.
entity_country: NL
entity_exchange: NASDAQ
```

The `NL` country evidence passed without special handling. ASML is a bounded
non-US governance validation ticker, not an implementation special case.

## Acquisition and Staging Evidence

Acquisition:

```text
completed_count: 3
blocked_count: 0
rejected_count: 0
provider_error_count: 0
unsupported_count: 0
network_used: false
provider_calls_performed: false
production_write_performed: false
```

Staging:

```text
total_inspected_entries: 3
accepted_entries: 3
rejected_entries: 0
stale_count: 0
hash_mismatch_count: 0
size_mismatch_count: 0
```

No separate importer copy was required. The acquisition packages were already
local cached-source packages, and the existing staging validator classified all
three as accepted for cached-source staging.

## Tests and Validation

```text
2 passed - ME-RUN27 runner tests
12 passed - automated cached-source acquisition tests
21 passed - cached-source local execution tests
114 passed - tests/market_engine/run
511 passed - tests/market_engine
1178 passed - full pytest
git diff --check passed
```

## Generated Artifact Policy

The run root is local evidence under `artifacts/market_engine/`. Generated
acquisition packages, dry-run artifacts, logs, and summary artifacts are not
committed. The committed evidence consists of the deterministic runner and
audit, backlog, roadmap, and test documents.

The narrow `.gitignore` pattern
`artifacts/market_engine/me-run27-company-profile-cross-ticker-*/` prevents
accidental staging while preserving the local evidence.

## Safety and Non-Goals

ME-RUN27 adds no live provider calls, network access, yfinance, SEC/EDGAR
access, Telegram sending, production writes, broker actions, portfolio/watchlist
mutation, or source refresh behavior.

ME-RUN27 adds no investment interpretation, recommendation, target, ranking,
urgency, conviction, scoring, setup progression, allocation, or Decision Engine
authority.

## Follow-Up

Run evidence confirms one general next contract boundary:

```text
ME-SA07 - Allow company_profile observations into Analysis Review as descriptive context only
```

ME-SA07 must preserve the same non-advisory and ticker-agnostic constraints.
