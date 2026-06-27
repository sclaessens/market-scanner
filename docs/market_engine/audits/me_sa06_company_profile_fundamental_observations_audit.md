# ME-SA06 - Company Profile Fundamental Observations Audit

Sprint ID: ME-SA06
Status: COMPLETED BY ME-SA06
Job family: ME-SA / Fundamental Observations
Date: 2026-06-27
Branch: `me-sa06-derive-company-profile-observations`

## Objective

ME-SA06 derives bounded descriptive observations from a consumed
`market-engine-company-profile-source-context-v1` Source Context.

The implementation follows the ME-SA03 compatibility contract, relies on the
ME-SA04 gate outcome, and consumes only the trusted Source Context produced by
ME-SA05.

## Implementation

ME-SA06 introduces:

```text
market-engine-company-profile-fundamental-observations-v1
```

The typed observation contract records:

* source family and consumed Source Context state;
* ticker and symbol;
* Source Context and snapshot format versions;
* source snapshot identity and path;
* provider and cached-source provenance;
* as-of and retrieval timestamps;
* deterministic informational observations;
* an explicit descriptive-only authority boundary.

## Observation Scope

Observations are emitted in a fixed order when source fields are present:

```text
company_profile_identity_observed
company_profile_symbol_observed
company_profile_exchange_observed
company_profile_sector_observed
company_profile_industry_observed
company_profile_country_observed
company_profile_currency_observed
company_profile_description_available
company_profile_website_observed
company_profile_provenance_retained
company_profile_as_of_retained
```

Missing optional profile fields do not produce fabricated values or
observations.

## Consumption State Behavior

`absent_optional`:

* no company-profile observations are produced;
* the existing SEC CompanyFacts Fundamental Observations path is unchanged.

`blocked`:

* no profile body is trusted;
* no company-profile observations are produced;
* Source Context gate and blocker metadata remain traceable.

`consumed`:

* descriptive observations are built from the validated Source Context only;
* provenance, source formats, ticker binding, and as-of metadata are retained.

## Profile-Only Dry-Run Behavior

A valid profile-only cached-source run now completes:

```text
Source Context
Fundamental Observations
```

The run then blocks at Derived Observations with:

```text
company_profile_fundamental_observations_do_not_provide_derived_financial_evidence
```

Setup Detection, Analysis Review, Recommendation Review, Portfolio Review,
Decision Engine handoff, and Delivery remain not started.

## Backward Compatibility

The existing `sec-companyfacts-fundamental-observations-v1` contract and builder
are unchanged. The dry-run validator accepts the new company-profile
observation version in addition to the existing SEC version.

The current cached-source command accepts one snapshot path. ME-SA06 does not
introduce hidden discovery or joining of separate SEC and company-profile
packages. Multi-source run orchestration remains explicit follow-up work.

## Non-Advisory Guarantee

Company-profile observations are informational source descriptions only. They
contain no price, performance, valuation, momentum, setup, scoring, allocation,
or action inference.

ME-SA06 does not change Recommendation Review, Portfolio Review, Decision
Engine, setup, handoff, or delivery semantics.

## Tests

Coverage includes:

* complete consumed profile observation derivation;
* deterministic optional-field omission;
* refusal of non-consumed Source Context;
* targeted advisory-language exclusion;
* profile-only dry-run stage progression;
* artifact provenance visibility;
* blocked and malformed input behavior;
* symbol and provenance gate regressions;
* absent-optional SEC CompanyFacts regression.

Validation:

```text
4 passed - tests/market_engine/fundamental_observations/test_company_profile_observations.py
21 passed - tests/market_engine/run/test_me_run10_cached_source_local_execution.py
112 passed - tests/market_engine/run
509 passed - tests/market_engine
1176 passed - full pytest
```

## Safety Boundary

No provider calls, network access, yfinance, SEC/EDGAR access, Telegram,
production writes, broker actions, portfolio/watchlist mutation, source refresh,
or delivery side effects were added.

## Final Status

```text
PASS
```

## Follow-Up

```text
ME-RUN27 - Run NVDA/AMD/ASML with company_profile Source Context and Fundamental Observations
```
