# ME-RUN27 - Company Profile Cross-Ticker Dry-Run Backlog Entry

Sprint ID: ME-RUN27
Status: COMPLETED WITH CONTROLLED STOP BY ME-RUN27
Job family: ME-RUN / Run and orchestration
Date: 2026-06-27

## Summary

ME-RUN27 executed the bounded `NVDA`, `AMD`, and `ASML` validation set through
deterministic local company-profile acquisition, staging validation,
compatibility gating, Source Context, and Fundamental Observations.

Result:

```text
acquisition: 3 completed
staging validation: 3 accepted
compatibility gate: 3 allowed
Source Context: 3 consumed
Fundamental Observations: 3 completed
controlled stop: 3 at Derived Observations
overall: completed_with_controlled_stop
```

Every ticker produced descriptive company-profile observations. Every ticker
stopped for the same contract reason:

```text
company_profile_fundamental_observations_do_not_provide_derived_financial_evidence
```

## Evidence

Committed:

```text
.gitignore
scripts/market_engine/me_run27_company_profile_cross_ticker_dry_run.py
tests/market_engine/run/test_me_run27_company_profile_cross_ticker_dry_run.py
docs/market_engine/audits/me_run27_company_profile_cross_ticker_dry_run_audit.md
docs/market_engine/backlog/me_run27_company_profile_cross_ticker_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run27_company_profile_cross_ticker_dry_run_roadmap_entry.md
```

Local generated run evidence:

```text
artifacts/market_engine/me-run27-company-profile-cross-ticker-20260627T150000Z
```

## Validation

```text
2 passed - ME-RUN27 runner tests
12 passed - automated cached-source acquisition tests
21 passed - cached-source local execution tests
114 passed - tests/market_engine/run
511 passed - tests/market_engine
1178 passed - full pytest
```

## Safety Boundary

No ticker-specific runtime fix, live provider call, network access, production
write, Telegram send, portfolio/watchlist mutation, broker action, or investment
logic was added.

## Next Sprint

```text
ME-SA07 - Allow company_profile observations into Analysis Review as descriptive context only
```
