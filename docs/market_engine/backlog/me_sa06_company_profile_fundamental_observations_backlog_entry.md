# ME-SA06 - Company Profile Fundamental Observations Backlog Entry

Sprint ID: ME-SA06
Status: COMPLETED BY ME-SA06
Job family: ME-SA / Fundamental Observations
Date: 2026-06-27

## Summary

ME-SA06 derives deterministic, informational company-profile observations from
consumed Company Profile Source Context.

Implemented:

* typed `market-engine-company-profile-fundamental-observations-v1` contract;
* fixed-order descriptive identity and profile observations;
* provenance, timestamp, format, snapshot, and ticker retention;
* no observations for absent or blocked profile context;
* profile-only dry-run completion through Fundamental Observations;
* controlled block at Derived Observations;
* unchanged SEC CompanyFacts Fundamental Observations;
* focused unit and cached-source integration coverage.

## Safety Boundary

The implementation adds no provider or network calls, source refresh, live data,
production writes, delivery, broker actions, portfolio/watchlist mutation,
investment interpretation, scoring, ranking, or allocation authority.

## Validation

```text
4 passed - company-profile Fundamental Observations tests
21 passed - cached-source local execution tests
112 passed - tests/market_engine/run
509 passed - tests/market_engine
1176 passed - full pytest
```

## Next Sprint

```text
ME-RUN27 - Run NVDA/AMD/ASML with company_profile Source Context and Fundamental Observations
```
