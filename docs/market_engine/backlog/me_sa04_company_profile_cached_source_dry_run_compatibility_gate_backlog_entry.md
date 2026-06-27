# ME-SA04 - Company Profile Cached-Source Dry-Run Compatibility Gate Backlog Entry

Sprint ID: ME-SA04
Status: COMPLETED BY ME-SA04
Job family: ME-SA / Source Acquisition
Date: 2026-06-26

## Summary

ME-SA04 implements a fail-closed compatibility gate for `company_profile` cached-source snapshots in the local `cached_source_snapshot` dry-run route.

Implemented:

* `company_profile` payload detection before SEC CompanyFacts context building;
* payload and provenance validation;
* sibling manifest validation;
* source-family and ticker consistency checks;
* optional payload hash and size verification;
* explicit rejection for provider/network provenance violations;
* explicit blocked state for valid-but-not-yet-consumable `company_profile` snapshots;
* regression tests for existing SEC CompanyFacts cached-source dry-run behavior.

## Outcome

ME-SA04 replaces the previous implicit SEC CompanyFacts metadata failure with an explicit compatibility gate result for `company_profile` snapshots.

The gate distinguishes:

1. malformed or unsafe `company_profile` packages;
2. structurally valid packages that are not yet semantically consumable by dry-run;
3. non-`company_profile` snapshots that remain handled by the existing source-context loader.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_me_run10_cached_source_local_execution.py -q
12 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run -q
103 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
496 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1163 passed
```

## Implemented documentation

```text
docs/market_engine/audits/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_audit.md
docs/market_engine/backlog/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_backlog_entry.md
docs/market_engine/roadmap/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_roadmap_entry.md
```

## Safety Boundary

ME-SA04 did not add provider calls, network calls, live data fetching, source refresh, production writes, delivery sends, broker behavior, portfolio/watchlist mutation, Decision Engine semantics, Recommendation Review semantics, allocation authority, action authority, or downstream `company_profile` consumption.

## Next Sprint

```text
ME-RUN27 - Implement or validate company_profile cached-source package consumption for local dry-run
```
