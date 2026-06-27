# ME-SA04 - Company Profile Cached-Source Dry-Run Compatibility Gate Audit

Sprint ID: ME-SA04
Status: COMPLETED BY ME-SA04
Job family: ME-SA / Source Acquisition
Date: 2026-06-26

## Objective

ME-SA04 implements the ME-SA03 compatibility contract as a fail-closed gate in the local `cached_source_snapshot` dry-run route.

The gate prevents `company_profile` snapshots from being silently coerced into SEC CompanyFacts Source Context and records explicit blocked reasons when a package is not safe or not yet semantically consumable.

## Scope Completed

Implemented:

* local-only `company_profile` snapshot detection before SEC CompanyFacts context building;
* manifest presence validation for `company_profile` cached-source packages;
* payload format, source-family, ticker, profile object, and provenance checks;
* fail-closed network/provider-call provenance checks;
* fail-closed side-effect intent checks for production, portfolio, watchlist, and broker flags when present;
* manifest/payload source-family and ticker consistency checks;
* optional manifest hash and size verification when present;
* explicit blocked state for structurally valid `company_profile` snapshots whose dry-run consumption is not implemented;
* regression coverage preserving existing SEC CompanyFacts cached-source dry-run behavior.

## Non-Goals Preserved

ME-SA04 did not implement `company_profile` downstream consumption, live provider access, network access, yfinance access, SEC/EDGAR access, production writes, delivery sends, broker actions, portfolio/watchlist mutation, or source refresh.

ME-SA04 did not change Decision Engine semantics, Recommendation Review semantics, Portfolio Review semantics, allocation authority, action authority, ranking, scoring, urgency, conviction, position sizing, target price, or tradeability behavior.

## Files Changed

Runtime:

```text
src/market_engine/run/cached_source_execution.py
```

Tests:

```text
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Documentation:

```text
docs/market_engine/audits/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_audit.md
docs/market_engine/backlog/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_backlog_entry.md
docs/market_engine/roadmap/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_roadmap_entry.md
```

## Compatibility Outcome

Valid SEC CompanyFacts cached-source snapshots continue through the existing local dry-run pipeline.

`company_profile` snapshots are now evaluated by an explicit compatibility gate:

* malformed or unsafe packages are rejected with contract-specific blocked reasons;
* structurally valid packages are blocked with `blocked_company_profile_consumption_not_implemented`;
* the blocked state is explicit and no longer depends on SEC CompanyFacts metadata failure.

## Validation

Commands run:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_me_run10_cached_source_local_execution.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

Results:

```text
12 passed
103 passed
496 passed
1163 passed
```

## Conclusion

```text
PASS
```

ME-SA04 makes the `company_profile` cached-source dry-run boundary explicit, local-only, provenance-aware, and fail-closed while preserving current SEC CompanyFacts cached-source dry-run behavior.

## Next Sprint

```text
ME-RUN27 - Implement or validate company_profile cached-source package consumption for local dry-run
```
