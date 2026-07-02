# ME-SA12 - Generic Supported-Universe Cached-Source Coverage Contract Audit

Sprint ID: ME-SA12
Status: COMPLETED DOCS-ONLY CONTRACT
Job family: ME-SA / Source Acquisition and Source Coverage
Date: 2026-07-02
Branch: `me-sa12-generic-supported-universe-cached-source-coverage-contract`

## Purpose

ME-SA12 converts the ME-RUN28 universe-scale evidence into a generic,
future-ticker-safe coverage contract.

The governing principle is:

```text
tickers are data, not logic
```

ME-RUN28 ticker names remain documentation/regression examples only.

## Source Basis

Source main commit:

```text
5afdefb Merge pull request #417 from sclaessens/me-run28-expanded-supported-universe-acquisition-dry-run-classification
```

Reviewed contract families:

```text
ME-UNI04 editable Professional Swing Universe
ME-SR05/06 source-support classification
ME-SA01 automated cached-source acquisition
ME-SR08 acquisition manifest
ME-SR10 staging validation
ME-SA09/10/11 analysis-context readiness
ME-RR01 through ME-RR04 Recommendation Review
ME-PR01/02 Portfolio Review
ME-DE01/02 Decision Engine handoff
ME-RUN28 expanded-universe run evidence
```

## Contract Decisions

ME-SA12:

* defines generic universe, requirement, snapshot, coverage, readiness,
  blocker, and pipeline-stage concepts;
* selects generic coverage requirements through governed capability profiles,
  never ticker;
* preserves current source-support and readiness status names;
* marks cross-cutting aggregate terms as proposed;
* separates acquired-source families, derived evidence, and downstream
  contract inputs;
* separates source availability from validation, consumability, completeness,
  analysis readiness, Recommendation Review eligibility, actionability, and
  Decision Engine readiness;
* preserves ME-RUN28 as a regression-class family only;
* reserves implementation for ME-SA13.

## Authority Boundary

ME-SA12 does not activate `actionable_review`, `actionable`,
`decision_ready`, or `de_ready`.

Coverage and readiness remain classification only. Recommendation Review
remains non-actionable. Decision Engine remains the only allocation authority.

## Files Changed

```text
docs/market_engine/source_support/me_sa12_generic_supported_universe_cached_source_coverage_contract.md
docs/market_engine/audits/me_sa12_generic_supported_universe_cached_source_coverage_contract_audit.md
docs/market_engine/backlog/me_sa12_generic_supported_universe_cached_source_coverage_contract_backlog_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/me_sa12_generic_supported_universe_cached_source_coverage_contract_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

No runtime or test file changes.

## Governance Grep

Command:

```text
rg -n 'ticker\s*==|ticker\s+in\s+\[|symbol\s*==|symbol\s+in\s+\[' src tests scripts docs/market_engine
```

Pre-change interpretation:

```text
src/market_engine: no hits
scripts: no source-text hits
src/market_scanner: three existing empty-string identity validation hits
tests: fixture assertions, fixture lookup, and one test-only zero-value branch
docs/market_engine: no hits
```

The new contract intentionally adds documentation hits showing prohibited
patterns. Those examples are governance guidance, not runtime behavior.

ME-SA12 adds no ticker-specific runtime logic.

Mandatory Decision Engine authority greps:

```text
grep -R "BUY" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "SELL" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "tradeable" scripts/ | grep -v decision_engine.py
Ignored bytecode matches only.
```

No file under `scripts/` changed in ME-SA12.

The complete staged-diff safety grep returned documentation-only matches:
contract boundaries, current contract names, explicit non-goals, and legacy
grep interpretation. Inspection confirmed that no provider/network, Telegram,
broker/order, portfolio/watchlist, or production-write behavior was added.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
546 passed in 2.29s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1213 passed in 3.34s

git diff --check
PASS

final governance grep
PASS - no new runtime hit; new hits are prohibited-pattern examples in the
ME-SA12 contract
```

## Safety

ME-SA12 is docs/contract-only. It adds no provider or network access, live
source consumption, production write, Telegram behavior, portfolio/watchlist
side effect, source validation relaxation, recommendation output, allocation
behavior, Decision Engine change, or ticker-specific runtime control flow.

## Follow-Up

```text
ME-SA13 - Implement generic cached-source coverage classification model
```

Expanded acquisition coverage remains downstream of the generic classifier.

## Final Status

```text
PASS
```
