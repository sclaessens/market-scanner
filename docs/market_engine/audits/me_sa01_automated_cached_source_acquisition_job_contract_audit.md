# ME-SA01 — Automated Cached-Source Acquisition Job Contract Audit

Sprint ID: ME-SA01  
Status: PASS once docs are applied  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26  

## 1. Audit Purpose

This audit records the expected scope, constraints, files, validation steps, and conclusion for ME-SA01.

ME-SA01 is a docs-only contract and architecture sprint. It defines the automated cached-source acquisition job contract so that ME-SA02 can implement a bounded, safe first job without changing downstream analysis, recommendation, portfolio, decision, delivery, or execution semantics.

## 2. Product-Owner Decision

The application must be able to retrieve source data itself through an automated job.

The primary route is now:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

Manual operator-supplied source packages are no longer the primary route. ME-SR13A is superseded as the primary route and remains only a possible fallback/manual diagnostic candidate.

## 3. Scope

ME-SA01 defines:

- automated cached-source acquisition job purpose;
- architectural chain;
- job boundary;
- request contract;
- approved run modes;
- ticker input rules;
- source family rules;
- provider/source adapter policy;
- output contract;
- snapshot package compatibility;
- provenance requirements;
- freshness/staleness policy;
- missing-data policy;
- failure model;
- safety and side-effect rules;
- handoff to existing import/staging/dry-run flow;
- acceptance criteria for ME-SA02;
- recommended next sprints.

## 4. Non-Goals

ME-SA01 does not:

- implement runtime code;
- modify tests;
- perform provider calls;
- fetch live data;
- use yfinance;
- call SEC/EDGAR;
- use internet access;
- create source data files;
- create fake NVDA/AMD/ASML data;
- send Telegram messages;
- write portfolio state;
- write watchlist state;
- write production outputs;
- modify Decision Engine semantics;
- modify Recommendation Review semantics;
- modify Portfolio Review semantics;
- modify Delivery semantics;
- introduce BUY / SELL / HOLD;
- introduce target price;
- introduce allocation;
- introduce position sizing;
- introduce ranking;
- introduce urgency;
- introduce conviction;
- introduce tradeability authority.

## 5. Expected Files

ME-SA01 expects the following docs-only files:

```text
docs/market_engine/source_data/me_sa01_automated_cached_source_acquisition_job_contract.md
docs/market_engine/audits/me_sa01_automated_cached_source_acquisition_job_contract_audit.md
docs/market_engine/backlog/me_sa01_automated_cached_source_acquisition_job_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa01_automated_cached_source_acquisition_job_contract_roadmap_entry.md
```

ME-SA01 may also update consolidated backlog and roadmap documents by inserting the prepared markdown blocks into:

```text
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## 6. Validation Expected

Required validation:

```bash
git diff --check
```

Expected result:

```text
no whitespace errors
```

No pytest run is required because ME-SA01 is docs-only and does not modify runtime code or tests.

A pytest run is allowed as an optional sanity check, but it is not required for ME-SA01 acceptance.

## 7. Governance Checks

| Check | Expected Result |
|---|---|
| Docs-only | PASS |
| Runtime code unchanged | PASS |
| Tests unchanged | PASS |
| No provider calls | PASS |
| No live data fetched | PASS |
| No yfinance | PASS |
| No SEC/EDGAR call | PASS |
| No internet use | PASS |
| No source data files created | PASS |
| No fake NVDA/AMD/ASML data created | PASS |
| No Telegram send | PASS |
| No portfolio/watchlist writes | PASS |
| No production writes | PASS |
| No downstream decision/recommendation semantics changed | PASS |
| No BUY / SELL / HOLD authority introduced | PASS |
| No target price/allocation/position sizing introduced | PASS |
| No ranking/urgency/conviction/tradeability introduced | PASS |

## 8. Acceptance Criteria

ME-SA01 is accepted when:

1. The contract document exists.
2. The audit document exists.
3. The backlog entry exists.
4. The roadmap entry exists.
5. Consolidated backlog and roadmap blocks are prepared.
6. `git diff --check` passes.
7. No runtime code is changed.
8. No tests are changed.
9. The next logical sprint is clearly identified as ME-SA02.

## 9. Conclusion

PASS once the ME-SA01 docs are applied and `git diff --check` reports no whitespace errors.

ME-SA01 establishes the governance contract for automated cached-source acquisition and safely bridges the project from operator-supplied packages toward application-owned acquisition.

Next sprint:

```text
ME-SA02 — Implement first bounded automated cached-source acquisition job
```
