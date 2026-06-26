# ME-SA01 — Automated Cached-Source Acquisition Job Contract Backlog Entry

Status: COMPLETED BY ME-SA01  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26  

## Outcome

ME-SA01 defined the docs-only contract for the automated cached-source acquisition job.

The sprint establishes the primary route:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

This supersedes ME-SR13A as the primary route. Manual operator-supplied packages remain only a possible fallback/manual diagnostic candidate.

## Implemented Docs

ME-SA01 adds:

```text
docs/market_engine/source_data/me_sa01_automated_cached_source_acquisition_job_contract.md
docs/market_engine/audits/me_sa01_automated_cached_source_acquisition_job_contract_audit.md
docs/market_engine/backlog/me_sa01_automated_cached_source_acquisition_job_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa01_automated_cached_source_acquisition_job_contract_roadmap_entry.md
```

## Contract Defined

ME-SA01 defines:

- acquisition job purpose;
- architectural chain;
- job boundary;
- request format `market-engine-automated-cached-source-acquisition-request-v1`;
- result format `market-engine-automated-cached-source-acquisition-result-v1`;
- approved run modes;
- ticker input rules;
- source family rules;
- provider/source adapter policy;
- snapshot package compatibility;
- provenance requirements;
- freshness/staleness policy;
- missing-data policy;
- failure model;
- safety and side-effect constraints;
- handoff to existing import/staging/dry-run flow;
- ME-SA02 acceptance criteria.

## Non-Goals

ME-SA01 did not:

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

## Follow-Up Candidates

### ME-SA02 — Implement first bounded automated cached-source acquisition job

Next logical sprint.

Expected scope:

- bounded ticker list, initially `NVDA`, `AMD`, `ASML`, or smaller;
- at least one approved source family;
- deterministic fake adapter in tests;
- no real provider calls in tests;
- no network calls in tests;
- no yfinance;
- no SEC/EDGAR;
- local non-production artifact writing only;
- snapshot package compatible with existing import/staging validation.

### ME-RUN26 — Run automated cached-source acquisition for NVDA/AMD/ASML through staging validation and local dry-run

Expected after ME-SA02.

Expected scope:

- execute bounded acquisition;
- pass output through existing import/staging validation;
- attempt cached-source local dry-run;
- record PASS/BLOCKED audit.

### ME-TP01 — Produce terminal-visible operator preview from real cached-source dry-run artifacts

Expected after ME-RUN26 when real cached-source dry-run artifacts are available.

Expected scope:

- terminal-visible preview;
- Telegram-style formatting only;
- no Telegram send unless separately approved.

## Next Logical Sprint

```text
ME-SA02 — Implement first bounded automated cached-source acquisition job
```
