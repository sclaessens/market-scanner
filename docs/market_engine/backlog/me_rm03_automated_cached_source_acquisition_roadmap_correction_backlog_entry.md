# ME-RM03 - Automated cached-source acquisition roadmap correction backlog entry

Sprint ID: ME-RM03

Status: COMPLETED BY ME-RM03

Job family: ME-RM / Roadmap Governance

## Decision

Automated cached-source acquisition by a dedicated Market Engine application job is now the primary product direction.

Manual operator-supplied cached-source packages remain useful as fallback, diagnostic, or recovery inputs, but they are no longer the primary operating model.

## Superseded Primary Candidate

```text
ME-SR13A - Prepare real-world operator-supplied cached-source input package for NVDA, AMD, ASML
```

ME-SR13A may remain a fallback/manual diagnostic candidate only.

## Next Active Sprint

```text
ME-SA01 - Define automated cached-source acquisition job contract
```

## Recommended Sequence

```text
ME-SA01 - Define automated cached-source acquisition job contract
ME-SA02 - Implement first bounded automated cached-source acquisition job for approved sample tickers/source families
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
ME-TP01 - Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

## ME-SA01 Scope

ME-SA01 must define:

* acquisition job inputs;
* approved ticker universe or bounded ticker list input;
* approved source families;
* approved provider/source adapters;
* provenance requirements;
* retrieval timestamp and source timestamp;
* freshness/staleness policy;
* missing-data handling;
* cached-source snapshot output location;
* manifest compatibility with the existing validator/import flow;
* fail-closed behavior;
* no downstream side effects;
* no analysis or decision authority.

## Implemented documentation

```text
docs/market_engine/audits/me_rm03_automated_cached_source_acquisition_roadmap_correction.md
docs/market_engine/backlog/me_rm03_automated_cached_source_acquisition_roadmap_correction_backlog_entry.md
docs/market_engine/roadmap/me_rm03_automated_cached_source_acquisition_roadmap_correction_roadmap_entry.md
```
