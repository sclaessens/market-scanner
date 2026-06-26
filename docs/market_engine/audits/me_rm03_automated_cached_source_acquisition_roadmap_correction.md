# ME-RM03 - Automated cached-source acquisition roadmap correction

Date: 2026-06-26

## Objective

ME-RM03 records the product-owner roadmap correction after ME-SR13. Automated cached-source acquisition by a dedicated Market Engine application job is now the primary direction.

## Context

ME-RUN25 proved that a valid cached-source snapshot package can be imported, staged, validated, and used by the existing local `cached_source_snapshot` dry-run flow.

ME-SR13 attempted the first real-world operator-supplied cached-source sample import for `NVDA`, `AMD`, and `ASML`, but the run was blocked because the expected local operator input package was missing:

```text
operator_input/market_engine/me-sr13-real-world-sample/
```

The blocked ME-SR13 result is valid evidence that manual operator package preparation should not be the primary operating model.

## Product-Owner Decision

The Market Engine application must own automated acquisition of required source data through a dedicated application job.

Corrected primary chain:

```text
automated cached-source acquisition job
-> cached-source snapshot package
-> existing import/staging validator
-> existing cached_source_snapshot dry-run
-> terminal-visible / Telegram-style operator preview
```

The manual operator-supplied route remains useful as a fallback, diagnostic, or recovery path. It is no longer the primary next route toward real cached-source analysis.

## Roadmap Correction

Superseded primary candidate:

```text
ME-SR13A - Prepare real-world operator-supplied cached-source input package for NVDA, AMD, ASML
```

Next active sprint:

```text
ME-SA01 - Define automated cached-source acquisition job contract
```

Recommended sequence:

```text
ME-SA01 - Define automated cached-source acquisition job contract
ME-SA02 - Implement first bounded automated cached-source acquisition job for approved sample tickers/source families
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
ME-TP01 - Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

## Existing Route Preserved

The ME-SR12 / ME-RUN25 route remains valid:

* ME-SR12 import command;
* ME-SR10 staging validator;
* existing `cached_source_snapshot` dry-run input mode;
* local-only validation and dry-run boundaries.

The future automated acquisition job should produce cached-source snapshot packages that remain compatible with the existing manifest, validation, import, and dry-run bridge instead of bypassing it.

## ME-SA01 Scope

ME-SA01 must define:

* acquisition job inputs;
* approved ticker universe or bounded ticker list input;
* approved source families;
* approved provider/source adapters;
* provenance requirements;
* retrieval timestamp and source timestamp requirements;
* freshness and staleness policy;
* missing-data handling;
* cached-source snapshot output location;
* manifest compatibility with the existing validator/import flow;
* fail-closed behavior;
* no downstream side effects;
* no analysis or decision authority.

## Safety Boundaries

ME-RM03 is documentation-only. It does not change runtime code, tests, provider access, source-data fetching, yfinance behavior, SEC/EDGAR behavior, Telegram behavior, production portfolio/watchlist data, Decision Engine logic, recommendation semantics, portfolio review semantics, or delivery semantics.

## Conclusion

```text
COMPLETED BY ME-RM03
```

ME-SA01 is now the next active sprint. ME-SR13A is retained only as a fallback/manual diagnostic candidate.
