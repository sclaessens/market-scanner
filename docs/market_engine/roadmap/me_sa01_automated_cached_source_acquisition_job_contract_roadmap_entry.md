# ME-SA01 — Automated Cached-Source Acquisition Job Contract Roadmap Entry

Status: ACTIVE ROADMAP AFTER ME-SA01  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26  

## Roadmap Position

ME-SA01 completes the docs-only contract for the automated cached-source acquisition job.

The active route after ME-SA01 is:

```text
ME-SA01
  -> ME-SA02
  -> ME-RUN26
  -> ME-TP01
```

## ME-SA01 Completed

ME-SA01 defined the contract for:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

The contract preserves existing fail-closed cached-source validation and dry-run controls while moving the primary route away from manual operator-supplied source packages.

## Next Active Sprint

```text
ME-SA02 — Implement first bounded automated cached-source acquisition job
```

ME-SA02 should implement the first safe, bounded, local, non-production acquisition job according to the ME-SA01 contract.

Expected ME-SA02 implementation boundaries:

- bounded ticker list;
- at least one approved source family;
- deterministic fake adapter in tests;
- no real provider calls in tests;
- no network calls in tests;
- no yfinance;
- no SEC/EDGAR;
- no Telegram send;
- no portfolio/watchlist writes;
- no production writes;
- no Decision Engine, Recommendation Review, Portfolio Review, or Delivery semantic changes;
- snapshot package compatible with existing import/staging validation.

## Following Sprint

```text
ME-RUN26 — Run automated cached-source acquisition for NVDA/AMD/ASML through staging validation and local dry-run
```

ME-RUN26 should validate whether ME-SA02 output can pass through the existing cached-source chain.

Expected route:

```text
ME-SA02 acquisition output
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> audit result
```

## Preview Sprint

```text
ME-TP01 — Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

ME-TP01 should produce terminal-visible operator preview output from cached-source dry-run artifacts.

Telegram-style formatting may be used, but actual Telegram sending remains out of scope unless separately approved by a later sprint.

## Superseded Primary Route

ME-SR13A is no longer the primary route.

It may remain as a fallback/manual diagnostic candidate for operator-supplied source packages, but the roadmap now prioritizes application-owned acquisition.

## Roadmap Conclusion

After ME-SA01, the active roadmap should proceed to ME-SA02.

ME-SA02 is the bridge from contract to implementation.
ME-RUN26 is the bridge from implementation to validated cached-source dry-run.
ME-TP01 is the bridge from validated dry-run artifacts to operator-visible preview.
