# ME04-PREP Archive Active Docs Audit

Owner role: Governance Auditor

Status: ME04-PREP AUDIT

## Purpose

This audit records `ME04-PREP - Archive old active documentation and make Market Engine the only active docs root`.

## Moved

The former active documentation tree was moved with `git mv`:

```text
docs/active/
-> docs/archive/market_scanner_reference/active/
```

## Files Created

- `docs/archive/market_scanner_reference/README.md`
- `docs/market_engine/reference_extraction/legacy_reference_map.md`
- `docs/market_engine/audits/me04prep_archive_active_docs_audit.md`

## Files Updated

- `docs/market_engine/backlog/market_engine_backlog.md`

## Active Root Confirmation

`docs/market_engine/` remains the only active Market Engine documentation root.

The old `docs/active/` content is preserved as historical reference material under `docs/archive/market_scanner_reference/active/`.

Archived documents do not authorize implementation, provider calls, reporting, Telegram, portfolio/watchlist mutation, Decision Engine behavior, or runtime behavior.

## Boundary Confirmations

- No Python files were changed.
- No test files were changed.
- No provider calls were executed.
- No yfinance calls were executed.
- No SEC or EDGAR calls were executed.
- No scanner commands were executed.
- No reporting commands were executed.
- No Telegram commands were executed.
- No portfolio or watchlist commands were executed.
- No runtime commands were executed.
- No production writes were introduced.
- No reports were generated.
- No Telegram delivery was triggered.
- No portfolio data was mutated.
- No watchlist data was mutated.
- No Decision Engine behavior was changed.
- `docs/market_engine/` was not archived or moved.
- No old documentation was deleted.

## Known Limitations

Existing Market Engine extraction documents may still mention `docs/active/...` because those paths were accurate when the extraction occurred. ME04-PREP preserves those historical references rather than rewriting extraction evidence.

Old archived documents may contain language that described them as active at the time they were written. The archive README and Market Engine legacy reference map supersede that authority for Market Engine work.

## Readiness For ME04

ME04 can proceed with a single active Market Engine documentation root and a preserved reference archive for old v2, BL, and reset documentation.

## Next Recommended Sprint

ME04 - Extract and write Market Engine technical, coding, and testing architecture.
