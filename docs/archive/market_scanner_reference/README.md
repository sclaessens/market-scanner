# Market Scanner Reference Archive

Owner role: Governance Auditor / Scrum Master

Status: LEGACY REFERENCE ARCHIVE

## Purpose

This archive preserves pre-Market Engine market-scanner documentation as historical reference material.

The active Market Engine documentation root is now:

```text
docs/market_engine/
```

Documents in this archive are not active Market Engine authority by default. They may be used only as reference sources for explicit Market Engine extraction, specification, audit, or implementation work.

## Archive Principle

The Market Engine line keeps old documentation as evidence and knowledge, but does not treat it as the active product direction.

Archived reference material may contain useful lessons about:

- operator goals;
- source-data readiness;
- provider governance;
- missing-data handling;
- scanner boundaries;
- fundamental-analysis boundaries;
- Decision Engine authority;
- reporting and Telegram boundaries;
- portfolio/watchlist boundaries;
- testing conventions;
- runtime and side-effect risks.

Archived reference material must not be blindly copied into Market Engine implementation.

## Active Documentation Rule

From the Market Engine line onward:

```text
docs/market_engine/ = the only active Market Engine documentation root
```

The former `docs/active/` tree is historical v2/BL/reset documentation and is archived under:

```text
docs/archive/market_scanner_reference/active/
```

## Use Rules

When using archived documents:

1. Inspect the archived source.
2. Extract useful logic or lessons.
3. Decide whether Market Engine keeps, rejects, or defers the concept.
4. Record implementation implications.
5. Record testing implications.
6. Keep Market Engine authority inside `docs/market_engine/`.

## Non-Authority Statement

This archive does not authorize:

- Python implementation;
- test implementation;
- provider calls;
- yfinance calls;
- SEC / EDGAR calls;
- production writes;
- report generation;
- Telegram delivery;
- portfolio mutation;
- watchlist mutation;
- Decision Engine behavior changes;
- BUY / SELL / HOLD behavior;
- allocation;
- urgency;
- conviction;
- tradeability;
- recommendation behavior.
