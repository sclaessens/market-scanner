# Market Scanner Reference Archive

Owner role: Governance Auditor

Status: HISTORICAL REFERENCE

## Purpose

This folder preserves old v2, BL, and reset documentation as historical reference material.

`docs/market_engine/` is the only active Market Engine documentation root.

`docs/archive/market_scanner_reference/active/` contains the former `docs/active/` tree.

## Reference Rules

Archived documents are reference sources only. They preserve institutional evidence, prior decisions, prior constraints, and lessons learned, but they do not authorize new Market Engine implementation.

Archived documents do not authorize:

- Python code changes;
- test changes;
- provider calls;
- yfinance calls;
- SEC or EDGAR calls;
- scanner or runtime execution;
- production writes;
- report generation;
- Telegram delivery;
- portfolio mutation;
- watchlist mutation;
- Decision Engine behavior changes;
- BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, or recommendation behavior.

Old documents may be used only through explicit Market Engine extraction. Future Market Engine work must inspect archived reference material, extract useful lessons, decide keep / reject / defer, record implementation and testing implications, and then continue from the active Market Engine specifications.
