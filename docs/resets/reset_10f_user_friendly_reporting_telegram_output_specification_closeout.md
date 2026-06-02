# RESET-10F — User-Friendly Reporting and Telegram Output Specification Closeout

## Purpose

RESET-10F defines the target v2 user experience for Telegram and reporting communication.

The specification is based on the RESET-10E0 review and the agreed compact Telegram structure.

## Result

Created:

- `docs/active/reporting_telegram_ux.md`

The document defines a compact, portfolio-first Telegram output structure:

```text
Market Scanner

Portfolio
ASML: +8.4% | price €XXX | target €XXX | REVIEW
Thales: +14.2% | price €XXX | target €XXX | HOLD
Costco: -3.1% | price $XXX | target $XXX | REVIEW
ETFs: +2-8% | keep accumulating

Buy now
No candidates today.

Buy on pullback
AMD below $XXX
ASML below €XXX

Buy on breakout
NVIDIA above $XXX
Meta above $XXX

Data status
Fundamental data incomplete -> many REVIEW.
```

## Decision

Decision: USE_PORTFOLIO_FIRST_COMPACT_TELEGRAM_UX_AS_V2_BASELINE

The legacy Telegram output must not be translated directly into v2.

## Scope Confirmation

Documentation-only.

No files were modified under:

- `scripts/`
- `src/`
- `tests/`
- `data/`
- `reports/`
- `.github/workflows/`

No runtime was executed.
No Telegram message was sent.
No report artifact was generated.
No production pipeline, SEC/provider/network/broker/Telegram/live data call was made.

## Recommended Next Action

RESET-9C2D — Translate Reporting and Telegram UX Requirements to V2 Contracts.

Executor: Codex/local implementation.
