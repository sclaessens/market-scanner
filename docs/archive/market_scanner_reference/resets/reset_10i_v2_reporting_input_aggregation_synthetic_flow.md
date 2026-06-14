# RESET-10I — V2 Reporting Input Aggregation Synthetic Flow

## Purpose

This document records the synthetic in-memory flow proven by RESET-10I.

It demonstrates that explicit reporting input records can be assembled into the v2 Telegram renderer input without reading real data, generating reports, or sending Telegram messages.

## Synthetic Input Categories

RESET-10I uses only synthetic in-memory records:

- portfolio display input;
- candidate display input;
- source-data status input.

Each input carries source reference metadata required by the reporting input aggregation contract.

## Adapter Routing

The synthetic adapter routes:

- portfolio display inputs to the Telegram Portfolio section;
- candidate display inputs with `BUY_NOW` to the Buy now section;
- candidate display inputs with `BUY_ON_PULLBACK` to the Buy on pullback section;
- candidate display inputs with `BUY_ON_BREAKOUT` to the Buy on breakout section;
- source-data status input to the Data status section.

The adapter preserves display-ready values exactly as supplied.

## Renderer Output

The synthetic flow renders:

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

## Synthetic Data Confirmation

All records in the RESET-10I flow are synthetic.

No real portfolio data, provider data, SEC data, generated CSV output, report artifact, or Telegram message is used.

## Adapter Boundary

The adapter does not calculate:

- decisions;
- target prices;
- buy thresholds;
- breakout thresholds;
- profit/loss;
- rankings;
- scores;
- allocation;
- execution instructions;
- urgency;
- conviction;
- tradeability.

The adapter does not read files, write files, generate reports, send Telegram messages, import legacy `scripts`, or call providers/network services.
