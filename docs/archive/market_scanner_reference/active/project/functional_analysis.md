# Functional Analysis

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the functional expectations for v2 from the operator perspective.

## Operator Goals

The operator needs to:

- run a deterministic market scan;
- see preserved opportunities, not silently filtered outputs;
- understand structural validation, context, fundamentals readiness, timing state, and portfolio state;
- receive final decisions from one authority only;
- understand why final decisions are REVIEW, BUY, HOLD, SELL, or other approved states;
- receive reporting that explains decisions without changing them;
- review historical outcomes under an explicit research model.

## Functional Flow

The v2 functional flow should be:

```text
operator input
-> approved source inputs
-> opportunity discovery
-> classification layers
-> portfolio context
-> Decision Engine
-> reporting
-> optional research feedback loop
```

## Acceptance Requirements

v2 must satisfy these functional requirements:

- every opportunity row must retain traceable identity where the contract requires it;
- each layer must explain what it adds and what it does not decide;
- source-data readiness must be separated from investment quality;
- portfolio presence must be descriptive and must not allocate;
- final action semantics must come only from the Decision Engine;
- reporting must not suppress or reinterpret source decisions;
- generated outputs must be clearly labelled as generated.

## Functional Non-Requirements

v2 does not need live trading, broker execution, fully automated portfolio rebalancing, or real-time market monitoring during the reset sequence.

## Operator Visibility

The v2 design should support better visibility than the old pipeline:

- clear terminal or run status;
- clear freshness status of required inputs and generated outputs;
- clear row counts by layer;
- clear source-data insufficiency reasons;
- clear final decision reason summaries;
- clear Telegram/reporting communication status.

## Research and Learning

Prediction tracking and self-analysis are research-only until a governed Decision Engine integration exists. Research outputs must not create allocation authority by themselves.
