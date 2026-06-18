# ME-UNI01 Backlog Entry — Canonical ticker universe sequence

Job family: ME-UNI — Ticker Universe

Status: PLANNED

## Backlog insertion reason

ME-UNI01 is inserted after ME-RUN15 because ME-RUN15 completed operator-facing cached-source batch dry-run command visibility, but the project still lacks a canonical, editable and extensible ticker universe.

Current batch input modes are suitable for technical dry-runs but not sufficient for repeatable analysis scope, Telegram preview eligibility or future delivery workflows.

## Required sequence

```text
ME-UNI01 — Define canonical ticker universe contract
ME-UNI02 — Implement canonical ticker universe loading and validation
ME-RUN16 — Execute first real cached-source batch dry-run using canonical ticker universe
ME-RUN17 — Broader cached-source batch review using canonical ticker universe
ME-TG01 — Define Telegram preview contract
ME-TG02 — Implement Telegram render-only preview
ME-TG03 — Implement gated Telegram delivery
```

## ME-UNI01 scope

ME-UNI01 is documentation-only.

It must define the canonical ticker universe path, file format, required and optional columns, active/inactive semantics, priority semantics, source-policy semantics, validation rules, downstream RUN integration requirements and ME-UNI02 implementation requirements.

Canonical ticker universe path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## ME-UNI02 scope

ME-UNI02 must implement loading and validation only after ME-UNI01 defines the contract.

ME-UNI02 may introduce runtime code and tests for loading the canonical ticker universe, validating required columns, rejecting duplicate tickers, rejecting invalid tickers, filtering active tickers, preserving deterministic ordering and exposing validated ticker selections to RUN jobs.

## Dependency rule

ME-RUN16 must be considered blocked until ME-UNI02 is completed.

ME-TG01 must be considered blocked until ME-UNI02 and initial canonical-universe cached-source RUN validation are completed.
