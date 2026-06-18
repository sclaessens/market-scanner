# ME-UNI01 Roadmap Entry — Canonical ticker universe sequence

Job family: ME-UNI — Ticker Universe

Status: PLANNED

## Roadmap change

After ME-RUN15, the next strategic gap is not command visibility but analysis scope governance.

Market Engine needs a canonical ticker universe before broader real cached-source batch analysis, Telegram previews or Telegram delivery are allowed.

Canonical ticker universe path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Planned order

```text
ME-UNI01 — Define canonical ticker universe contract
ME-UNI02 — Implement canonical ticker universe loading and validation
ME-RUN16 — Execute first real cached-source batch dry-run using canonical ticker universe
ME-RUN17 — Broader cached-source batch review using canonical ticker universe
ME-TG01 — Define Telegram preview contract
ME-TG02 — Implement Telegram render-only preview
ME-TG03 — Implement gated Telegram delivery
```

## Blocking rule

ME-RUN16 is blocked until ME-UNI02 is completed.

ME-TG01 is blocked until ME-UNI02 and initial canonical-universe RUN validation are completed.

Telegram send behavior remains blocked until render-only previews and explicit safe gates are validated.
