# ME-SR13 - Real-world operator-supplied cached-source sample import roadmap entry

BLOCKED BY ME-SR13.

## Roadmap position

```text
ME-SR12 -> ME-RUN25 -> ME-SR13 -> ME-SR13A -> ME-SR13 rerun or ME-SR14
```

ME-SR13 was intended to move from the ME-RUN25 fixture-backed validation flow to real operator-supplied cached-source samples for `NVDA`, `AMD`, and `ASML`.

## Result

```text
BLOCKED
```

The expected local input root was missing:

```text
/Users/sclaessens/Documents/market-scanner/operator_input/market_engine/me-sr13-real-world-sample/
```

No import, staging validation, or local cached-source dry-run was attempted because the sprint was not allowed to fabricate source files or silently fall back to fixture data.

## Next logical sprint

```text
ME-SR13A - Prepare real-world operator-supplied cached-source input package for NVDA, AMD, ASML
```

ME-SR13A should prepare the local input package, verify per-ticker directory availability, confirm manifest/payload contract compatibility, and preserve the same local-only, provider-free, production-safe boundaries.
