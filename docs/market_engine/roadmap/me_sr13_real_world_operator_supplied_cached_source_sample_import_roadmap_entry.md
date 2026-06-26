# ME-SR13 - Real-world operator-supplied cached-source sample import roadmap entry

BLOCKED BY ME-SR13.

## Roadmap position

```text
ME-SR12 -> ME-RUN25 -> ME-SR13 -> ME-RM03 -> ME-SA01
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

## Corrected next logical sprint

```text
ME-SA01 - Define automated cached-source acquisition job contract
```

ME-RM03 supersedes ME-SR13A as the primary next sprint. ME-SR13A remains available only as a fallback/manual diagnostic candidate.
