# Resolve Live SEC CompanyFacts Smoke Failure

## Backlog Item

RESET-10L-BL52 — Resolve Live SEC CompanyFacts Smoke Failure

## Purpose

Resolve the BL51 controlled live SEC CompanyFacts smoke failure without broadening runtime behavior, enabling persistence, changing Decision Engine authority, modifying Telegram delivery, or touching portfolio/watchlist data.

## BL51 Failure Summary

The BL51 controlled live SEC CompanyFacts one-ticker smoke failed closed before network execution.

The local operator-supplied `SEC_USER_AGENT` value was missing.

No live SEC request was executed.

No SEC payload was retrieved.

No data was written.

No raw payload, cache, report, Telegram artifact, portfolio update, watchlist update, workflow integration, scanner integration, or recommendation behavior was created or modified.

## Resolution Decision

BL52 confirms that the BL51 failure was correct fail-closed behavior, not a runtime defect.

The canonical live-smoke boundary already requires all pre-flight gates to pass before a request can be executed.

The missing `SEC_USER_AGENT` remains an operator configuration blocker.

BL52 therefore resolves the failure by documenting the required remediation path:

1. keep live smoke disabled by default;
2. require explicit local invocation;
3. require the approved NVDA / CIK `0001045810` target;
4. require a locally supplied `SEC_USER_AGENT`;
5. execute at most one SEC CompanyFacts request only after explicit operator approval;
6. write no production data;
7. preserve Decision Engine, Telegram, portfolio, and watchlist isolation.

## Code Inspection Summary

The controlled live smoke implementation is located at:

* `src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py`

The associated unit tests are located at:

* `tests/unit/test_v2_sec_companyfacts_live_smoke.py`

Inspection confirms that:

* live execution is disabled by default;
* missing `SEC_USER_AGENT` fails closed before network execution;
* malformed `SEC_USER_AGENT` fails closed before network execution;
* wrong ticker or CIK fails closed before network execution;
* the only approved target remains NVDA / CIK `0001045810`;
* the network fetcher is injectable for unit tests;
* the controlled path performs at most one request;
* results redact raw SEC payload details and the User-Agent;
* module import has no filesystem side effects;
* the module does not import script-era files;
* the module does not depend on `requests`, `yfinance`, or `yf`.

## Validation

Targeted validation was executed:

```text
pytest tests/unit/test_v2_sec_companyfacts_live_smoke.py
```

Result:

```text
13 passed in 0.10s
```

## Live Network Status

No live SEC request is executed by this BL52 documentation-only resolution.

A future controlled live retry may be performed only with explicit operator approval and a locally supplied `SEC_USER_AGENT`.

## Data Write Status

No data is written.

No cache is created.

No raw SEC payload is committed.

No generated report is produced.

No Telegram artifact is produced.

No portfolio or watchlist file is modified.

## Decision Engine Status

Decision Engine authority remains unchanged.

A successful future live smoke will not equal SEC CompanyFacts source approval.

SEC CompanyFacts may not influence final recommendations until source approval, persistence governance, completeness/freshness validation, and Decision Engine authority review are complete.

## Follow-up

Recommended next step:

RESET-10L-BL53 — Document Live Provider Smoke Governance

Optional controlled retry:

A future controlled live SEC CompanyFacts one-ticker smoke may be executed only if:

* the operator explicitly approves one live request;
* `SEC_USER_AGENT` is locally configured;
* the target remains NVDA / CIK `0001045810`;
* no production data is written;
* no cache or raw payload is committed;
* the result is documented as audit evidence.
