# V2 Persistence Integration Review

Status: ACTIVE
Reset stage: RESET-10L-BL16

## Purpose

This document reviews whether the controlled synthetic persistence boundary
created in `RESET-10L-BL15` may be integrated with upstream provider output, and
under which constraints a later sprint may proceed.

This is a review-only governance artifact. It does not authorize code changes,
test changes, production data writes, provider calls, SEC or EDGAR calls, broker
calls, network calls, production pipeline execution, report generation, Telegram
delivery, Decision Engine behavior, scoring, recommendations, BUY, SELL, HOLD,
allocation, conviction, urgency, target-price, or tradeability logic.

## Reviewed Inputs

The review is based on the completed RESET-10L chain:

- `RESET-10L-BL11 — Real-Source Capture Persistence Design`;
- `RESET-10L-BL12 — Persistence Contract and Fixture Design`;
- `RESET-10L-BL13 — Synthetic Persistence Contract Tests`;
- `RESET-10L-BL14 — Controlled Persistence Implementation Design`;
- `RESET-10L-BL15 — Controlled Synthetic Persistence Implementation`.

Relevant implementation boundary:

```text
src/market_scanner/fundamentals/fundamentals_persistence.py
```

Relevant test boundary:

```text
tests/unit/test_v2_fundamentals_persistence.py
tests/contract/test_v2_persistence_raw_evidence_contracts.py
tests/contract/test_v2_persistence_normalized_fundamentals_contracts.py
tests/contract/test_v2_persistence_readiness_contracts.py
tests/contract/test_v2_persistence_fixture_contracts.py
```

Relevant fixture boundary:

```text
tests/fixtures/fundamentals/persistence/
```

## Review Finding

The controlled synthetic persistence boundary is suitable for a future,
separately approved integration step with upstream provider output, but only
under strict synthetic and no-production-write constraints.

The current BL15 implementation should be treated as a persistence safety
boundary, not as a production storage layer.

Approved current capability:

- validate raw evidence-shaped records;
- validate normalized fundamentals-shaped records;
- validate readiness-shaped records;
- preserve provenance linkage;
- preserve explicit missing-value states;
- reject forbidden production, report, Telegram, and workflow paths;
- write synthetic records only to caller-provided temporary roots;
- return deterministic write metadata;
- remain disconnected from provider execution, reports, Telegram, the production
  pipeline, and the Decision Engine.

Not approved current capability:

- production writes under `data/`;
- automatic provider execution;
- real provider capture persistence;
- report generation;
- Telegram delivery;
- pipeline integration;
- portfolio or watchlist integration;
- Decision Engine investment use;
- BUY, SELL, HOLD, allocation, conviction, urgency, target-price, scoring,
  recommendation, or tradeability behavior.

## Integration Boundary Assessment

A future integration step may connect provider-boundary output to the persistence
boundary only as an explicit, in-memory handoff.

Allowed future integration pattern:

```text
provider/source response
-> provider adapter normalization
-> in-memory raw evidence / normalized fundamentals / readiness records
-> persistence validators
-> synthetic temporary persistence write in tests only
```

Forbidden future integration pattern:

```text
provider/source response
-> automatic production data write
-> report generation
-> Telegram delivery
-> Decision Engine action
```

The persistence boundary must remain passive. It must not call providers, fetch
source data, inspect portfolios, run reports, trigger Telegram, or invoke the
Decision Engine.

## Conditions for a Future Integration Sprint

A future integration sprint may be approved only if it remains limited to:

- synthetic provider-boundary output;
- in-memory handoff into the persistence boundary;
- pytest-managed temporary directories;
- contract tests proving no production paths are touched;
- no live provider calls;
- no credentials;
- no raw live payloads;
- no report generation;
- no Telegram artifacts;
- no production pipeline execution;
- no Decision Engine investment behavior.

The next sprint should not yet persist real provider output to production data
paths.

## Required Future Test Coverage

A future integration sprint should add tests proving that:

- provider adapter output can be transformed into persistence-ready raw evidence
  records;
- normalized fundamentals retain raw evidence linkage;
- readiness records retain raw and normalized linkage;
- missing values remain explicit;
- missing values are not converted to zero;
- partial, invalid, stale, and provenance-gap states remain neutral;
- forbidden investment semantics are rejected or flagged;
- synthetic writes remain inside pytest `tmp_path`;
- forbidden production paths are rejected;
- report and Telegram paths are rejected;
- provider calls are not made by persistence code;
- network calls are not made by persistence code;
- pipeline behavior is not invoked;
- Decision Engine behavior is not imported or called.

## Production Write Readiness

Production persistence is not ready yet.

Before production writes can be considered, the project still needs:

1. an integration contract between provider adapter output and persistence input;
2. synthetic integration tests using fake provider output;
3. explicit path policy for production persistence;
4. approval of append/version semantics;
5. rollback and cleanup design;
6. audit and retention design;
7. no-side-effect tests for production pipeline, reports, Telegram, and Decision
   Engine boundaries;
8. a separate governance decision approving production data writes.

## Risk Assessment

Primary risks if integration proceeds too quickly:

- raw source evidence and normalized fundamentals may become coupled;
- missing values may be hidden or zero-filled;
- production data files may be modified before path policy is approved;
- provider execution may become coupled to persistence;
- persistence may accidentally trigger downstream reporting or Telegram output;
- readiness may be misread as investment quality;
- Decision Engine authority boundaries may be bypassed.

Mitigation:

- keep next step synthetic-only;
- keep writes inside pytest temporary directories;
- keep provider output fake or fixture-based;
- keep integration tests explicit and narrow;
- require no-side-effect assertions;
- do not connect to pipeline, reports, Telegram, portfolio, watchlist, or Decision
  Engine flows.

## Integration Decision

Decision: proceed only to a synthetic provider-to-persistence integration contract
step.

Approved next candidate:

```text
RESET-10L-BL17 — Synthetic Provider-to-Persistence Integration Contracts
```

The next candidate may add synthetic integration tests that pass fake provider
adapter output into the controlled persistence boundary and write only to pytest
`tmp_path`.

The next candidate must not add production writes, live provider calls, pipeline
hooks, reports, Telegram delivery, portfolio/watchlist updates, or Decision
Engine behavior.

## Non-Goals

RESET-10L-BL16 does not:

- add or modify code;
- add or modify tests;
- add or modify fixtures;
- write production data;
- execute provider calls;
- execute SEC, EDGAR, broker, or network calls;
- add credentials;
- commit raw live payloads;
- generate reports;
- create Telegram artifacts;
- run the production pipeline;
- integrate with portfolio or watchlist flows;
- add Decision Engine behavior;
- approve investment analysis;
- add BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring,
  target-price, or recommendation behavior.

## Conclusion

RESET-10L-BL16 approves the next step only as synthetic integration contract
coverage.

The project should not yet move to production persistence or real analysis. The
safe next step is to prove that fake provider-boundary output can be handed to
the controlled persistence boundary while preserving provenance, missing-value
behavior, neutral readiness, and no-side-effect guardrails.
