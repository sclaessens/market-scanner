# ME-UNI09 - Expand Professional Swing Universe From Candidates

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI09

## Goal

Implement controlled, deterministic Professional Swing Universe expansion from non-actionable candidate-classification output.

## Outcome

ME-UNI09 added `market-engine-professional-swing-universe-expansion-v1` as a pure universe-maintenance builder.

Implemented behavior:

* consumes `market-engine-candidate-classification-v1`;
* preserves existing editable Professional Swing Universe entries;
* includes only eligible `ready_for_manual_candidate_review` candidates;
* validates proposed universe rows through the existing editable Professional Swing Universe loader;
* excludes already-present, duplicated, manual-review-only, ambiguous, unsupported, non-equity, missing-source, malformed, and ineligible candidates with explicit reasons;
* fails closed on unknown candidate buckets, malformed summaries, unsupported format versions, invalid tickers, conflicting identities, unsafe paths, and invalid proposed universe entries;
* returns deterministic summary counts and auditable per-candidate decisions;
* performs no file writes and does not mutate the canonical universe CSV.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/ticker_universe -q
```

Result:

```text
67 passed in 0.08s
```

Full Market Engine validation is recorded in the final ME-UNI09 implementation notes.

## Boundaries

ME-UNI09 does not add provider calls, live data calls, source refresh, broker integration, Telegram/email delivery, reporting delivery, portfolio writes, watchlist writes, production data writes, scheduler behavior, UI behavior, Decision Engine changes, action-oriented instructions, ranking, scoring, target prices, urgency, conviction, allocation, order, or execution semantics.

## Next

```text
ME-SR06 - Classify source support for expanded Professional Swing Universe
ME-RUN23 - Execute expanded supported-universe cached-source scan with readable report and candidate classification
```
