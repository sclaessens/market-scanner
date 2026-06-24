# ME-SR06 — Expanded Professional Swing Universe source-support backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR06

## Goal

Classify cached-source support for expanded/proposed Professional Swing Universe entries produced by ME-UNI09.

## Scope

ME-SR06 is source-support-only. It consumes ME-UNI09 expansion output and reuses ME-SR05 Professional Swing source-support classification rules.

In scope:

* ME-UNI09 `market-engine-professional-swing-universe-expansion-v1` input;
* expanded `final_universe_entries` validation;
* existing-universe vs expansion-candidate provenance;
* supported/missing/unsupported/malformed/ambiguous/manual-review/excluded source-support states;
* deterministic summary counts;
* tests and documentation.

Out of scope:

* provider calls;
* SEC/EDGAR fetch;
* yfinance;
* broker calls;
* Telegram;
* portfolio/watchlist mutation;
* Decision Engine changes;
* BUY/SELL/HOLD;
* target prices;
* ranking/scoring/urgency/conviction/tradeability;
* allocation/order/execution semantics;
* generated artifact commits.

## Outcome

ME-SR06 implemented `market-engine-expanded-professional-swing-source-support-v1` as a thin wrapper over the existing ME-SR05 source-support classifier.

The implementation:

* validates ME-UNI09 expansion result input;
* converts `final_universe_entries` into a temporary editable Professional Swing Universe CSV shape;
* invokes the existing ME-SR05 classifier;
* enriches each ticker result with expanded-universe provenance;
* preserves explicit source-support states and summary counts;
* remains deterministic and auditable.

## Next candidate

The active next candidate remains:

```text
ME-RUN23 - Execute expanded supported-universe cached-source scan with readable report and candidate classification
```

ME-SR06 did not insert a refinement sprint ahead of ME-RUN23 because no blocker was discovered.

## Local validation

Steven must run the local test suite in his checkout before merge because the GitHub-only implementation was not executed inside Steven's local `.venv` or artifact tree.
