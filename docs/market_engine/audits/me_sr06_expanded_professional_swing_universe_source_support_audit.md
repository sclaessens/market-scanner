# ME-SR06 — Expanded Professional Swing Universe source-support audit

Status: COMPLETED BY ME-SR06

## Audit scope

ME-SR06 was audited for source-support-only behavior over the ME-UNI09 expanded/proposed Professional Swing Universe.

The audit checks that the sprint:

* reuses existing ME-SR05 source-support classification as much as possible;
* consumes ME-UNI09 expansion output safely;
* preserves explicit missing/unsupported/malformed/manual-review/ambiguous source states;
* keeps deterministic, auditable output;
* does not introduce provider, broker, delivery, portfolio, watchlist, Decision Engine, or trade-authority behavior.

## Runtime review

Implemented runtime:

```text
src/market_engine/source_support/expanded_professional_swing.py
src/market_engine/source_support/__init__.py
```

The implementation creates a narrow wrapper over the existing ME-SR05 classifier instead of duplicating source-support rules. The wrapper writes the expanded ME-UNI09 `final_universe_entries` into a temporary CSV that matches the editable Professional Swing Universe shape, invokes the existing ME-SR05 classifier, and enriches each result with expanded-universe provenance.

## Safety boundaries reviewed

Confirmed boundaries:

* no provider calls;
* no SEC/EDGAR fetch;
* no yfinance;
* no broker calls;
* no Telegram;
* no portfolio/watchlist mutation;
* no Decision Engine changes;
* no BUY/SELL/HOLD output;
* no target-price output;
* no ranking/scoring/urgency/conviction/tradeability semantics;
* no allocation/order/execution semantics;
* no artifact commits.

## Source-support behavior reviewed

ME-SR06 preserves ME-SR05 statuses:

* `supported_cached`;
* `missing_snapshot`;
* `unsupported_sec_companyfacts`;
* `missing_required_source_field`;
* `malformed_or_unreadable_source_artifact`;
* `ambiguous_identity`;
* `manual_review_only`;
* `excluded`.

ME-SR06 adds expanded-universe provenance:

* `existing_universe` for entries already present in the input Professional Swing Universe;
* `expansion_candidate` for entries included by ME-UNI09 candidate expansion;
* candidate source id/reference when available;
* original input-universe row reference for existing rows;
* candidate classification path and inclusion decision for newly included candidates.

## Fail-closed behavior reviewed

The implementation fails closed for unsupported input contract, malformed `final_universe_entries`, duplicate ticker/market keys, unsafe parent traversal in paths, invalid referenced input universe, and lost provenance.

Malformed or unreadable cached source artifacts remain handled by the reused ME-SR05 classifier and are surfaced as explicit source-support states.

## Test review

Implemented tests:

```text
tests/market_engine/source_support/test_expanded_professional_swing_source_support.py
```

The tests cover supported cached candidates, missing snapshots, manual-review-only rows, unsupported SEC CompanyFacts states, malformed source artifacts, existing rows, newly included ME-UNI09 candidates, deterministic ordering, summary counts, unsupported input contracts, duplicate ticker/market fail-closed behavior, no provider/network imports, and no forbidden action-authority wording in normal output.

## Consolidated planning note

ME-SR06 is the completed scale-first source-support layer after ME-UNI09. The next active candidate remains:

```text
ME-RUN23 - Execute expanded supported-universe cached-source scan with readable report and candidate classification
```

No refinement sprint was inserted ahead of ME-RUN23 because ME-SR06 did not document a blocker requiring insertion.

## Local validation requirement

This audit was prepared through GitHub edits only. Full local validation remains required in Steven's checkout because this environment does not have access to Steven's local `.venv` or local artifact tree.
