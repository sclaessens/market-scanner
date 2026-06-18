# ME-RUN12 — Safe all-ticker cached-source batch dry-run contract audit

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN12

## Audit purpose

This audit verifies that ME-RUN12 defines a safe future contract for all-ticker or broader cached-source batch dry-runs without approving implementation, provider calls, production execution, delivery, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or action/allocation authority.

## Scope audited

Audited document:

```text
docs/market_engine/run/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract.md
```

Synchronized planning documents:

```text
docs/market_engine/backlog/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_backlog_entry.md
docs/market_engine/roadmap/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Baseline continuity check

ME-RUN12 correctly builds on the approved chain:

```text
ME-RUN05 -> optional local dry-run artifact persistence
ME-RUN06 -> local_snapshot_fixture input wrapper
ME-RUN07 -> realistic non-production local fixture artifact execution
ME-RUN08 -> deterministic local fixture matrix coverage
ME-RUN09 -> cached-source local execution contract
ME-RUN10 -> cached_source_snapshot local execution implementation
ME-RUN11 -> deterministic small ticker-bundle validation through per-ticker command invocation
```

The audit confirms that ME-RUN12 does not replace the existing per-ticker output contract:

```text
market-engine-end-to-end-dry-run-v1
```

The audit confirms that ME-RUN12 preserves the existing local artifact contracts:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Contract boundary check

ME-RUN12 defines the future batch-level wrapper contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

The audit confirms that this contract is defined as a local, deterministic, review-only batch summary over already-existing cached source snapshots.

The audit confirms that the contract is explicitly not any of the following:

* source refresh contract;
* provider adapter contract;
* live-data execution contract;
* all-ticker production execution contract;
* delivery or notification contract;
* portfolio or watchlist contract;
* scheduler or UI contract;
* broker, trading, allocation, ranking, scoring, conviction, urgency, target-price, target-weight, or tradeability contract.

## Input boundary check

ME-RUN12 limits approved future batch input to already-existing cached source snapshots under:

```text
data/market_engine/source_snapshots/
```

The audit confirms that local override roots remain allowed only when explicitly supplied, path-contained, local, non-production, and side-effect-free.

The audit confirms that ME-RUN12 does not approve provider refresh, automatic cache refresh, automatic cache cleanup, portfolio writes, watchlist writes, delivery writes, scheduler writes, UI writes, broker calls, or production report writes.

## Ticker universe check

ME-RUN12 defines approved ticker universe sources as:

* explicit operator ticker list;
* explicit local ticker-list JSON or CSV file;
* explicit discovery of cached snapshots inside the approved snapshot root;
* bounded local test fixture.

The audit confirms that ME-RUN12 forbids ticker universe discovery from broker APIs, broker scraping, live market-data providers, yfinance, SEC/EDGAR, Telegram/email, production watchlists unless a future contract explicitly approves a read-only boundary, hidden all-market universes, internet search, or external APIs.

## Discovery and ambiguity check

ME-RUN12 requires deterministic cached-source discovery rules for:

* filename or manifest references;
* source snapshot format version;
* source family;
* ticker identity;
* CIK identity when applicable;
* snapshot timestamp;
* retrieval timestamp when present;
* source snapshot age;
* stale-for-review state;
* required source fields;
* duplicate source snapshots;
* ambiguous ticker-to-CIK relationships;
* unsupported or obsolete snapshot versions.

The audit confirms that ME-RUN12 forbids silent arbitrary source selection.

The audit confirms that ambiguity must either be blocked, resolved by explicit operator reference, or resolved by a deterministic and audited selection policy.

## Per-ticker output check

ME-RUN12 preserves per-ticker execution as the unit of work.

The final per-ticker output remains:

```text
market-engine-end-to-end-dry-run-v1
```

The audit confirms that the batch wrapper may summarize execution state only and may not merge ticker evidence into a ranking, score, shortlist, trading queue, action list, allocation list, recommendation list, or tradeability output.

## Failure isolation check

ME-RUN12 requires ticker-level failure isolation.

The audit confirms that one ticker failure may not corrupt, overwrite, mutate, or invalidate successful per-ticker results for other tickers.

The audit confirms that non-completed ticker states must include deterministic reasons.

The audit confirms that unexpected local errors must remain local ticker-level failures and must not trigger provider calls, delivery, portfolio writes, watchlist writes, broker access, or automatic remediation.

## Batch output and artifact check

ME-RUN12 defines minimum batch output fields for:

* contract version;
* batch id;
* input mode;
* source snapshot root;
* requested tickers;
* execution state;
* counts;
* per-ticker results;
* warnings and blocked reasons;
* artifact reference;
* forbidden-side-effect confirmation;
* authority-boundary confirmation;
* provenance.

The audit confirms that artifact writing remains opt-in only.

The audit confirms that no artifact directory, per-ticker artifact, or batch manifest may be written when artifact writing is not explicitly requested.

The audit confirms that future artifact writing must preserve path containment, overwrite protection, deterministic timestamps, per-ticker artifact identity, and local-only semantics.

## Missing, stale, blocked, and numeric-zero check

ME-RUN12 preserves all existing data-safety requirements:

* missing fields remain explicit missing-data markers;
* stale cached snapshots remain explicit stale-data markers;
* numeric zero values remain valid evidence;
* unsupported source snapshots fail closed;
* incomplete source snapshots fail closed or produce blocked dry-run states;
* blocked stages record blocked reasons;
* local execution may not repair, enrich, normalize, or backfill data with live provider calls.

The audit confirms that batch summaries may count these states but may not hide per-ticker evidence.

## Forbidden behavior check

ME-RUN12 does not approve:

* Python runtime changes;
* test changes;
* fixture changes;
* source refresh jobs;
* SEC/EDGAR provider calls;
* yfinance or live market data calls;
* external API calls;
* automatic cache refresh;
* automatic cache cleanup;
* broker integration;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* all-ticker production execution;
* Decision Engine decisions;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

## Future implementation readiness

ME-RUN12 defines the logical next implementation sprint as:

```text
ME-RUN13 — Implement safe cached-source batch dry-run path
```

The audit confirms that ME-RUN13 is allowed only if it preserves the ME-RUN12 contract boundaries and remains local, deterministic, cached-source-only, provider-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

## Audit conclusion

ME-RUN12 satisfies its documentation-only acceptance criteria.

It defines the safe all-ticker cached-source batch dry-run contract, including input boundaries, discovery and ambiguity rules, ticker-level failure isolation, batch output expectations, artifact behavior, operator visibility, provenance, fail-closed behavior, forbidden side effects, and future ME-RUN13 implementation requirements.

No code, tests, fixtures, provider calls, live data calls, production writes, delivery behavior, portfolio behavior, watchlist behavior, scheduler behavior, UI behavior, or Decision Engine action/allocation authority is introduced.
