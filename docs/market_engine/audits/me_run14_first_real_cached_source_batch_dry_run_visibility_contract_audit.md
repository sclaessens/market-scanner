# ME-RUN14 - First real cached-source batch dry-run execution and visibility contract audit

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN14

## Audit purpose

This audit verifies that ME-RUN14 defines the first real cached-source batch dry-run execution and visibility contract without introducing runtime code, tests, provider calls, source refresh, production execution, delivery channels, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or action/allocation authority.

## Scope audited

Audited contract document:

```text
docs/market_engine/run/me_run14_first_real_cached_source_batch_dry_run_visibility_contract.md
```

Synchronized planning documents:

```text
docs/market_engine/backlog/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_backlog_entry.md
docs/market_engine/roadmap/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_roadmap_entry.md
```

## Baseline continuity check

ME-RUN14 correctly builds on:

```text
ME-RUN09 -> cached-source local execution contract
ME-RUN10 -> cached-source per-ticker local execution implementation
ME-RUN11 -> deterministic ticker-bundle validation
ME-RUN12 -> safe broader cached-source batch dry-run contract
ME-RUN13 -> safe cached-source batch dry-run runtime implementation
```

The audit confirms that ME-RUN14 does not replace the existing batch contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

The audit confirms that ME-RUN14 does not replace the existing per-ticker output contract:

```text
market-engine-end-to-end-dry-run-v1
```

The audit confirms that local artifact identities remain:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Visibility contract check

ME-RUN14 defines the operator execution visibility contract:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

The audit confirms that this is a documentation and acceptance contract only. It is not a new runtime output contract and does not require a new data model in ME-RUN14.

## First real run check

The audit confirms that ME-RUN14 defines the first real cached-source batch dry-run as a local review run over already-existing real cached source snapshots.

The audit confirms that the first real run must answer:

* which cached-source root was used;
* which tickers were requested or discovered;
* which tickers completed;
* which tickers were blocked;
* why each blocked ticker was blocked;
* which artifacts were written when artifact writing was explicitly enabled;
* which command was run;
* which branch and commit produced the evidence;
* whether forbidden side effects stayed closed.

## Input boundary check

ME-RUN14 correctly preserves the approved cached-source root category:

```text
data/market_engine/source_snapshots/
```

The audit confirms that local override roots remain allowed only when explicitly supplied, local, path-contained, readable, and non-production.

The audit confirms that ME-RUN14 does not approve live data fetches, source refresh, repair, enrichment, normalization, or backfilling through provider calls.

## Ticker-set check

ME-RUN14 approves only explicit local ticker-set modes:

* explicit operator ticker list;
* explicit local ticker-list file;
* deterministic cached-source discovery under the approved root.

The audit confirms that ME-RUN14 forbids ticker selection from hidden broker portfolios, watchlists, yfinance universes, SEC/EDGAR discovery, external APIs, internet search, Telegram/email input, or implicit all-market universes.

## Terminal visibility check

The audit confirms that ME-RUN14 requires operator-visible terminal sections for:

```text
RUN CONTEXT
INPUT DISCOVERY
SELECTED TICKERS
EXECUTION PROGRESS
BATCH SUMMARY
BLOCKED / FAILED TICKERS
ARTIFACTS
FORBIDDEN SIDE-EFFECT CONFIRMATION
NEXT REVIEW ACTIONS
```

The audit confirms that JSON-only output is not sufficient if it prevents quick operator scanning. A human-readable summary is required.

The audit confirms that ME-RUN14 preserves the operator workflow requirement to support terminal output that can be shown and copied, for example with:

```bash
<command> | tee /dev/tty | pbcopy
```

## Artifact visibility check

ME-RUN14 keeps artifact writing opt-in only.

The audit confirms that no artifact may be written unless explicitly enabled by the operator.

When enabled, the required artifact visibility remains local and non-production:

```text
<artifact_output_root>/<batch_id>/batch_manifest.json
<artifact_output_root>/<batch_id>/<ticker>/dry_run.json
<artifact_output_root>/<batch_id>/<ticker>/manifest.json
```

The audit confirms that ME-RUN14 does not approve production report writes or committed generated artifacts.

## Evidence bundle check

The audit confirms that ME-RUN14 defines a reviewable evidence bundle containing:

* final command used;
* branch and commit reference;
* relevant test command output for the changed path in the implementation sprint;
* terminal output copied from the run;
* artifact tree listing when artifacts are enabled;
* batch manifest path when artifacts are enabled;
* completed per-ticker artifact path when artifacts are enabled;
* blocked ticker list when applicable;
* explicit forbidden side-effect confirmation.

## Failure triage check

The audit confirms that blocked and failed ticker states must be visible without opening Python internals.

Required fields include:

* ticker;
* state;
* blocked or failure code;
* human-readable reason;
* source snapshot reference when available;
* execution boundary;
* artifact write state;
* retry safety without provider access.

The audit confirms that unexpected local errors remain ticker-level failures unless the batch cannot safely continue.

## Batch fail-closed check

The audit confirms that ME-RUN14 defines batch-level fail-closed conditions for missing roots, unsafe paths, empty non-discovery input, invalid artifact roots, overwrite conflicts, ambiguous arguments, contradictory arguments, and attempted live/provider/data-refresh options.

## Forbidden behavior check

ME-RUN14 does not introduce or approve:

* Python runtime code;
* test code;
* fixtures;
* provider refresh;
* SEC/EDGAR live calls;
* yfinance calls;
* live market data calls;
* external API calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* automatic cache refresh;
* automatic cache cleanup;
* generated artifact commits;
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
* tradeability authority.

## Next sprint readiness

ME-RUN14 defines the next implementation sprint as:

```text
ME-RUN15 - Implement first real cached-source batch dry-run command visibility
```

The audit confirms that ME-RUN15 should remain a narrow implementation of the ME-RUN14 visibility and command execution contract only.

## Audit conclusion

ME-RUN14 satisfies its documentation-only acceptance criteria.

It defines the first real cached-source batch dry-run execution and visibility contract, including terminal visibility, artifact visibility, evidence bundle requirements, blocked/failure triage, batch fail-closed behavior, approved local input boundaries, and next implementation requirements.

No code, tests, fixtures, provider calls, live data calls, production writes, delivery behavior, portfolio behavior, watchlist behavior, scheduler behavior, UI behavior, or Decision Engine action/allocation authority is introduced.
