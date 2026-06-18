# ME-RUN12 — Safe all-ticker cached-source batch dry-run contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN12

## Purpose

ME-RUN12 defines the safe contract for a future all-ticker or broader cached-source batch dry-run path.

The contract exists because ME-RUN10 implemented a single cached-source local execution path and ME-RUN11 validated that path against a small deterministic ticker bundle by invoking the approved per-ticker command path ticker-by-ticker. A broader batch behavior needs an explicit contract before implementation so cached-source discovery, ambiguity handling, failure isolation, artifact behavior, operator visibility, and production guardrails remain controlled.

ME-RUN12 is documentation-only. It does not add Python code, tests, fixtures, commands, runtime behavior, provider calls, source refresh behavior, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, or action/allocation authority.

## Approved baseline

The approved RUN baseline before ME-RUN12 is:

```text
ME-RUN05 -> optional local dry-run artifact persistence
ME-RUN06 -> local_snapshot_fixture input wrapper
ME-RUN07 -> realistic non-production local fixture artifact execution
ME-RUN08 -> deterministic local fixture matrix coverage
ME-RUN09 -> cached-source local execution contract
ME-RUN10 -> cached_source_snapshot local execution implementation
ME-RUN11 -> deterministic small ticker-bundle validation through per-ticker command invocation
```

The existing approved per-ticker output remains:

```text
market-engine-end-to-end-dry-run-v1
```

The existing local artifact contracts remain:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

ME-RUN12 does not replace these contracts. It defines the future batch wrapper around repeated safe per-ticker cached-source dry-runs.

## New future contract family

A future implementation may introduce this batch-level local contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

This contract represents a local, deterministic, review-only batch summary over already-existing cached source snapshots.

It is not a source refresh contract.

It is not a provider adapter contract.

It is not a live-data execution contract.

It is not an all-ticker production execution contract.

It is not a delivery, notification, portfolio, watchlist, scheduler, UI, broker, trading, allocation, ranking, scoring, conviction, urgency, target-price, target-weight, or tradeability contract.

## Approved batch input boundary

A future batch implementation may consume only cached source snapshots that already exist on disk before the batch command starts.

Approved cached-source category:

```text
data/market_engine/source_snapshots/
```

A future implementation may support a local development override root only when all of the following remain true:

* the override is explicitly supplied by the operator;
* path containment is enforced;
* the root is local and non-production;
* no provider refresh is triggered;
* no automatic cache cleanup is triggered;
* no portfolio, watchlist, delivery, scheduler, UI, broker, or production report write is triggered;
* every accepted snapshot still passes the same single-snapshot validation requirements from ME-RUN09 and ME-RUN10.

## Approved batch command shape

A future implementation may add a batch mode to the existing local dry-run command family or an adjacent ME-RUN command module only if it preserves all existing per-ticker modes.

Approved future input mode:

```text
--input-mode cached_source_batch
```

Approved future operator inputs may include:

```text
--source-snapshot-root <path>
--ticker-list-json <path>
--ticker-list-csv <path>
--ticker <ticker>
--ticker-limit <positive integer>
--dry-run-batch-id <id>
--generated-at <timestamp>
--write-local-artifact
--artifact-output-root <path>
--artifact-created-at <timestamp>
--continue-on-ticker-error
--fail-on-ambiguous-source
```

The exact command interface may be refined during implementation, but it must preserve these contract boundaries:

* batch execution must be explicitly requested;
* per-ticker cached-source execution must remain the underlying unit of work;
* existing `synthetic_contract_fixture`, `local_snapshot_fixture`, `explicit_in_memory_payload`, and `cached_source_snapshot` behaviors must remain unchanged;
* artifact writing must remain opt-in only;
* the default behavior must not write artifacts, production reports, portfolio files, watchlists, delivery payloads, or scheduler state.

## Ticker universe boundary

A future batch run may derive its ticker universe only from explicitly local and reviewable inputs.

Approved ticker universe sources are:

* explicit operator ticker list;
* explicit local ticker-list JSON or CSV file;
* explicit discovery of cached snapshots inside the approved snapshot root;
* a bounded test fixture used only in local tests.

Forbidden ticker universe sources are:

* broker portfolio API calls;
* Bolero or other broker scraping;
* live market-data provider calls;
* yfinance calls;
* SEC/EDGAR calls;
* Telegram/email content;
* production watchlist reads unless a future sprint explicitly defines and approves a read-only local watchlist contract;
* hidden default all-market universes;
* implicit internet search or external API discovery.

The batch contract must distinguish these concepts:

```text
requested_tickers
eligible_cached_source_tickers
discovered_but_unrequested_tickers
requested_but_missing_cached_source_tickers
ambiguous_source_tickers
unsupported_source_tickers
skipped_tickers
executed_tickers
blocked_tickers
completed_tickers
completed_with_limitations_tickers
```

## Cached-source discovery rules

Discovery must be deterministic and explainable.

A future implementation must define and enforce rules for:

* approved filename patterns or manifest references;
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

When multiple snapshots match one ticker, the implementation must not silently choose an arbitrary file.

Approved resolution options are:

* select the newest valid snapshot only when this rule is explicit, deterministic, and documented in the batch manifest;
* block the ticker as ambiguous;
* require an explicit operator snapshot reference.

The default safe rule should be to block ambiguity unless the implementation provides a deterministic, audited selection policy.

## Per-ticker execution boundary

Each ticker remains an isolated unit of work.

For every eligible ticker, a future batch implementation must execute or construct the equivalent of the approved ME-RUN10 per-ticker path:

```text
cached source snapshot
-> Source Context
-> Fundamental Observations
-> Derived Observations
-> Setup Detection
-> Analysis Review
-> Recommendation Review
-> Portfolio Review
-> Decision Engine handoff
-> Delivery / Reporting
-> End-to-end dry-run summary
```

The final per-ticker payload must remain:

```text
market-engine-end-to-end-dry-run-v1
```

The batch wrapper must not merge ticker evidence into a new investment ranking, score, shortlist, trading queue, action list, allocation list, or recommendation list.

The batch wrapper may summarize execution state only.

## Failure isolation requirements

A future batch implementation must isolate failures per ticker.

A failure for one ticker must not corrupt, overwrite, mutate, or invalidate successful per-ticker results for other tickers.

Each ticker result must end in one of the approved batch execution states:

```text
completed
completed_with_limitations
blocked_missing_cached_source
blocked_ambiguous_cached_source
blocked_unsupported_cached_source
blocked_invalid_cached_source
blocked_stale_source_without_safe_downstream_contract
blocked_downstream_contract_failure
failed_unexpected_local_error
skipped_by_operator_limit
skipped_by_operator_filter
```

Every non-completed state must record deterministic reasons.

Unexpected local errors must be captured as ticker-level failures and summarized at batch level. They must not trigger provider calls, delivery, portfolio writes, watchlist writes, broker access, or automatic remediation.

## Batch output contract

A future `market-engine-cached-source-batch-dry-run-v1` payload must include, at minimum:

```text
contract_version
batch_id
generated_at
input_mode
source_snapshot_root
operator_ticker_input_reference
requested_tickers
batch_execution_state
batch_counts
per_ticker_results
batch_blocked_reasons
batch_warnings
artifact_manifest_reference
forbidden_side_effect_confirmation
authority_boundary_confirmation
provenance
```

`batch_counts` must include, at minimum:

```text
requested_count
discovered_cached_source_count
eligible_count
executed_count
completed_count
completed_with_limitations_count
blocked_count
failed_count
skipped_count
missing_cached_source_count
ambiguous_cached_source_count
unsupported_cached_source_count
stale_source_count
```

Each `per_ticker_results` item must include, at minimum:

```text
ticker
cik
source_snapshot_reference
source_snapshot_format_version
source_snapshot_created_at
source_snapshot_age_days
source_snapshot_stale_for_review
execution_state
blocked_reasons
warnings
end_to_end_dry_run_reference
artifact_reference
missing_data_markers
stale_data_markers
numeric_zero_evidence_present
provenance
```

The batch output may reference per-ticker outputs by local artifact reference when artifacts are explicitly written, or by in-memory logical reference when artifacts are not written.

## Artifact behavior

Artifact writing remains opt-in only through an explicit artifact flag.

When artifact writing is not requested:

* the batch may emit stdout or return an in-memory payload;
* no artifact directory may be created;
* no per-ticker artifact may be written;
* no batch manifest may be written.

When artifact writing is explicitly requested:

* each successfully materialized per-ticker payload must preserve `market-engine-local-dry-run-artifact-v1`;
* the existing per-ticker manifest contract must preserve `market-engine-local-dry-run-artifact-manifest-v1` semantics;
* a future batch-level manifest may be added, but it must only reference local dry-run artifacts and batch metadata;
* overwrite protection must remain default-on;
* artifact paths must remain inside the approved artifact root;
* artifact timestamps must remain deterministic when supplied by the operator or tests.

Approved future artifact layout:

```text
artifacts/market_engine/dry_runs/<batch_id>/batch_manifest.json
artifacts/market_engine/dry_runs/<batch_id>/<ticker>/dry_run.json
artifacts/market_engine/dry_runs/<batch_id>/<ticker>/manifest.json
```

This layout is a contract direction, not an implementation requirement until a future sprint implements it.

## Operator visibility requirements

A future batch implementation must provide enough operator visibility to make the run reviewable from terminal output and/or local artifacts.

Required visibility categories:

* batch id;
* requested ticker count;
* discovered cached-source count;
* executed ticker count;
* completed ticker count;
* completed-with-limitations count;
* blocked ticker count;
* failed ticker count;
* missing cached-source tickers;
* ambiguous cached-source tickers;
* stale cached-source tickers;
* artifact root when artifact writing is explicitly requested;
* confirmation that no provider, broker, delivery, portfolio, watchlist, scheduler, UI, or production report side effects occurred.

Operator visibility may not introduce ranking, scoring, urgency, conviction, actionability, tradeability, target prices, target weights, or allocation advice.

## Missing, stale, blocked, and numeric-zero behavior

The batch wrapper must preserve all existing ME-RUN09 and ME-RUN10 data-safety behavior:

* missing fields remain explicit missing-data markers;
* stale cached snapshots remain explicit stale-data markers;
* numeric zero values remain valid evidence and may not be converted to missing values;
* unsupported source snapshots fail closed at ticker level;
* incomplete source snapshots fail closed or produce blocked dry-run states;
* blocked stages record blocked reasons;
* local execution may not silently repair, enrich, normalize, or backfill data with live provider calls.

Batch-level summaries may count missing/stale/blocked conditions, but may not hide per-ticker evidence.

## Provenance requirements

A future batch implementation must preserve provenance at both batch and per-ticker level.

Required batch provenance categories:

* batch contract version;
* batch id;
* command/input mode;
* source snapshot root;
* operator ticker input reference;
* discovery policy;
* ambiguity policy;
* artifact policy;
* generated timestamp;
* implementation version or module identity when available.

Required per-ticker provenance categories:

* ticker;
* CIK when applicable;
* source snapshot path or logical reference;
* source snapshot format version;
* source family;
* retrieval timestamp when present;
* source snapshot timestamp;
* selected snapshot rule;
* downstream stage identifiers;
* omitted-stage reasons when a stage cannot be constructed;
* per-ticker artifact reference when artifact writing is explicitly requested.

## Fail-closed requirements

A future batch implementation must fail closed at batch level when:

* the source snapshot root does not exist;
* the source snapshot root escapes the approved local input boundary;
* the ticker universe cannot be determined safely;
* the ticker list file is malformed;
* the ticker list contains unsupported identifiers that cannot be represented safely;
* the batch id is missing or unsafe when artifact writing is requested;
* artifact writing would overwrite an existing batch artifact directory without explicit override;
* provider access would be required;
* broker, Telegram, email, scheduler, UI, portfolio, or watchlist access would be required;
* production report generation would be required;
* action/allocation/ranking/scoring/tradeability semantics would be introduced outside approved guardrail text.

A future batch implementation must fail closed at ticker level when:

* the ticker has no matching cached source snapshot;
* the ticker has ambiguous cached source snapshots and no approved deterministic selection rule applies;
* the source snapshot file is not valid JSON;
* the source snapshot format version is unsupported;
* required source identity metadata is missing;
* the snapshot ticker/CIK does not match the requested ticker identity;
* required numeric values cannot be parsed without unsafe coercion;
* required downstream contract identity cannot be produced;
* missing or stale source evidence cannot be represented safely by downstream contracts.

## Explicit non-goals

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

## Future implementation requirements

The logical next implementation sprint is:

```text
ME-RUN13 — Implement safe cached-source batch dry-run path
```

ME-RUN13 should be allowed only if it preserves the ME-RUN12 contract boundaries.

Minimum implementation requirements for ME-RUN13:

* add an explicit local batch invocation path without changing existing per-ticker command behavior;
* discover cached source snapshots deterministically inside an approved local root;
* accept an explicit operator ticker universe or explicit cached-source discovery mode;
* validate path containment, JSON shape, source identity, format version, ticker/CIK identity, staleness, missing fields, and ambiguity;
* run each ticker through the approved cached-source per-ticker dry-run path;
* preserve per-ticker `market-engine-end-to-end-dry-run-v1` output identity;
* produce a batch-level `market-engine-cached-source-batch-dry-run-v1` summary;
* isolate per-ticker failures;
* keep artifact writing opt-in only;
* add deterministic local tests for discovery, missing cached source, ambiguous cached source, malformed source, unsupported version, stale source, numeric-zero preservation, per-ticker failure isolation, batch summary counts, artifact default-off behavior, opt-in batch artifact writing, and forbidden side-effect confirmation;
* add implementation documentation and audit;
* synchronize backlog and roadmap.

## Acceptance criteria for ME-RUN12

ME-RUN12 is complete when:

* the future batch input boundary is defined;
* cached-source discovery and ambiguity rules are defined;
* per-ticker failure isolation requirements are defined;
* batch output and manifest expectations are defined;
* artifact behavior remains opt-in and local-only;
* no production writes or live provider calls are approved;
* the final per-ticker output relationship to `market-engine-end-to-end-dry-run-v1` is defined;
* ME-RUN13 implementation requirements are defined;
* backlog and roadmap entries are synchronized;
* no runtime code, tests, fixtures, provider calls, live data calls, broker calls, delivery behavior, portfolio behavior, watchlist behavior, scheduler behavior, UI behavior, production report generation, or Decision Engine action/allocation behavior is introduced.

## Conclusion

ME-RUN12 moves Market Engine closer to broad local review by defining how a future batch runner may safely orchestrate already-existing cached-source snapshots. It deliberately stops before implementation, provider refresh, live data access, production execution, delivery, portfolio mutation, watchlist mutation, scheduling, UI behavior, and action/allocation authority.
