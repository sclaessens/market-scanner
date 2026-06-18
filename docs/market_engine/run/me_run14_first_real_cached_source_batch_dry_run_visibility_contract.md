# ME-RUN14 - First real cached-source batch dry-run execution and visibility contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN14

## Purpose

ME-RUN14 defines the first real cached-source batch dry-run execution contract after ME-RUN13.

ME-RUN13 implemented the safe local cached-source batch dry-run runtime behavior, but it did not define the exact operator-visible execution procedure for the first real cached-source batch run, the required terminal output, the artifact inspection expectations, or the minimum evidence that must be captured before the Market Engine can be treated as practically usable for real local analysis review.

ME-RUN14 closes that gap as a documentation-only contract sprint.

## Contract identity

ME-RUN14 preserves the existing batch contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

ME-RUN14 preserves the existing per-ticker dry-run output contract:

```text
market-engine-end-to-end-dry-run-v1
```

ME-RUN14 preserves the existing local artifact contracts:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

ME-RUN14 defines an operator execution visibility contract named:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

This is not a new runtime output family. It is a documentation and acceptance contract for how the first real cached-source batch dry-run must be executed, displayed, captured, and reviewed.

## Baseline chain

ME-RUN14 builds on the approved chain:

```text
ME-RUN09 -> cached-source local execution contract
ME-RUN10 -> cached-source per-ticker local execution implementation
ME-RUN11 -> deterministic ticker-bundle validation
ME-RUN12 -> safe broader cached-source batch dry-run contract
ME-RUN13 -> safe cached-source batch dry-run runtime implementation
```

ME-RUN14 does not supersede any earlier contract. It narrows the next operational step.

## Scope

ME-RUN14 defines:

* the first real cached-source batch dry-run objective;
* the approved local input boundary;
* the minimum ticker-set selection rules;
* the required pre-run checks;
* the required terminal visibility;
* the required artifact visibility;
* the required run summary fields;
* the required failure triage format;
* the required evidence bundle for review;
* the Codex implementation handoff requirements for the next sprint.

ME-RUN14 is documentation-only.

## First real batch-run objective

The first real cached-source batch dry-run must prove that the Market Engine can process already-existing real cached source snapshots in a broader local batch without provider access, production writes, delivery, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or action authority.

The run is successful only if it produces operator-visible evidence that answers these questions:

* Which cached-source root was used?
* Which tickers were requested or discovered?
* Which tickers completed?
* Which tickers were blocked?
* Why was each blocked ticker blocked?
* Which artifacts were written, if artifact writing was explicitly enabled?
* Where are those artifacts located?
* Which command was run?
* Which environment and branch produced the evidence?
* Were forbidden side effects kept closed?

## Approved input boundary

The first real batch dry-run may only consume already-existing local cached source snapshots.

Approved root category:

```text
data/market_engine/source_snapshots/
```

A local override root is allowed only when explicitly supplied by the operator and when the implementation validates that it is local, path-contained, readable, and non-production.

The first real run may not fetch, refresh, repair, enrich, normalize, or backfill source data through live calls.

## Approved ticker-set modes

The first real batch dry-run may use one of these explicit ticker-set modes:

1. explicit operator ticker list;
2. explicit local ticker-list file;
3. deterministic cached-source discovery under the approved root.

The first real run must not use hidden broker portfolios, watchlists, yfinance universes, SEC/EDGAR discovery, external APIs, internet search, Telegram/email input, or any implicit all-market universe.

## Minimum first-run ticker-set requirement

The first real run must be broader than the ME-RUN11 deterministic bundle and must use real cached source snapshots.

Minimum review target:

```text
at least 10 cached-source tickers when available
```

If fewer than 10 eligible cached-source tickers exist locally, the run may proceed only if the terminal output and artifact evidence explicitly state:

* discovered ticker count;
* eligible ticker count;
* missing or blocked reason for the shortfall;
* confirmation that no live provider refresh was attempted.

## Required pre-run checks

Before execution, the operator-visible command path must confirm:

* active git branch;
* working tree status;
* cached-source root path;
* whether the root exists;
* discovered cached-source count;
* selected ticker count;
* artifact writing flag state;
* artifact output root if enabled;
* overwrite-protection setting;
* provider/network side effects remain forbidden.

These checks must be visible in the terminal or captured in the run evidence bundle.

## Terminal visibility contract

The first real cached-source batch dry-run must produce terminal output that is useful during execution, not only after execution.

Required terminal sections:

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

The terminal output may include JSON, but it must not be JSON-only if that makes operator scanning difficult. A human-readable summary is required.

The command should support copying output to the macOS clipboard in the operator workflow, for example:

```bash
<command> | tee /dev/tty | pbcopy
```

The implementation sprint must document the final exact command once the command interface exists.

## Execution progress visibility

The first real run must show per-ticker progress.

Minimum progress fields:

* ticker;
* source snapshot reference or local path reference;
* execution state;
* blocked reason when applicable;
* artifact write state when applicable.

Progress output must not include recommendations, rankings, scores, BUY / SELL / HOLD labels, target prices, target weights, allocation advice, position sizing, order instructions, urgency, conviction, or tradeability labels.

## Batch summary visibility

The batch summary must include:

* batch contract version;
* batch id;
* input mode;
* cached-source root;
* requested tickers;
* discovered tickers when discovery mode is used;
* selected tickers;
* completed count;
* completed-with-limitations count when applicable;
* blocked count;
* failed count;
* skipped count when applicable;
* artifact root when enabled;
* artifact write count;
* forbidden side-effect confirmation;
* authority-boundary confirmation.

## Artifact visibility contract

Artifact writing remains opt-in only.

The first real batch dry-run may not write artifacts unless the operator explicitly enables local artifact writing.

When artifact writing is enabled, terminal output and the batch summary must show:

```text
<artifact_output_root>/<batch_id>/batch_manifest.json
<artifact_output_root>/<batch_id>/<ticker>/dry_run.json
<artifact_output_root>/<batch_id>/<ticker>/manifest.json
```

The artifact section must also state:

* whether overwrite protection was active;
* whether any artifact was skipped because it already existed;
* whether any artifact write failed;
* that no artifact was written to a production report location;
* that no generated artifact is committed by default.

## Evidence bundle requirement

The first real run must produce a reviewable evidence bundle.

The evidence bundle must include:

* final command used;
* branch and commit reference;
* test command output relevant to the changed command or runtime path;
* terminal output copied from the run;
* artifact tree listing when artifacts are enabled;
* batch manifest path when artifacts are enabled;
* at least one completed per-ticker artifact path when artifacts are enabled;
* blocked ticker list when applicable;
* explicit forbidden side-effect confirmation.

Recommended operator capture commands:

```bash
git status --short --branch | tee /dev/tty | pbcopy
```

```bash
find artifacts/market_engine -maxdepth 4 -type f | sort | tee /dev/tty | pbcopy
```

The final implementation sprint may refine these commands once the exact artifact output root is finalized.

## Failure triage contract

Blocked or failed ticker states must be reviewable without opening Python internals.

Each blocked or failed ticker must expose:

* ticker;
* state;
* blocked or failure code;
* human-readable reason;
* source snapshot reference when available;
* whether execution stopped before or after per-ticker dry-run invocation;
* whether an artifact was written;
* whether retry is safe without provider access.

Unexpected local errors must remain ticker-level failures unless the batch itself cannot safely continue.

## Batch-level fail-closed conditions

The batch must fail closed before ticker execution when:

* cached-source root is missing;
* cached-source root is outside approved/local boundaries;
* ticker input is empty and discovery mode is not explicitly enabled;
* artifact output root is invalid when artifact writing is enabled;
* overwrite protection detects an unsafe target;
* command arguments are ambiguous or contradictory;
* any live/provider/data-refresh option is requested.

## Forbidden side effects

ME-RUN14 does not approve:

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

## Acceptance criteria

ME-RUN14 is complete when:

* the first real cached-source batch dry-run execution contract is documented;
* terminal visibility requirements are documented;
* artifact visibility requirements are documented;
* evidence bundle requirements are documented;
* blocked/failure triage requirements are documented;
* forbidden side effects are explicitly preserved;
* roadmap and backlog entry documents are created;
* audit document confirms that the sprint is documentation-only.

## Next implementation sprint

Recommended next sprint:

```text
ME-RUN15 - Implement first real cached-source batch dry-run command visibility
```

ME-RUN15 should implement only the narrow operator-facing command and visibility behavior defined here. It must not broaden into source refresh, production execution, delivery, portfolio/watchlist mutation, scheduling, UI, or action/allocation authority.
