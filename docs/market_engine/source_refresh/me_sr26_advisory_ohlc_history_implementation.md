# ME-SR26 Advisory OHLC History Implementation

## Architecture

`advisory_ohlc_history.py` validates the destination before acquisition, uses
the existing yfinance daily-history adapter with `auto_adjust=False`, excludes
`Adj Close`, performs one batch request, and permits no more than 25 individual
fallbacks. Files are immutable per run and live below the approved advisory
artifact root. The loader accepts a path, never a caller mapping, and returns a
private frozen validated context after checksum, policy, universe, identity,
and full semantic replay. The replay derives the complete index, eligibility,
global status, lag, failure, coverage, and observation bindings from the actual
bars. Re-signing a false index, manifest, and checksum index therefore cannot
turn old bars into current evidence. Producer-time status is immutable;
effective freshness is recalculated at load time.

Production acquisition internally selects `_acquire_with_existing_adapter` and
uses an internal canonical UTC clock. Public history build/load interfaces
expose neither provider injection nor time overrides. Acquisition start and
completion are measured separately. Deterministic providers and times exist
only on private `_build_advisory_ohlc_history_impl()` and
`_load_advisory_ohlc_history_impl()` test seams, which reject non-canonical UTC
values and are never called from the CLI with caller-derived authority.

`current_technical_screening.py` is the only ME-SR26 consumer. It rebuilds the
existing MA20/50/200, ATR20, setup classification, score, and deterministic
ranking under a versioned, checksum-bound screening policy. Missing volume
remains nullable and is never converted to zero. It never reads
`data/processed` and does not copy RUN30 indicator or ranking values. RUN30
appears only in a drift report marked audit-only.

The same module validates SR25 evidence and emits
`market-engine-technical-price-reconciliation-v1` plus
`market-engine-run33-grounded-candidate-input-v1`. Optional portfolio context
can only bind an authoritative private ledger path; derived caller mappings
are rejected.

The canonical handoff loader reloads and replays history, screening, SR25
price, reconciliation, all 952 identities, approval validation, execution
proof, and downstream after-authority. Only a private validated context in
`ready_for_run33` state exposes candidate input. Pending approval, approved
without refreshed downstream authority, invalid downstream authority, and
ready states are separate and reachable.

Public screening and RUN33 build/load interfaces capture one internal UTC
evaluation time per operation. That same internal value drives history
effective freshness and the existing SR25 `trusted_now` compatibility call.
Public callers may select artifact paths but cannot select, backdate, or alias
the freshness evaluation time. Private deterministic helpers remain test-only.

## Artifact contracts

History produces `manifest.json`, `history_index.json`, per-instrument series,
`screening_eligibility.json`, and `checksum_index.json`. Screening produces a
manifest, full universe index, setup/outcome summaries, blocker report, newly
calculated ranking and Markdown views, RUN30 drift, and checksum index. Handoff
produces a manifest, technical/price reconciliation, candidate input, and
checksum index.

## Workflow

`advisory-ohlc-history.yml` runs daily at 09:30 UTC and supports manual
dispatch. The later window responds to SR25 run `32104872490`, where a 05:30
UTC job was technically successful but 944 records were stale by one session.
The workflow has read-only contents permission, no publication job, no Git
write, 14-day retention, and `cancel-in-progress: false`. Artifact upload uses
`always()` so blocker evidence survives. A runtime semantic replay gate allows
`usable` evidence at the versioned 0.99 fresh-coverage threshold while keeping
each isolated ticker blocked. Widespread one-session lag or global provider
failure remains `unusable` and blocks screening.

No workflow or provider canary was run during ME-SR26 implementation.
