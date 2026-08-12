# ME-PR03 Manual Portfolio Transaction Ledger Audit

Status: COMPLETED

## Scope and base

| Evidence | Result |
|---|---|
| Base | `4342c35461bb3f9e4f311611324a82182002e9be` (`origin/main`) |
| ME-RM08 | Present through merge commit `4342c354` |
| Canonical story | `docs/market_engine/backlog/me_pr03_manual_portfolio_transaction_ledger.md` |
| Branch | `codex/me-pr03-manual-portfolio-transaction-ledger` |
| Existing user changes | Preserved untouched in the original worktree; implementation isolated in a clean worktree |

## Acceptance evidence

| Criterion | Evidence |
|---|---|
| Confirmed events only | Preview digest/token must exactly match explicit confirm input |
| Append-only private ledger | Exclusive create, locked single-line append, `fsync`, `0600`, no edit/delete API |
| Sole transaction truth | Manual transaction role is authoritative; legacy position source is migration input only |
| Deterministic positions | Ledger-only moving-weighted-average rebuild using exact decimal arithmetic |
| Corrections preserve history | Correction/reversal are new references; target remains in the ledger |
| Fail-closed validation | Stable `LedgerIssueCode` values and no append on error |
| Identity | Canonical registry-derived instrument ID, ticker, currency, mapped aliases |
| Private persistence | External private paths or ignored `data/portfolio/private/` only |
| Portfolio Review adapter | Emits validated `market-engine-portfolio-context-v1` with full provenance |
| Candidate context | Held, not-held, closed, partial, and unknown are explicit and non-actionable |
| Price absence | Market price/value/unrealized result/exposure remain unavailable without blocking position state |
| Authority | No ranking, filtering, recommendation, allocation, sizing, order, execution, or Decision Engine behavior |

## Test coverage

The focused tests cover valid purchase, multiple purchases and moving average,
partial and full sale, oversell, available zero and unavailable fees, duplicate
IDs and replay, deterministic/idempotent rebuild, corrections, full reversal,
unknown and duplicate reversal, multiple accounts, authoritative identity,
mapped and unsupported aliases, ambiguity, decimal precision, invalid/future
dates, missing versus zero, unsupported/mismatched currencies, unavailable FX,
non-deterministic ordering, corrupt/incompatible ledgers, confirmation mismatch,
confirmation-required persistence, private/tracked paths, derived-file
non-authority, schema alignment, Portfolio Review provenance, descriptive state
distinctions, price unavailability, CLI flow, and absence of provider, broker,
network, notification, or publication side effects.

| Command | Passed | Failed | Skipped | Duration | Result |
|---|---:|---:|---:|---:|---|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/portfolio_review/test_manual_transaction_ledger.py -q --tb=short` | 39 | 0 | 0 | 0.10 s | Passed |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/portfolio_review tests/market_engine/recommendation_review tests/market_engine/decision_engine_handoff tests/contract/test_v2_portfolio_contracts.py tests/contract/test_v2_portfolio_source_of_truth_contracts.py tests/portfolio -q --tb=short` | 118 | 0 | 0 | 0.47 s | Passed |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1,592 | 1 | 0 | 7.92 s | One pre-existing missing historical artifact |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2,259 | 1 | 0 | 9.06 s | Same pre-existing missing historical artifact |
| Single failing test on detached `origin/main` at `4342c354` | 0 | 1 | 0 | 0.02 s | Identical failure reproduced on the correct base |
| `git diff --check` | n/a | 0 | n/a | <0.1 s | Passed |

The sole broad-suite failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`. It expects
the absent historical file
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
The exact assertion and path reproduce in a clean detached worktree at the
current `origin/main` head. ME-PR03 neither reads nor changes that evidence
contract.

The mandatory `AGENTS.md` greps report only unchanged legacy historical
transaction parsing and portfolio-manager `BUY`/`SELL` literals under
`scripts/portfolio/`; the `tradeable` grep is empty. ME-PR03's ledger
`BUY`/`SELL` values label already completed user-reported transactions and do
not execute purchase/sale logic or emit advice.

## Privacy and repository review

No real transaction or position file was created, read by the ME-PR03 runtime,
copied, migrated, logged, staged, or committed. Tests use temporary directories
and synthetic portfolio/account/instrument values. The new repository-local
private root is ignored. The command response reports only a redacted ledger
filename and event digest.

The implementation does not import or call legacy `scripts/portfolio/` code.
Legacy tracked portfolio files remain untouched and are not promoted as
authority.

## PR #477 review remediation

Reviewed head: `2b221e000270424591771f6e436a595118199cdb`.

| Finding | Original cause | Remediation and guarantee | Regression evidence |
|---|---|---|---|
| P1: stored ledger semantics | Loader validated only shape/version/event type before projection | One canonical validator now covers preview, confirmation, append, load, and rebuild; exact header/event fields, event semantics, canonical finite decimals, fees, currencies, identities, timestamps, references, append order, and portfolio ownership fail closed before projection | Negative/zero/float/NaN/Infinity/exponent quantity, negative price/fee, invalid type/source/currency/fee state, malformed/future timestamps, economic reversal, incomplete correction, bad portfolio/reference, header, and extra-field cases |
| P1: derived projection authority | Public context adapter accepted an arbitrary projection mapping | Public adapter accepts only a ledger path, loads and rebuilds it itself, and derives digest/references/position values from that rebuild; projection mappings are rejected | Forged quantity `999999`, forged digest, modified derived export, instrument/ticker mismatch, portfolio/account mismatch, real ledger context, and all state mappings |
| P1: private path depends on `cwd` | Repository discovery began at the process working directory | Resolved target path drives enclosing-worktree discovery; `.git` directory/file, Git repository identity, ignore and tracked checks, other repositories, symlinks, and Git errors are all fail closed; caller repository-root override was removed | External `cwd`, tracked target/parent, unignored target, allowed ignored target, second repository, external non-Git location, linked worktree, unsafe symlink, and Git failure |
| P2: future execution time | Execution time was checked only for timezone and trade date | Recorded time cannot be future; execution must be UTC-normalizable, on the trade date, and no later than recorded time; confirmation and load repeat the check | Later same-day, exact equality, historical offset, future recorded time, future-after-UTC-normalization, changed preview, and corrupted stored relationship |
| P2: reopened cost basis | Unknown old purchase fee permanently set the position's cost-basis flag to unknown | Exact zero quantity closes the old cost-basis cycle at exact zero; a new fully known purchase starts a known cycle while historical fee and realized-result uncertainty remains independent | Unavailable-fee purchase, full sale, known-fee repurchase, plus partial-sale control case |

The event and projection schemas parse as JSON and runtime transaction,
correction, reversal, and projection payloads are validated against their
actual constraints. Invalid event and forged projection instances are rejected
by the same schema tests.

| Remediation command | Passed | Failed | Skipped | Duration | Result |
|---|---:|---:|---:|---:|---|
| Focused manual-ledger tests | 72 | 0 | 0 | 0.70 s | Passed |
| Relevant portfolio, recommendation, handoff, contract, and legacy portfolio tests | 151 | 0 | 0 | 1.06 s | Passed |
| Market Engine tests | 1,625 | 1 | 0 | 9.20 s | Known base failure only |
| Full repository tests | 2,292 | 1 | 0 | 10.48 s | Same known base failure only |
| Exact failing test on base `4342c354` | 0 | 1 | 0 | 0.03 s | Identical assertion and missing path |

### Remaining P1 private-path false-positive remediation

Reviewed head `1d9b068628e65cae386c8f4321bdc919154981d5` still used
`git ls-files --error-unmatch` for every parent path. Git treats a directory
pathspec such as `data` as a match when any tracked descendant exists, so the
real repository's tracked `data/processed/` and `data/local/` content
incorrectly blocked the valid ignored target
`data/portfolio/private/ledger.jsonl`.

Tracked-target validation now asks Git for matching tracked paths and compares
the normalized returned names with the exact ledger target. Parent safety is a
separate filesystem check: every existing parent component must be a directory,
and an existing target must be a regular file. A tracked or untracked file used
as a required parent therefore remains fail closed, while an ordinary
directory with unrelated tracked descendants does not create a false
conflict. Ignore verification, repository identity, target resolution,
worktree `.git` files, other-repository rejection, symlink resolution, exact
tracked-target rejection, and Git-error fail-closed behavior are unchanged.

The realistic regression repository tracks and commits both `.gitignore` and
`data/processed/reference.csv`, then proves that the ignored private ledger
target is allowed. Negative coverage proves that an exact tracked ledger,
tracked and untracked file parents, an unignored target, a path outside the
private root, another repository, an unsafe symlink, and Git verification
errors remain rejected. The targeted path set passes 9 tests; the complete
ledger suite passes 73 tests; and the relevant portfolio boundary passes 152
tests. Market Engine passes 1,626 tests and the complete repository passes
2,293 tests, with the same single missing historical ME-DATA06 fixture failure
documented below.

The broad-suite failure remains
`test_compact_checksums_match_committed_files_and_local_full_runs` at
`tests/market_engine/data/test_operator_pilot_compact_evidence.py:70`. The
missing path remains
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
No test was removed, skipped, or weakened, and the historical fixture remains
outside ME-PR03 scope.

No residual review-remediation limitation can reintroduce a derived projection
as authority or permit repository safety to depend on `cwd`. The unchanged
product limitations remain private local file authority without broker,
provider, live price, FX, tax, corporate-action, exposure, notification,
workflow, publication, or Decision Engine behavior.

## Cross-job changes

The only Portfolio Review contract change adds the descriptive `closed`
position state and recognizes it as non-actionable not-held context. A delayed
Recommendation Review import resolves a package initialization cycle exposed by
the new command entry point; contract behavior remains unchanged. No Analysis,
Recommendation, Decision Engine, Delivery, Source Refresh, provider, workflow,
or data path is otherwise changed.

## Remaining follow-ups

ME-SR25 remains next and may add advisory price freshness only. ME-PI01 remains
responsible for later exposure and concentration intelligence. FX, corporate
actions, transfers, taxes, dividends, short positions, options, broker import,
and migration of any real legacy portfolio remain unavailable and require
separate governance.
