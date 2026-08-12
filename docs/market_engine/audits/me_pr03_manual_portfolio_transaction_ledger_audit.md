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
