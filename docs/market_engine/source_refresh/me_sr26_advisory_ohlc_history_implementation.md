# ME-SR26 Advisory OHLC History Implementation

## Architecture

`advisory_ohlc_history.py` validates the destination before acquisition, uses
the existing yfinance daily-history adapter with `auto_adjust=False`, excludes
`Adj Close`, performs one batch request, and permits no more than 25 individual
fallbacks. Files are immutable per run and live below the approved advisory
artifact root. The loader accepts a path, never a caller mapping, and returns a
private frozen validated context after checksum, policy, universe, identity,
series, and effective-freshness replay.

`current_technical_screening.py` is the only ME-SR26 consumer. It rebuilds the
existing MA20/50/200, ATR20, setup classification, score, and deterministic
ranking. It never reads `data/processed` and does not copy RUN30 indicator or
ranking values. RUN30 appears only in a drift report marked audit-only.

The same module validates SR25 evidence and emits
`market-engine-technical-price-reconciliation-v1` plus
`market-engine-run33-grounded-candidate-input-v1`. Optional portfolio context
can only bind an authoritative private ledger path; derived caller mappings
are rejected.

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
`always()` so blocker evidence survives; the final quality gate fails unless
the history manifest is analytically completed.

No workflow or provider canary was run during ME-SR26 implementation.
