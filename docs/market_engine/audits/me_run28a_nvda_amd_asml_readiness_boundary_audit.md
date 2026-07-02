# ME-RUN28A - NVDA/AMD/ASML Readiness-Boundary Audit

Sprint ID: ME-RUN28A
Status: COMPLETED WITH CONTROLLED STOP BY ME-RUN28A
Job family: ME-RUN / Run and orchestration
Date: 2026-07-02
Branch: `me-run28a-nvda-amd-asml-readiness-boundary-audit`

## Purpose

ME-RUN28A runs the bounded `NVDA`, `AMD`, and `ASML` validation set through
the existing deterministic cached-source/local dry-run flow after ME-SA11.
The sprint verifies persisted `analysis_context_readiness` metadata and the
Recommendation Review boundary without changing runtime behavior.

## Source Basis

Source main commit:

```text
63a5e99 Merge pull request #415 from sclaessens/me-sa11-analysis-context-readiness-adapter-artifact-metadata
```

The run reuses the existing ME-RUN27 runner because it is the canonical
network-free flow for the same bounded ticker set:

```text
deterministic company-profile acquisition
staging validation
cached-source compatibility gate
local dry-run
local artifact persistence
```

No new acquisition, import, provider, or execution flow was introduced.

## Run Identity and Commands

```text
run_id: me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z
generated_at: 2026-07-02T11:31:04Z
artifact_root: artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z
bounded_tickers: NVDA, AMD, ASML
input_mode: cached_source_snapshot
source_family: company_profile
```

Run command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/market_engine/me_run27_company_profile_cross_ticker_dry_run.py --run-id me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z --generated-at 2026-07-02T11:31:04Z --artifact-root artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z
```

The run completed acquisition for three of three entries, accepted all three
staging entries, allowed all three compatibility gates, consumed all three
Source Context payloads, and completed Analysis Review for every ticker.
Each dry-run then stopped at Recommendation Review under the existing
company-profile-only boundary.

## Persisted Artifact Results

For every ticker, the artifact exists and contains:

```text
artifact["payload"]["analysis_context_readiness"]
```

| Ticker | Artifact path | Run state | Readiness | Present evidence | Missing evidence | Readiness blocked reasons | Recommendation Review | Actionable review allowed? | Decision Engine ready? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | `artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z/dry_runs/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-nvda/artifacts/market_engine_dry_run_me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-nvda_2026-07-02.json` | `dry_run_blocked` | `descriptive_only` | `company_profile` | `fundamentals`, `setup_price_market`, `provenance_manifest_staleness` | `stale_or_unprovenanced_analysis_context`; `company_profile_only_context_non_actionable` | `blocked` with `company_profile_only_context_non_actionable` | no | no | `provenance_valid=true`; `context_stale=true` |
| AMD | `artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z/dry_runs/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-amd/artifacts/market_engine_dry_run_me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-amd_2026-07-02.json` | `dry_run_blocked` | `descriptive_only` | `company_profile` | `fundamentals`, `setup_price_market`, `provenance_manifest_staleness` | `stale_or_unprovenanced_analysis_context`; `company_profile_only_context_non_actionable` | `blocked` with `company_profile_only_context_non_actionable` | no | no | `provenance_valid=true`; `context_stale=true` |
| ASML | `artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z/dry_runs/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-asml/artifacts/market_engine_dry_run_me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z-asml_2026-07-02.json` | `dry_run_blocked` | `descriptive_only` | `company_profile` | `fundamentals`, `setup_price_market`, `provenance_manifest_staleness` | `stale_or_unprovenanced_analysis_context`; `company_profile_only_context_non_actionable` | `blocked` with `company_profile_only_context_non_actionable` | no | no | `provenance_valid=true`; `context_stale=true` |

The machine-readable and human-readable local run summaries are:

```text
artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z/me_run27_summary.json
artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z/me_run27_summary.md
```

The filenames retain `me_run27` because ME-RUN28A deliberately reuses the
existing runner without changing runtime or adding a duplicate run helper.

## Recommendation Review Boundary

Analysis Review completed with descriptive company-profile context for every
ticker. Recommendation Review exists as an explicit stage result and is
blocked for every ticker with:

```text
company_profile_only_context_non_actionable
```

A recursive key inspection of all three payloads found no:

```text
buy
sell
hold
entry_price
stop_loss
target_price
conviction
position_sizing
trade_plan
```

No actionable recommendation, entry, stop, target, conviction, sizing, trade
plan, or allocation field was inferred from insufficient company-profile-only
context.

Every readiness result persists:

```text
recommendation_review_eligible: false
actionable_review_allowed: false
decision_engine_ready: false
```

`actionable_review` and `decision_ready` remain reserved and unreachable.
The Decision Engine remains the only allocation authority.

## Staleness and Provenance Interpretation

Every result reports:

```text
provenance_valid: true
context_stale: true
```

The deterministic fake company-profile acquisition retains provider,
retrieval, snapshot, path, and ticker provenance but has no source timestamp.
The readiness adapter therefore does not count
`provenance_manifest_staleness` as present and fails closed with
`stale_or_unprovenanced_analysis_context`.

This conservative result does not block the audit purpose and does not justify
loosening cached-source validation or readiness classification.

## Artifact Inspection

A local one-off Python inspection:

* located exactly three persisted dry-run artifacts;
* asserted that every readiness level is one of `descriptive_only`,
  `partial_analysis`, or `recommendation_eligible`;
* asserted that no result is `actionable_review` or `decision_ready`;
* asserted `actionable_review_allowed=false`;
* asserted `decision_engine_ready=false`;
* asserted Recommendation Review is blocked;
* recursively checked that no actionable recommendation-field keys exist.

Result:

```text
PASS - 3 artifacts inspected
PASS - 3 descriptive_only readiness results
PASS - 3 blocked Recommendation Review stage results
PASS - 0 forbidden actionable recommendation keys
```

## Validation

Commands and results:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/analysis_review -q
51 passed in 0.08s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/recommendation_review -q
16 passed in 0.06s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run -q
114 passed in 1.84s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1213 passed in 3.86s

git diff --check
PASS
```

Mandatory repository governance greps:

```text
grep -R "BUY" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "SELL" scripts/ | grep -v decision_engine.py
Existing legacy matches only in scripts/portfolio/parse_trade_commands.py,
scripts/portfolio/portfolio_manager.py, and ignored bytecode.

grep -R "tradeable" scripts/ | grep -v decision_engine.py
Ignored bytecode matches only.
```

No file under `scripts/` changed in ME-RUN28A. The complete staged-diff
governance grep returned documentation-only matches: safety confirmations,
the recorded legacy-grep interpretation, and the deferred unsent Telegram
preview. Inspection confirmed that no provider/network, Telegram,
broker/order, portfolio/watchlist, or production-write behavior was added.

## Generated Artifact Policy

Generated acquisition packages, validation output, dry-run artifacts, and
summary artifacts remain local under:

```text
artifacts/market_engine/me-run28a-nvda-amd-asml-readiness-boundary-20260702T113104Z
```

They are not committed. A narrow `.gitignore` entry prevents accidental
staging of ME-RUN28A run evidence.

## Safety

The acquisition safety record confirms:

```text
network_used: false
provider_calls_performed: false
production_write_performed: false
telegram_sent: false
portfolio_written: false
watchlist_written: false
broker_action_performed: false
```

ME-RUN28A changes no runtime or tests and adds no provider/network access,
cached-source relaxation, trading or allocation logic, Telegram sending,
production writes, or portfolio/watchlist mutation.

## Known Limitations and Follow-Up

This bounded run uses deterministic company-profile evidence only. It does not
prove readiness behavior for valid fundamentals plus setup/price/market
evidence, and it does not validate a materially expanded supported universe.
The missing source timestamp keeps the provenance/staleness evidence family
unsatisfied by design.

Recommended next sprint:

```text
ME-RUN28 - Expanded supported-universe acquisition and dry-run classification
```

The expanded run should preserve the same fail-closed readiness and
Recommendation Review assertions. A non-production Telegram preview remains
unsent and deferred to:

```text
ME-DL03 - Telegram preview artifact, no sending
```

## Final Status

```text
PASS
```
