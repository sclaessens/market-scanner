# ME-RUN30 Full Canonical-Universe Analysis and Candidate Ranking Audit

Status: `completed_with_blockers`

## Executive Summary

ME-RUN30 now runs the first broad deterministic Market Engine technical setup
screening across the complete current canonical universe. The corrected run
consumes the 952-instrument canonical universe and validated local
price-history CSVs maintained through ME-DATA05, attempts every instrument,
analyses every instrument with valid current local history, reports precise
price-history blockers, and writes a deterministic traceable candidate ranking.

The PR review found that the original ME-RUN30 implementation duplicated
advice semantics, used canonical-looking advice labels without canonical
advice readiness, made closing-breakdown detection practically unreachable,
collapsed stale and insufficient history statuses, overstated ranking evidence
claims, overwrote artifacts by default, and underreported throughput. The
original run artifacts were removed and replaced by the corrected run below.

ME-RUN30 does not call providers, does not call model APIs, does not download
live prices, does not use synthetic data, does not mutate portfolio or
watchlist state, does not create broker orders, and does not send Telegram or
production delivery output.

## Component Reuse Strategy

The chosen strategy is technical setup screening.

The canonical deterministic advice engine requires valid dry-run artifacts and
fundamental context before assigning full advice labels. The full canonical
universe does not yet have those artifacts for every instrument. ME-RUN30
therefore does not produce canonical labels such as `buy_candidate`,
`wait_for_price`, `watchlist`, or `avoid_for_now`.

ME-RUN30 reuses the canonical setup/price/market context extractor through an
explicit adapter. The adapter supplies a canonical
`setup_price_market_context` payload derived from local price history and then
passes it through `extract_setup_price_market_context`. ME-RUN30 aggregation
and ranking consume that canonical context payload and emit only technical
screening labels:

```text
technical_setup_candidate
technical_wait_for_entry
technical_watch
technical_risk_exclusion
unable_to_analyse
```

All ranked candidates remain `full_advice_ready: false` until a future sprint
connects complete fundamental, portfolio, and market context.

## Replaced Run Artifacts

Invalidated original artifact directory:

```text
artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-20260714T100000Z/
```

Replacement artifact directory:

```text
artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z/
```

Replacement artifacts:

```text
manifest.json
universe_analysis_index.json
universe_analysis_summary.md
throughput_report.json
setup_detection_summary.json
analysis_outcome_distribution.json
blocker_report.json
candidate_ranking.json
candidate_ranking.md
top_candidates.md
unable_to_analyse.md
```

## Input Contract

ME-RUN30 reads:

```text
config/market_engine/universes/canonical_universe.json
data/processed
```

The replacement run used:

```text
run_id: me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z
cutoff_date: 2026-07-10
universe_version: me-data04-complete-canonical-local-market-dataset-v1
ranking_scope: technical_setup_screening
```

## Coverage and Processing Status

| Metric | Count |
|---|---:|
| Total canonical instruments | 952 |
| Attempted instruments | 952 |
| Eligible analysed | 946 |
| Blocked insufficient history | 4 |
| Blocked stale history | 2 |
| Blocked missing history | 0 |
| Blocked invalid history | 0 |
| Blocked unsupported mapping | 0 |
| Failed | 0 |
| Ranked candidates | 330 |

Final processing status distribution:

```text
eligible_analyzed: 946
blocked_insufficient_history: 4
blocked_stale_history: 2
```

Blocked instruments:

```text
BLD: blocked_stale_history
FDXF: blocked_insufficient_history
HONA: blocked_insufficient_history
JHG: blocked_stale_history
Q: blocked_insufficient_history
SOLS: blocked_insufficient_history
```

## Technical Screening Outcomes

Output label distribution:

| Output label | Count |
|---|---:|
| `technical_setup_candidate` | 105 |
| `technical_wait_for_entry` | 257 |
| `technical_watch` | 407 |
| `technical_risk_exclusion` | 177 |
| `unable_to_analyse` | 6 |

Confidence distribution:

| Confidence | Count |
|---|---:|
| `high` | 0 |
| `medium` | 539 |
| `low` | 413 |

Blocker distribution:

| Blocker | Count |
|---|---:|
| `no_clear_setup` | 407 |
| `price_or_risk_not_preferred` | 257 |
| `weak_or_high_risk_setup` | 177 |
| `insufficient_history` | 4 |
| `insufficient_forward_data` | 2 |

Compared with the original PR artifact, label counts by shape are unchanged,
but their meaning is corrected from canonical-looking advice labels to
technical screening labels. Ranked candidates dropped from 527 to 330 because
risk exclusions and low-scoring technical watches are no longer ranking
eligible.

## Breakdown Detection

The support-break calculation now excludes the latest bar from the support
window:

```text
prior_twenty_day_low = low.iloc[:-1].tail(20).min()
support_break_pct = (latest_close - prior_twenty_day_low) / prior_twenty_day_low
support_window_excludes_latest_bar: true
```

This makes a close below prior support detectable. An intraday low below prior
support does not become `below_support_or_breakdown` unless the close also
breaks support.

## Ranking Contract

The ranking policy is:

```json
{
  "ranking_scope": "technical_setup_screening",
  "incomplete_evidence_allowed": true,
  "missing_data_positive_evidence": false,
  "blocked_or_failed_instruments_excluded": true,
  "full_advice_ready": false
}
```

Each candidate records:

```text
ranking_scope
ranking_eligible
full_advice_ready
missing_evidence
positive_components
penalties
raw_score
candidate_score
exclusion_reasons
traceability
```

Missing fundamental, portfolio, and market context is visible on each ranked
candidate and is penalised. It is never counted as positive evidence. Blocked
or failed instruments are not ranking eligible.

Tie-breakers:

```text
candidate_score desc
symbol asc
instrument_id asc
```

Top five candidates in the replacement run:

| Rank | Symbol | Score | Output label | Full advice ready |
|---:|---|---:|---|---|
| 1 | ASB | 75 | `technical_setup_candidate` | false |
| 2 | ASH | 75 | `technical_setup_candidate` | false |
| 3 | ATR | 75 | `technical_setup_candidate` | false |
| 4 | AXP | 75 | `technical_setup_candidate` | false |
| 5 | BIO | 75 | `technical_setup_candidate` | false |

## Throughput

Replacement run throughput:

| Metric | Value |
|---|---:|
| Total runtime seconds | 2.966189 |
| Attempted instruments | 952 |
| Analysed instruments | 946 |
| Blocked instruments | 6 |
| Failed instruments | 0 |
| Wall-clock average seconds per ticker | 0.00311574 |
| Measured mean ticker runtime seconds | 0.00310106 |
| Median ticker runtime seconds | 0.00309958 |
| P95 ticker runtime seconds | 0.00329283 |
| Minimum ticker runtime seconds | 0.00033487 |
| Maximum ticker runtime seconds | 0.00815979 |
| Tickers per second | 320.950557 |
| Tickers per minute | 19257.033449 |
| Successful analysis rate | 0.993697 |
| Failure rate | 0.0 |

Runtime values are machine-dependent. Classifications, scores, blockers, and
ranking order are deterministic for identical inputs.

## Overwrite Policy

The Python API and CLI now default to `allow_overwrite=False`. Existing output
directories fail closed with `FileExistsError` or CLI exit code 2 unless
`--allow-overwrite` is explicitly passed. Artifact writing uses a temporary
directory and only replaces the final run directory after the complete required
artifact set is present, preventing partial directory mixing.

## Determinism

A comparison rerun was written to:

```text
/private/tmp/market-engine-run30-determinism/me-run30-determinism-20260714T143209Z/
```

Programmatic comparison confirmed:

```text
per-instrument status/output-label/score/blocker tuples match
candidate ranking order and scores match
```

The comparison printed:

```text
ME-RUN30 deterministic comparison passed
```

## Validation

Expanded executable coverage includes:

```text
canonical setup/context extractor reuse
absence of canonical advice labels in ME-RUN30 technical screening
breakdown support-window exclusion
intraday-low-without-closing-break behavior
invalid and insufficient OHLCV handling
specific price-history blocker status mapping
ranking score components and penalties
blocked and failed ranking exclusion
stable tie-breakers
overwrite fail-closed behavior
atomic artifact replacement behavior
throughput mean, median, p95, min, max, rates, and slowest ordering
artifact count consistency
governance guardrails
```

Executed validation:

```text
tests/market_engine/run/test_full_canonical_universe_analysis.py: 21 passed
tests/market_engine/advice tests/market_engine/run: 199 passed
```

Full Market Engine and repository-wide validation were run after the review
fix implementation and are recorded in the PR update.

## Governance

ME-RUN30 preserves the project doctrine:

```text
classification upstream
allocation downstream
Decision Engine = only allocation authority
```

The candidate ranking is a technical review queue, not full deterministic
investment advice, not allocation authority, not an order instruction, and not
portfolio mutation.

## Recommended Next Sprint

Recommended next sprint: connect full-advice readiness inputs for broad
universe candidates by adding cached fundamental, market, and portfolio context
adapters that can feed the existing deterministic advice engine without live
provider calls or Decision Engine authority changes.
