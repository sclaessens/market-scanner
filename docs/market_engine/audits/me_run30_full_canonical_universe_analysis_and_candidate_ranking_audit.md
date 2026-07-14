# ME-RUN30 Full Canonical-Universe Analysis and Candidate Ranking Audit

Status: `completed_with_blockers`

## Executive Summary

ME-RUN30 ran the first broad deterministic Market Engine analysis across the
complete current canonical universe. The run consumed the 952-instrument
canonical universe and local price-history CSVs maintained through ME-DATA05,
attempted every instrument, analysed all instruments with valid current local
history, produced setup and advice distributions, reported blockers, and wrote
a deterministic traceable candidate ranking.

The sprint did not call providers, did not use synthetic forward data, did not
mutate portfolio or watchlist state, did not create broker orders, and did not
send Telegram or production delivery output. It produced deterministic advice
labels as analysis output, not allocation decisions.

## Run Artifacts

Primary run:

```text
artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-20260714T100000Z/
```

Idempotency comparison run:

```text
artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-idempotency-20260714T101000Z/
```

Primary artifacts:

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

The run used:

```text
cutoff_date: 2026-07-10
universe_version: me-data04-complete-canonical-local-market-dataset-v1
```

## Coverage and Processing Status

| Metric | Count |
|---|---:|
| Total canonical instruments | 952 |
| Attempted instruments | 952 |
| Eligible analysed | 946 |
| Insufficient history | 6 |
| Blocked | 0 |
| Failed | 0 |
| Unsupported mappings | 0 |
| Ranked candidates | 527 |

Final processing status distribution:

```text
eligible_analyzed: 946
insufficient_history: 6
```

The six insufficient-history instruments are explicitly retained as
`unable_to_analyse` rather than ranked.

## Setup and Advice Outcomes

Advice label distribution:

| Advice label | Count |
|---|---:|
| `buy_candidate` | 105 |
| `wait_for_price` | 257 |
| `watchlist` | 407 |
| `avoid_for_now` | 177 |
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

## Candidate Ranking Policy

Candidate ranking is deterministic and traceable. It uses only local
price-history-derived setup observations and explicit evidence metadata:

```text
trend_state
setup_state
price_position
risk_state
history_depth
missing_evidence_penalty
```

Missing evidence is never treated as positive evidence. Every candidate in the
current run carries missing fundamental, portfolio, and market context, so the
score includes an explicit missing-evidence penalty. Instruments with
insufficient local history are excluded from ranking.

Tie-breakers:

```text
candidate_score desc
symbol asc
instrument_id asc
```

Top five candidates in the primary run:

| Rank | Symbol | Score | Advice | Setup |
|---:|---|---:|---|---|
| 1 | ASB | 87 | `buy_candidate` | breakout candidate / uptrend |
| 2 | ASH | 87 | `buy_candidate` | breakout candidate / uptrend |
| 3 | ATR | 87 | `buy_candidate` | breakout candidate / uptrend |
| 4 | AXP | 87 | `buy_candidate` | breakout candidate / uptrend |
| 5 | BIO | 87 | `buy_candidate` | breakout candidate / uptrend |

The human-readable review package is:

```text
top_candidates.md
```

## Throughput

The primary run records total runtime, per-ticker runtime, and throughput in:

```text
throughput_report.json
```

Observed runtime on this local execution was approximately 2.8 to 3.1 seconds
for 952 attempted instruments. Runtime is expected to vary by machine, while
processing outcomes and candidate ordering remain deterministic for identical
inputs.

## Idempotency

The second run used the same cutoff and local inputs. Programmatic comparison
confirmed:

```text
final processing status counts match
advice label counts match
per-instrument status/advice/score tuples match
candidate ranking order and scores match
```

The comparison printed:

```text
ME-RUN30 deterministic comparison passed
```

## Governance

ME-RUN30 preserves the project doctrine:

```text
classification upstream
allocation downstream
Decision Engine = only allocation authority
```

The candidate ranking is an evidence-based review queue, not allocation
authority, not an order instruction, and not portfolio mutation.

## Remaining Limitations

The analysis is currently price-history-led. Fundamental, portfolio, and market
context are recorded as missing evidence and penalised in ranking. A later
sprint can connect broader cached-source/fundamental context once the
canonical-universe analysis path is stable.

## Acceptance Summary

| Criterion | Result |
|---|---|
| Complete canonical universe attempted | pass |
| Every ticker has explicit final status | pass |
| No ticker-specific analysis logic | pass |
| Throughput and runtime reported | pass |
| Setup, advice, and blocker distributions reported | pass |
| Deterministic traceable candidate ranking generated | pass |
| Top candidates reviewable without opening ticker artifacts | pass |
| Rerun produced equivalent ordering and outcomes | pass |
| No production side effects | pass |
