# Pipeline Contracts

Status: ACTIVE

This document is a compact operational contract map for the certified market-scanner pipeline.

## Purpose

Pipeline contracts define what each layer may read, write, and mean. They preserve deterministic behavior, row traceability, separation of concerns, and Decision Engine authority.

This document is an operational overview. Detailed schema changes still require Governance v2 classification before implementation.

## Layer Contract Table

| Layer | Authoritative input | Authoritative output | Allowed semantics | Forbidden semantics | Preservation expectation |
|---|---|---|---|---|---|
| Scanner | Ticker universe and market data | `data/processed/scanner_ranked.csv` | Opportunity discovery and setup metadata | Tradeability, allocation, conviction, final action, hidden strategic filtering | Preserve discovered opportunity distribution subject to explicit scanner contract rules |
| Validation | `scanner_ranked.csv` | `data/processed/validation_layer.csv`; `data/processed/entry_quality_metrics.csv` | Structure and entry-quality classification | Allocation eligibility, execution quality, urgency, final action | Preserve ticker/date identity and classify without allocation authority |
| Context | `scanner_ranked.csv`; optional sector relative-strength source | `data/processed/context_strength.csv` | Leadership and relative-strength classification | Blocking opportunities, tradeability, allocation simulation | Preserve scanner ticker/date universe |
| Fundamental | `context_strength.csv` | `data/processed/fundamental_quality.csv` | Fundamental quality metadata | Ranking, conviction, allocation priority, final action | Preserve context ticker/date universe |
| Timing State | `fundamental_quality.csv`; `entry_quality_metrics.csv` | `data/processed/timing_state_layer.csv` | Timing condition and setup-state classification | Execution gating, urgency, tradeability, allocation readiness | Preserve upstream ticker/date universe and append timing metadata |
| Portfolio Intelligence | `timing_state_layer.csv`; portfolio position source | `data/processed/portfolio_intelligence.csv` | Descriptive portfolio presence, exposure, and state metadata | BUY/SELL authority, allocation gating, opportunity removal | Preserve timing rows and add descriptive portfolio metadata |
| Decision Engine | `portfolio_intelligence.csv` | `data/processed/final_decisions.csv`; `data/logs/decision_engine_log.csv` | Allocation, execution, arbitration, final action, rationale, provenance | Hidden filtering, upstream mutation | Preserve input row count, ticker/date universe, order, and traceability |
| Stability | `final_decisions.csv` | `data/processed/stability_state.csv`; `data/logs/stability_layer_log.csv` | Persistence and stability metadata | Allocation override, suppression, final-action mutation | Preserve decision input identity and never mutate decisions |
| Reporting | `final_decisions.csv`; optional `stability_state.csv` | `data/processed/reporting_dashboard_data.csv`; `data/logs/reporting_layer_log.csv`; `reports/daily/telegram_message.txt` | Communication, representation, grouping, traceability | Decision logic, allocation priority, urgency interpretation, source decision mutation | Represent source decisions without omission or semantic override |

## Boundary Rules

- Upstream layers classify only.
- Portfolio Intelligence is descriptive only.
- Stability is persistence metadata only.
- Decision Engine is the only allocation, execution, arbitration, and final-action authority.
- Reporting communicates Decision Engine outputs only.

## Forbidden Semantics Outside Decision Engine

Outside `scripts/core/decision_engine.py`, do not introduce:

- tradeability
- conviction
- BUY logic
- SELL logic
- REMOVE logic
- allocation gating
- execution urgency
- hidden filtering
- ranking or scoring authority that changes opportunity treatment

## Row Preservation

Pipeline layers must preserve row identity where their contract requires it. If a layer cannot preserve row count, ticker/date universe, or source ordering, the change is at least Governance Level 2 and must be reviewed before implementation.

The Decision Engine must preserve its input row count, ticker/date universe, and order unless a future formally governed architecture change explicitly changes that contract.

## Determinism

Outputs must be deterministic for the same inputs. Any nondeterministic source, ordering rule, generated timestamp, or external dependency must be explicit, traceable, and tested or documented.

## Reporting Boundary

Reporting may format, group, summarize, truncate, and deliver communications. It must preserve source traceability and must not reinterpret, prioritize, suppress, or override Decision Engine decisions.
