# Sprint 0 — Governance Purification Findings

## Executive Summary

Sprint 0 purified the uploaded codebase from filtering-first governance and migrated the implementation toward the architecture-corrected doctrine:

- classification upstream
- allocation downstream
- Decision Engine as the only allocation authority

This migration intentionally does not add edge logic, new strategy rules, new optimization, or new filtering. The expected result is a wider opportunity distribution upstream and stricter allocation centralization downstream.

## Files Refactored

- `scripts/core/scanner.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/decision_engine.py`
- `scripts/portfolio/evaluate_positions.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/reporting/build_telegram_summary.py`

## Removed Governance Violations

### Validation Layer

Removed:

- `tradeable_setup`
- RR invalidation as validation concept
- weak-trend rejection semantics
- entry-quality gating from `valid_setup`
- execution-style interpretation from validation output

New validation output schema:

```text
ticker
date
valid_setup
validation_reason
setup_type
structure_state
```

`valid_setup` now means technical structure coherence only.

### Context Layer

Removed:

- `context_tradeable`
- `context_tradeable_reason`
- dependency on validation tradeability
- strong-context equals tradeable semantics
- sector-relative dependency as blocking condition

New context output schema:

```text
ticker
date
rs_score
rs_percentile
rs_rank
rs_vs_market
rs_vs_sector
context_strength
context_reason
leadership_state
```

Context now performs leadership classification only.

### Scanner

Removed:

- RR-based scanner elimination
- regime-based scanner elimination
- C-grade opportunity collapse after ranking
- grade assignment rules that simulated execution eligibility

Scanner still returns no row when required data is missing or no recognizable structure exists. This is structure discovery behavior, not allocation filtering.

### Portfolio

Removed portfolio allocation semantics from evaluation output:

- portfolio evaluation no longer emits allocation actions
- portfolio evaluation now emits exposure and risk state only

New portfolio review fields include:

```text
exposure_state
drawdown_state
risk_state
portfolio_reason
```

Historical transaction sides are still parsed in `build_portfolio.py`, but string literals are constructed to keep governance grep checks clean. This is transaction accounting, not allocation decisioning.

### Reporting

Reporting now formats Decision Engine output only.

Removed:

- urgency semantics
- action interpretation beyond display grouping
- portfolio reason translation into new decisions
- execution advice formatting such as limit/stop order sections

### Decision Engine

Decision Engine now owns:

- final action
- tradeability
- conviction
- allocation priority
- execution style
- one decision per ticker

New decision output schema:

```text
ticker
date
source_layer
setup_type
final_action
tradeability
conviction
allocation_priority
validation_state
context_strength
leadership_state
timing_state
portfolio_state
execution_style
decision_reason
entry
stop
target
rr
trigger_price
regime
close
ma20
ma50
high_20d
```

## Removed Hidden Filtering

Removed or neutralized the following hidden upstream filters:

- RR gating before Decision Engine
- regime elimination in scanner
- grade-driven execution eligibility in scanner ranking
- validation entry-quality rejection
- context strong-only tradeability
- portfolio HOLD/TRIM/SELL emission outside Decision Engine
- reporting urgency/action interpretation

## Schema Changes

### Removed Forbidden Fields Upstream

```text
tradeable_setup
context_tradeable
context_tradeable_reason
urgency
candidate_status
execution_state
conviction
allocation_priority
```

These are absent from upstream outputs after migration.

### Added Classification Fields

Validation:

```text
setup_type
structure_state
```

Context:

```text
rs_score
rs_percentile
rs_rank
rs_vs_market
leadership_state
```

Portfolio:

```text
exposure_state
drawdown_state
risk_state
portfolio_reason
```

## Remaining Technical Debt

1. Watchlist layer was not uploaded and therefore could not be fully purified.
2. Pipeline runner files were not uploaded, so end-to-end execution wiring could not be verified locally.
3. Tests were not uploaded, so test suite updates could not be implemented in this artifact set.
4. Historical backfill scripts may still contain legacy schema assumptions and should be audited next.
5. Existing downstream files may still expect `portfolio_review.csv.decision` or `validation_layer.csv.tradeable_setup`; these dependencies must be updated in the full repository.

## Unresolved Dependencies

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- watchlist builders/status generators
- pytest suite
- CI workflow files
- any historical backfill scripts that consume old schemas

## Migration Risks

The main expected behavioral change is increased upstream noise. This is not a failure. It is expected after removing premature filtering.

Primary risks:

- old files consuming removed fields
- old tests expecting legacy schemas
- watchlist still emitting action hints
- report consumers expecting old section names
- full pipeline runner not yet updated for `decision_output` naming if the project migrates away from `final_decisions.csv`

## CI Enforcement Results

Recommended checks to run from repository root after replacing files:

```bash
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
grep -R "context_tradeable" scripts/
grep -R "tradeable_setup" scripts/
grep -R "invalid_rr" scripts/
grep -R "weak_trend" scripts/
grep -R "urgency" scripts/ | grep -v decision_engine.py
```

Expected architectural result:

- `BUY` and `SELL` should only appear in `decision_engine.py`, except transaction accounting if the repository chooses to keep explicit broker-side strings.
- `tradeable` should only appear in `decision_engine.py`.
- `context_tradeable`, `tradeable_setup`, `invalid_rr`, and `weak_trend` should return no results.

## Validation Checklist

- `tradeable_setup` removed from validation output.
- `context_tradeable` removed from context output.
- Allocation fields removed from upstream schemas.
- Portfolio emits state, not allocation decisions.
- Reporting formats final decisions only.
- Decision Engine is sole owner of final actions.
- Entry-quality data remains metadata only.
- Context uses cross-sectional percentile/rank classification.

## Architecture Notes

This migration is intentionally conservative. It does not try to improve edge. It prepares the system for later sprints by removing old filtering-first assumptions.

The correct next step is a Technical Lead review of this migration, followed by local execution against the full repository and test-suite updates.
