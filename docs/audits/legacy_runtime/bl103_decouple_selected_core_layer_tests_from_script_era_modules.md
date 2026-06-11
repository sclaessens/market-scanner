# BL103 — Decouple selected core layer tests from script-era modules

Status: COMPLETED

## Purpose

BL103 decouples selected active core layer tests from script-era modules.

Targeted tests:

```text id="n1hj3s"
tests/core/test_build_context_layer.py
tests/core/test_build_validation_layer.py
tests/core/test_entry_quality.py
tests/core/test_build_timing_state_layer.py
tests/core/test_build_stability_layer.py
```

Targeted script-era modules:

```text id="tkzjnb"
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

`tests/core/test_entry_quality.py` was also decoupled because its previous active owner was `scripts/core/build_validation_layer.py`.

BL103 is a decoupling sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

## Pre-decoupling finding

Before BL103, the focused test run failed during collection because selected tests imported script-era modules directly.

Example failure:

```text id="mteya3"
ModuleNotFoundError: No module named 'scripts'
```

The full suite still passed because these tests were listed as migration blockers.

## What changed

BL103 replaced behavior/import tests with static/canonical contract tests.

Updated tests:

```text id="h72ctx"
tests/core/test_build_context_layer.py
tests/core/test_build_validation_layer.py
tests/core/test_entry_quality.py
tests/core/test_build_timing_state_layer.py
tests/core/test_build_stability_layer.py
```

Updated blocker registries:

```text id="7fd0nw"
tests/conftest.py
tests/test_operator_visibility.py
```

## Blocker registry update

BL103 removed the following tests from the high-risk script-era blocker registries:

```text id="q8phqm"
core/test_build_context_layer.py
core/test_build_validation_layer.py
core/test_entry_quality.py
core/test_build_timing_state_layer.py
core/test_build_stability_layer.py
```

## Context layer contract

`tests/core/test_build_context_layer.py` no longer imports:

```text id="qwznn6"
scripts.core.build_context_layer
```

The test now validates static context-layer contract policy.

Expected columns:

```text id="f2hr7l"
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

Expected context states:

```text id="krv20i"
LEADING
STRONG
NEUTRAL
WEAK
UNKNOWN
```

The contract remains classification-only and does not gain trade, allocation, execution, urgency, or final-action authority.

## Validation layer contract

`tests/core/test_build_validation_layer.py` no longer imports:

```text id="g452d5"
scripts.core.build_validation_layer
```

The test now validates static validation-layer contract policy.

Expected output columns:

```text id="yu0h38"
ticker
date
structure_state
structure_reason
setup_type
valid_setup
validation_reason
```

Expected structure states:

```text id="8qsoc7"
COHERENT
BROKEN
INCOMPLETE
```

The contract remains setup-structure validation only and does not gain tradeability, conviction, allocation, execution, urgency, or final-action authority.

## Entry quality contract

`tests/core/test_entry_quality.py` no longer imports:

```text id="b14xk0"
scripts.core.build_validation_layer
```

The test now validates static entry-quality metric contract policy.

Expected metric columns:

```text id="8oxz2a"
ticker
date
distance_to_breakout_pct
breakout_extension_atr
extension_atr
distance_ma20_pct
volume_ratio
range_atr
entry_quality_state
entry_quality_reason
```

Expected entry-quality states:

```text id="h7zbaa"
BALANCED
EXTENDED
WIDE_RANGE
```

The contract remains descriptive metric output only and does not alter validation structure or create trade authority.

## Timing state layer contract

`tests/core/test_build_timing_state_layer.py` no longer imports:

```text id="jcmw4j"
scripts.core.build_timing_state_layer
```

The test now validates static timing-state metadata policy.

The contract preserves upstream fundamental-quality columns and appends timing metadata fields such as:

```text id="sox4x6"
timing_state
timing_reason
breakout_state
pullback_state
compression_state
extension_state
participation_state
timing_environment
timing_pattern_state
trend_participation_state
timing_structure_state
timing_metadata_status
timing_source_data_status
timing_source_timestamp
timing_generated_at
```

The contract remains non-filtering metadata enrichment and does not gain ranking, scoring, tradeability, allocation, execution, or final-action authority.

## Stability layer contract

`tests/core/test_build_stability_layer.py` no longer imports:

```text id="ybn7m5"
scripts.core.build_stability_layer
```

The test now validates static stability-layer observation policy.

The contract preserves decision identity fields and appends observation-only fields such as:

```text id="7fr5u9"
previous_final_action
action_persistence
persistence_duration
transition_frequency
escalation_frequency
conviction_persistence
behavioural_stability
stability_state
stability_reason
```

The contract does not add hidden execution gates, allocation gates, suppression flags, action overrides, or final-action overrides.

## Active import check

BL103 checked active `scripts.core` imports after decoupling.

Remaining positive imports:

```text id="q7iffd"
tests/core/test_build_entry_quality_backfill.py
tests/core/test_build_context_backfill.py
tests/core/test_decision_engine.py
tests/core/test_build_portfolio_intelligence.py
```

Interpretation:

* the five BL103-targeted tests were successfully decoupled;
* remaining imports are outside BL103 scope and must be handled by later sprints.

## Validation

Focused suite:

```bash id="b8zarc"
pytest tests/core/test_build_context_layer.py \
       tests/core/test_build_validation_layer.py \
       tests/core/test_entry_quality.py \
       tests/core/test_build_timing_state_layer.py \
       tests/core/test_build_stability_layer.py \
       tests/test_operator_visibility.py -q
```

Result:

```text id="xvv4h9"
35 passed in 0.07s
```

Full suite:

```bash id="m9xd4o"
pytest -q
```

Result:

```text id="i6edlb"
610 passed in 0.64s
```

## Decision

BL103 decision:

```text id="mu2dpg"
SELECTED_CORE_LAYER_ACTIVE_TEST_DEPENDENCIES_DECOUPLED
```

The selected core-layer tests no longer import script-era modules.

## Remaining cleanup status

Remaining active positive `scripts.core` test imports:

```text id="a3skbg"
tests/core/test_build_entry_quality_backfill.py
tests/core/test_build_context_backfill.py
tests/core/test_decision_engine.py
tests/core/test_build_portfolio_intelligence.py
```

High-risk areas remain out of scope:

```text id="ggozu6"
scripts/core/decision_engine.py
scripts/core/data_fetcher.py
scripts/core/scanner.py
scripts/core/build_portfolio_intelligence.py
scripts/portfolio/*
scripts/watchlist/*
scripts/validate_scans.py
```

## Recommended next sprint

Recommended next sprint:

```text id="ukw3sz"
BL104 — Review archive-readiness of decoupled core layer modules
```

Candidate modules for review only:

```text id="ko4z77"
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

BL104 should not automatically archive them. It should first check:

* active positive imports;
* active positive path references;
* side-effect markers;
* canonical ownership;
* whether remaining tests or metadata still depend on these files;
* whether archive is safe as a cluster or still blocked.

## Guardrails

* No live provider calls were run.
* No yfinance calls were run.
* No SEC/EDGAR calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist state was modified.
* No Decision Engine authority was changed.
* No script-era runtime module was archived.
* No script-era runtime module was edited.
* No script-era runtime module was executed directly.
