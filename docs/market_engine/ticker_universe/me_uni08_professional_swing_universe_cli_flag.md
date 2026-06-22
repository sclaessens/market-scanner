# ME-UNI08 - Professional Swing Universe CLI Flag

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

## Purpose

ME-UNI08 adds a first-class local runtime CLI flag for the editable Professional Swing Universe.

The sprint builds on:

```text
ME-UNI06 - Implement editable universe loader and validation
ME-UNI07 - Wire editable universe into local Market Engine runtime input
```

The new flag lets an operator select the approved Professional Swing Universe without manually remembering the CSV path or using the ME-UNI07 bridge helper directly.

## CLI Flag

The cached-source batch dry-run command now supports:

```text
--professional-swing-universe
```

The flag routes through the existing ME-UNI07 runtime-input builder:

```text
build_professional_swing_runtime_input()
```

That builder uses the ME-UNI06 loader and validation path:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

## Selection Behavior

ME-UNI08 preserves the existing approved editable-universe selection predicate:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

The flag produces local cached-source batch requested-ticker input only. It does not classify source support, promote rows into the canonical SEC CompanyFacts universe, call providers, or create action authority.

## Conflict Behavior

The new flag is part of the existing mutually exclusive ticker-input group.

The command fails closed if `--professional-swing-universe` is combined with any other ticker input mode, including a custom `--canonical-ticker-universe <path>` input.

No silent priority or implicit override is allowed.

## Runtime Metadata

Command results now preserve editable-universe runtime-input metadata under:

```text
run_context.editable_universe_runtime_input
```

This metadata includes:

* editable runtime-input format version;
* Professional Swing Universe source contract version;
* source path;
* selection policy;
* loaded row count;
* selected row count;
* excluded ticker groups;
* no-provider and no-canonical-promotion authority markers.

## Non-Goals

ME-UNI08 does not introduce:

* provider calls;
* SEC or EDGAR live calls;
* yfinance calls;
* source refresh;
* source-support classification;
* Telegram or email delivery;
* reporting output;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Validation

Focused validation:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_cached_source_batch_dry_run_command.py tests/market_engine/run/test_editable_universe_runtime_input.py tests/market_engine/ticker_universe/test_professional_swing_universe.py -q
```

Full validation:

```text
python -m pytest tests/market_engine -q
```

## Next Sprint

Recommended next sprint:

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

ME-SR05 should classify actual local cached-source support for Professional Swing Universe rows. It must not treat the operator `source_policy_hint` column as authoritative source truth.
