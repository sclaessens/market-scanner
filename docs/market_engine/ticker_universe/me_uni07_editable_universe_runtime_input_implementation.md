# ME-UNI07 - Editable universe runtime input implementation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI07

## Purpose

ME-UNI07 wires the validated editable Professional Swing Universe into local Market Engine runtime input.

The sprint bridges the ME-UNI06 loader output to the existing local cached-source batch runtime by producing explicit runtime ticker input from the editable universe selection policy.

## Runtime implemented

```text
src/market_engine/run/editable_universe_runtime_input.py
```

Public API:

```text
EDITABLE_UNIVERSE_RUNTIME_INPUT_FORMAT_VERSION
PROFESSIONAL_SWING_RUNTIME_INPUT_SELECTION_POLICY
EditableUniverseRuntimeInput
EditableUniverseRuntimeInputError
build_professional_swing_runtime_input
build_cached_source_batch_argv_from_professional_swing_universe
selected_tickers_from_runtime_input
```

## Runtime input contract

```text
market-engine-editable-universe-runtime-input-v1
```

## Source contract consumed

```text
market-engine-editable-professional-swing-universe-v1
```

## Selection behavior

The runtime input builder loads the editable Professional Swing Universe with `include_inactive=True` and then applies the approved runtime selection predicate:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

This preserves the ME-UNI06 ordering behavior from the editable universe loader and produces deterministic requested ticker input for local cached-source batch runs.

## Output behavior

`build_professional_swing_runtime_input` returns a bounded `EditableUniverseRuntimeInput` record containing:

* source contract identity;
* source path;
* selection policy;
* requested tickers;
* loaded and selected row counts;
* excluded inactive tickers;
* excluded source-mapping-required tickers;
* excluded manual-review-only tickers;
* excluded unsupported tickers;
* explicit authority-boundary fields.

`build_cached_source_batch_argv_from_professional_swing_universe` converts the selected editable-universe tickers into an argv tuple for the existing local cached-source batch command path.

## Authority boundaries

The editable universe runtime input has:

```text
source_policy_hint_authority=operator_hint_not_source_support_authority
canonical_promotion_authority=false
provider_call_authority=false
runtime_input_authority=local_cached_source_batch_requested_tickers_only
```

ME-UNI07 does not make editable-universe rows canonical, source-supported, actionable, ranked, scored, tradeable, report-deliverable, or eligible for execution.

## Tests implemented

```text
tests/market_engine/run/test_editable_universe_runtime_input.py
```

Coverage includes:

* valid editable-universe to runtime-input conversion;
* runtime payload boundary preservation;
* cached-source batch argv construction;
* positive ticker-limit helper behavior;
* zero ticker-limit fail-closed behavior;
* missing editable-universe file fail-closed behavior.

## Connector implementation note

This connector execution did not mutate the existing `cached_source_batch_dry_run_command.py` parser directly. Instead, ME-UNI07 adds a safe bridge module that converts editable-universe output into explicit cached-source batch runtime input.

A later hardening sprint may add a first-class `--professional-swing-universe` CLI flag if desired, but the runtime input wiring is now available without changing the established cached-source batch command surface.

## Boundaries preserved

ME-UNI07 does not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, source refresh, cached-source refresh, reporting delivery, Telegram/email delivery, production writes, portfolio writes, watchlist writes, scheduler behavior, UI behavior, canonical-universe promotion, source-support authority, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.
