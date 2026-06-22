# ME-UNI07 Audit - Editable universe runtime input

Sprint: ME-UNI07 - Wire editable universe into local Market Engine runtime input

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI07

Branch: `me-uni07-wire-editable-universe-runtime-input-connector`

## Goal

Wire the editable Professional Swing Universe into local Market Engine runtime input while preserving the boundary between operator-managed candidates, canonical universe membership, source support, review, reporting, and execution authority.

## Files inspected

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
src/market_engine/ticker_universe/professional_swing.py
src/market_engine/ticker_universe/__init__.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
docs/market_engine/ticker_universe/me_uni06_editable_universe_loader_validation_implementation.md
```

## Files introduced

```text
src/market_engine/run/editable_universe_runtime_input.py
tests/market_engine/run/test_editable_universe_runtime_input.py
docs/market_engine/ticker_universe/me_uni07_editable_universe_runtime_input_implementation.md
docs/market_engine/audits/me_uni07_editable_universe_runtime_input_audit.md
docs/market_engine/backlog/me_uni07_editable_universe_runtime_input_backlog_entry.md
docs/market_engine/roadmap/me_uni07_editable_universe_runtime_input_roadmap_entry.md
```

## Existing runtime finding

The existing cached-source batch command already accepts explicit local ticker input through `--tickers` and then executes the established local cached-source batch path.

Therefore ME-UNI07 adds a bridge from editable-universe selection to explicit runtime ticker input instead of duplicating batch-run behavior.

## Implementation audit

ME-UNI07 implemented:

* an editable-universe runtime input format identity;
* fail-closed conversion from Professional Swing Universe validation errors to runtime input errors;
* selected ticker extraction from the editable universe using the approved ME-UNI06 predicate;
* loaded, selected, and excluded ticker metadata;
* explicit authority-boundary metadata;
* argv construction for the existing cached-source batch command surface;
* ticker-limit helper validation.

## Selection audit

The implemented selection predicate is:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

Rows with `source_mapping_required`, `manual_review_only`, `unsupported`, inactive state, or non-runtime statuses are not promoted into requested local runtime tickers.

## Boundary audit

The runtime input object explicitly records:

```text
source_policy_hint_authority=operator_hint_not_source_support_authority
canonical_promotion_authority=false
provider_call_authority=false
runtime_input_authority=local_cached_source_batch_requested_tickers_only
```

This prevents the editable universe from being misread as canonical source support, tradeability, recommendation authority, or execution authority.

## Validation status

Validation was not run in the connector environment because the macOS checkout and `.venv` were not mounted here, and the sandbox could not clone GitHub due DNS/network restrictions.

Required local validation:

```bash
git fetch origin
git checkout me-uni07-wire-editable-universe-runtime-input-connector
git pull origin me-uni07-wire-editable-universe-runtime-input-connector
git diff --check | tee /dev/tty | pbcopy
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_editable_universe_runtime_input.py tests/market_engine/ticker_universe/test_professional_swing_universe.py -q | tee /dev/tty | pbcopy
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q | tee /dev/tty | pbcopy
```

## Non-goals confirmed

ME-UNI07 does not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, source refresh, cached-source refresh, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, canonical-universe promotion, source-support authority, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Conclusion

ME-UNI07 wires the editable Professional Swing Universe into local Market Engine runtime input by producing explicit cached-source batch ticker input from validated editable-universe rows.

Recommended next sprint:

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

Optional hardening candidate:

```text
ME-UNI08 - Add first-class professional-swing-universe CLI flag
```
