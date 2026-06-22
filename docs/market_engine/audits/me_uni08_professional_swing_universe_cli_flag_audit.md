# ME-UNI08 Audit - Professional Swing Universe CLI Flag

Sprint: ME-UNI08 - Add first-class Professional Swing Universe CLI flag

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

Branch: `me-uni08-professional-swing-universe-cli-flag`

## Goal

Add a first-class CLI flag to select the approved editable Professional Swing Universe in the local cached-source batch runtime while preserving ME-UNI06/ME-UNI07 loader and validation behavior.

## Files Changed

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
docs/market_engine/ticker_universe/me_uni08_professional_swing_universe_cli_flag.md
docs/market_engine/audits/me_uni08_professional_swing_universe_cli_flag_audit.md
docs/market_engine/backlog/me_uni08_professional_swing_universe_cli_flag_backlog_entry.md
docs/market_engine/roadmap/me_uni08_professional_swing_universe_cli_flag_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
docs/market_engine/backlog/market_engine_backlog_me_uni07_append.md
docs/market_engine/roadmap/market_engine_roadmap_me_uni07_append.md
```

## Runtime Behavior Implemented

The cached-source batch dry-run command now supports:

```text
--professional-swing-universe
```

The flag uses the existing ME-UNI07 `build_professional_swing_runtime_input()` path, which uses the ME-UNI06 Professional Swing Universe loader and validator.

## Conflict Behavior

The new flag is in the existing mutually exclusive ticker-input group. Combining it with a custom `--canonical-ticker-universe <path>` input fails closed through argparse with a clear mutual-exclusion error.

No silent override behavior was introduced.

## Tests Added

Tests cover:

* first-class flag acceptance;
* default Professional Swing Universe source resolution;
* custom canonical universe path preservation;
* explicit conflict between `--professional-swing-universe` and `--canonical-ticker-universe <path>`;
* CLI help text containing the new flag.

## Validation Commands

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_cached_source_batch_dry_run_command.py tests/market_engine/run/test_editable_universe_runtime_input.py tests/market_engine/ticker_universe/test_professional_swing_universe.py -q
python -m pytest tests/market_engine -q
git diff --check
```

## Boundaries Preserved

ME-UNI08 did not introduce provider calls, SEC or EDGAR live calls, yfinance calls, source refresh, source-support classification, Telegram or email delivery, reporting output, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Next Sprint

Recommended next sprint:

```text
ME-SR05 - Classify source support for Professional Swing Universe
```
