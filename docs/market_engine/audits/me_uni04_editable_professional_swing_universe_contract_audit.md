# ME-UNI04 Audit - Editable Professional Swing Universe contract

Sprint: ME-UNI04 - Define editable Professional Swing Universe contract

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI04

Branch: `me-uni04-define-editable-professional-swing-universe-contract`

## Goal

Define a safe editable Professional Swing Universe contract before seed-list import, loader implementation, source-support classification, cached-source scanning, readable output, or candidate classification work.

## Files inspected

```text
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_contract.md
docs/market_engine/ticker_universe/me_uni03_initial_canonical_ticker_universe_csv.md
docs/market_engine/backlog/me_sr04_ho_canonical_universe_source_identity_decision_backlog_entry.md
docs/market_engine/roadmap/me_sr04_ho_canonical_universe_source_identity_decision_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Contract added

```text
market-engine-editable-professional-swing-universe-v1
```

Contract document:

```text
docs/market_engine/ticker_universe/me_uni04_editable_professional_swing_universe_contract.md
```

## Decisions

ME-UNI04 separates the Professional Swing Universe from the existing canonical ticker universe:

* canonical universe remains the approved cached-source RUN execution universe;
* Professional Swing Universe is an editable candidate universe for future expansion and review;
* overlap is allowed but must be explicit;
* Professional Swing Universe membership does not imply canonical membership;
* Professional Swing Universe membership does not imply source support, delivery eligibility, preview eligibility, tradeability, ranking, scoring or action authority.

ME-UNI04 reserves this path category:

```text
data/market_engine/ticker_universe/professional_swing_universe/
```

and defines this future default seed path:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

ME-UNI04 does not create that CSV.

## Validation coverage defined

The contract defines fail-closed validation for:

* required columns;
* malformed UTF-8 CSV input;
* duplicate or unnamed headers;
* required values;
* ticker normalization and character set;
* allowed market values;
* allowed asset type values;
* allowed active values;
* allowed universe statuses;
* allowed source-policy hints;
* integer operator priority;
* allowed swing, liquidity, volatility and market-cap profiles;
* duplicate normalized ticker and market rows;
* default-selection exclusions.

## Downstream sequence preserved

ME-UNI04 preserves this next sequence:

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```

## Boundaries preserved

ME-UNI04 is documentation and contract only.

It does not introduce Python code, tests, CSV data, fixtures, provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated artifact commits, source-support authority, canonical-universe promotion, Decision Engine action labels, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Validation performed

Repository tests were not run from this connector session because the macOS checkout is not mounted in the runtime. This is acceptable for ME-UNI04 because it is documentation-only and introduces no Python code, tests, CSV data or runtime behavior.

Local validation to run before merge:

```text
git diff --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```

## Files changed

```text
docs/market_engine/ticker_universe/me_uni04_editable_professional_swing_universe_contract.md
docs/market_engine/audits/me_uni04_editable_professional_swing_universe_contract_audit.md
docs/market_engine/backlog/me_uni04_editable_professional_swing_universe_contract_backlog_entry.md
docs/market_engine/roadmap/me_uni04_editable_professional_swing_universe_contract_roadmap_entry.md
```

## Conclusion

ME-UNI04 defines the editable Professional Swing Universe contract without granting runtime, source-refresh, reporting, delivery, portfolio, watchlist, Decision Engine, action, allocation, ranking, scoring, target-price, position-sizing, tradeability or execution authority.

Next sprint:

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
```
