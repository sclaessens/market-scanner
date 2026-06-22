# ME-SR04 - HO Canonical-Universe Source Identity Decision

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR04

## Purpose

ME-SR04 resolves the remaining HO blocker exposed by ME-RUN19 and ME-SR03.

The sprint decides whether HO has an approved SEC CompanyFacts cached-source identity or must be moved out of default canonical SEC CompanyFacts cached-source execution.

This is a source identity and canonical-universe governance sprint. It does not introduce provider calls, live SEC or EDGAR access, yfinance, source fabrication, portfolio writes, watchlist writes, Telegram or email delivery, production reports, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, ranking, scoring, conviction, urgency, tradeability, or execution advice.

## Starting Evidence

The canonical universe listed HO as:

```text
ticker: HO
name: Thales
market: EURONEXT
source_policy: cached_source_only
```

ME-SR02 source-refresh evidence recorded HO as unsupported:

```text
HO is Euronext Thales and has no approved SEC CompanyFacts CIK in ME-SR02 bounded snapshot bundle
```

The approved local snapshot bundle is:

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/
```

The raw snapshot directory contains no `HO_companyfacts.json` file and no approved alternate HO source snapshot.

## Decision

HO is excluded from default canonical SEC CompanyFacts cached-source execution.

HO remains present in the canonical ticker universe as an active, portfolio-relevant manual-review entry:

```text
ticker: HO
name: Thales
market: EURONEXT
source_policy: manual_review_only
portfolio_relevant: true
telegram_preview_eligible: false
telegram_delivery_eligible: false
```

This is an exclusion from the automated SEC CompanyFacts cached-source execution set, not a deletion from the canonical universe.

## Rationale

No approved local evidence supports a SEC CompanyFacts identity for HO.

ME-SR04 did not find:

* an approved SEC CompanyFacts CIK for Euronext Thales;
* a local cached raw SEC CompanyFacts snapshot for HO;
* a matching approved snapshot under another ticker or identifier;
* repository evidence proving HO was intended to represent a different supported security.

Because the source identity is unsupported, forcing HO through the cached-source pipeline would require inventing identity or source evidence. ME-SR04 explicitly rejects that path.

## Updated Canonical Universe Behavior

Default canonical SEC CompanyFacts cached-source execution now selects 12 tickers:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
AVGO
TSM
```

Manual-review-only exclusions:

```text
HO
SMCI
```

HO can only return to default cached-source execution through a future approved source identity or source-refresh/backfill sprint.

## Validation Run

Command:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --canonical-ticker-universe \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-sr04-canonical-identity-check-20260622T120000Z \
  --generated-at 2026-06-22T12:00:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine
```

Result:

```text
canonical_loaded_rows=14
canonical_selected_rows=12
excluded_manual_review_only=HO, SMCI
discovered_cached_source_tickers=12
requested_count=12
executed_count=12
completed_count=12
blocked_count=0
failed_count=0
batch_state=completed
```

Generated local artifacts:

```text
artifacts/market_engine/me-sr04-canonical-identity-check-20260622T120000Z/
```

Generated artifacts are local run evidence and are intentionally not committed.

## Files Updated

```text
data/market_engine/ticker_universe/ticker_universe.csv
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_contract.md
docs/market_engine/source_data/me_sr04_ho_canonical_universe_source_identity_decision.md
docs/market_engine/audits/me_sr04_ho_canonical_universe_source_identity_decision_audit.md
docs/market_engine/backlog/me_sr04_ho_canonical_universe_source_identity_decision_backlog_entry.md
docs/market_engine/roadmap/me_sr04_ho_canonical_universe_source_identity_decision_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Boundary Confirmation

ME-SR04 did not introduce:

* provider calls;
* live SEC or EDGAR calls;
* yfinance calls;
* live market data calls;
* broker calls;
* portfolio writes;
* watchlist writes;
* Telegram or email delivery;
* production reports;
* scheduler behavior;
* UI behavior;
* Decision Engine action semantics;
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

## Conclusion

HO is excluded from default canonical SEC CompanyFacts cached-source execution because no approved SEC CompanyFacts source identity or local cached source snapshot exists.

The canonical cached-source run now completes for the 12 supported active cached-source tickers with zero blocked tickers.

Recommended next sprint:

```text
ME-TG01 - Define Telegram preview contract
```

ME-TG01 remains contract-only and must not implement delivery.
