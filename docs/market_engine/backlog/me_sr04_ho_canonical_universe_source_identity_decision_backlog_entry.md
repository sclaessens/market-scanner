# ME-SR04 Backlog Entry - HO Canonical-Universe Source Identity Decision

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR04

## Goal

Decide whether HO should receive an approved SEC CompanyFacts source identity or be moved out of default canonical cached-source execution until a valid source exists.

## Outcome

HO remains in the canonical ticker universe as Thales on Euronext, but it is now `manual_review_only`.

This removes HO from default canonical SEC CompanyFacts cached-source execution while preserving the ticker for manual portfolio/source identity review.

## Evidence

ME-SR02 recorded HO as unsupported:

```text
HO is Euronext Thales and has no approved SEC CompanyFacts CIK in ME-SR02 bounded snapshot bundle
```

No approved local `HO_companyfacts.json` raw snapshot exists in the bounded ME-SR02 source snapshot bundle.

No repository evidence proves that HO should be remapped to a different supported security.

## Validation Outcome

Canonical cached-source rerun after the source-policy decision:

```text
canonical_loaded_rows=14
canonical_selected_rows=12
excluded_manual_review_only=HO, SMCI
requested_count=12
discovered_cached_source_tickers=12
executed_count=12
completed_count=12
blocked_count=0
failed_count=0
```

## Boundaries

ME-SR04 did not introduce provider calls, live SEC or EDGAR calls, yfinance calls, source fabrication, portfolio writes, watchlist writes, Telegram or email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Recommended Sprint

### ME-TG01 - Define Telegram preview contract

Status: RECOMMENDED NEXT AFTER ME-SR04

Job family: ME-TG - Telegram / Preview Governance

Goal: define a render-only Telegram preview contract after canonical cached-source execution has a clean supported-ticker run.

Scope: documentation and contract only. ME-TG01 must not implement delivery, send Telegram messages, create scheduler behavior, mutate portfolios or watchlists, introduce provider calls, or introduce Decision Engine action semantics.
