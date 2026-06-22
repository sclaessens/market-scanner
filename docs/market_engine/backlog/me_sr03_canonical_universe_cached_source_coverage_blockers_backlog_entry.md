# ME-SR03 Backlog Entry - Canonical-Universe Cached-Source Coverage Blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR03

## Goal

Resolve or precisely document the canonical-universe cached-source coverage blockers exposed by ME-RUN19:

```text
HO
ASML
TSM
```

## Outcome

ASML and TSM are resolved through minimal source mapping remediation against existing cached SEC CompanyFacts payloads.

HO remains blocked because no approved local SEC CompanyFacts snapshot exists in the ME-SR02 snapshot bundle.

## Implemented Changes

ME-SR03 updated:

```text
src/market_engine/source_intake/sec_companyfacts_fields.py
```

Implemented source-only mapping support for:

* annual `20-F` and `20-F/A` filings;
* raw `EUR` unit preservation;
* explicit `ifrs-full` aliases for cached TSM-like source facts.

No currency conversion, derived values, estimates, provider refresh, or fabricated values were introduced.

## Validation Outcome

Narrow check:

```text
ASML completed
TSM completed
HO blocked_missing_cached_source
```

Canonical rerun:

```text
requested_count=13
discovered_cached_source_count=12
executed_count=12
completed_count=12
blocked_count=1
failed_count=0
```

Remaining blocked ticker:

```text
HO
```

## Boundaries

ME-SR03 did not introduce provider calls, live SEC or EDGAR calls, yfinance calls, live market data calls, broker calls, portfolio writes, watchlist writes, Telegram or email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Candidate

### ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision

Status: NEXT CANDIDATE AFTER ME-SR03

Job family: ME-SR - Source Refresh / Source Coverage

Goal: decide whether HO should receive an approved source identity/backfill path or be moved out of default cached-source execution until a valid source exists.

## Planned Sequence After ME-SR04

The operator clarified that the future minimum scan universe must be editable and must support adding, disabling, removing, and classifying tickers without Python hardcoding.

The planned sequence after ME-SR04 is now recorded in:

```text
docs/market_engine/backlog/me_sr03_next_sprint_sequence_universe_management_backlog_entry.md
```

Required follow-up sequence:

```text
ME-UNI04 - Define editable Professional Swing Universe contract
ME-UNI05 - Import and normalize Professional Swing Universe seed list
ME-UNI06 - Implement editable universe loader and validation
ME-SR05  - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```

This sequence must not be skipped in favor of hardcoded large-universe scans, ranking, scoring, entry analysis, or buy planning.
