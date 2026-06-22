# ME-SR03 - Canonical-Universe Cached-Source Coverage Blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR03

## Purpose

ME-SR03 investigates and minimally remediates the canonical-universe cached-source coverage blockers exposed by ME-RUN19:

```text
HO
ASML
TSM
```

The sprint remains source-coverage only. It does not introduce provider calls, live SEC or EDGAR access, yfinance, portfolio writes, Telegram/email delivery, production reports, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, ranking, scoring, conviction, urgency, tradeability, or execution advice.

## Starting Evidence

ME-RUN19 completed 10 tickers and blocked 3:

```text
ASML
HO
TSM
```

ASML and TSM blocked at Recommendation Review because Source Context was `MISSING` after all four canonical fields were missing:

```text
revenue
net_income
operating_cash_flow
capital_expenditures
```

HO blocked before per-ticker execution because no cached source snapshot was found.

## Source Snapshot Investigation

Source snapshot root:

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/
```

ME-SR02 raw snapshot files include:

```text
ASML_companyfacts.json
TSM_companyfacts.json
```

No `HO_companyfacts.json` file exists.

ME-SR02 provider errors record:

```text
HO is Euronext Thales and has no approved SEC CompanyFacts CIK in ME-SR02 bounded snapshot bundle
```

## HO Outcome

HO remains unresolved by ME-SR03.

Reason:

```text
blocked_missing_cached_source
```

The canonical universe ticker is:

```text
HO
```

No local cached SEC CompanyFacts snapshot was found under another approved ticker or identifier. ME-SR03 does not fabricate source data.

HO requires a future approved source-refresh or universe-identity decision before it can participate in cached-source dry-runs.

## ASML Outcome

ASML is resolved by ME-SR03.

The cached ASML payload contains the four canonical source fields as annual `20-F` source facts in the `us-gaap` namespace with `EUR` units:

```text
revenue -> RevenueFromContractWithCustomerExcludingAssessedTax
net_income -> NetIncomeLoss
operating_cash_flow -> NetCashProvidedByUsedInOperatingActivities
capital_expenditures -> PaymentsToAcquirePropertyPlantAndEquipment
```

ME-SR03 extends source mapping to preserve annual `20-F` / `20-F/A` filings and raw `EUR` units without currency conversion.

## TSM Outcome

TSM is resolved by ME-SR03.

The cached TSM payload contains the four canonical source fields as annual `20-F` source facts in the `ifrs-full` namespace with `USD` units:

```text
revenue -> Revenue
net_income -> ProfitLoss
operating_cash_flow -> CashFlowsFromUsedInOperatingActivities
capital_expenditures -> PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities
```

ME-SR03 adds explicit IFRS aliases for these source-only mappings.

No values are derived, estimated, combined, converted, or inferred.

## Commands Run

Narrow remediation check:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --tickers ASML,TSM,HO \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-sr03-coverage-check-20260622T110000Z \
  --generated-at 2026-06-22T11:00:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json
```

Narrow result:

```text
requested_count=3
executed_count=2
completed_count=2
blocked_count=1
missing_cached_source_count=1
```

Canonical rerun:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --canonical-ticker-universe \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-sr03-canonical-coverage-20260622T111500Z \
  --generated-at 2026-06-22T11:15:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine
```

Canonical result:

```text
requested_count=13
discovered_cached_source_count=12
executed_count=12
completed_count=12
blocked_count=1
failed_count=0
missing_cached_source_count=1
```

Blocked ticker:

```text
HO
```

## Generated Artifacts

Generated local artifact roots:

```text
artifacts/market_engine/me-sr03-coverage-check-20260622T110000Z/
artifacts/market_engine/me-sr03-canonical-coverage-20260622T111500Z/
```

Generated artifacts are local run evidence and are intentionally not committed.

## Boundary Confirmation

ME-SR03 did not introduce:

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

ME-SR03 resolves ASML and TSM by preserving already-cached foreign-issuer source facts through source mapping.

HO remains blocked because no approved local SEC CompanyFacts snapshot exists.

Recommended next sprint:

```text
ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision
```
