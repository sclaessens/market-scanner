# ME-SR03 Audit - Canonical-Universe Cached-Source Coverage Blockers

Owner roles: Governance Auditor / Technical Architect / QA Lead

Status: SOURCE COVERAGE AUDIT CREATED BY ME-SR03

## Audit Target

ME-SR03 investigates and remediates source coverage blockers from ME-RUN19 for:

```text
HO
ASML
TSM
```

## Files Changed

```text
src/market_engine/source_intake/sec_companyfacts_fields.py
tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py
docs/market_engine/source_contracts/sec_companyfacts_field_mapping_contract.md
docs/market_engine/source_data/me_sr03_canonical_universe_cached_source_coverage_blockers.md
docs/market_engine/audits/me_sr03_canonical_universe_cached_source_coverage_blockers_audit.md
docs/market_engine/backlog/me_sr03_canonical_universe_cached_source_coverage_blockers_backlog_entry.md
docs/market_engine/roadmap/me_sr03_canonical_universe_cached_source_coverage_blockers_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Python Code Changed

Yes.

Changed module:

```text
src/market_engine/source_intake/sec_companyfacts_fields.py
```

Change type:

* source mapping only;
* annual `20-F` and `20-F/A` form support;
* raw `EUR` unit preservation;
* explicit `ifrs-full` aliases for TSM-like cached source facts;
* no currency conversion;
* no derived values;
* no analysis, recommendation, portfolio, delivery, or Decision Engine behavior.

## Tests Changed

Yes.

Changed test file:

```text
tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py
```

Coverage added:

* foreign-issuer US GAAP `20-F` source facts in `EUR`;
* IFRS `20-F` source facts in `USD`;
* unit, taxonomy namespace, filing form, tag selection and raw value preservation.

## HO Audit

HO remains blocked.

No `HO_companyfacts.json` snapshot exists under the ME-SR02 raw snapshot directory.

ME-SR02 `provider_errors.csv` records HO as unsupported because it is Euronext Thales and has no approved SEC CompanyFacts CIK in the bounded snapshot bundle.

ME-SR03 did not fabricate a source snapshot.

## ASML Audit

ASML is resolved.

The cached ASML source payload contained annual `20-F` `us-gaap` facts in `EUR` for:

```text
RevenueFromContractWithCustomerExcludingAssessedTax
NetIncomeLoss
NetCashProvidedByUsedInOperatingActivities
PaymentsToAcquirePropertyPlantAndEquipment
```

After ME-SR03 mapping, ASML completed the narrow dry-run and canonical rerun.

## TSM Audit

TSM is resolved.

The cached TSM source payload contained annual `20-F` `ifrs-full` facts in `USD` for:

```text
Revenue
ProfitLoss
CashFlowsFromUsedInOperatingActivities
PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities
```

After ME-SR03 mapping, TSM completed the narrow dry-run and canonical rerun.

## Commands Run

Narrow check:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command --tickers ASML,TSM,HO --source-snapshot-root data/market_engine/source_snapshots --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json --batch-id me-sr03-coverage-check-20260622T110000Z --generated-at 2026-06-22T11:00:00Z --write-local-artifacts --artifact-output-root artifacts/market_engine --emit-json
```

Canonical rerun:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command --canonical-ticker-universe --source-snapshot-root data/market_engine/source_snapshots --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json --batch-id me-sr03-canonical-coverage-20260622T111500Z --generated-at 2026-06-22T11:15:00Z --write-local-artifacts --artifact-output-root artifacts/market_engine
```

Tests:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py -q
```

Result:

```text
16 passed
```

Full validation:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```

Result:

```text
307 passed
```

## Side-Effect Audit

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

## Audit Conclusion

ME-SR03 resolves the ASML and TSM source coverage blockers using only existing cached source data and explicit source mapping extensions.

HO remains unresolved and requires a future approved source identity or exclusion decision.

Recommended next sprint:

```text
ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision
```
