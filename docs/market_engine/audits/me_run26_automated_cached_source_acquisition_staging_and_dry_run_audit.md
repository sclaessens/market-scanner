# ME-RUN26 - Automated cached-source acquisition through staging and local dry-run audit

Sprint ID: ME-RUN26

Status: COMPLETED WITH BLOCKED OUTCOME BY ME-RUN26

Date: 2026-06-26

Branch:

```text
me-run26-run-automated-cached-source-acquisition-through-staging-and-local-dry-run
```

Commit:

```text
ME-RUN26 run automated cached-source acquisition through staging and local dry-run
```

## Scope

ME-RUN26 executed the ME-SA02 automated cached-source acquisition job through the existing cached-source staging validator and then attempted the existing `cached_source_snapshot` local dry-run path.

Input tickers:

```text
NVDA
AMD
ASML
```

Input source family:

```text
company_profile
```

ME-RUN26 is a run/audit validation sprint. It does not broaden source-family semantics, provider access, dry-run consumption behavior, reporting behavior, portfolio/watchlist behavior, or Decision Engine behavior.

## Commands

### Wrapper Invocation

```text
ME_RUN26_TIMESTAMP=20260626T120200Z ME_RUN26_GENERATED_AT=2026-06-26T12:02:00Z ME_RUN26_RUN_ID=me-run26-automated-acquisition-20260626T120200Z scripts/market_engine/me_run26_run_automated_cached_source_acquisition.sh
```

### Acquisition Command

The wrapper invokes the existing ME-SA02 function entrypoint:

```text
market_engine.source_acquisition.automated_cached_source_acquisition.run_automated_cached_source_acquisition
```

The request forced:

```text
tickers: NVDA, AMD, ASML
source_families: company_profile
run_mode: dry_run
allow_provider_calls: false
allow_network: false
allow_production_writes: false
allow_telegram_send: false
allow_portfolio_writes: false
allow_watchlist_writes: false
allow_broker_actions: false
```

### Staging Validation Command

The wrapper invokes the existing staging validator function:

```text
market_engine.source_refresh.cached_source_snapshot_staging_validator.build_cached_source_snapshot_staging_validation
```

with:

```text
staging_root: artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition
tickers: NVDA, AMD, ASML
validated_at: 2026-06-26T12:02:00Z
output_json: artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/staging_validation.json
```

Note: an earlier CLI attempt against `market_engine.source_refresh.cached_source_snapshot_staging_validator_command` exposed an existing argparse `append` default issue for repeated `--ticker` arguments. The final validated run used the existing validator function directly and did not bypass validation logic.

### cached_source_snapshot Local Dry-Run Commands

The wrapper attempted the existing dry-run command once per ticker:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/NVDA/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition --dry-run-id me-run26-nvda-company-profile-dry-run --generated-at 2026-06-26T12:02:00Z --artifact-output-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/dry_runs --artifact-created-at 2026-06-26T12:02:00Z --write-local-artifact --compact
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/AMD/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition --dry-run-id me-run26-amd-company-profile-dry-run --generated-at 2026-06-26T12:02:00Z --artifact-output-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/dry_runs --artifact-created-at 2026-06-26T12:02:00Z --write-local-artifact --compact
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/ASML/company_profile/company_profile.json --source-snapshot-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition --dry-run-id me-run26-asml-company-profile-dry-run --generated-at 2026-06-26T12:02:00Z --artifact-output-root artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/dry_runs --artifact-created-at 2026-06-26T12:02:00Z --write-local-artifact --compact
```

## Artifact Root

```text
artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z
```

Generated artifacts are local run evidence and are not committed.

## Generated Package Paths

| Ticker | Snapshot path | Manifest path | Payload path | Acquisition status |
|---|---|---|---|---|
| NVDA | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/NVDA/company_profile` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/NVDA/company_profile/manifest.json` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/NVDA/company_profile/company_profile.json` | completed |
| AMD | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/AMD/company_profile` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/AMD/company_profile/manifest.json` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/AMD/company_profile/company_profile.json` | completed |
| ASML | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/ASML/company_profile` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/ASML/company_profile/manifest.json` | `artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z/acquisition/ASML/company_profile/company_profile.json` | completed |

## Acquisition Result

```text
completed_count: 3
blocked_count: 0
rejected_count: 0
provider_error_count: 0
unsupported_count: 0
network_used: false
provider_calls_performed: false
production_write_performed: false
```

Acquisition result: PASS.

## Staging Validation Result

```text
total_inspected_entries: 3
accepted_entries: 3
rejected_entries: 0
all_entries_accepted: true
```

Per ticker:

| Ticker | Source family | Staging status | Issues |
|---|---|---|---|
| NVDA | company_profile | accepted | none |
| AMD | company_profile | accepted | none |
| ASML | company_profile | accepted | none |

Staging validation result: PASS.

## cached_source_snapshot Local Dry-Run Result

Per ticker:

| Ticker | Result | Exit code | Exact blocker |
|---|---:|---:|---|
| NVDA | blocked | 2 | `cannot build SEC CompanyFacts Source Context from snapshot: SEC CompanyFacts snapshot metadata is missing` |
| AMD | blocked | 2 | `cannot build SEC CompanyFacts Source Context from snapshot: SEC CompanyFacts snapshot metadata is missing` |
| ASML | blocked | 2 | `cannot build SEC CompanyFacts Source Context from snapshot: SEC CompanyFacts snapshot metadata is missing` |

The existing `cached_source_snapshot` local dry-run path expects SEC CompanyFacts snapshot metadata and does not currently consume ME-SA02 `company_profile` payloads.

## Overall Result

```text
BLOCKED
```

Acquisition passed and staging validation passed. The existing `cached_source_snapshot` local dry-run path blocked for every ticker because `company_profile` packages are structurally valid cached-source packages but are not semantically consumable by the current SEC CompanyFacts-only dry-run path.

## Exact Blocker

```text
company_profile cached-source packages are accepted by staging validation, but the existing cached_source_snapshot local dry-run path attempts to build SEC CompanyFacts Source Context and blocks with: SEC CompanyFacts snapshot metadata is missing.
```

## Safety Confirmation

ME-RUN26 performed no production writes, Telegram sends, portfolio/watchlist writes, broker/execution actions, provider calls, network calls, yfinance calls, SEC/EDGAR calls, BUY/SELL/HOLD behavior, target price behavior, allocation behavior, position sizing behavior, ranking behavior, urgency behavior, conviction behavior, or tradeability authority.

## Tests and Checks

Commands run:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_acquisition/test_automated_cached_source_acquisition.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py -q
```

Results:

```text
12 passed
19 passed
99 passed - tests/market_engine/run
492 passed - tests/market_engine
1159 passed - full pytest
git diff --check passed
```

## Governance Grep

Command:

```text
grep -RInE "BUY|SELL|HOLD|target price|allocation|position sizing|ranking|urgency|conviction|tradeability|broker|telegram|yfinance|SEC|EDGAR|watchlist|portfolio" docs/market_engine/audits/me_run26_automated_cached_source_acquisition_staging_and_dry_run_audit.md docs/market_engine/backlog/me_run26_automated_cached_source_acquisition_staging_and_dry_run_backlog_entry.md docs/market_engine/roadmap/me_run26_automated_cached_source_acquisition_staging_and_dry_run_roadmap_entry.md scripts/market_engine/me_run26_run_automated_cached_source_acquisition.sh
```

The ME-RUN26 governance grep was run against the new audit, backlog entry, roadmap entry, and wrapper script.

Allowed matches:

* safety confirmations for forbidden concepts;
* `SEC` / `SEC CompanyFacts` in the exact blocker;
* `Telegram`, `portfolio`, `watchlist`, `broker`, `yfinance`, `SEC`, and `EDGAR` in safety confirmation text.

No new action, trading, allocation, ranking, conviction, urgency, tradeability, broker, delivery, provider, or network behavior is introduced.

## Next Active Sprint

```text
ME-SA03 - Define company_profile cached-source dry-run consumption compatibility contract
```

ME-SA03 is the governance-correct next sprint because the current blocker is a source-family consumption contract gap, not a wrapper or staging-validator defect.
