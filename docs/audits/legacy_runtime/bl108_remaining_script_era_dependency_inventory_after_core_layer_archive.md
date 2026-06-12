# BL108 Remaining Script-Era Dependency Inventory After Core-Layer Archive

## Sprint

BL108 — Remaining script-era dependency inventory after core-layer archive

## Scope

Review-only inventory after BL106/BL107 archived the fail-closed core-layer module cluster.

In scope:

- remaining active `scripts/` imports and references;
- test-only dependencies on script-era modules;
- archive-readiness classification for the next cleanup cluster;
- documentation-only audit output.

Out of scope:

- archiving files;
- moving files;
- modifying Python code;
- changing tests;
- running live providers;
- yfinance calls;
- SEC/EDGAR calls;
- credential access;
- production data writes;
- report generation;
- Telegram delivery;
- portfolio/watchlist state mutation;
- Decision Engine authority changes;
- scanner/provider, portfolio intelligence, scan validation, or portfolio runtime changes.

## Repository baseline

Baseline commit reviewed:

- `d0a1ed0d5bc94ba3c21a0761de83e07edfee0375`
- Merge commit for PR #285, which completed BL106/BL107.

BL106/BL107 result confirmed in the prior audit/backlog state:

- the four fail-closed core-layer modules were moved to `archive/legacy_runtime/scripts/core/`;
- no canonical runtime under `src/market_scanner/` was changed;
- remaining active `scripts.core` imports reference other modules outside the BL107 archive cluster.

## Commands represented by this review

The equivalent local review commands for this sprint are:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.|import scripts\.)" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

```bash
grep -RInE \
  "scripts\.core|scripts\.portfolio|scripts\.scanner|scripts\.reporting|scripts\.telegram|scripts\.fundamentals|scripts\.data_sources" \
  src tests .github docs scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

This ChatGPT-executed BL108 used GitHub repository inspection instead of local pytest/runtime execution. No production command, provider command, runtime pipeline, report, Telegram command, or data write was executed.

## Active imports found

### 1. Entry quality backfill

Active test import:

```text
tests/core/test_build_entry_quality_backfill.py: from scripts.core.build_entry_quality_backfill import (...)
```

Classification:

- `ACTIVE_TEST_DEPENDENCY`
- `MANUAL_OR_HISTORICAL_BACKFILL_RISK`
- `YFINANCE_RISK_PRESENT_IN_MODULE`

Evidence:

- the test imports `EntryQualityConfig`, `calculate_entry_quality`, `compute_point_in_time_indicators`, `load_scans`, and `validate_output` from `scripts.core.build_entry_quality_backfill`;
- the script-era module imports `yfinance` behind an optional import and contains historical backfill behavior;
- tests use synthetic/pandas data paths, but the module itself still contains provider-facing code paths.

Decision:

- Not archive-ready.
- Candidate for BL109 test decoupling or canonical extraction review before any archive decision.

### 2. Context backfill

Active test import:

```text
tests/core/test_build_context_backfill.py: from scripts.core import build_context_backfill as b
```

Classification:

- `ACTIVE_TEST_DEPENDENCY`
- `MANUAL_OR_HISTORICAL_BACKFILL_RISK`
- `YFINANCE_RISK_LIKELY_OR_REQUIRES_CONFIRMATION`

Evidence:

- active tests import the script-era module directly;
- tests exercise schema, governance-clean output columns, point-in-time behavior, missing price data behavior, output writes to temporary files, and monkeypatched download behavior;
- because the imported script-era module appears to represent backfill behavior, archive-readiness requires a separate focused review.

Decision:

- Not archive-ready.
- Candidate for BL109 alongside entry quality backfill if the cluster is scoped as historical backfill test decoupling.

### 3. Decision Engine

Active test import:

```text
tests/core/test_decision_engine.py: from scripts.core import decision_engine
```

Classification:

- `ACTIVE_TEST_DEPENDENCY`
- `HIGH_GOVERNANCE_RISK`
- `DECISION_ENGINE_AUTHORITY_BOUNDARY`

Evidence:

- tests import the script-era Decision Engine directly;
- tests monkeypatch input, output, and log paths;
- tests validate output schema, no hidden filtering, source provenance, rationale completeness, no watchlist/reporting/portfolio authority leakage, and allowed decision values;
- the module is in the explicit out-of-scope governance area.

Decision:

- Not archive-ready.
- Do not include in a generic archive sprint.
- Requires a separate Decision Engine authority migration/review sprint if it is ever touched.

### 4. Portfolio intelligence

Active test imports:

```text
tests/core/test_build_portfolio_intelligence.py: from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py: from scripts.core import build_portfolio_intelligence
```

Classification:

- `ACTIVE_TEST_DEPENDENCY`
- `HIGH_GOVERNANCE_RISK`
- `PORTFOLIO_INTELLIGENCE_BOUNDARY`

Evidence:

- active tests import the script-era portfolio intelligence module directly;
- tests monkeypatch processed, portfolio, metadata, output, and log paths;
- tests validate portfolio metadata preservation and absence of forbidden investment semantics;
- this area is explicitly out of scope for BL108 archive action.

Decision:

- Not archive-ready.
- Requires a dedicated portfolio intelligence boundary review before any decoupling/archive path.

### 5. Portfolio build/source contract

Active test import:

```text
tests/portfolio/test_portfolio_source_contract.py: from scripts.portfolio import build_portfolio
```

Classification:

- `ACTIVE_TEST_DEPENDENCY`
- `PORTFOLIO_STATE_SOURCE_CONTRACT`
- `PRODUCTION_DATA_READ_RISK`

Evidence:

- the test reads active portfolio CSV files under `data/portfolio/portfolio_transactions.csv` and `data/portfolio/portfolio_positions.csv`;
- the test monkeypatches output to a temporary file, but still validates against active portfolio source files;
- portfolio/watchlist state is explicitly out of scope for blind cleanup.

Decision:

- Not archive-ready.
- Requires separate portfolio source-contract review and possibly fixture isolation before any archive decision.

### 6. Script-era trade command parser

Relevant script-era file observed:

```text
scripts/portfolio/parse_trade_commands.py
```

Classification:

- `ACTIVE_SCRIPT_TREE_REMAINDER`
- `MANUAL_TRADE_COMMAND_RISK`
- `PORTFOLIO_WRITE_RISK`

Evidence:

- module imports `log_trade` from `scripts.portfolio.portfolio_manager`;
- parser supports `BUY` and `SELL` command formats;
- `parse_trade_command()` calls `log_trade(...)`;
- direct manual execution prompts for trade commands.

Decision:

- Not archive-ready.
- High priority for a fail-closed/manual-entrypoint review before archive consideration.
- Must not be touched in a broad cleanup sprint because it can affect portfolio transaction state.

## Documentation and archive-only references

Search results also include many references under:

- `docs/audits/legacy_runtime/`
- `docs/archive/`
- `docs/legacy/`
- `docs/resets/`
- `archive/legacy_runtime/`

Classification:

- `DOCUMENTATION_ONLY_REFERENCE`
- `ARCHIVED_HISTORICAL_REFERENCE`

Decision:

- These do not block cleanup by themselves.
- They should not be rewritten unless a documentation rationalization sprint explicitly requires it.

## BL108 decision

Decision: `NO_ARCHIVE_SPRINT_APPROVED`

Rationale:

- active `scripts/` dependencies remain;
- multiple dependencies are test imports, not documentation-only references;
- several remaining modules sit in explicitly high-risk governance areas: Decision Engine, portfolio intelligence, portfolio state, and manual trade command handling;
- historical backfill modules appear to retain provider/backfill behavior and should be decoupled or fail-closed before archive consideration;
- no tests were changed and no runtime behavior was changed in BL108.

## Recommended next sprint

### BL109 — Decouple historical backfill tests from script-era modules

Category: Legacy Runtime Cleanup / Test Decoupling Review and Implementation

Scope proposal:

- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_context_backfill.py`
- tests that directly import those modules

Rationale:

- these are the smallest coherent remaining core-layer-ish historical backfill dependencies;
- they are lower authority than Decision Engine and portfolio intelligence;
- they likely need test decoupling and manual/provider-risk review before archive readiness.

Strict BL109 guardrails:

- no live yfinance calls;
- no provider calls;
- no production data writes;
- no report generation;
- no Telegram;
- no Decision Engine authority changes;
- no portfolio/watchlist state changes;
- no archiving in BL109 unless a separate review gate explicitly approves it.

## Deferred high-risk clusters

Do not bundle these with BL109:

1. Decision Engine script-era dependency
   - `scripts/core/decision_engine.py`
   - `tests/core/test_decision_engine.py`

2. Portfolio intelligence script-era dependency
   - `scripts/core/build_portfolio_intelligence.py`
   - `tests/core/test_build_portfolio_intelligence.py`
   - `tests/portfolio/test_portfolio_source_contract.py`

3. Portfolio source/command handling
   - `scripts/portfolio/build_portfolio.py`
   - `scripts/portfolio/parse_trade_commands.py`
   - `scripts/portfolio/portfolio_manager.py`

Each cluster requires a separate review-first sprint.

## Validation status

No pytest suite was executed by ChatGPT during BL108 because this was a GitHub documentation/review-only sprint executed through repository inspection, not a local runtime session.

Validation remains inherited from BL107 for baseline health:

- BL107 post-archive full suite: `610 passed`

BL108 changed documentation only.

## Final BL108 result

- Archive approval: `NO`
- Next action: BL109 historical backfill decoupling review/implementation
- Code changed: `NO`
- Runtime behavior changed: `NO`
- Production data changed: `NO`
- Decision Engine authority changed: `NO`
