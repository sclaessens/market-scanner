# ME-SA02 - Bounded automated cached-source acquisition job implementation

Date: 2026-06-26

## Purpose

ME-SA02 implements the first bounded automated cached-source acquisition job according to the ME-SA01 contract.

The implementation is local, deterministic, non-production, and limited to the first approved source family:

```text
company_profile
```

## Runtime

Implemented module:

```text
src/market_engine/source_acquisition/automated_cached_source_acquisition.py
```

The job accepts `market-engine-automated-cached-source-acquisition-request-v1` request payloads and writes `market-engine-automated-cached-source-acquisition-result-v1` result payloads.

The first implementation supports:

* explicit bounded ticker lists;
* ticker validation;
* approved source-family validation;
* dry-run and local non-production run modes;
* deterministic fake `company_profile` adapter output;
* explicit destination roots;
* per-ticker snapshot package writing;
* payload hash and size recording;
* provenance recording;
* retrieval timestamp and source timestamp handling;
* freshness state recording;
* fail-closed request validation;
* explicit unsupported, blocked, and provider-error entry statuses.

## Snapshot Package Shape

Each completed entry writes:

```text
<destination_root>/<TICKER>/company_profile/manifest.json
<destination_root>/<TICKER>/company_profile/company_profile.json
```

The manifest uses the existing cached-source snapshot acquisition manifest format:

```text
market-engine-cached-source-snapshot-acquisition-manifest-v1
```

The package includes ticker, snapshot ID, batch ID, source family, source identity, retrieval timestamp, payload hash, payload size, validation status, staleness status, and usable flag fields required by the existing staging validator.

## Downstream Compatibility

ME-SA02 verifies that the completed `company_profile` package shape is accepted by the existing ME-SR10 staging validator.

The implementation does not claim that the current `cached_source_snapshot` dry-run can semantically consume `company_profile` payloads end-to-end. If downstream consumption blocks, ME-RUN26 must record that as an explicit blocked reason instead of bypassing validation.

## Safety Boundaries

ME-SA02 does not perform real provider calls, network calls, yfinance calls, SEC/EDGAR calls, live data retrieval, production writes, Telegram sends, portfolio writes, watchlist writes, broker or execution actions, Decision Engine changes, Recommendation Review changes, Portfolio Review changes, Delivery changes, or action/allocation authority.

ME-SA02 does not introduce BUY / SELL / HOLD, target price, allocation, position sizing, ranking, urgency, conviction, or tradeability authority.

## Validation

Local validation:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_acquisition/test_automated_cached_source_acquisition.py -q
12 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py -q
19 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
492 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1159 passed

git diff --check
passed
```

## Next Sprint

```text
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
```

ME-RUN26 should execute the ME-SA02 acquisition job output through the existing staging validation and local dry-run route, recording pass/block evidence without introducing provider calls, delivery, production writes, or downstream decision authority.
