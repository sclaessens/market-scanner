# ME-SR05 Audit - Professional Swing Universe source-support classification

Sprint: ME-SR05 - Classify source support for Professional Swing Universe

Job family: ME-SR - Source Refresh / Source Coverage

Status: Completed

## Goal

Implement a deterministic source-support classification layer for the editable Professional Swing Universe using only approved local cached SEC CompanyFacts artifacts.

## Files Inspected

* `src/market_engine/ticker_universe/professional_swing.py`
* `src/market_engine/source_refresh/sec_companyfacts_snapshots.py`
* `src/market_engine/source_intake/sec_companyfacts_fields.py`
* `src/market_engine/run/cached_source_batch_execution.py`
* `docs/market_engine/ticker_universe/me_uni04_editable_professional_swing_universe_contract.md`
* `docs/market_engine/ticker_universe/me_uni06_editable_universe_loader_validation_implementation.md`
* `docs/market_engine/ticker_universe/me_uni07_editable_universe_runtime_input_implementation.md`
* `docs/market_engine/ticker_universe/me_uni08_professional_swing_universe_cli_flag.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Files Changed

Python:

* `src/market_engine/source_support/__init__.py`
* `src/market_engine/source_support/professional_swing.py`

Tests:

* `tests/market_engine/source_support/test_professional_swing_source_support.py`

Documentation:

* `docs/market_engine/source_support/me_sr05_professional_swing_universe_source_support_classification.md`
* `docs/market_engine/audits/me_sr05_professional_swing_universe_source_support_classification_audit.md`
* `docs/market_engine/backlog/me_sr05_professional_swing_universe_source_support_classification_backlog_entry.md`
* `docs/market_engine/roadmap/me_sr05_professional_swing_universe_source_support_classification_roadmap_entry.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Implementation Summary

ME-SR05 adds `classify_professional_swing_universe_source_support(...)`.

The classifier:

* loads the validated Professional Swing Universe;
* discovers local SEC CompanyFacts snapshot envelopes under `data/market_engine/source_snapshots`;
* reads local provider error records when present;
* maps approved source fields using the existing SEC CompanyFacts mapper;
* emits `market-engine-professional-swing-source-support-v1`;
* preserves source artifact references and universe row references;
* keeps missing, malformed, unsupported, ambiguous, manual-review-only, and excluded states explicit.

## Statuses Implemented

* `supported_cached`
* `missing_snapshot`
* `unsupported_sec_companyfacts`
* `missing_required_source_field`
* `malformed_or_unreadable_source_artifact`
* `ambiguous_identity`
* `manual_review_only`
* `excluded`

No source staleness status was implemented because no current Professional Swing source-support contract defines a freshness threshold.

## Requirements Satisfied

* Local cached artifacts only.
* Deterministic classification.
* Provenance preservation.
* Explicit missing-source and missing-field states.
* Malformed artifact handling.
* Numeric-zero safety.
* Fail-closed invalid-universe behavior.
* No source mutation.

## Tests Added

The new source-support tests cover:

* fully supported ticker;
* unsupported ticker through local provider error;
* missing source snapshot;
* missing required source field;
* malformed source artifact;
* numeric zero preservation;
* manual-review-only row;
* excluded row;
* ambiguous identity;
* deterministic output ordering;
* invalid universe fail-closed behavior;
* absence of provider/network/legacy dependencies.

## Validation

Validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_support -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff --check
```

Results are recorded in the final sprint report.

## Boundaries Preserved

ME-SR05 did not introduce provider calls, SEC or EDGAR live access, yfinance usage, source refresh, synthetic source facts, cached-source execution, production writes, Telegram/email delivery, reporting output, portfolio/watchlist mutation, Recommendation Review changes, Portfolio Review changes, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, ranking, scoring, urgency, conviction, tradeability, position sizing, order generation, or execution behavior.

## Next Sprint

ME-RUN20 - Execute clean supported-universe cached-source scan.
