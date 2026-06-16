# ME-AR03 — Setup Detection input contract audit

## Status

COMPLETED BY ME-AR03

## Sprint

ME-AR03 — Extend Analysis Review contract for Setup Detection input

## Job family

ME-AR — Analysis Review jobs

## Purpose

This audit records the ME-AR03 documentation-only contract sprint.

ME-AR03 extends the Analysis Review contract so a future ME-AR04 implementation can consume Setup Detection output.

The sprint remains contract-only and does not introduce runtime behavior.

## Files added

Contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

Audit:

* `docs/market_engine/audits/me_ar03_setup_detection_input_contract_audit.md`

## Files updated before completion

Backlog:

* `docs/market_engine/backlog/market_engine_backlog.md`

Roadmap:

* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Source documents reviewed

ME-AR03 is based on the completed Analysis Review and Setup Detection documents:

* `docs/market_engine/analysis_review/me_ar01_analysis_review_contract.md`
* `docs/market_engine/analysis_review/me_ar02_analysis_review_implementation.md`
* `docs/market_engine/setup_detection/me_sd01_setup_detection_contract.md`
* `docs/market_engine/audits/me_sd02_setup_detection_implementation_audit.md`

## Contract extended

ME-AR03 keeps the existing Analysis Review output contract:

* `sec-companyfacts-analysis-review-v1`

ME-AR03 extends the approved Analysis Review input families with:

* ME-SD — Setup Detection

ME-AR03 extends the approved Analysis Review input contracts with:

* `sec-companyfacts-setup-detection-v1`

## Existing input contracts preserved

The existing Analysis Review input contracts remain approved:

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`

ME-AR03 does not remove or replace the existing ME-AR01 / ME-AR02 input model.

## Setup Detection input boundary

ME-AR03 defines Setup Detection output as an additional upstream Analysis Review input.

The Setup Detection input must be treated as:

* non-actionable;
* descriptive;
* provenance-bearing;
* missing-data-aware;
* numeric-zero-safe;
* human-review routable where needed.

The Setup Detection input must not be treated as:

* a recommendation signal;
* a score;
* a ranking;
* a portfolio instruction;
* a Decision Engine instruction;
* a delivery trigger.

## Required input alignment

ME-AR03 requires a future ME-AR04 implementation to validate alignment across:

* Fundamental Observations;
* Derived Observations;
* Setup Detection.

Alignment requirements include:

* ticker;
* CIK where available;
* provider name where available;
* source context format version where available;
* source context state where available;
* source refresh snapshot ID where available;
* upstream format versions;
* Setup Detection format version.

Mismatched input evidence must fail closed or emit controlled limitation output.

A future implementation must not silently merge mismatched evidence.

## Setup Detection-aware review categories defined

ME-AR03 approves these additional Analysis Review categories for future implementation:

* `SETUP_DETECTION_REVIEW`
* `SETUP_EVIDENCE_COMPLETENESS_REVIEW`
* `SETUP_LIMITATION_REVIEW`
* `SETUP_HUMAN_REVIEW_REQUIREMENT`

These categories extend the existing ME-AR01 categories and do not replace them.

## Setup Detection-aware review states defined

ME-AR03 approves these additional Analysis Review states for future implementation:

* `SETUP_DETECTED`
* `SETUP_PARTIALLY_DETECTED`
* `SETUP_NOT_DETECTED`
* `SETUP_CONFLICTED`
* `SETUP_BLOCKED_BY_MISSING_DATA`
* `SETUP_NOT_ASSESSED`
* `SETUP_REQUIRES_HUMAN_REVIEW`

These states are Analysis Review states only.

They do not create recommendation, score, ranking, allocation, execution, delivery, or Decision Engine authority.

## State mapping defined

ME-AR03 defines a deterministic mapping from Setup Detection states to Analysis Review states:

* `setup_detected` → `SETUP_DETECTED`
* `setup_partially_detected` → `SETUP_PARTIALLY_DETECTED`
* `setup_not_detected` → `SETUP_NOT_DETECTED`
* `setup_conflicted` → `SETUP_CONFLICTED`
* `setup_blocked_by_missing_data` → `SETUP_BLOCKED_BY_MISSING_DATA`
* `setup_not_assessed` → `SETUP_NOT_ASSESSED`

The contract also allows `SETUP_REQUIRES_HUMAN_REVIEW` when Setup Detection is partial, conflicted, blocked, or not assessed.

## Category mapping defined

ME-AR03 defines a deterministic mapping from Setup Detection categories to Analysis Review categories:

* `cash_generation_setup` → `SETUP_DETECTION_REVIEW`
* `fundamental_availability_setup` → `SETUP_EVIDENCE_COMPLETENESS_REVIEW`
* `profitability_evidence_setup` → `SETUP_DETECTION_REVIEW`
* `revenue_evidence_setup` → `SETUP_DETECTION_REVIEW`
* `balance_sheet_evidence_setup` → `SETUP_DETECTION_REVIEW`
* `data_limitation_setup` → `SETUP_LIMITATION_REVIEW`
* `not_assessed_setup` → `SETUP_HUMAN_REVIEW_REQUIREMENT`

The original Setup Detection category and state must remain preserved in references or evidence fields.

## Review item requirements

ME-AR03 defines future Setup Detection-aware Analysis Review item requirements.

A future ME-AR04 implementation should preserve:

* existing Analysis Review review item fields;
* Fundamental Observation references;
* Derived Observation references;
* Setup Detection references;
* setup categories;
* setup states;
* setup evidence;
* setup limitations;
* missing observations;
* non-recommendation boundary markers.

## Missing-data handling

ME-AR03 requires Analysis Review to preserve Setup Detection missingness.

Analysis Review must not:

* convert missing values to zero;
* omit missing-observation lists;
* treat missing setup evidence as negative evidence;
* treat missing setup evidence as a recommendation reason;
* silently proceed as if setup evidence were complete.

If Setup Detection is blocked by missing data, Analysis Review should emit a blocked or human-review state.

## Numeric-zero handling

ME-AR03 preserves the numeric-zero rule.

Numeric zero remains a valid observed or derived value.

A future ME-AR04 implementation must not treat numeric zero as missing.

If a setup is neutral because of a zero value, Analysis Review may describe the evidence as neutral or zero, but must not emit `HOLD` or imply a neutral recommendation.

## Conflict handling

ME-AR03 requires conflicted Setup Detection evidence to remain visible.

If Setup Detection emits `setup_conflicted`, Analysis Review should emit:

* `SETUP_CONFLICTED`; and
* a human-review routing state where appropriate.

Conflicted evidence must not be converted into a positive or negative recommendation.

## Backward compatibility

ME-AR03 requires ME-AR04 to preserve existing ME-AR02 behavior unless ME-AR04 explicitly scopes a new Setup Detection-aware builder.

Allowed ME-AR04 strategies include:

* a new Setup Detection-aware builder;
* an optional Setup Detection argument on the existing builder;
* a separate contract-preserving wrapper.

ME-AR04 must choose the strategy that best fits the active Market Engine implementation patterns.

Existing Analysis Review tests must not be broken without explicit justification.

## ME-AR04 requirements defined

ME-AR03 defines ME-AR04 implementation requirements.

ME-AR04 must:

* consume `sec-companyfacts-setup-detection-v1`;
* validate input alignment;
* preserve existing Fundamental Observation and Derived Observation behavior;
* emit Setup Detection-aware review items;
* preserve setup categories and states;
* preserve setup evidence and limitations;
* preserve missing observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-recommendation and non-actionable boundary markers;
* fail closed or emit controlled limitation output for unsupported Setup Detection input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes.

## Explicit boundaries preserved

ME-AR03 does not introduce:

* Python implementation;
* tests;
* runtime behavior;
* provider calls;
* live SEC calls;
* EDGAR calls;
* yfinance calls;
* data writes;
* generated artifacts;
* Source Refresh changes;
* Source Context changes;
* Fundamental Observation changes;
* Derived Observation changes;
* Setup Detection runtime changes;
* Recommendation Review behavior;
* Portfolio Review behavior;
* Delivery behavior;
* Telegram behavior;
* reporting behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD;
* target price;
* rating;
* score;
* ranking;
* conviction;
* urgency;
* tradeability;
* allocation;
* position sizing;
* execution advice;
* order generation;
* watchlist mutation;
* portfolio mutation.

## Validation to perform before commit

Required validation commands:

```bash
git diff --check
grep -n "ME-AR03\|ME-AR04\|Status: RECOMMENDED NEXT" docs/market_engine/backlog/market_engine_backlog.md
grep -n "ME-AR03\|ME-AR04\|Status: RECOMMENDED NEXT" docs/market_engine/roadmap/market_engine_roadmap.md
git status --short
git diff --stat
```

Expected validation result:

* whitespace check passes;
* ME-AR03 is marked completed in backlog and roadmap;
* ME-AR04 is the only recommended next sprint in backlog and roadmap;
* changed files are documentation-only;
* no `src/`, `tests/`, or `data/` files are changed.

## Expected next sprint

After ME-AR03, the expected next sprint is:

* `ME-AR04 — Implement Analysis Review consumption of Setup Detection`

ME-AR04 is an implementation sprint and must be executed only after this contract is merged.

## Conclusion

ME-AR03 extends Analysis Review with a Setup Detection input contract while preserving Analysis Review as descriptive, provenance-preserving, missing-data-aware, numeric-zero-safe, non-recommendation, and non-actionable.

The sprint prepares ME-AR04 implementation without changing runtime behavior.
