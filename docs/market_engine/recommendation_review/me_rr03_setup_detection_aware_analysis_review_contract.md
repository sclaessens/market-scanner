# ME-RR03 — Recommendation Review contract for Setup Detection-aware Analysis Review

## Status

COMPLETED BY ME-RR03

## Sprint

ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

## Job family

ME-RR — Recommendation Review jobs

## Purpose

ME-RR03 extends the Recommendation Review contract so a future Recommendation Review implementation can consume Setup Detection-aware Analysis Review output.

The existing Recommendation Review input contract remains:

```text
sec-companyfacts-analysis-review-v1
```

ME-RR03 does not introduce `sec-companyfacts-analysis-review-v2`.

ME-RR03 does not introduce `sec-companyfacts-recommendation-review-v2`.

Recommendation Review remains downstream of Analysis Review.

Recommendation Review remains non-actionable, human-review-only, source-grounded, missing-data-aware, numeric-zero-safe, and provenance-preserving.

## Governing upstream contracts

Existing Recommendation Review contract:

* `docs/market_engine/recommendation_review/me_rr01_recommendation_review_contract.md`

Existing Recommendation Review implementation audit:

* `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`

Setup Detection-aware Analysis Review contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

Setup Detection-aware Analysis Review implementation documentation:

* `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`

Setup Detection-aware Analysis Review implementation audit:

* `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`

## Architectural position

The approved architectural chain remains:

```text
Source Refresh / raw snapshots
→ Source Context
→ Fundamental Observations
→ Derived Observations
→ Setup Detection
→ Analysis Review
→ Recommendation Review
→ Portfolio Review
→ Decision Engine handoff / action authority
→ Delivery / reporting
```

ME-RR03 does not move Recommendation Review.

Recommendation Review may consume only validated Analysis Review output.

Recommendation Review must not consume Setup Detection output directly.

Recommendation Review must not bypass Analysis Review.

Recommendation Review must not reinterpret raw source snapshots, Fundamental Observations, Derived Observations, or Setup Detection runtime output directly.

## Approved input contract

The approved input contract remains:

```text
sec-companyfacts-analysis-review-v1
```

A future ME-RR04 implementation may consume Setup Detection-aware Analysis Review only when the artifact still declares:

```text
sec-companyfacts-analysis-review-v1
```

This means Setup Detection-aware Analysis Review is an allowed extension of the existing Analysis Review contract, not a new input contract family for Recommendation Review.

If the Analysis Review artifact declares a missing, malformed, unknown, or unsupported contract, Recommendation Review must fail closed or emit controlled non-actionable invalid-input output according to the existing Recommendation Review contract.

## Approved Setup Detection-aware input interpretation

Recommendation Review may interpret Setup Detection-aware Analysis Review items only as Analysis Review evidence.

Recommendation Review may inspect Analysis Review review items that contain:

* Setup Detection-aware Analysis Review categories;
* Setup Detection-aware Analysis Review states;
* Setup Detection references;
* setup categories;
* setup states;
* setup evidence;
* setup limitations;
* missing setup observations;
* source observation references;
* derived observation references;
* non-actionable boundary markers.

Recommendation Review must preserve the distinction between:

* direct Analysis Review evidence;
* Setup Detection-aware Analysis Review evidence;
* upstream Setup Detection provenance embedded in Analysis Review.

Recommendation Review must not treat Setup Detection-aware Analysis Review as:

* portfolio advice;
* action advice;
* execution advice;
* Decision Engine input;
* ranking input;
* score input;
* conviction input;
* urgency input;
* tradeability input;
* allocation input;
* position-sizing input.

## Setup-aware provenance requirements

A future Recommendation Review artifact must preserve upstream setup-aware provenance when available.

Minimum setup-aware provenance fields or equivalent structured references should preserve:

* input Analysis Review contract;
* input Analysis Review run identifier, when available;
* input Analysis Review artifact path or identifier, when available;
* Setup Detection format version, when available from Analysis Review;
* Setup Detection run identifier, when available from Analysis Review;
* Setup Detection references embedded in Analysis Review;
* setup category;
* setup state;
* setup message;
* setup evidence;
* setup limitations;
* missing setup observations;
* source observation references;
* derived observation references;
* non-actionable Setup Detection boundary marker;
* non-recommendation Analysis Review boundary marker.

Recommendation Review must not flatten setup-aware evidence in a way that loses traceability.

Recommendation Review must not hide Setup Detection limitations.

Recommendation Review must not silently drop Setup Detection warnings, missing observations, conflicts, partial states, not-assessed states, or boundary markers.

## Review-state routing from Setup Detection-aware Analysis Review

Recommendation Review review states remain non-actionable.

ME-RR03 does not introduce trade states.

A future ME-RR04 implementation may map Setup Detection-aware Analysis Review states into existing Recommendation Review routing states as follows.

| Analysis Review setup-aware state | Recommendation Review routing state |
| --- | --- |
| `SETUP_DETECTED` | `human_review_required` |
| `SETUP_PARTIALLY_DETECTED` | `human_review_required` |
| `SETUP_NOT_DETECTED` | `insufficient_evidence` |
| `SETUP_CONFLICTED` | `human_review_required` |
| `SETUP_BLOCKED_BY_MISSING_DATA` | `blocked_by_missing_data` |
| `SETUP_NOT_ASSESSED` | `insufficient_evidence` |
| `SETUP_REQUIRES_HUMAN_REVIEW` | `human_review_required` |

These mappings are human-review routing semantics only.

They must not imply:

* buy;
* sell;
* hold;
* attractiveness;
* quality ranking;
* score;
* conviction;
* urgency;
* tradeability;
* allocation;
* execution;
* delivery.

## Review-category routing from Setup Detection-aware Analysis Review

A future ME-RR04 implementation may map Setup Detection-aware Analysis Review categories into Recommendation Review categories as follows.

| Analysis Review setup-aware category | Recommendation Review category |
| --- | --- |
| `SETUP_DETECTION_REVIEW` | `analysis_supportive_but_not_actionable` |
| `SETUP_EVIDENCE_COMPLETENESS_REVIEW` | `analysis_blocked_by_missing_data` |
| `SETUP_LIMITATION_REVIEW` | `analysis_mixed_or_conflicted` |
| `SETUP_HUMAN_REVIEW_REQUIREMENT` | `analysis_mixed_or_conflicted` |

When setup evidence is complete and detected, Recommendation Review may route the ticker to human review.

When setup evidence is partial, conflicted, blocked, or not assessed, Recommendation Review must preserve that uncertainty in the review candidate.

Recommendation Review must not upgrade a detected setup into an action recommendation.

Recommendation Review must not downgrade a not-detected, blocked, partial, conflicted, or not-assessed setup into a negative investment recommendation.

## Detected setup handling

If Analysis Review contains `SETUP_DETECTED`, Recommendation Review may emit:

* review state: `human_review_required`;
* review category: `analysis_supportive_but_not_actionable`.

Allowed language:

* “Setup-aware Analysis Review supports human review.”
* “Setup evidence is present in Analysis Review and remains non-actionable.”
* “Further review requires downstream Portfolio Review and Decision Engine boundaries.”

Forbidden interpretation:

* buy;
* enter position;
* increase allocation;
* rank higher;
* assign conviction;
* trigger delivery;
* trigger Decision Engine action.

## Partial setup handling

If Analysis Review contains `SETUP_PARTIALLY_DETECTED`, Recommendation Review may emit:

* review state: `human_review_required`;
* review category: `analysis_mixed_or_conflicted` or `analysis_blocked_by_missing_data`, depending on missingness.

Recommendation Review must preserve:

* partial setup state;
* missing setup observations;
* incomplete evidence;
* source and derived references;
* setup limitations.

A partial setup must not become a complete thesis.

A partial setup must not be treated as actionable.

## Conflicted setup handling

If Analysis Review contains `SETUP_CONFLICTED`, Recommendation Review may emit:

* review state: `human_review_required`;
* review category: `analysis_mixed_or_conflicted`.

Recommendation Review must preserve the conflict explanation and upstream setup references.

Conflicted setup evidence must not be converted into a positive or negative recommendation.

## Blocked setup handling

If Analysis Review contains `SETUP_BLOCKED_BY_MISSING_DATA`, Recommendation Review must emit or preserve:

* review state: `blocked_by_missing_data`;
* review category: `analysis_blocked_by_missing_data`;
* explicit missing-data entries.

Recommendation Review must not:

* convert missing setup evidence to zero;
* infer missing observations;
* treat blocked state as a negative investment conclusion;
* continue as if setup evidence were complete.

## Not-assessed setup handling

If Analysis Review contains `SETUP_NOT_ASSESSED`, Recommendation Review may emit:

* review state: `insufficient_evidence`;
* review category: `analysis_not_supported` or `analysis_blocked_by_missing_data`, depending on the Analysis Review evidence.

Recommendation Review must preserve the reason the setup could not be assessed.

A not-assessed setup must not become a recommendation, ranking, exclusion, or portfolio decision.

## Not-detected setup handling

If Analysis Review contains `SETUP_NOT_DETECTED`, Recommendation Review may emit:

* review state: `insufficient_evidence`;
* review category: `analysis_not_supported`.

Recommendation Review must not interpret not-detected setup evidence as:

* sell signal;
* avoid signal;
* watchlist removal;
* negative ranking;
* low conviction;
* reduced allocation.

## Missing-data and numeric-zero handling

Missing setup data must remain explicit.

Recommendation Review must preserve missing setup observations from Analysis Review.

Recommendation Review must not convert missing setup values into numeric zero.

Recommendation Review must not treat numeric zero as missing.

Numeric zero remains a valid observed or derived value when source-grounded and explicitly present.

If Analysis Review preserves numeric zero in setup evidence, Recommendation Review must preserve that value or reference.

## Controlled output language

Allowed Setup Detection-aware Recommendation Review language:

* “Setup-aware Analysis Review supports human review.”
* “Setup-aware Analysis Review is partial and requires human review.”
* “Setup-aware Analysis Review is conflicted and requires human review.”
* “Recommendation Review is blocked by missing setup evidence.”
* “Setup Detection could not be assessed in the Analysis Review input.”
* “The candidate remains non-actionable.”
* “Portfolio Review and Decision Engine authority are outside this layer.”

Forbidden language and semantics remain forbidden:

* BUY / SELL / HOLD;
* target price;
* rating;
* ranking;
* score;
* conviction;
* urgency;
* tradeability;
* allocation;
* position sizing;
* execution advice;
* order generation;
* portfolio mutation;
* watchlist mutation;
* Telegram;
* reporting;
* delivery;
* Decision Engine action.

## Persistence expectations

The Recommendation Review persistence root remains:

```text
data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json
```

ME-RR03 does not create or write this path.

A future ME-RR04 implementation may preserve Setup Detection-aware provenance in the existing `sec-companyfacts-recommendation-review-v1` artifact shape.

If persistence is touched in ME-RR04, it must:

* preserve input Analysis Review references;
* preserve Setup Detection-aware Analysis Review references;
* preserve setup-aware provenance;
* refuse overwrite by default;
* use temporary directories in tests;
* avoid production data writes.

## Backward compatibility

ME-RR04 must preserve existing ME-RR02 behavior for Analysis Review artifacts that do not contain Setup Detection-aware review items.

ME-RR04 may extend the existing Recommendation Review builder rather than create a new contract version, provided it preserves:

* existing input contract validation;
* existing output contract;
* existing review states and categories;
* existing missing-data behavior;
* existing numeric-zero behavior;
* existing non-actionable boundary;
* existing persistence overwrite protection.

## ME-RR04 implementation requirements

ME-RR04 must implement Setup Detection-aware Recommendation Review behavior according to this contract.

ME-RR04 must:

* consume only validated `sec-companyfacts-analysis-review-v1`;
* preserve existing ME-RR02 behavior when Setup Detection-aware Analysis Review items are absent;
* detect Setup Detection-aware Analysis Review items where present;
* preserve setup-aware provenance;
* preserve setup categories and states;
* preserve setup evidence and limitations;
* preserve missing setup observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-actionable boundary markers;
* route detected setup evidence to human review only;
* route partial setup evidence to human review with explicit uncertainty;
* route conflicted setup evidence to human review with explicit conflict;
* route blocked setup evidence to blocked-by-missing-data;
* route not-assessed setup evidence to insufficient-evidence or blocked routing;
* fail closed for unsupported Analysis Review input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-RR04 must test:

* Setup Detection-aware detected Analysis Review creates non-actionable human-review Recommendation Review;
* partial setup-aware Analysis Review preserves partiality and missing setup evidence;
* conflicted setup-aware Analysis Review preserves conflict and routes to human review;
* blocked setup-aware Analysis Review preserves missing data and blocks Recommendation Review;
* not-assessed setup-aware Analysis Review remains not assessed or insufficient evidence;
* not-detected setup-aware Analysis Review does not become a negative recommendation;
* unsupported Analysis Review contract fails closed;
* numeric zero remains present and is not treated as missing;
* setup-aware provenance is preserved;
* existing ME-RR02 behavior remains valid for non-setup-aware Analysis Review;
* persistence preserves setup-aware provenance if persistence is touched;
* forbidden action-authority terms are not emitted;
* no legacy `scripts` or old `market_scanner` imports are introduced.

## Explicit non-scope

ME-RR03 does not authorize:

* Python implementation;
* tests;
* runtime behavior;
* provider calls;
* live SEC calls;
* EDGAR calls;
* yfinance calls;
* data writes;
* generated artifacts;
* Analysis Review runtime changes;
* Setup Detection runtime changes;
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

## Acceptance criteria

ME-RR03 is complete when:

* the Setup Detection-aware Recommendation Review contract is documented;
* `sec-companyfacts-analysis-review-v1` remains the approved input contract;
* Setup Detection-aware Analysis Review interpretation is documented;
* setup-aware evidence and provenance preservation is documented;
* detected setup routing is documented;
* partial setup routing is documented;
* conflicted setup routing is documented;
* blocked setup routing is documented;
* not-assessed setup routing is documented;
* not-detected setup routing is documented;
* missing-data handling is documented;
* numeric-zero handling is documented;
* controlled output language is documented;
* persistence expectations are documented;
* backward compatibility expectations are documented;
* ME-RR04 implementation requirements are documented;
* backlog marks ME-RR03 completed;
* backlog marks ME-RR04 as the only recommended next sprint;
* roadmap status is updated to `ACTIVE ROADMAP AFTER ME-RR03`;
* roadmap completed chain includes ME-RR03;
* roadmap marks ME-RR04 as the only recommended next sprint;
* future sprint sequence remains ME-RR04, ME-PR01, ME-PR02, ME-DE01, ME-DE02, ME-DL01, ME-DL02;
* no runtime, test, provider, data, portfolio, delivery, Telegram, reporting, or Decision Engine behavior is changed.

## Completion outcome

ME-RR03 extends the Recommendation Review contract so ME-RR04 can later implement consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output.

The contract keeps Recommendation Review downstream of Analysis Review, preserves setup-aware evidence and provenance, keeps missing setup data explicit, preserves numeric-zero semantics, routes setup states only to non-actionable human-review outcomes, and prevents action, portfolio, delivery, ranking, scoring, or Decision Engine authority.