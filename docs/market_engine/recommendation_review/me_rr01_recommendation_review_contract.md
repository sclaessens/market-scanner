ME-RR01 — Recommendation Review Contract from Analysis Review

Sprint status

Status: completed docs-only contract
Sprint code: ME-RR01
Job family: Recommendation Review
Runtime impact: none
Test impact: none
Data write impact: none
Provider impact: none
Decision Engine impact: none
Portfolio impact: none
Delivery impact: none

Purpose

ME-RR01 defines the first contract boundary for the future Recommendation Review layer.

The Recommendation Review layer may consume validated Analysis Review output and produce a structured recommendation-review candidate for human review.

This layer does not create trading authority.

This layer does not create portfolio authority.

This layer does not create execution authority.

This layer does not create Telegram, reporting, or delivery authority.

This layer does not modify the Decision Engine.

This layer exists only to define a controlled review boundary between non-decision analysis interpretation and future decision-authority systems.

Upstream context

The current Market Engine flow already contains these source-grounded and non-decision layers:

1. Source Context
    * Relevant current contract family: SEC CompanyFacts Source Context.
2. Fundamental Observations
    * Relevant current contract family: SEC CompanyFacts Fundamental Observations.
3. Derived Observations
    * Relevant current contract family: SEC CompanyFacts Derived Cash Generation Observations.
4. Analysis Review
    * Relevant current contract family: SEC CompanyFacts Analysis Review.

Current upstream Analysis Review module:

src/market_engine/analysis_review/sec_companyfacts_analysis_review.py

Current upstream Analysis Review tests:

tests/market_engine/analysis_review/

ME-RR01 does not modify either of these paths.

Scope

ME-RR01 is documentation-only.

Allowed scope:

* Define the Recommendation Review contract boundary.
* Define the allowed input contract.
* Define the proposed output contract name.
* Define the proposed future output path.
* Define review states.
* Define review categories.
* Define allowed and forbidden message semantics.
* Define provenance requirements.
* Define boundaries with Portfolio Review, Delivery, Reporting, Telegram, and the Decision Engine.
* Define implementation requirements for a future ME-RR02 sprint.

Forbidden scope:

* Python code.
* Tests.
* Provider calls.
* SEC calls.
* EDGAR calls.
* yfinance calls.
* Runtime behavior.
* Data writes.
* Generated artifacts.
* Portfolio review.
* Portfolio action.
* Allocation.
* Position sizing.
* Order execution.
* Execution advice.
* Telegram.
* Reporting.
* Delivery.
* Decision Engine behavior changes.
* Watchlist mutation.
* Portfolio mutation.
* BUY, SELL, or HOLD as direct trading instructions.
* Score, ranking, conviction, urgency, tradeability, allocation, or execution output.

Contract name

Recommended future output contract name:

sec-companyfacts-recommendation-review-v1

This contract name is source-specific and versioned.

It must not be treated as a final investment decision contract.

It must not be treated as a portfolio allocation contract.

It must not be treated as an execution contract.

It represents a review candidate only.

Input contract

Allowed input contract:

sec-companyfacts-analysis-review-v1

A future ME-RR implementation may consume only validated Analysis Review output that explicitly declares this contract.

A future implementation must fail closed when the input contract is missing, unknown, malformed, or unsupported.

A future implementation must not infer Analysis Review content from raw source snapshots, Fundamental Observations, Derived Observations, legacy scanner files, portfolio files, watchlist files, reports, or Telegram output.

Proposed future output path

Recommended future output path:

data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json

Path semantics:

* recommendation_review_run_id identifies one isolated Recommendation Review run.
* <ticker> identifies the reviewed ticker.
* recommendation_review.json contains one structured recommendation-review candidate.
* The file is a generated artifact only in a future implementation sprint.
* ME-RR01 does not create this path.
* ME-RR01 does not write this file.

Proposed future top-level output shape

A future sec-companyfacts-recommendation-review-v1 artifact should contain these top-level fields:

{
  "contract": "sec-companyfacts-recommendation-review-v1",
  "ticker": "NVDA",
  "recommendation_review_run_id": "example-run-id",
  "input_contract": "sec-companyfacts-analysis-review-v1",
  "input_provenance": {},
  "review_state": "human_review_required",
  "review_category": "analysis_supportive_but_not_actionable",
  "review_summary": [],
  "supporting_factors": [],
  "blocking_factors": [],
  "missing_data": [],
  "risk_notes": [],
  "forbidden_actions": [],
  "boundary_notes": [],
  "created_at": "ISO-8601 timestamp"
}

This shape is illustrative for ME-RR01 and not yet an implementation schema.

A future ME-RR02 implementation must define the final exact schema before writing runtime code.

Review states

Recommendation Review may define review states.

These states are not trade actions.

human_review_required

Meaning:

The candidate contains enough structured Analysis Review material to route the ticker to a human reviewer, but no autonomous action is allowed.

Allowed language:

* “Requires human review.”
* “Candidate for review.”
* “Analysis supports further review.”
* “Review cannot be converted into action without Portfolio Review and Decision Engine checks.”

Forbidden interpretation:

* Buy now.
* Sell now.
* Hold now.
* Increase allocation.
* Reduce allocation.
* Execute trade.

insufficient_evidence

Meaning:

The upstream Analysis Review does not contain enough evidence to support a useful recommendation-review candidate.

Allowed language:

* “Insufficient evidence for recommendation review.”
* “Required observations are missing.”
* “Analysis Review does not support review routing.”

Forbidden interpretation:

* Avoid the stock as an investment decision.
* Sell the position.
* Remove from watchlist.
* Penalize ranking.
* Lower conviction.

blocked_by_missing_data

Meaning:

One or more required source-grounded inputs are missing, stale, malformed, or explicitly unavailable.

Allowed language:

* “Blocked by missing data.”
* “Recommendation Review cannot proceed because required data is missing.”
* “Missing data remains explicit.”

Forbidden interpretation:

* Treat missing data as zero.
* Infer missing values.
* Fill with fallback assumptions.
* Convert blocked state into a negative investment recommendation.

not_applicable

Meaning:

The available Analysis Review does not apply to the Recommendation Review contract, for example because the input contract is unsupported or the ticker context is not eligible.

Allowed language:

* “Input not applicable for Recommendation Review.”
* “Unsupported input contract.”
* “No Recommendation Review candidate created.”

Forbidden interpretation:

* Portfolio exclusion.
* Watchlist removal.
* Sell signal.
* Negative ranking.

Review categories

Recommendation Review may classify a candidate into review categories.

These categories are for human-review routing only.

They are not scores.

They are not ranks.

They are not conviction levels.

They are not urgency levels.

They are not tradeability grades.

analysis_supportive_but_not_actionable

Meaning:

The Analysis Review contains supportive non-decision evidence, but the evidence is not sufficient to produce portfolio action or execution advice.

analysis_mixed_or_conflicted

Meaning:

The Analysis Review contains both supportive and limiting factors, or the evidence does not point in one clear review direction.

analysis_blocked_by_missing_data

Meaning:

The Analysis Review cannot support Recommendation Review because required evidence is missing or explicitly unavailable.

analysis_not_supported

Meaning:

The Analysis Review does not support a Recommendation Review candidate.

input_contract_invalid

Meaning:

The input is missing, malformed, unsupported, or not declared as sec-companyfacts-analysis-review-v1.

Allowed message semantics

Recommendation Review may produce controlled review language.

Allowed message types:

* Human-review routing message.
* Evidence summary.
* Supporting-factor summary.
* Blocking-factor summary.
* Missing-data summary.
* Risk note.
* Boundary note.
* Provenance note.
* Contract validation note.

Allowed terms:

* “candidate”
* “review”
* “human review”
* “requires review”
* “blocked”
* “insufficient evidence”
* “supporting factor”
* “blocking factor”
* “missing data”
* “non-actionable”
* “not execution advice”
* “not portfolio advice”

Forbidden message semantics

Recommendation Review must not produce direct investment, portfolio, or execution language.

Forbidden terms and semantics:

* “buy”
* “sell”
* “hold”
* “strong buy”
* “strong sell”
* “accumulate”
* “trim”
* “exit”
* “enter position”
* “increase position”
* “reduce position”
* “take profit”
* “stop loss”
* “price target”
* “target allocation”
* “position size”
* “portfolio weight”
* “conviction score”
* “urgency score”
* “tradeability score”
* “ranking”
* “top pick”
* “best candidate”
* “execute”
* “order”
* “rebalance”
* “send alert”
* “send Telegram”
* “publish report”

A future implementation may include these words only inside an explicit forbidden-action list or boundary warning, never as recommendation output.

Missing-data rules

Missing data must remain explicit.

A future ME-RR implementation must preserve upstream missing-data states.

A future ME-RR implementation must not convert missing values into numeric zero.

A future ME-RR implementation must not treat numeric zero as missing.

A future ME-RR implementation must not infer missing fundamentals from derived observations.

A future ME-RR implementation must fail closed when required review inputs are unavailable.

Numeric zero rule

Numeric zero is a valid numeric value when it is source-grounded and explicitly present.

Numeric zero must not be interpreted as missing.

Missing values must be represented through explicit missing-data fields, nulls, absence markers, or upstream contract-defined missing states.

A future ME-RR implementation must keep this distinction intact.

Provenance requirements

Every future Recommendation Review artifact must include provenance.

Minimum provenance requirements:

* Input contract name.
* Input contract version.
* Input ticker.
* Input Analysis Review artifact path or identifier.
* Input Analysis Review run identifier, when available.
* Source Context lineage, when available from upstream.
* Fundamental Observation lineage, when available from upstream.
* Derived Observation lineage, when available from upstream.
* Creation timestamp.
* Recommendation Review contract name.
* Recommendation Review run identifier.

Recommendation Review must not hide that its evidence comes from SEC CompanyFacts-derived Analysis Review.

Recommendation Review must not imply that it has consumed portfolio, price, ranking, watchlist, execution, or delivery context.

Boundary with Analysis Review

Analysis Review is upstream of Recommendation Review.

Analysis Review interprets source-grounded observations.

Recommendation Review may summarize and route Analysis Review output into a human-review candidate.

Recommendation Review must not mutate Analysis Review output.

Recommendation Review must not reinterpret raw source snapshots directly.

Recommendation Review must not bypass Analysis Review.

Boundary with Portfolio Review

Portfolio Review is a separate future layer.

Recommendation Review must not inspect portfolio holdings.

Recommendation Review must not inspect allocation.

Recommendation Review must not inspect exposure.

Recommendation Review must not inspect concentration.

Recommendation Review must not inspect account cash.

Recommendation Review must not inspect watchlist state.

Recommendation Review must not decide whether a ticker fits the portfolio.

Recommendation Review may only state that portfolio context is outside its authority.

Boundary with Decision Engine

The Decision Engine remains the only future authority for final action decisions, subject to its own explicit contracts.

Recommendation Review must not create Decision Engine behavior.

Recommendation Review must not call the Decision Engine.

Recommendation Review must not simulate the Decision Engine.

Recommendation Review must not produce a Decision Engine input unless a later sprint explicitly defines such a boundary.

Recommendation Review must not claim that its output is actionable by the Decision Engine.

Recommendation Review may only produce a non-actionable review candidate.

Boundary with Delivery, Reporting, and Telegram

Delivery is a separate future layer.

Reporting is a separate future layer.

Telegram is a separate future delivery channel.

Recommendation Review must not send messages.

Recommendation Review must not create Telegram alerts.

Recommendation Review must not create reports.

Recommendation Review must not write delivery artifacts.

Recommendation Review must not format output for end-user delivery unless a later ME-DL contract explicitly consumes Recommendation Review artifacts.

Boundary with provider calls

Recommendation Review must not call live providers.

Forbidden provider access:

* SEC live calls.
* EDGAR live calls.
* yfinance calls.
* Broker calls.
* Portfolio provider calls.
* Watchlist provider calls.
* Any network call.

Recommendation Review may only consume local, validated Analysis Review artifacts in a future implementation.

Boundary with legacy runtime

Recommendation Review must not reintroduce legacy runtime dependencies.

Forbidden active dependencies:

* scripts
* market_scanner
* legacy scanner modules
* legacy Decision Engine modules
* legacy reporting modules
* legacy Telegram modules
* legacy portfolio modules

A future ME-RR implementation must live under the src/market_engine/ namespace.

Future recommended module boundary

A future ME-RR02 implementation may create a new stable job-family boundary under:

src/market_engine/recommendation_review/

Recommended future module name:

src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py

This is not created in ME-RR01.

A new Python file is justified in ME-RR02 only because Recommendation Review is a stable contract family boundary separate from Analysis Review.

Future recommended test boundary

A future ME-RR02 implementation may create tests under:

tests/market_engine/recommendation_review/

Tests must be local and synthetic.

Tests must not call providers.

Tests must not write production data.

Tests must not depend on legacy runtime modules.

Tests must validate fail-closed behavior, missing-data preservation, numeric-zero preservation, contract validation, and forbidden-message guardrails.

ME-RR02 implementation requirements

A future ME-RR02 sprint should be allowed only after this contract is accepted.

ME-RR02 should implement the minimum viable Recommendation Review builder.

Required ME-RR02 acceptance criteria:

1. Implement a sec-companyfacts-recommendation-review-v1 builder.
2. Consume only sec-companyfacts-analysis-review-v1.
3. Reject missing or unsupported input contracts.
4. Preserve explicit missing-data states.
5. Preserve numeric zero as a valid value.
6. Produce only non-actionable recommendation-review candidates.
7. Include provenance.
8. Include boundary notes.
9. Include forbidden-action guardrails.
10. Avoid BUY, SELL, HOLD, allocation, sizing, ranking, score, urgency, tradeability, execution, Telegram, reporting, watchlist, and portfolio mutation semantics.
11. Use only local synthetic tests.
12. Avoid all live provider calls.
13. Avoid all prod data writes unless a later sprint explicitly scopes generated artifact writes.
14. Avoid legacy scripts and market_scanner dependencies.

Acceptance criteria for ME-RR01

ME-RR01 is complete when:

* This contract document exists under docs/market_engine/recommendation_review/.
* The audit document exists under docs/market_engine/audits/.
* The backlog marks ME-RR01 as completed.
* The document defines the Recommendation Review boundary.
* The document defines the allowed input contract.
* The document defines the proposed output contract.
* The document defines the proposed future output path.
* The document defines review states.
* The document defines review categories.
* The document defines allowed and forbidden message semantics.
* The document defines provenance requirements.
* The document defines boundaries with Portfolio Review, Delivery, Reporting, Telegram, and the Decision Engine.
* The document defines future ME-RR02 implementation requirements.
* No Python files are changed.
* No tests are changed.
* No runtime behavior is changed.
* No generated data is written.
* No provider calls are introduced.
* No Decision Engine authority is introduced.
* No portfolio, watchlist, reporting, delivery, Telegram, allocation, position sizing, or execution behavior is introduced.

Final ME-RR01 boundary statement

ME-RR01 defines Recommendation Review as a non-actionable, source-grounded, human-review routing layer.

It may later help structure which tickers deserve human attention.

It may not decide what to buy, sell, hold, allocate, size, execute, report, deliver, or send.

Recommendation Review is not the Decision Engine.

Recommendation Review is not Portfolio Review.

Recommendation Review is not Delivery.

Recommendation Review is not execution advice.