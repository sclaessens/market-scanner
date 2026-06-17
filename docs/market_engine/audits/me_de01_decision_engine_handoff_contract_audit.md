# ME-DE01 - Decision Engine handoff contract audit

## Status

COMPLETED BY ME-DE01

## Sprint

ME-DE01 - Define Decision Engine handoff contract

## Branch

me-de01-define-decision-engine-handoff-contract

## Sprint goal

Define the formal Decision Engine handoff contract downstream of Portfolio Review while preserving Decision Engine as the only action and allocation authority.

## Reason for sprint

ME-PR02 implemented Portfolio Review and produced non-actionable `sec-companyfacts-portfolio-review-v1` output suitable for later Decision Engine handoff review.

Before any implementation can prepare a handoff payload, Market Engine needs a formal contract that defines eligible Portfolio Review input, blocked states, handoff payload shape, fail-closed behavior, numeric-zero handling, provenance requirements, and the authority boundary between Market Engine and the future Decision Engine.

## Files added

Documentation:

* `docs/market_engine/decision_engine/me_de01_decision_engine_handoff_contract.md`
* `docs/market_engine/audits/me_de01_decision_engine_handoff_contract_audit.md`

## Files changed

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Sources inspected

Inspected upstream contract and implementation context:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`
* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`
* `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`
* `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`
* `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`
* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`
* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`
* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`

## Contract defined

Defined the first Decision Engine handoff contract:

* approved upstream input: `sec-companyfacts-portfolio-review-v1`;
* required portfolio-context family: `market-engine-portfolio-context-v1`;
* future handoff payload: `market-engine-decision-engine-handoff-v1`;
* required Portfolio Review eligibility criteria;
* blocked handoff states;
* handoff payload shape;
* prohibited payload fields;
* Market Engine request boundary;
* Decision Engine-only authority;
* fail-closed rules;
* numeric-zero safety;
* audit and traceability requirements;
* ME-DE02 implementation requirements.

## Approved handoff readiness states

ME-DE01 defined these readiness states:

* `ready_for_decision_engine_review`
* `blocked_missing_portfolio_review`
* `blocked_invalid_portfolio_review_contract`
* `blocked_unapproved_portfolio_review`
* `blocked_missing_portfolio_context`
* `blocked_stale_portfolio_context`
* `blocked_incomplete_provenance`
* `blocked_ticker_mismatch`
* `blocked_insufficient_evidence`
* `not_applicable`

These states are handoff-readiness states only and do not grant action or allocation authority.

## Backlog updates

Backlog changes:

* marked `ME-DE01 - Define Decision Engine handoff contract` as completed;
* recorded the contract document and audit document;
* moved `ME-DE02 - Implement controlled Decision Engine handoff` to the only `Status: RECOMMENDED NEXT`;
* preserved `ME-DL01` and `ME-DL02` as planned future sprints;
* preserved the possible `ME-PR03` follow-up candidate without inserting it ahead of ME-DE02.

## Roadmap updates

Roadmap changes:

* changed roadmap status to `ACTIVE ROADMAP AFTER ME-DE01`;
* added ME-DE01 to the completed chain;
* documented the ME-DE01 contract outcome;
* moved ME-DE02 to the recommended next sprint;
* preserved the future sequence through Delivery / Reporting.

## Boundaries preserved

Confirmed ME-DE01 did not introduce:

* Python runtime code;
* tests;
* provider calls;
* broker calls;
* web API calls;
* generated data;
* portfolio writes;
* watchlist writes;
* portfolio mutation;
* Telegram behavior;
* reporting delivery behavior;
* Decision Engine runtime behavior;
* BUY / SELL / HOLD decisions;
* action recommendations;
* allocation advice;
* target weights;
* order generation;
* position sizing instructions;
* urgency;
* conviction;
* tradeability;
* ranking;
* scoring;
* execution advice.

## Validation performed

Validation commands:

```bash
git diff --check
git status --short
git diff --name-only
grep -n "Status: RECOMMENDED NEXT" docs/market_engine/backlog/market_engine_backlog.md
grep -n "ME-DE01\|ME-DE02\|ME-DL01\|ME-DL02" docs/market_engine/roadmap/market_engine_roadmap.md
```

Results:

* `git diff --check` passed.
* `git status --short` showed only planned documentation changes.
* `git diff --name-only` showed only the ME-DE01 contract, audit, backlog, and roadmap files.
* The backlog has one `Status: RECOMMENDED NEXT` marker for ME-DE02.
* The roadmap preserves ME-DE02, ME-DL01, and ME-DL02 in sequence.

No standard docs-specific lint command was found or run.

## Conclusion

ME-DE01 is complete as a documentation-only contract sprint.

The repository now has a formal Decision Engine handoff contract downstream of Portfolio Review and upstream of future Decision Engine behavior.

## Next recommended sprint

```text
ME-DE02 - Implement controlled Decision Engine handoff
```
