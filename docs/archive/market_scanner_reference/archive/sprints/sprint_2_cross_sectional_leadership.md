# Sprint 2 — Cross-Sectional Leadership Layer

## 1. Sprint Status

Status: CERTIFIED COMPLETE

Sprint 2 is certified complete. See `docs/sprints/sprint_2_closeout.md`.

Sprint 2 confirmed that active Context runtime was already governance-clean. Implementation strengthened Context tests and documented historical artifact hygiene without changing runtime Context code.

Sprint 1 is certified complete. Sprint 2 may proceed only through the standard workflow:

audit  
→ impact analysis  
→ scoped Technical Lead specification  
→ developer execution  
→ validation  
→ governance review  
→ closeout

## 2. Governance Inheritance From Sprint 0 And Sprint 1

Sprint 2 inherits the certified Sprint 0 doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine

Sprint 2 also inherits Sprint 1 certification:

- `structure_state` and `structure_reason` are authoritative Validation Layer fields
- `valid_setup` and `validation_reason` are compatibility-only
- Validation output may not be reinterpreted by Context as tradeability or allocation eligibility
- hidden filtering remains forbidden

Context Layer responsibility is leadership classification only.

## 3. Sprint Objective

Sprint 2 prepares the Context Layer for governance-safe cross-sectional leadership alignment.

The sprint objective is to audit and, only if later approved, refine Context Layer documentation, schema expectations, tests, and governance checks so context remains a pure leadership and relative-strength classification layer.

Sprint 2 must preserve:

- opportunity distribution
- deterministic outputs
- separation of concerns
- Decision Engine centralized allocation authority

Sprint 2 must not introduce trading decisions, allocation eligibility, execution readiness, or hidden filters.

## 4. Context Layer Responsibilities

Context Layer may classify:

- leadership
- relative strength
- cross-sectional strength
- sector-relative leadership
- market-relative leadership

Context Layer may emit descriptive classification metadata such as:

- `rs_score`
- `rs_rank`
- `rs_percentile`
- `rs_vs_market`
- `rs_vs_sector`
- `context_strength`
- `context_reason`
- `leadership_state`

Context Layer may enrich observations with sector-relative strength where available, but sector data must not become a blocking dependency.

## 5. Explicitly Forbidden Context Layer Responsibilities

Context Layer may not determine:

- tradeability
- allocation eligibility
- conviction
- urgency
- execution readiness
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- final actions
- portfolio actions
- watchlist removal

Context Layer may not:

- filter opportunities out of the pipeline
- collapse opportunity distribution through hidden rules
- reinterpret Validation output as capital readiness
- override scanner, validation, watchlist, portfolio, Decision Engine, or reporting responsibilities
- create action labels or execution instructions

## 6. Classification-First Doctrine

Sprint 2 must keep Context as a classification layer.

Allowed:

- rank relative strength
- classify leadership distribution
- describe market-relative leadership
- describe sector-relative leadership
- expose deterministic context metadata
- log distribution and missing-data observations

Forbidden:

- execution logic
- filtering logic
- tradeability logic
- allocation ranking
- urgency logic
- conviction scoring
- BUY/SELL/HOLD/TRIM/REMOVE behavior

Weak context means weak leadership classification only. It does not mean rejection.

Strong or leading context means strong leadership classification only. It does not mean tradeability.

## 7. Leadership/Context Contract

Current governance-clean runtime contract observed in `data/processed/context_strength.csv`:

- `ticker`
- `date`
- `rs_score`
- `rs_percentile`
- `rs_rank`
- `rs_vs_market`
- `rs_vs_sector`
- `context_strength`
- `context_reason`
- `leadership_state`

Sprint 2 governance audit must confirm whether the current runtime contract already satisfies Sprint 2 objectives before any developer execution is scoped.

Any proposed contract additions must remain descriptive and classificatory. Proposed fields may not imply tradeability, actionability, allocation, conviction, urgency, or execution readiness.

## 8. Expected Schema

Expected active Context Layer schema:

| Field | Governance Meaning |
|---|---|
| `ticker` | identifier |
| `date` | observation date |
| `rs_score` | relative-strength score metadata |
| `rs_percentile` | cross-sectional distribution percentile |
| `rs_rank` | cross-sectional rank |
| `rs_vs_market` | market-relative strength metadata |
| `rs_vs_sector` | sector-relative strength metadata, nullable if sector data is unavailable |
| `context_strength` | leadership classification |
| `context_reason` | classification reason |
| `leadership_state` | leadership-state classification |

Potential future fields such as `leadership_bucket` or `leadership_persistence` require Technical Lead approval before implementation and must remain descriptive metadata only.

## 9. Descriptive Metadata Policy

Context metadata must remain:

- descriptive
- classificatory
- deterministic
- non-allocative
- non-executory

Context metadata may inform downstream Decision Engine interpretation, but the Context Layer itself may not decide capital allocation, tradeability, urgency, or final actions.

Sector-relative values may enrich context classification. Missing sector-relative data must be handled as missing metadata, not as a reason to block or remove an opportunity.

## 10. Forbidden Field Policy

Forbidden Context Layer output fields:

- `context_tradeable`
- `tradeability`
- `conviction`
- `allocation_priority`
- `final_action`
- `urgency`
- `actionable`
- `BUY`
- `SELL`
- `HOLD`
- `TRIM`
- `REMOVE`

Forbidden fields must not appear in active Context Layer runtime output.

Tests may reference forbidden terms only when asserting absence.

Historical/archive references may remain if clearly contextualized as legacy or anti-patterns.

## 11. Distribution-Preservation Doctrine

Sprint 2 must preserve opportunity distribution.

Context may:

- classify the full scanner universe available to it
- rank observations
- expose leadership distributions
- expose missing-data counts
- expose sector-relative availability

Context may not:

- drop rows because leadership is weak
- suppress rows because sector data is missing
- remove rows because benchmark-relative strength is poor
- filter rows because they appear non-actionable
- cap output to only strong or leading names

Distribution changes are observability findings, not permission to add hidden filters.

## 12. Required Observability

Sprint 2 governance audit and any later implementation spec should require clear observability for:

- total Context Layer rows
- `context_strength` distribution
- `leadership_state` distribution
- `rs_percentile` distribution
- missing sector-relative data count
- top-decile or leadership cohort count
- schema drift
- forbidden-field drift

Observability must not alter runtime trading logic, allocation behavior, Context output eligibility, or pipeline outputs.

## 13. In-Scope Items

In scope for Sprint 2 planning and future governance audit:

- review current Context Layer implementation
- review current Context Layer tests
- review `context_strength.csv` schema
- verify no `context_tradeable` or allocation fields exist in active context output
- verify sector-relative data enriches but does not block classification
- verify cross-sectional ranking and percentile behavior
- verify weak leadership is classification only
- verify strong leadership is classification only
- identify test gaps
- define future developer-spec scope if required

In scope for later developer execution only after Technical Lead specification:

- narrowly scoped Context Layer schema/test alignment
- forbidden-field assertions
- deterministic output validation
- observability/logging alignment
- documentation updates

## 14. Out-of-Scope Items

Out of scope for Sprint 2 preparation:

- runtime implementation
- Context Layer rewrite
- scanner changes
- validation changes
- watchlist changes
- portfolio changes
- Decision Engine allocation changes
- reporting or Telegram changes
- strategy scoring changes
- trading threshold optimization
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- new allocation logic
- new filters
- execution-readiness logic

Out of scope for Context Layer generally:

- tradeability
- allocation eligibility
- conviction
- urgency
- final actions
- execution instructions

## 15. Risks And Controls

| Risk | Severity | Control |
|---|---|---|
| Strong context is interpreted as tradeability | HIGH | State that leadership classification is not capital allocation |
| Weak context is treated as rejection | HIGH | Test and document weak context as classification only |
| Sector data becomes a blocking dependency | HIGH | Require nullable sector-relative metadata and non-blocking behavior |
| Benchmark-relative strength becomes the sole model when cross-sectional ranking is available | MEDIUM | Audit percentile/rank behavior before implementation |
| Leadership ranking becomes allocation priority | HIGH | Forbid allocation ranking outside Decision Engine |
| Context output is filtered to only top leadership names | HIGH | Enforce full opportunity-distribution preservation |
| Legacy `context_tradeable` terminology leaks from archive docs | MEDIUM | Treat archive references as historical anti-patterns; forbid active output |
| Sprint scope expands into Decision Engine behavior | HIGH | Require Technical Lead specification before developer execution |

## 16. Acceptance Criteria

Sprint 2 preparation passes when:

- Sprint 2 document inherits Sprint 0 and Sprint 1 governance
- Context responsibilities are limited to leadership classification
- forbidden Context output fields are explicit
- expected schema is governance-clean
- metadata policy is descriptive and non-allocative
- distribution-preservation doctrine is explicit
- observability requirements are defined without changing runtime logic
- in-scope and out-of-scope boundaries are clear
- governance risks are documented
- final recommendation is ready for Sprint 2 governance audit

Future Sprint 2 implementation may be accepted only if:

- runtime inspection happens before code changes
- no Context tradeability fields are emitted
- no allocation/action/conviction/urgency fields are emitted
- weak/strong context remains classification-only
- tests pass
- governance grep checks pass or are interpreted as absence-tests only
- generated artifacts are not blindly committed

## 17. Definition Of Done

Sprint 2 preparation is done when:

- active Sprint 2 planning documentation is governance-aligned
- no runtime code is changed
- no generated data is changed
- contradictions or governance risks are documented
- Scrum Master recommendation is recorded

Sprint 2 implementation is not started by this document.

## 18. Sprint 2 Preparation Recommendation

READY FOR SPRINT 2 GOVERNANCE AUDIT

The current active Context Layer appears governance-clean at the schema level:

- no `context_tradeable`
- no tradeability fields
- no conviction fields
- no allocation fields
- no final-action fields

The next step should be a Sprint 2 governance audit, not developer execution. The audit should confirm whether runtime implementation and tests already satisfy the Sprint 2 leadership-classification contract before any developer specification is created.
