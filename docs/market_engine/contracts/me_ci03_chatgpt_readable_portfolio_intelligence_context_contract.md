# ME-CI03 - ChatGPT-readable Portfolio Intelligence Context Contract v1

## Purpose

ME-CI03 defines the ChatGPT-readable Portfolio Intelligence Context: a
controlled, traceable, evidence-backed portfolio subcontext that may be embedded
inside, or referenced beside, `chatgpt-advisory-context-v1`.

Approved identity:

```text
contract_name: chatgpt_portfolio_intelligence_context
contract_version: v1
schema_version: chatgpt-portfolio-intelligence-context-v1
artifact_type: market-engine-chatgpt-portfolio-intelligence-context
```

The contract selects, normalizes, and bounds already-proven portfolio
information for advisory consumption. It does not calculate holdings, exposure,
cash, allocation, target weight, maximum weight, portfolio fit, position size,
or rebalancing actions.

## Architectural position

Approved chain:

```text
Market Engine / Portfolio Review / Governor
-> ChatGPT Portfolio Intelligence Context
-> ChatGPT Advisory Context
-> ChatGPT Advisory Layer
```

Market Engine, Portfolio Review, Governor, and Decision Engine-facing contracts
prove and classify portfolio state. ME-CI03 only structures that proven context
for later advisory interpretation.

ChatGPT may explain supplied portfolio intelligence. It must not reconstruct,
complete, optimize, or override it.

## Relationship to ME-CI02

`chatgpt-advisory-context-v1` remains the top-level advisory envelope.

Portfolio Intelligence Context v1 supports three inclusion modes:

| Mode | Meaning | Required handling |
| --- | --- | --- |
| `embedded_subcontext` | A compact Portfolio Intelligence Context object is embedded in `portfolio_context_boundary` or a future dedicated field of `chatgpt-advisory-context-v1`. | Embedded fields must preserve schema identity, provenance, freshness, missingness, and permission boundary. |
| `referenced_companion_artifact` | The advisory context contains a stable artifact reference and selected summary fields. | The companion artifact is canonical for portfolio details. Summary conflicts fail closed. |
| `absent` | No approved portfolio intelligence context is provided. | Portfolio-specific advisory is unavailable. Absence must not be converted to zero holdings or zero cash. |

Both embedded and companion modes are allowed. A future assembler must choose one
explicitly per advisory artifact and must not silently mix conflicting embedded
and referenced facts.

## Source-of-truth matrix

| Information family | Upstream existence | Canonical source | Information class | ME-CI03 value handling | Conflict behavior |
| --- | --- | --- | --- | --- | --- |
| Portfolio identity | Available when supplied by local portfolio context or Portfolio Review | `market-engine-portfolio-context-v1`; Portfolio Review provenance | Raw fact / provenance | Include directly when versioned and provenanced | Block if identity conflicts across context, review, and handoff |
| Portfolio snapshot identity | Available in portfolio context timestamp/run fields | `market-engine-portfolio-context-v1` | Raw fact | Include directly | Block on contradictory snapshots for same context |
| Holdings / position state | Available per reviewed ticker | `MarketEnginePortfolioContext.position_state`; Portfolio Review position review | Raw fact / review state | Include per ticker only when explicitly supplied | Unknown remains unknown; absent from partial context does not mean not held |
| Quantity | Available for per-ticker context when supplied | `current_quantity` in `market-engine-portfolio-context-v1` | Raw fact | Include with provenance and timestamp | Do not convert missing to zero |
| Position value | Available when supplied | `current_market_value` in `market-engine-portfolio-context-v1` | Raw fact / valuation fact | Include only with currency and valuation freshness | Stale or unknown valuation downgrades usage |
| Position weight | Available when supplied | `current_ticker_exposure_pct`; Portfolio Review exposure review | Derived observation | Include only when upstream supplied | Do not calculate from value and total unless future contract authorizes |
| Cost basis | Not canonical in current Market Engine portfolio context | None approved for ME-CI03 | Unavailable | Represent as missing or not provided | Any supplied unprovenanced cost basis blocks that claim |
| Unrealized result | Not canonical in current Market Engine portfolio context | None approved for ME-CI03 | Unavailable | Represent as missing or not provided | Do not derive from market value and missing cost basis |
| Cash | Not present in current `market-engine-portfolio-context-v1` except broader legacy data outside approved ME path | No approved ME-CI03 canonical cash artifact yet | Unavailable unless future approved source supplies it | Represent explicit missingness unless an approved artifact provides amount, currency, and timestamp | Cash unavailable is not cash zero |
| Exposure | Limited ticker exposure and optional buckets exist | `current_ticker_exposure_pct`, `exposure_buckets`, Portfolio Review exposure review | Derived observation / review state | Include only supplied buckets and measurements | Do not create sector, theme, currency, factor, or geography exposure |
| Concentration | Limited threshold review exists | `concentration_thresholds`; Portfolio Review concentration review | Review judgement | Include measured/reviewed concentration context only | Missing threshold means review limitation, not safe concentration |
| Allocation | Current per-ticker exposure can exist; target allocation is not approved here | `current_ticker_exposure_pct` only for current exposure | Raw/derived current state | Include current allocation-like facts only as current exposure | Never create target or recommended allocation |
| Constraints | Optional policy constraints and concentration thresholds exist | `policy_constraints`, `concentration_thresholds`, Governor context when present | Policy context / review judgement | Include directly with source and status | Do not invent thresholds |
| Portfolio fit | Portfolio Review emits review states, not investment optimization | `sec-companyfacts-portfolio-review-v1` | Review judgement | Include review state and blockers | Portfolio fit is not recommendation or sizing |
| Position management | Governor explanation may exist for held/not-held context | `market-engine-governor-buy-zone-position-management-explanation-v1` | Governance interpretation | Include explanation state and limitations only | Governor conflicts block or downgrade advisory |
| Recommendation relation | Available by combining recommendation identity with proven position state | Recommendation Review, Portfolio Review, Structured Decision Output | Relationship classification | Include relation only when both sides are proven | Candidate ticker is not a holding by default |
| Missing context | Available in Portfolio Review and handoff markers | `missing_portfolio_context_fields`, handoff blocked reasons | Limitation marker | Include explicitly | Missingness must remain visible |
| Uncertainty | Available as missing/stale/blocked/conflict markers | Portfolio Review, handoff, ME-CI02 uncertainty context | Limitation marker | Include explicit machine-readable reason codes | Free text is not enough |
| Freshness | Available per snapshot/stale fields, not as one global truth | `portfolio_snapshot_timestamp`, `stale_portfolio_context_fields`, review/handoff stale markers | Freshness marker | Include per family | Fresh context generation does not refresh portfolio evidence |
| Provenance | Available in Portfolio Review and local context fixture metadata | `context_provenance`, Portfolio Review provenance, handoff audit provenance | Lineage | Required for material claims | Free-floating facts block the affected claim |

Legacy `data/portfolio` CSV files are not automatically canonical ME-CI03 inputs.
They may become sources only after an explicit approved portfolio source contract
validates them and produces `market-engine-portfolio-context-v1` or a later
approved successor.

## Contract identity fields

Every artifact must contain:

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Must equal `chatgpt-portfolio-intelligence-context-v1`. |
| `artifact_type` | yes | Must equal `market-engine-chatgpt-portfolio-intelligence-context`. |
| `generated_at` | yes | Context generation time. Not portfolio freshness. |
| `run_id` | yes | Portfolio intelligence context run identity. |
| `portfolio_identity` | yes | Portfolio/account identity when known; explicit unknown when not approved. |
| `portfolio_snapshot_identity` | yes | Snapshot/run identity and snapshot timestamp when known. |
| `source_artifact_refs` | yes | Versioned artifacts used to assemble the context. |
| `currency_context` | yes | Base/reporting currency and any known valuation currency limits. |
| `timestamp_context` | yes | Distinguishes generation, snapshot, valuation, Portfolio Review, Governor, and handoff times. |

`generated_at`, `portfolio_snapshot_time`, `market_valuation_time`,
`portfolio_review_time`, and `governor_interpretation_time` are distinct.
Consumers must not treat them as equivalent.

## Portfolio context availability

Allowed availability states:

| State | Meaning |
| --- | --- |
| `available` | Required portfolio identity, provenance, and relevant context families are present for the declared advisory scope. |
| `partial` | Some portfolio context exists, but material families are missing, stale, or incomplete. |
| `unavailable` | No approved portfolio context exists for portfolio-specific interpretation. |
| `blocked` | Context is malformed, conflicting, unsupported, stale in a critical family, or lacks required provenance. |

Hard semantic boundaries:

```text
portfolio context unavailable != zero holdings
portfolio cash unavailable != cash = 0
position absent from provided partial context != user owns no position
```

## Holdings context

Holdings entries may include only facts already present in approved upstream
context:

```text
ticker
instrument
position_state
quantity
quantity_unit
currency
market_value
portfolio_weight_pct
cost_basis
unrealized_result
source_snapshot_ref
valuation_timestamp
freshness
provenance
missing_fields
```

Current v1 canonical support is limited. `position_state`, quantity, market
value, total value, and ticker exposure can be included when supplied through
`market-engine-portfolio-context-v1`. Cost basis and unrealized result are
reserved as explicit missing or unsupported fields unless a future approved
source provides them.

ME-CI03 must not:

* convert quantity without a contract;
* infer FX;
* reconstruct market value from stale price;
* estimate cost basis;
* infer acquisition date;
* infer ownership from watchlist membership;
* infer holdings from historical text or generated reports.

## Position context

ME-CI03 separates four layers:

| Layer | Example | Authority |
| --- | --- | --- |
| Holding fact | `current_quantity = 4` | Approved portfolio context only |
| Position intelligence | `current_ticker_exposure_pct = 8.3` | Approved portfolio context / Portfolio Review |
| Review judgement | `concentration_requires_review` | Portfolio Review |
| Governance interpretation | `hold_context` or `add_review_context` | Governor explanation |

ChatGPT may explain these layers, but it may not collapse them into an action.
`add_review_context` does not specify how much to buy. `reduce_review_context`
does not specify what to sell. `exit_review_context` does not create an order.

## Exposure context

Approved v1 exposure families are limited to:

| Exposure family | V1 status |
| --- | --- |
| Single-name ticker exposure | Supported when `current_ticker_exposure_pct` is supplied. |
| Generic supplied exposure buckets | Supported as provided by `exposure_buckets`. |
| Sector exposure | Supported only when explicitly supplied as a bucket. |
| Industry exposure | Supported only when explicitly supplied as a bucket. |
| Geographic exposure | Not calculated by ME-CI03. |
| Currency exposure | Not calculated by ME-CI03. |
| Thematic exposure | Not calculated by ME-CI03. |
| Factor exposure | Not calculated by ME-CI03. |
| Asset-class exposure | Supported only when explicitly supplied as a bucket. |

Each exposure entry must contain:

```text
exposure_type
exposure_key
value
unit
source_ref
timestamp
completeness
confidence_or_uncertainty
```

ME-CI03 does not create a new exposure engine. Missing exposure families remain
missing.

## Concentration context

Concentration context must distinguish:

| Class | Meaning |
| --- | --- |
| `measured_concentration` | A supplied measurement such as current ticker exposure. |
| `classified_concentration` | A Portfolio Review state such as `concentration_within_context`. |
| `warning` | A limitation or review-required marker. |
| `advisory_explanation` | Natural-language explanation later produced by ChatGPT. |

ChatGPT may say that upstream context flags concentration for review. It may
not independently decide that adding a candidate makes the portfolio too risky
unless an approved upstream review supports that statement.

## Cash context

Cash is optional and high-risk. Current v1 has no canonical Market Engine cash
source approved for ME-CI03.

When a future approved source exists, cash context must include:

```text
amount
currency
snapshot_timestamp
source_ref
available_for_investment_semantics
restricted_or_reserved_cash
cash_floor_constraint
freshness
uncertainty
provenance
```

Hard boundary:

```text
cash balance != deployable cash
deployable cash != recommended position size
```

ChatGPT must never infer position size from cash alone.

## Allocation context

ME-CI03 separates:

| Allocation family | V1 treatment |
| --- | --- |
| Current allocation / exposure | Allowed only when supplied as current weight or exposure. |
| Policy allocation | Allowed only when approved policy context supplies it. |
| Target allocation | Not generated by ME-CI03. |
| Recommended allocation | Decision Engine / approved downstream authority only. |

Current exposure is not a recommendation. Target weight and recommended
allocation must remain null, unavailable, or referenced-only unless a future
approved Decision Engine contract provides them.

## Portfolio constraints

Constraints may be included only when upstream approved evidence supplies them.

Required fields:

```text
constraint_type
source_ref
status
current_measurement
threshold
interpretation
severity
blocker_status
provenance
```

ME-CI03 may expose `concentration_thresholds` and `policy_constraints` from
`market-engine-portfolio-context-v1`, but it must not invent missing thresholds.

## Portfolio fit context

ME-CI03 consumes Portfolio Review semantics:

```text
portfolio_context_supported
portfolio_context_partial
portfolio_context_missing
portfolio_context_stale
portfolio_context_invalid
position_already_held
position_not_held
position_unknown
exposure_known
exposure_missing
concentration_within_context
concentration_requires_review
blocked_by_missing_portfolio_context
blocked_by_invalid_input
ready_for_decision_engine_handoff_review
```

Semantic separations:

```text
instrument recommendation != portfolio fit
portfolio fit != position sizing
position sizing != allocation authority
```

## Recommendation-to-position relationship

Allowed relationship states:

| State | Meaning |
| --- | --- |
| `portfolio_relation_unknown` | Portfolio context is absent, partial, stale, invalid, or does not prove position relation. |
| `held_in_approved_context` | The instrument is explicitly held in approved context. |
| `not_held_in_approved_context` | The instrument is explicitly not held in approved context. |
| `held_status_partial` | Some position information exists but ownership or exposure remains incomplete. |
| `recommendation_for_held_instrument` | Recommendation identity and proven held state refer to the same instrument. |
| `recommendation_for_non_held_instrument` | Recommendation identity and explicit not-held state refer to the same instrument. |
| `blocked_relation_conflict` | Recommendation identity and portfolio identity conflict. |

The relationship class does not remake the recommendation. It only describes
how a proven recommendation candidate relates to proven portfolio context.

## Missingness and uncertainty

Allowed missingness states:

```text
known_absent
unknown
not_applicable
not_provided
stale
conflicting
blocked
partial
unsupported
```

Required missingness families:

```text
missing_holdings
partial_holdings
missing_valuation
stale_valuation
unknown_cash
missing_fx_context
incomplete_portfolio_universe
unknown_external_accounts
unknown_liabilities
incomplete_cost_basis
unavailable_target_allocation
unavailable_portfolio_constraints
contradictory_snapshots
```

ChatGPT must disclose material missingness and must not present incomplete
portfolio context as complete.

## Provenance contract

Every material portfolio claim must trace to lineage:

```text
portfolio_snapshot
holdings
valuation
cash
exposure
concentration
portfolio_review
governor_context
position_management_explanation
decision_engine_handoff
```

Provenance entries must include, where available:

```text
artifact_ref
run_id
snapshot_id
source_type
generated_at
observed_at
as_of
derived_from_refs
```

Free-floating portfolio facts are invalid.

## Freshness contract

Freshness is per family:

```text
holdings_freshness
valuation_freshness
cash_freshness
portfolio_review_freshness
governor_interpretation_freshness
exposure_freshness
concentration_freshness
```

Allowed statuses:

```text
fresh
mixed
stale
unknown
blocked
not_applicable
```

Degradation examples:

| Condition | Allowed result |
| --- | --- |
| Fresh holdings plus stale valuation | Ownership may be explainable; current market value and weight are descriptive-only or blocked. |
| Fresh price plus unknown holdings freshness | Ownership is not proven; portfolio-specific advisory is unavailable. |
| Missing cash freshness | Cash-specific interpretation is unavailable. |
| Stale Portfolio Review | Portfolio-fit interpretation is blocked or descriptive-only according to upstream state. |
| Fresh context generation but stale snapshot | The context may be current as an artifact, but the portfolio facts remain stale. |

## Advisory permission boundary

Machine-readable permissions must use these groups:

| Permission group | Allowed examples |
| --- | --- |
| `explain_allowed` | Explain position weight, held/not-held state, concentration warning, Portfolio Review state, Governor position-management explanation, known constraints, and known cash context. |
| `conditional_interpretation_allowed` | Discuss portfolio relation only when Recommendation Review, Portfolio Review, freshness, and provenance support it. |
| `determination_forbidden` | Target allocation, target weight, max weight unless upstream-proven, order size, shares to buy, cash to deploy, rebalance instruction, tax strategy, broker action, synthetic stop, synthetic target price, unsupported sell urgency, unsupported add urgency. |

## Use-case matrix

| User question | Required context | Allowed answer class | Blocked without |
| --- | --- | --- | --- |
| "I own ASML. Should I add?" | Proven ASML holding, recommendation context, Portfolio Review, Governor position-management context, freshness, blockers | Explain upstream held-state, recommendation relation, portfolio-fit limits, and Governor review context | Proven holding, Portfolio Review, or Governor context |
| "Does AMD fit beside NVDA and ASML?" | Holdings context, exposure buckets, concentration context, candidate Portfolio Review, missingness disclosure | Explain only upstream portfolio-fit and exposure/concentration evidence | Multi-holding/exposure evidence |
| "How much should I add?" | Approved downstream position-sizing/allocation authority | Usually blocked; may explain that sizing is unavailable | Decision Engine or approved sizing output |
| "Which holding should I sell to make room?" | Approved comparative Portfolio Review or Decision Engine authority | Usually blocked; may explain missing comparative authority | Comparative review or allocation authority |
| "I have EUR 3,000 cash. Where should I invest it?" | Proven cash plus approved Decision Engine allocation/recommendation context | Cash explanation only unless downstream authority exists | Allocation authority and portfolio-fit context |

## Fail-closed matrix

| Condition | Result |
| --- | --- |
| Missing schema version | `context_invalid` |
| Unsupported schema version | `context_invalid` |
| Missing portfolio identity | `portfolio_advisory_blocked` unless explicitly scoped to non-portfolio explanation |
| Missing snapshot identity | `portfolio_advisory_blocked` |
| Missing run identity | `context_invalid` |
| Missing holdings provenance | `portfolio_specific_advice_unavailable` |
| Partial holdings universe | `eligible_with_mandatory_caveat` or `descriptive_only_allowed` |
| Stale holdings | `portfolio_advisory_blocked` for ownership-sensitive use cases |
| Stale valuation | `descriptive_only_allowed` for ownership; block current value or weight |
| Unknown valuation timestamp | Block valuation and weight claims |
| Missing cash provenance | Cash interpretation unavailable |
| Currency mismatch | Block affected value/cash/exposure claim unless approved FX evidence exists |
| Missing FX evidence | Block cross-currency valuation aggregation |
| Conflicting portfolio snapshots | `portfolio_advisory_blocked` |
| Missing Portfolio Review | Portfolio-fit interpretation unavailable |
| Blocked Portfolio Review | `portfolio_advisory_blocked` for portfolio-fit answers |
| Malformed Governor position context | Governor position-management interpretation unavailable or blocked |
| Conflicting position state | `portfolio_advisory_blocked` |
| Missing advisory context linkage | Companion artifact not consumable by ME-CI02 |
| Unsupported allocation fields | Ignore field and record validation issue; block if material |
| Unsupported sizing fields | Ignore field and record validation issue; block sizing answer |

## Prohibited inputs

ME-CI03 must not consume:

* unvalidated broker exports;
* raw portfolio CSV files without approved contract conversion;
* generated reports as holdings source of truth;
* watchlist membership as ownership;
* stale snapshots without stale markers;
* cash balances without source identity;
* market prices without timestamp and source;
* prompt-only user claims as portfolio facts;
* ChatGPT-generated portfolio state.

## Prohibited inferences

ChatGPT and future assemblers must not infer:

* missing holdings from absent rows;
* zero cash from unavailable cash;
* not-held state from partial portfolio context;
* deployable cash from cash balance;
* target weight from current weight;
* maximum weight from concentration threshold unless explicitly defined;
* position size from cash, confidence, or recommendation;
* rebalancing actions from concentration warnings;
* sell candidates from current holdings;
* tax strategy from unrealized result;
* portfolio fit from standalone recommendation;
* Decision Engine approval from handoff readiness.

## Validation object

Required validation fields:

```text
validation.contract_valid
validation.validation_state
validation.errors
validation.warnings
validation.blocked_reasons
validation.unsupported_fields
validation.semantic_boundary_version
```

Unsupported future fields must not be interpreted unless a later compatibility
contract approves them.

## Implementation decision

ME-CI03 is a docs-only contract sprint.

Rationale:

* ME-CI02 defines the advisory envelope but no runtime assembler exists yet.
* Existing runtime already has `market-engine-portfolio-context-v1`,
  Portfolio Review, and Decision Engine handoff contracts.
* A typed schema, validator, deterministic assembler, and ChatGPT prompt
  contract belong in later explicit sprints.
* Implementing a context assembler now would create premature coupling between
  legacy portfolio data, local non-production fixtures, Portfolio Review, and
  advisory delivery.

## Next sprint

Recommended next sprint:

```text
ME-CI04 - Define explainability/change-rationale contract
```

Recommended later Portfolio Intelligence work:

```text
ME-PI01 - Define Portfolio Intelligence exposure contract
```
