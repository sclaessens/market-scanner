# ME-DS01 - Dispatch Station Governor Report Output Contract

Owner roles: Product Owner / Technical Architect / Functional Analyst / Financial Analyst / QA Lead / Governance Auditor

Job family: ME-DS - Dispatch Station

Status: COMPLETED DOCS-ONLY CONTRACT

Contract version: `market-engine-dispatch-station-governor-report-v1`

## Purpose

ME-DS01 defines the first output contract for Dispatch Station. Dispatch Station consumes completed, versioned Governor evaluation output and transforms it into deterministic, operator-readable report artifacts without changing investment semantics or introducing delivery authority.

The product flow remains:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

Dispatch Station is a presentation and packaging boundary. It must preserve Governor evidence, states, limitations, blockers, provenance, and authority flags exactly enough for operator inspection.

ME-DS01 is docs-only. It does not implement runtime report generation, delivery, publishing, notifications, UI behavior, portfolio mutation, watchlist mutation, broker behavior, scheduling, or Decision Engine authority.

## Approved upstream input

Dispatch Station v1 may consume only a versioned Governor output that has passed the Governor contract chain and contains the fields required for rendering.

The current approved upstream contract family is rooted in:

```text
market-engine-governor-investment-evaluation-v1
```

and may include approved extensions produced by later Governor contracts for:

```text
factor evaluations and factor scoring
recommendation-state mapping
buy-zone explanation
position-management explanation
```

Dispatch Station must not consume raw provider payloads, unvalidated staged packages, arbitrary operator notes, broker output, watchlist state, or hidden application state as substitutes for Governor output.

## Core boundary

Dispatch Station may:

* select and order approved Governor fields for presentation;
* format values, labels, headings, and sections deterministically;
* emit machine-readable and human-readable representations of the same report meaning;
* preserve blockers, missing evidence, conflicts, risk, limitations, provenance, and authority state;
* create channel-neutral preview payloads for later delivery adapters;
* omit unavailable optional sections only when the omission is explicit in report metadata.

Dispatch Station may not:

* calculate or alter Governor scores;
* invent an overall score or rank;
* reinterpret recommendation states;
* turn explanation states into executable instructions;
* invent price levels, targets, stops, quantities, weights, or allocations;
* determine urgency, conviction, tradeability, or order priority;
* call providers or fetch live prices;
* mutate portfolio or watchlist state;
* send Telegram or email messages;
* publish production reports;
* route broker orders;
* make Decision Engine decisions.

## Report identity

Every report payload must contain:

```text
report_contract_version
report_id
report_generated_at
source_evaluation_id
source_contract_references
subject
report_state
rendering_profile
sections
risk_and_limitations
missing_evidence
blocked_reasons
authority_boundary
provenance
```

Required identity rules:

1. `report_contract_version` equals `market-engine-dispatch-station-governor-report-v1`.
2. `report_id` is deterministic for a fixed report input and generation identity policy.
3. `source_evaluation_id` preserves the Governor evaluation identity.
4. `source_contract_references` lists every Governor contract version represented in the report.
5. `subject` identifies ticker, market, and company name when provided upstream.
6. `report_generated_at` is report metadata and must not be presented as market-data freshness.

## Report states

Allowed v1 report states are exactly:

```text
blocked
partial_report
descriptive_report
evaluation_report
```

Meaning:

| State | Meaning |
| --- | --- |
| `blocked` | Governor output is malformed, unknown, contradictory, or lacks minimum identity/provenance required for safe rendering |
| `partial_report` | Valid Governor output exists but material sections are unavailable or blocked; report must expose the gaps |
| `descriptive_report` | Upstream output is descriptive-only and must not be presented as a full investment evaluation |
| `evaluation_report` | A completed non-actionable Governor evaluation can be rendered with all applicable sections and explicit limitations |

Dispatch Station must not create states named or implying `actionable`, `trade_ready`, `order_ready`, `decision_ready`, or `de_ready`.

## Canonical section model

A v1 report may contain the following ordered sections:

```text
1. executive_summary
2. evidence_readiness
3. factor_review
4. recommendation_review
5. buy_zone_review
6. position_management_review
7. risk_and_limitations
8. missing_evidence_and_blockers
9. provenance_and_authority
```

### Executive summary

The executive summary must summarize upstream states without changing their meaning.

It may include:

```text
evaluation_state
recommendation_state
buy_zone_state
position_management_state
primary_strengths
primary_risks
primary_blockers
```

It must not introduce new scores, rankings, target prices, allocation guidance, or execution language.

### Evidence readiness

This section must preserve upstream readiness evidence, including where available:

```text
source validity
provenance validity
freshness
consumability
completeness
Analyzer integrity
Recommendation Review boundary
portfolio-context availability
authority state
```

A report must not visually hide stale, missing, invalid, or non-consumable evidence.

### Factor review

Each factor row or block must preserve:

```text
factor name
factor state
score when upstream-authorized and non-null
score scale/version when present
evidence references
limitations
missing evidence
conflicts
```

Dispatch Station must not compute weighted totals, averages, composite scores, or ranks unless a later explicit Governor contract produces and authorizes them upstream.

### Recommendation review

Recommendation output must be rendered as an interpretive Governor state, not as an order instruction.

The report must preserve:

```text
eligibility
state
reasons
critical factor context
risk context
data-confidence context
conflicts
limitations
actionable flag
Decision Engine readiness flag
```

Any fixed-false or blocked authority field remains fixed-false or blocked in every rendering.

### Buy-zone review

Buy-zone rendering may display only price/setup levels supplied by approved Governor evidence.

It may present:

```text
state
eligibility
pullback context
breakout context
acceptable-zone context
extension context
invalidation context
current position relative to zone when supplied upstream
evidence references
limitations
```

Wording must remain conditional and explanatory. Terms such as `buy now`, `place order`, `set stop`, `risk X percent`, or `allocate Y percent` are forbidden unless a later explicit authority contract changes the boundary.

### Position-management review

Position-management rendering may display only approved upstream explanation for explicit position context.

It may present:

```text
position-context state
eligibility
review state
hold/add/reduce/exit review context
invalidation context
risk context
conflicts
limitations
evidence references
```

The report must distinguish review context from execution authority.

### Risk and limitations

Risk, limitations, conflicts, and uncertainty are first-class report content.

The report must surface:

```text
critical risks
material limitations
data-confidence limits
staleness warnings
conflicting evidence
missing factor evidence
missing portfolio context
blocked downstream authority
```

A formatter may improve readability but may not suppress negative or blocking evidence because a report would otherwise appear incomplete.

### Provenance and authority

Every human-readable report must have a corresponding inspectable authority/provenance section or metadata block.

Minimum authority representation:

```text
actionable: false unless explicitly authorized upstream
execution_authorized: false
order_generation_authorized: false
portfolio_mutation_authorized: false
watchlist_mutation_authorized: false
decision_engine_ready: false unless explicitly authorized by a later contract
```

For v1, any absent authority field must fail closed rather than default true.

## Canonical machine-readable shape

The v1 report payload target is:

```text
{
  "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
  "report_id": "...",
  "report_generated_at": "...",
  "source_evaluation_id": "...",
  "source_contract_references": [...],
  "subject": {
    "ticker": "...",
    "market": "...",
    "company_name": "..."
  },
  "report_state": "...",
  "rendering_profile": "...",
  "sections": {
    "executive_summary": {...},
    "evidence_readiness": {...},
    "factor_review": [...],
    "recommendation_review": {...},
    "buy_zone_review": {...},
    "position_management_review": {...},
    "risk_and_limitations": [...],
    "missing_evidence_and_blockers": {...},
    "provenance_and_authority": {...}
  },
  "risk_and_limitations": [...],
  "missing_evidence": [...],
  "blocked_reasons": [...],
  "authority_boundary": {...},
  "provenance": {...}
}
```

The duplicated top-level risk/blocker/authority summaries are allowed for safe machine inspection. They must be derived directly from the same upstream report content and must not diverge from section rendering.

## Rendering profiles

ME-DS01 defines channel-neutral rendering profiles, not delivery behavior.

Allowed initial profiles:

```text
canonical_json
operator_markdown
compact_preview
```

### `canonical_json`

Purpose: complete machine-readable report artifact.

Requirements:

* preserves all required contract fields;
* stable key semantics;
* deterministic ordering policy where serialized ordering is tested;
* no channel-specific message limits;
* suitable as the source for other renderers.

### `operator_markdown`

Purpose: full local human-readable operator report.

Requirements:

* preserves the canonical section order;
* shows report state prominently;
* surfaces blockers and limitations before any interpretive detail that could be mistaken for action guidance;
* clearly labels recommendation, buy-zone, and position-management output as evaluation/review context;
* contains provenance and authority information.

### `compact_preview`

Purpose: bounded preview text for later delivery adapters or dashboards.

Requirements:

* contains subject identity;
* contains report/evaluation state;
* contains recommendation state when available;
* contains buy-zone and position-management state when available;
* includes the highest-priority blocker or limitation when present;
* includes an explicit non-actionable authority marker;
* must not silently truncate away blockers, risk warnings, or authority boundaries.

The compact preview is not a Telegram message, email, notification, or production publish action. Delivery remains a separate sprint boundary.

## Determinism requirements

For identical normalized Governor input, rendering profile, contract version, and explicit generation metadata:

* report state must be identical;
* section inclusion must be identical;
* field-to-label mapping must be identical;
* blocker ordering must be deterministic;
* missing-evidence ordering must be deterministic;
* factor ordering must follow the approved Governor taxonomy order;
* authority representation must be identical;
* JSON semantics and Markdown meaning must remain equivalent.

A renderer may vary whitespace only where determinism tests explicitly permit it.

## Fail-closed rules

Dispatch Station must emit `blocked` rather than a misleading report when any of these occur:

* unknown report input contract;
* missing Governor evaluation identity;
* malformed evaluation state;
* missing subject identity required by the selected profile;
* contradictory recommendation state and eligibility;
* non-null price levels without approved evidence references;
* authority flags indicating unsupported execution/order/mutation permission;
* missing provenance for material report sections;
* impossible or unknown factor states;
* renderer cannot preserve a material blocker or limitation.

A blocked report must include deterministic blocked reasons and must preserve any usable source identity/provenance needed for audit.

## Cross-format equivalence

Every human-readable representation must be traceable to a canonical report payload.

Rules:

1. Markdown and preview output must not contain a stronger recommendation than canonical JSON.
2. Preview output must not omit a critical blocker while showing favorable evaluation text.
3. Numeric price levels must match canonical JSON exactly.
4. Missing values must remain missing, unavailable, null, or explicitly blocked; they must not be replaced with zero.
5. Authority flags must not change between profiles.
6. No renderer may infer a position state from absent portfolio context.

## Redaction and sensitive-output boundary

ME-DS01 does not define personal-data processing. Report artifacts should contain investment-evaluation content and technical provenance only.

A future delivery contract must define any recipient-specific redaction, destination policy, retention, and transport controls before reports are sent outside local non-production artifact boundaries.

## Relationship with ME-DS02

ME-DS02 may implement local non-production artifact generation only after conforming to this contract.

ME-DS02 should produce at minimum:

```text
dispatch_report.json
dispatch_report.md
```

and may optionally produce a local compact preview fixture/artifact.

ME-DS02 must prove:

* deterministic rendering;
* report-state mapping;
* blocker/limitation preservation;
* factor/recommendation/explanation fidelity;
* price-level fidelity;
* cross-format semantic equivalence;
* fixed-false unsupported authority;
* no provider/network call;
* no Telegram/email send;
* no production publish;
* no portfolio/watchlist mutation;
* no broker/order behavior;
* no scheduler/UI behavior;
* no Decision Engine decision.

## Acceptance criteria

ME-DS01 is complete when:

* a versioned Dispatch Station report contract exists;
* approved Governor input boundaries are explicit;
* report states are explicit and fail closed;
* canonical sections are defined;
* JSON, Markdown, and compact-preview profiles are defined without authorizing delivery;
* blockers, missing evidence, risk, limitations, provenance, and authority are mandatory presentation concerns;
* cross-format semantic equivalence rules are explicit;
* Dispatch Station is prohibited from creating new investment semantics;
* ME-DS02 has implementable acceptance criteria;
* no runtime, tests, providers, delivery, publishing, mutation, broker, scheduler, UI, or Decision Engine behavior is changed.

## Next sprint

```text
ME-DS02 - Implement local non-production Governor report artifact
```
