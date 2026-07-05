# ME-GV02 - Governor Factor Taxonomy and Evidence Requirements

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

Contract family: `market-engine-governor-factor-taxonomy-v1`

## Purpose

ME-GV02 defines the factor taxonomy, factor states, minimum evidence requirements, downgrade rules, and factor-level fail-closed behavior for The Governor.

It follows ME-GV01, which defined the envelope for:

```text
market-engine-governor-investment-evaluation-v1
```

ME-GV02 does not implement runtime behavior, numeric scoring, weighting, ranking, recommendation-state mapping, buy-zone logic, position-management logic, delivery, portfolio mutation, watchlist mutation, broker behavior, or Decision Engine authority.

## Roadmap position

```text
ME-RUN29 - Expanded generic coverage classification evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
  -> ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
  -> ME-GV04 - Implement factor scoring from approved analysis evidence
```

ME-GV02 is the semantic prerequisite for ME-GV03 and ME-GV04.

ME-GV03 may only implement deterministic factor-state evaluation and evidence packaging. ME-GV04 is the first planned sprint that may introduce numeric factor scoring.

## Taxonomy principles

The Governor taxonomy must obey these rules:

1. Evidence sufficiency is evaluated before investment quality.
2. A factor may be `blocked` even when other factors are evaluable.
3. Missing evidence must remain missing and must never become zero.
4. Stale evidence must remain explicit and may downgrade or block the factor.
5. Unprovenanced evidence must not support factor evaluation.
6. Factor outputs must reference approved upstream evidence.
7. Factor evaluation must not bypass Recommendation Review or other authority boundaries.
8. Factor states are independent from numeric scoring.
9. No factor may silently infer another factor's result.
10. Ticker identity and market remain data only; they may not select factor rules.

## Canonical factor families

The Governor v1 factor taxonomy contains nine factor families:

| Factor | Purpose | Core upstream evidence |
| --- | --- | --- |
| `fundamentals` | Evaluate current business and financial quality evidence | Fundamental Observations, Derived Observations, Source Context |
| `growth` | Evaluate growth direction, persistence, and quality evidence | Fundamental Observations, Derived Observations |
| `valuation` | Evaluate valuation evidence relative to approved valuation inputs or comparisons | Approved valuation observations or derived valuation evidence |
| `trend` | Evaluate directional market/price trend evidence | Setup Detection and approved price/market observations |
| `momentum` | Evaluate acceleration, persistence, and confirmation evidence | Setup Detection and approved momentum observations |
| `risk` | Evaluate downside, uncertainty, evidence gaps, and explicit risk observations | Analysis Review, Derived Observations, Source/Refinery limitations |
| `technical_setup` | Evaluate setup quality and technical readiness evidence | Setup Detection and approved price/market context |
| `portfolio_fit` | Evaluate relationship to approved portfolio context | Approved Portfolio Context only |
| `data_confidence` | Evaluate evidence trust, freshness, completeness, and provenance | Refinery coverage/readiness, provenance, freshness, completeness |

The taxonomy defines factor boundaries only. It does not define weights or scores.

## Canonical factor states

Every factor evaluation must use exactly one state from this set:

```text
not_started
blocked
unavailable
insufficient_evidence
partial
qualitative_only
evaluable
```

State semantics:

| State | Meaning | Scoring eligibility |
| --- | --- | --- |
| `not_started` | Factor evaluation has not been attempted | No |
| `blocked` | Invalid, stale beyond policy, unprovenanced, malformed, non-consumable, or authority-blocked evidence prevents evaluation | No |
| `unavailable` | The required evidence family is absent or unsupported | No |
| `insufficient_evidence` | Some evidence exists but minimum evidence requirements are not met | No |
| `partial` | Minimum evidence exists for a limited non-actionable interpretation but completeness requirements for full evaluation are not met | No in ME-GV03; reserved for later scoring policy |
| `qualitative_only` | Approved descriptive evidence supports explanation but not numeric evaluation | No |
| `evaluable` | Required evidence gates are satisfied for future deterministic factor evaluation | Yes only after ME-GV04 explicitly defines scoring |

`evaluable` does not imply positive quality, a recommendation, or actionability. It only indicates sufficient evidence for later approved factor logic.

## Factor output contract

Each factor evaluation should follow this target shape:

```text
{
  "factor": "fundamentals",
  "state": "partial",
  "evidence_references": [...],
  "evidence_requirements": {...},
  "missing_evidence": [...],
  "blocked_reasons": [...],
  "limitations": [...],
  "qualitative_summary": "...",
  "score": null,
  "score_scale": null,
  "weight": null,
  "weighted_score": null,
  "provenance": {...}
}
```

Until ME-GV04:

```text
score = null
score_scale = null
weight = null
weighted_score = null
```

## Global evidence gates

Before any factor may be `partial`, `qualitative_only`, or `evaluable`, the following global checks must pass where applicable:

* approved contract identity;
* valid Refinery/staging evidence;
* source-family support or explicit approved fallback path;
* manifest validity;
* payload presence;
* provenance presence;
* freshness status within policy or explicitly accepted downgrade policy;
* consumability;
* structural validity;
* deterministic evidence references.

Global failures must propagate to affected factors as deterministic blockers.

## Fundamentals factor

### Purpose

Evaluate evidence about financial and business quality without performing scoring in ME-GV02.

### Minimum evidence for `partial`

At least one approved fundamental evidence family with valid provenance and explicit completeness metadata.

Examples of acceptable evidence categories:

* revenue evidence;
* profitability evidence;
* cash-flow evidence;
* balance-sheet evidence;
* capital structure evidence;
* margin evidence.

### Minimum evidence for `evaluable`

The factor requires a future ME-GV04-approved set of core subdomains. ME-GV02 requires, at minimum, evidence spanning multiple financial dimensions rather than a single metric.

The exact scoring formula is deferred, but `evaluable` must not be reached from company-profile descriptive evidence alone.

### Downgrade rules

* company-profile-only context -> `qualitative_only` at most;
* one narrow financial metric family -> `partial` at most;
* stale core financial evidence -> `blocked` or `insufficient_evidence` according to freshness policy;
* missing provenance -> `blocked`;
* missing values remain missing, never zero-filled.

## Growth factor

### Purpose

Evaluate evidence about growth direction, persistence, and quality.

### Required evidence categories

Growth evidence should distinguish, where available:

* revenue growth;
* earnings or profitability growth;
* cash-flow growth;
* margin development;
* growth persistence across periods;
* growth quality versus one-off effects when approved evidence supports that distinction.

### State rules

* descriptive company growth narrative without measured evidence -> `qualitative_only`;
* one valid growth series without persistence evidence -> `partial`;
* multiple approved growth dimensions with sufficient history -> `evaluable` candidate for future ME-GV04 logic;
* missing provenance or malformed period alignment -> `blocked`.

## Valuation factor

### Purpose

Evaluate approved valuation evidence without manufacturing valuation from absent price or financial inputs.

### Required evidence categories

At least one approved valuation basis and its required supporting inputs, for example future approved evidence for:

* earnings-based valuation;
* sales-based valuation;
* cash-flow-based valuation;
* enterprise-value-based valuation;
* approved historical or peer-relative comparison.

ME-GV02 does not approve any specific ratio, threshold, peer group, or valuation model.

### State rules

* no approved price/valuation evidence -> `unavailable`;
* incomplete numerator/denominator support -> `insufficient_evidence`;
* descriptive valuation narrative only -> `qualitative_only`;
* approved valuation evidence with complete inputs -> `evaluable` candidate for ME-GV04.

No target price, fair value, intrinsic value, upside percentage, or buy zone may be derived in ME-GV02.

## Trend factor

### Purpose

Evaluate approved evidence about directional trend.

### Required evidence categories

The factor may later consume approved evidence such as:

* price direction;
* moving-average relationships;
* higher-high / higher-low or lower-high / lower-low structure;
* relative trend evidence;
* market-regime context.

ME-GV02 does not authorize calculation formulas or thresholds.

### State rules

* missing approved price/market evidence -> `unavailable`;
* stale price context -> `blocked`;
* limited setup evidence without broader trend context -> `partial`;
* complete approved trend context -> `evaluable` candidate for ME-GV04.

## Momentum factor

### Purpose

Evaluate approved evidence about acceleration, persistence, and confirmation of market movement.

### Required evidence categories

Future approved evidence may include:

* rate-of-change evidence;
* momentum indicator evidence;
* volume/participation confirmation;
* relative-strength evidence;
* acceleration/deceleration context.

ME-GV02 does not authorize any specific indicator, lookback window, or threshold.

### State rules

* no approved momentum evidence -> `unavailable`;
* one isolated indicator without context -> `partial`;
* stale or unprovenanced market evidence -> `blocked`;
* complete approved momentum context -> `evaluable` candidate for ME-GV04.

## Risk factor

### Purpose

Aggregate approved evidence about downside, uncertainty, structural weakness, and evidence limitations.

### Required evidence categories

Risk may consume approved evidence about:

* leverage or balance-sheet risk;
* earnings/cash-flow fragility;
* concentration or cyclicality evidence;
* volatility or downside behavior where approved market evidence exists;
* setup invalidation evidence;
* stale, incomplete, or conflicting evidence;
* explicit Analyzer limitations and blocked reasons.

Risk is special: missing data may itself be a risk limitation, but missing data must not be numerically interpreted as high or low risk.

### State rules

* explicit validated risk evidence with gaps -> `partial`;
* only source-readiness limitations -> `qualitative_only` or `insufficient_evidence`, not a numeric risk conclusion;
* malformed/unprovenanced risk evidence -> `blocked`;
* sufficiently broad approved risk evidence -> `evaluable` candidate for ME-GV04.

## Technical setup factor

### Purpose

Evaluate the quality and readiness of an approved technical setup.

### Required evidence categories

Future approved evidence may include:

* setup type;
* trigger state;
* invalidation context;
* support/resistance context;
* price structure;
* volume/participation confirmation;
* market context.

### State rules

* setup not applicable -> `unavailable`;
* setup evidence absent -> `unavailable`;
* setup detected but missing price/market confirmation -> `partial`;
* stale setup evidence -> `blocked`;
* complete approved setup evidence -> `evaluable` candidate for ME-GV04.

No buy zone or execution instruction may be derived here.

## Portfolio fit factor

### Purpose

Evaluate compatibility with an approved portfolio context without changing the portfolio.

### Required evidence categories

Portfolio fit requires an approved Portfolio Context contract and may later consider evidence such as:

* existing position presence;
* concentration exposure;
* sector/industry concentration;
* geographic exposure;
* currency exposure;
* portfolio risk concentration;
* overlap or redundancy evidence.

ME-GV02 does not approve these as implemented calculations; it only defines the evidence categories.

### State rules

Until ME-PR03 or equivalent approved context exists:

```text
state = blocked
blocked_reason = blocked_missing_approved_portfolio_context
```

Ad hoc portfolio data must not be used.

Portfolio fit may never authorize allocation, target weight, position sizing, order generation, or mutation.

## Data confidence factor

### Purpose

Evaluate trust in the evidence set supporting the Governor evaluation.

### Required evidence categories

Data confidence must consider:

* source support;
* manifest validity;
* provenance;
* freshness;
* consumability;
* completeness;
* contract identity;
* conflicting evidence indicators;
* missing required source families;
* Analyzer-stage blockers and limitations.

### State rules

* invalid manifest, malformed evidence, or missing provenance -> `blocked`;
* stale or incomplete but inspectable evidence -> `partial` or `insufficient_evidence`;
* descriptive-only evidence -> `qualitative_only`;
* complete, fresh, provenanced, consumable evidence -> `evaluable` candidate for later confidence scoring.

Data confidence must never be used to upgrade absent investment evidence. High data confidence about limited evidence remains limited evidence.

## Cross-factor downgrade rules

The following rules apply across factors:

| Evidence condition | Maximum factor state |
| --- | --- |
| Missing required source family | `unavailable` or `insufficient_evidence` |
| Invalid manifest | `blocked` |
| Missing provenance | `blocked` |
| Non-consumable evidence | `blocked` |
| Malformed evidence | `blocked` |
| Stale evidence beyond approved policy | `blocked` |
| Descriptive-only company profile | `qualitative_only` |
| Partial analytical coverage | `partial` |
| Recommendation Review blocked | Factor evaluation may remain descriptive/partial, but recommendation mapping remains unavailable |
| Missing approved portfolio context | `portfolio_fit = blocked` |

Downgrade rules must be deterministic and reason-coded.

## Conflicting evidence

Conflicting evidence must not be silently averaged away.

A future scaffold must:

* preserve the conflicting evidence references;
* expose an explicit limitation or conflict marker;
* downgrade the factor when conflict prevents reliable evaluation;
* avoid numeric scoring until ME-GV04 defines conflict handling.

## Factor completeness

ME-GV02 distinguishes:

```text
factor evidence presence
factor evidence sufficiency
factor evaluability
factor scoring eligibility
```

These are separate concepts.

A factor can have evidence present while remaining insufficient. A factor can be evaluable while still having no numeric score before ME-GV04.

## Overall evaluation prerequisites

ME-GV02 does not define an overall score.

A future overall evaluation may only be considered after:

* factor taxonomy is implemented consistently;
* required factors have explicit states;
* blocked and unavailable factors remain explicit;
* factor scoring exists under ME-GV04;
* weighting policy is explicitly approved;
* missing-factor aggregation behavior is explicitly approved.

Until then:

```text
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
```

## Reserved scoring semantics

ME-GV02 reserves but does not define:

* numeric factor scales;
* factor weights;
* weighted aggregation;
* score bands;
* normalization;
* cross-sector normalization;
* ranking;
* confidence weighting;
* missing-factor imputation.

ME-GV04 must define these explicitly before implementation.

## Recommendation-state boundary

Factor states do not imply recommendation states.

Examples:

```text
evaluable fundamentals != BUY
strong future score != actionable
high data confidence != recommendation
complete technical setup != order instruction
```

Recommendation-state mapping remains blocked until ME-GV05.

## Buy-zone boundary

No factor definition in ME-GV02 authorizes:

* entry price;
* buy-under price;
* breakout trigger price;
* exceptional buy zone;
* stop loss;
* target price;
* position-management instruction.

Those remain outside scope until ME-GV06 or later explicitly approved contracts.

## Authority boundary

ME-GV02 does not introduce or authorize:

* runtime code;
* provider calls;
* live market data;
* source acquisition;
* snapshot import;
* staging validator changes;
* generic coverage classifier changes;
* Analyzer semantic changes;
* Recommendation Review semantic changes;
* Portfolio Review semantic changes;
* Dispatch Station behavior;
* Telegram/email sending;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* broker behavior;
* BUY / SELL / HOLD action semantics;
* numeric scoring;
* weighting;
* ranking;
* urgency;
* conviction;
* tradeability;
* target prices;
* target weights;
* allocation;
* position sizing;
* order generation;
* execution instructions;
* Decision Engine decisions.

## Acceptance criteria for ME-GV03

ME-GV03 may implement a non-actionable scaffold only if it:

* consumes approved evidence contracts only;
* emits the canonical factor families;
* emits only canonical factor states;
* preserves deterministic evidence references;
* applies fail-closed downgrade rules;
* preserves missing, stale, blocked, and conflicting evidence;
* leaves all score/weight fields null or absent;
* does not map factor states into recommendations;
* keeps buy-zone and position-management sections blocked;
* keeps actionable and Decision Engine-ready states unreachable.

## Next sprint

```text
ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
```

ME-GV03 should implement contract shape, factor-state evaluation, evidence references, and fail-closed behavior only. Numeric factor scoring remains deferred to ME-GV04.
