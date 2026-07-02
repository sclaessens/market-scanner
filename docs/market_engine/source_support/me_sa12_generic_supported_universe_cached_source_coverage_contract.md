# ME-SA12 - Generic Supported-Universe Cached-Source Coverage Contract

Sprint ID: ME-SA12
Status: COMPLETED DOCS-ONLY CONTRACT
Job family: ME-SA / Source Acquisition and Source Coverage
Date: 2026-07-02
Runtime impact: none
Test impact: none

## 1. Purpose

ME-SA12 defines how Market Engine must classify cached-source coverage for any
entry in an approved supported universe.

The contract connects existing universe, source-support, acquisition-manifest,
staging, cached-source consumption, analysis-readiness, Recommendation Review,
Portfolio Review, and Decision Engine handoff contracts without replacing
their source-specific schemas or authority boundaries.

ME-RUN28 exposed a bounded acquisition allowlist and incomplete analytical
coverage across a larger universe. Its ticker set is evidence for this
contract, not the target of special-case behavior.

This contract is future-ticker safe:

```text
the same contract fields
+ the same validation rules
+ the same readiness rules
= the same classification behavior
```

Changing a ticker value while keeping all validated capabilities and evidence
equal must not change the outcome.

## 2. Non-Goals

ME-SA12 does not:

* modify runtime code or tests;
* add ticker-specific fixes, branches, mappings, or allowlists;
* implement a new provider or source adapter;
* approve live provider, SEC, EDGAR, yfinance, or external API access;
* acquire, import, normalize, or persist new source payloads;
* weaken manifest, provenance, identity, freshness, or staging validation;
* turn `company_profile` into fundamental, setup, price, or market evidence;
* create a recommendation, price target, trade plan, or investment advice;
* activate `actionable_review`, `actionable`, `decision_ready`, or `de_ready`;
* extend Decision Engine handoff eligibility;
* modify Decision Engine, Portfolio Review, Recommendation Review, delivery,
  Telegram, portfolio, or watchlist behavior;
* perform production writes or other side effects.

This contract classifies coverage and readiness. It does not allocate capital.

## 3. Core Principle: Tickers Are Data, Not Logic

A ticker is an opaque instrument identifier carried by data contracts. It may
be used for identity validation, lookup, joins, artifact partitioning,
provenance, and audit output. It must not select business behavior.

Tickers may appear in:

* approved universe configuration;
* fixtures and test data;
* local acquisition requests;
* cached-source manifests;
* source artifacts;
* run artifacts;
* documentation and audit examples.

Tickers must not appear as runtime control flow that changes source support,
validation, consumability, readiness, Recommendation Review eligibility,
actionability, or Decision Engine readiness.

Forbidden runtime patterns:

```python
if ticker == "ASML":
    ...
```

```python
if ticker in ["NVDA", "AMD", "ASML"]:
    ...
```

The same prohibition applies to `symbol`, issuer name, exchange-specific
nicknames, or any disguised hard-coded instrument list used as business logic.

Allowed generic patterns:

```python
if coverage.has_required_source_families:
    ...
```

```python
if snapshot.provenance_status == "accepted":
    ...
```

```python
if readiness.status == "descriptive_only":
    ...
```

A provider-specific identifier map may exist as versioned configuration or
source metadata when a provider contract requires one. It must not grant
readiness or bypass generic validation.

## 4. Generic Coverage Model

ME-SA12 proposes this future contract identity:

```text
market-engine-supported-universe-cached-source-coverage-v1
```

ME-SA12 does not implement or persist it. ME-SA13 may implement the model after
this contract is approved.

### 4.1 `SupportedUniverseEntry`

Required concepts:

| Field | Meaning |
|---|---|
| `universe_contract` | Versioned universe contract identity. |
| `universe_id` | Stable identity of the configured universe. |
| `universe_version` | Version or content identity used for the classification. |
| `instrument_key` | Stable normalized key, including market/listing context when needed for disambiguation. |
| `ticker` | Opaque display/source-join identifier; data only. |
| `market` | Listing/market context used for identity and source mapping. |
| `asset_type` | Validated universe asset type. |
| `active` | Whether the entry participates in the configured classification run. |
| `universe_status` | Existing governed universe state. |
| `coverage_profile_id` | Reference to a generic requirement profile, never a ticker-specific branch. |
| `universe_row_reference` | Traceable reference to the source configuration row. |

Universe membership does not prove source support, consumability, analytical
readiness, Recommendation Review eligibility, or action authority.

### 4.2 `SourceFamily`

Required concepts:

| Field | Meaning |
|---|---|
| `source_family_id` | Stable source or downstream evidence-family identity. |
| `family_kind` | `acquired_source`, `derived_evidence`, or `downstream_contract_input`. |
| `contract_versions` | Approved input/output versions for the family. |
| `provided_capabilities` | Generic capabilities this family may satisfy after validation. |
| `adapter_policy_reference` | Approved adapter policy when acquisition applies. |
| `freshness_policy_reference` | Versioned freshness policy. |
| `consumption_contract_reference` | Approved parser/consumer contract. |
| `required_provenance_fields` | Family-specific lineage and identity requirements. |

Source-family presence alone does not satisfy a capability. The payload must
also be accepted, fresh enough, provenanced, identity-aligned, consumable, and
complete for the applicable requirement.

### 4.3 `SourceCoverageRequirement`

Required concepts:

| Field | Meaning |
|---|---|
| `requirement_id` | Stable generic requirement identity. |
| `coverage_profile_id` | Requirement profile selected by configured capabilities, not ticker. |
| `pipeline_capability` | Capability being evaluated. |
| `accepted_source_families` | Approved family or explicit family alternatives. |
| `minimum_validation_status` | Required manifest/staging validation state. |
| `minimum_freshness_status` | Required freshness state. |
| `provenance_required` | Whether complete approved lineage is mandatory. |
| `consumability_required` | Whether a current consumer contract must exist and accept the payload. |
| `completeness_rule` | Required fields/evidence collections for the capability. |
| `requirement_mode` | `required`, `conditional`, or `not_required`. |
| `failure_reason_code` | Deterministic blocker when the requirement is not met. |

Requirements may be conditional on approved strategy or pipeline capability.
They must not be conditional on a hard-coded ticker name.

### 4.4 `CachedSourceSnapshot`

Required concepts align with ME-SR08:

* snapshot and batch identity;
* source family and source identity;
* ticker/entity/market identity;
* manifest contract and path;
* payload path, SHA-256, size, MIME type, and encoding;
* acquisition mode;
* validation status and issues;
* freshness/staleness status and reason;
* provenance and retrieval/publication timestamps;
* local-use and source-governance constraints;
* `usable_for_cached_source_dry_run`;
* consumer contract/version compatibility.

An artifact that exists but fails any required gate remains visible and
non-consumable.

### 4.5 `CoverageStatus`

`CoverageStatus` is a per-entry, per-requirement result. It must contain:

* entry and requirement identity;
* source-family state;
* selected snapshot and manifest references, when any;
* validation, provenance, freshness, consumability, and completeness states;
* deterministic blockers;
* evidence references;
* aggregate coverage classification;
* no-action and no-authority boundary metadata.

### 4.6 `ReadinessStatus`

`ReadinessStatus` references the current ME-SA09/10/11 classifier result rather
than inventing a parallel analytical authority.

It must preserve:

* current canonical readiness level;
* evidence families present and missing;
* blocked reasons;
* Recommendation Review eligibility;
* actionable-review allowance;
* Decision Engine readiness;
* provenance validity;
* staleness state;
* prohibited inferences;
* non-authority boundary.

Coverage classification may supply validated evidence-family facts to a future
readiness adapter. It must not override the readiness classifier.

### 4.7 `Blocker`

Required concepts:

| Field | Meaning |
|---|---|
| `reason_code` | Deterministic machine-readable blocker. |
| `category` | Universe, acquisition, manifest, provenance, freshness, consumability, completeness, readiness, portfolio, or handoff. |
| `pipeline_stage` | Earliest affected stage. |
| `source_family_id` | Affected family when applicable. |
| `requirement_id` | Failed generic requirement. |
| `blocking` | Whether progression must stop. |
| `evidence_references` | Artifacts and validation facts supporting the blocker. |

Blockers are classifications, not negative investment opinions.

### 4.8 `PipelineStage`

A stage contract must declare:

* stable stage name and contract version;
* required input capabilities;
* accepted input contract versions;
* output contract version;
* status and deterministic blocked reasons;
* provenance references;
* forbidden side effects;
* downstream authority boundary.

Stages must consume generic capabilities and contract fields. They must not
select behavior from a ticker name.

## 5. Source and Evidence Family Mapping

ME-SA12 uses a generic coverage vocabulary while preserving current repository
contract names.

| ME-SA12 family/capability | Kind | Existing canonical mapping | Contract status |
|---|---|---|---|
| `company_profile` | acquired source | Current `company_profile` source family and descriptive ME-SA07/08/09 evidence | Existing; descriptive only |
| `fundamental_facts` | acquired/normalized source capability | Current `sec_companyfacts` source family and ME-SA09 `fundamentals` evidence family | Generic alias; source-specific contracts remain authoritative |
| `price_history` | acquired source capability | ME-SA01 `market_price_snapshot` concept | Proposed alias; no provider or implementation approval |
| `setup_detection_input` | derived/input capability | ME-SA09 `setup_price_market`; may require approved price, liquidity, trend, volatility, and setup inputs | Proposed aggregate capability; source contracts still required |
| `portfolio_context` | downstream context | `market-engine-portfolio-context-v1` | Existing downstream context |
| `recommendation_review_input` | downstream contract input | Approved Analysis Review/Recommendation Review contracts plus ME-SA09 eligibility | Proposed generic reference; not an acquisition family |
| `decision_engine_handoff_input` | downstream contract input | `sec-companyfacts-portfolio-review-v1` and `market-engine-decision-engine-handoff-v1` | Proposed generic reference; not an acquisition family |

The last two rows are downstream evidence/contract-input families. Acquisition
must not attempt to fetch or synthesize them.

No alias changes a current format version, approves a provider, or makes a
reserved readiness level reachable.

## 6. Coverage Statuses

Existing status vocabularies remain authoritative in their own layers.

### 6.1 Existing source-support statuses

ME-SR05/06 currently define:

```text
supported_cached
missing_snapshot
unsupported_sec_companyfacts
missing_required_source_field
malformed_or_unreadable_source_artifact
ambiguous_identity
manual_review_only
excluded
```

### 6.2 Existing validation and manifest states

Current contracts include:

```text
accepted
rejected
missing_manifest
malformed_manifest
fresh
stale
unknown
not_applicable
usable_for_cached_source_dry_run: true | false
```

### 6.3 Proposed generic aggregate coverage terms

ME-SA13 may emit these aggregate terms while preserving all underlying states:

| Proposed term | Meaning |
|---|---|
| `supported` | A generic requirement has an approved source-family path and consumer contract. It does not prove that a snapshot is available. |
| `unsupported` | No approved source-family or consumer contract can satisfy the requirement. |
| `available` | At least one candidate snapshot exists for the requirement. |
| `unavailable` | No candidate snapshot exists. Preserve `missing_snapshot` where applicable. |
| `partial` | Some required families or gates pass and at least one required family or gate does not. |
| `accepted` | Required validation, identity, provenance, freshness, and local-use gates pass. |
| `rejected` | A validator rejected the candidate artifact. |
| `stale` | The applicable freshness policy fails. |
| `unprovenanced` | Required approved lineage is missing or invalid. |
| `invalid_manifest` | Manifest is missing, malformed, unsupported, inconsistent, or integrity-invalid. |
| `missing_snapshot` | Required snapshot is unavailable. Existing ME-SR05 term. |
| `not_consumable` | A snapshot exists but no approved compatible consumer may use it. |
| `not_required` | The requirement profile explicitly marks the family unnecessary for this capability. |
| `blocked` | One or more mandatory gates prevent the requested capability. |

`descriptive_only` is an analysis-readiness result, not a raw coverage status.
It may be repeated in a combined report only as a referenced readiness result.

The aggregate term must never hide the more specific underlying reason.

## 7. Readiness Levels and Capability Labels

ME-SA12 does not replace the canonical ME-SA09 readiness levels:

```text
descriptive_only
partial_analysis
recommendation_eligible
actionable_review        reserved and unreachable
decision_ready           reserved and unreachable
```

The requested generic labels map as follows:

| Generic label | Contract interpretation |
|---|---|
| `unavailable` | Coverage outcome: required analytical source input is unavailable; no analytical readiness may be inferred. |
| `partial` | Coverage/readiness summary alias for current `partial_analysis`; preserve the canonical runtime value. |
| `descriptive_only` | Current readiness level. Descriptive evidence may be communicated but is never actionable. |
| `analysis_ready` | Proposed capability label meaning approved non-descriptive analysis input is consumable. It does not imply Recommendation Review eligibility. |
| `recommendation_review_ready` | Proposed capability label that requires current `recommendation_eligible` plus compatible Recommendation Review input. Stage completion alone does not satisfy it. |
| `actionable` | Reserved proposed capability. It requires a future approved actionable-review contract in addition to complete, fresh, provenanced coverage. It is currently false. |
| `de_ready` | Reserved proposed capability mapping to future `decision_ready` prerequisites plus valid Portfolio Review and handoff controls. It is currently false. |
| `blocked` | Orthogonal outcome preserving why a requested capability cannot proceed; not a higher or lower evidence level by itself. |

Rules:

1. `descriptive_only` is never actionable.
2. `partial` or `partial_analysis` is never automatically actionable.
3. Source-family count never establishes readiness.
4. `analysis_ready` cannot be inferred from a profile, manifest, or artifact
   path alone.
5. `recommendation_review_ready` requires sufficient fundamental/financial and
   setup/price/market evidence, strategy-required evidence, valid provenance,
   accepted freshness, and no blocking limitation.
6. A completed Recommendation Review stage is not equivalent to
   `recommendation_review_ready`.
7. A future `actionable` capability would require all Recommendation Review
   prerequisites and an explicitly approved actionable-review contract. The
   current ME-RR contracts prohibit this state.
8. A future `de_ready` capability would additionally require valid,
   non-stale portfolio context, an approved non-blocked Portfolio Review,
   complete upstream provenance, and accepted handoff controls.
9. Missing setup/price/market or required portfolio context fails closed.
10. Only Decision Engine may determine action or allocation after an approved
    handoff. Readiness is not a decision.

## 8. Source-Family Requirement Matrix

`G` below means accepted identity, manifest, provenance, integrity, freshness,
and consumability gates.

| Pipeline capability | Required source/evidence families | Minimum status | Failure classification |
|---|---|---|---|
| Universe classification | `SupportedUniverseEntry` | validated active/configured row | `blocked_invalid_universe_entry` (proposed) |
| Acquisition accepted | requested acquired family + manifest/provenance | package present; validation accepted; local use allowed | existing rejected/invalid manifest or provider/acquisition reason |
| Descriptive analysis | `company_profile` + G | accepted, consumable, sufficiently fresh for descriptive use | `descriptive_only` when valid; otherwise blocked by specific gate |
| Fundamental analysis | `fundamental_facts` + G | required facts complete under approved mapping | `missing_fundamental_evidence` or source-support reason |
| Setup detection | `setup_detection_input`, including required price/market input + G | accepted, fresh, complete, consumable | `missing_setup_or_price_context` |
| Analysis ready | approved analytical family set + G | non-descriptive analytical input accepted and consumable | `partial_analysis` or specific blocker |
| Recommendation Review ready | fundamentals + setup/price/market + strategy-required context + G | complete, fresh, provenanced, `recommendation_eligible` | `blocked_recommendation_review` (proposed aggregate) plus canonical reasons |
| Actionable output | future actionable Recommendation Review + future risk/context requirements + G | future approved contract only | `not_actionable`; currently always false |
| Portfolio Review ready | approved Recommendation Review input + `portfolio_context` + G | context present, aligned, fresh, non-blocked | existing portfolio-context blocker |
| Decision Engine handoff | future actionable prerequisite + valid Portfolio Review + `decision_engine_handoff_input` + G | accepted handoff contract and complete lineage | `not_de_ready`; preserve ME-DE01 blocked reason |

Alternative source families may satisfy a requirement only when the requirement
profile explicitly approves them. There is no implicit fallback.

## 9. Future Ticker Onboarding Contract

A future ticker is onboarded without runtime code changes:

1. Add the instrument to approved universe data/config under its versioned
   universe contract.
2. Validate ticker, market, asset type, activity, identity, and row provenance.
3. Select a generic `coverage_profile_id` from configured capabilities.
4. Resolve required source families from that profile.
5. Run acquisition only through approved source adapters and modes.
6. Validate snapshot manifest, identity, integrity, provenance, source-use
   constraints, and freshness.
7. Classify each source-family requirement and preserve every blocker.
8. Permit cached-source consumption only when the applicable snapshot is
   accepted and consumable.
9. Classify analysis readiness from approved evidence, not source presence.
10. Permit Recommendation Review evaluation only when the current readiness
    contract says it is eligible.
11. Permit Portfolio Review and handoff preparation only when their independent
    contracts and context requirements pass.
12. Permit Decision Engine evaluation only through an approved handoff.

Adding, removing, or renaming a ticker in universe data must not require
changing acquisition, validation, coverage, readiness, or handoff control
flow.

If an instrument needs a different source path, the difference must be
expressed as generic market/asset/source capabilities and a versioned
requirement profile. A one-row profile created only to disguise a ticker
exception is prohibited.

## 10. ME-RUN28 as a Regression Case

ME-RUN28 examples are regression fixtures for generic classes only.

| Example class | ME-RUN28 examples | Generic meaning | Expected contract outcome |
|---|---|---|---|
| Acquisition and validation accepted; direct source path remains descriptive | NVDA, AMD, ASML | Valid `company_profile` coverage is insufficient for analytical or action readiness | `descriptive_only`; not actionable; not DE-ready |
| Current acquisition unsupported; separate existing analytical cache permits partial downstream processing | AVGO, CLS, VRT, COST, META, MSFT, TSM, CRDO, IREN | Acquisition coverage and cached-source origin are separate; partial analytical evidence must preserve gaps | `partial_analysis` or blocked; not actionable; not DE-ready |
| Required cached snapshot unavailable | AAPL, GOOGL, AMZN, MU | Required cached source does not exist for the requested path | `missing_snapshot` / unavailable / blocked |

These names may be used in documentation and regression fixtures. They must
not appear in implementation branches, capability rules, fallback logic, or
readiness overrides.

ME-RUN28 also establishes a semantic regression:

```text
Recommendation Review stage status = completed
does not imply
recommendation_review_eligible = true
```

The generic model must preserve both facts without silently upgrading
readiness.

## 11. Guardrails for Future Implementation

Future implementation must enforce:

* no `if ticker == ...` runtime business logic;
* no `ticker in [...]` runtime business logic;
* no equivalent `symbol` or issuer-name branches;
* ticker lists only in config, fixtures, test data, docs, or run artifacts;
* runtime interpretation only of versioned generic contract fields;
* requirement profiles selected by governed capabilities, never ticker;
* all new source families routed through an approved contract and validator;
* fail-closed behavior for missing provenance, identity, freshness,
  consumability, completeness, or required source family;
* no silent fallback to `company_profile` or another family;
* no manual data patch to make one instrument pass;
* no source-family presence interpreted as analytical completeness;
* no Recommendation Review completion interpreted as actionability;
* no handoff readiness interpreted as a Decision Engine decision;
* deterministic row preservation and per-entry blockers.

Recommended PR grep:

```bash
rg -n 'ticker\s*==|ticker\s+in\s+\[|symbol\s*==|symbol\s+in\s+\[' src tests scripts docs/market_engine
```

Interpretation rules:

* runtime hits in `src/` or `scripts/` require review;
* empty-string identity validation is not ticker-specific business logic;
* tests may use named fixture tickers and equality assertions;
* test helpers may select fixture rows by ticker;
* docs may contain prohibited examples and regression examples;
* any new runtime hit must prove it performs identity validation or generic
  lookup only and does not select business behavior;
* ME-SA13 should add a changed-runtime scope check so no new ticker-specific
  control-flow hit is accepted silently.

ME-SA12 baseline grep result:

```text
src/market_engine: no hits
scripts: no source-text hits
src/market_scanner: three existing empty-string identity validation hits
tests: fixture assertions, fixture lookup, and one test-only zero-value branch
docs/market_engine: no pre-ME-SA12 hits
```

The ME-SA12 contract itself intentionally adds documentation hits for
prohibited examples. No runtime hit is added.

## 12. Acceptance Criteria

ME-SA12 is accepted when:

* this contract exists under the existing `source_support` documentation
  domain;
* the model is generic and future-ticker safe;
* ME-RUN28 is represented only as regression evidence;
* current source-support, manifest, readiness, Recommendation Review,
  Portfolio Review, and handoff vocabularies remain authoritative;
* new aggregate terms are explicitly marked proposed;
* `actionable` and `de_ready` remain false and unclaimable under current
  governance;
* backlog and roadmap identify a generic implementation follow-up;
* no runtime or test file changes;
* Market Engine and full test suites pass;
* `git diff --check` passes;
* governance grep is executed and interpreted;
* the PR confirms that no ticker-specific runtime logic was added.

## 13. Implementation Follow-Up

Recommended next sprint:

```text
ME-SA13 - Implement generic cached-source coverage classification model
```

ME-SA13 should implement
`market-engine-supported-universe-cached-source-coverage-v1` as a pure,
deterministic classifier over validated universe entries, generic coverage
profiles, and snapshot/manifest evidence.

It must not add provider access, broaden ticker coverage by special case,
activate reserved readiness levels, or change Decision Engine authority.

Expanded source acquisition may proceed only after the generic classifier can
represent supported, missing, invalid, stale, unprovenanced, non-consumable,
partial, and blocked outcomes without ticker-specific logic.
