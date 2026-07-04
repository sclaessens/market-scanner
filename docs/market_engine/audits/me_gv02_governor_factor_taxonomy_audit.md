# ME-GV02 - Governor Factor Taxonomy Audit

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: DOCS-ONLY AUDIT

## Audit summary

ME-GV02 defines the Governor factor taxonomy and evidence requirements required between ME-GV01 and the future ME-GV03 runtime scaffold.

The sprint is documentation-only and introduces no runtime behavior or scoring.

## Files added

```text
docs/market_engine/governor/me_gv02_governor_factor_taxonomy_and_evidence_requirements.md
docs/market_engine/backlog/me_gv02_governor_factor_taxonomy_backlog_entry.md
docs/market_engine/roadmap/me_gv02_governor_factor_taxonomy_roadmap_entry.md
docs/market_engine/audits/me_gv02_governor_factor_taxonomy_audit.md
```

## Placement check

PASS. ME-GV02 follows the approved sequence:

```text
ME-RUN29 -> ME-GV01 -> ME-GV02 -> ME-GV03 -> ME-GV04
```

ME-GV02 does not start runtime implementation or numeric scoring.

## Factor-family check

PASS. The contract defines nine canonical factor families:

```text
fundamentals
growth
valuation
trend
momentum
risk
technical_setup
portfolio_fit
data_confidence
```

These match the factor families reserved by ME-GV01.

## Factor-state check

PASS. The contract defines seven canonical factor states:

```text
not_started
blocked
unavailable
insufficient_evidence
partial
qualitative_only
evaluable
```

`evaluable` is explicitly defined as evidence sufficiency only. It does not imply investment quality, score, recommendation, or actionability.

## Evidence-gate check

PASS. The contract requires applicable global evidence gates before a factor may become partial, qualitative-only, or evaluable:

* approved contract identity;
* valid Refinery/staging evidence;
* source-family support or explicit approved fallback;
* manifest validity;
* payload presence;
* provenance;
* freshness;
* consumability;
* structural validity;
* deterministic evidence references.

## Missing-data check

PASS. Missing evidence remains missing and must not be converted to zero.

The contract distinguishes:

```text
factor evidence presence
factor evidence sufficiency
factor evaluability
factor scoring eligibility
```

These concepts may not be collapsed.

## Company-profile boundary check

PASS. Company-profile-only evidence is limited to `qualitative_only` at most for relevant factors and cannot create numeric factor evaluation.

## Portfolio-fit boundary check

PASS. `portfolio_fit` remains blocked without an approved portfolio-context contract:

```text
blocked_missing_approved_portfolio_context
```

No allocation, target weight, position sizing, order generation, or mutation authority is introduced.

## Data-confidence boundary check

PASS. Data confidence evaluates evidence trust and cannot upgrade absent investment evidence.

High confidence in a limited evidence set remains limited evidence.

## Conflict-handling check

PASS. Conflicting evidence must remain explicit and may downgrade a factor. ME-GV02 does not authorize silently averaging conflict away or numerically scoring conflict before ME-GV04 defines such behavior.

## Scoring boundary check

PASS. ME-GV02 does not define:

* numeric factor scales;
* weights;
* weighted aggregation;
* score bands;
* normalization;
* ranking;
* confidence weighting;
* missing-factor imputation.

These remain deferred to ME-GV04.

## Recommendation boundary check

PASS. Factor states do not imply recommendation states.

Recommendation-state mapping remains deferred to ME-GV05.

## Buy-zone boundary check

PASS. No factor definition authorizes entry price, buy-under price, breakout trigger, exceptional buy zone, stop loss, target price, or position-management instruction.

These remain deferred to ME-GV06 or later explicit contracts.

## Scope check

PASS. No runtime files, tests, source acquisition jobs, staging validators, classifiers, Analyzer modules, Recommendation Review modules, Portfolio Review modules, Dispatch Station modules, delivery modules, UI code, scheduler code, broker code, or artifact generators are changed.

## Authority boundary check

PASS. ME-GV02 does not authorize:

* provider calls;
* live market data;
* source acquisition;
* snapshot import;
* staging/classifier semantic changes;
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

## Test check

Not applicable. ME-GV02 is docs-only and changes no runtime or test files.

## Result

PASS. ME-GV02 defines the factor taxonomy and evidence requirements needed for ME-GV03 while preserving fail-closed behavior and keeping scoring deferred to ME-GV04.

## Next sprint

```text
ME-GV03 - Implement non-actionable Governor dry-run evaluation scaffold
```
