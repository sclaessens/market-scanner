Sprint 0 — Governance Purification

Institutional Architecture Migration — Developer Specification v1

1. Sprint Metadata
Field	Value
Sprint	Sprint 0
Sprint Name	Governance Purification
Priority	CRITICAL
Classification	Architecture Migration
Owner	Technical Lead
Status	ACTIVE
Blocking Future Sprints	YES
2. Sprint Objective

Het doel van deze sprint is:

de volledige codebase zuiveren van filtering-first governance

zodat de nieuwe institutionele architectuur correct enforced wordt.

3. Core Architecture Doctrine
HARD RULE
classification upstream
allocation downstream
HARD RULE
Decision Engine = enige allocatieautoriteit
HARD RULE

Geen enkele upstream layer mag:

tradeability bepalen
opportunities elimineren
conviction bepalen
allocation beslissingen nemen
urgency bepalen
execution logica uitvoeren
4. Sprint Scope

Deze sprint focust EXCLUSIEF op:

✅ governance purification
✅ layer isolation
✅ schema purification
✅ allocation centralisation
✅ filtering elimination

OUT OF SCOPE

❌ nieuwe features
❌ performance optimization
❌ nieuwe indicators
❌ AI logic
❌ ranking improvements
❌ nieuwe strategieën
❌ portfolio optimization
❌ reporting redesign

5. Target Architecture

De codebase moet migreren naar:

SCANNER
↓
pure discovery

VALIDATION
↓
pure structure classification

CONTEXT
↓
pure leadership classification

TIMING
↓
pure timing-state classification

PORTFOLIO
↓
pure exposure-state modelling

DECISION ENGINE
↓
ONLY allocation authority

REPORTING
↓
pure presentation
6. Critical Governance Problems Identified

De audit detecteerde:

🚨 allocation leakage
🚨 hidden filtering
🚨 layer contamination
🚨 duplicated logic
🚨 execution semantics upstream

Belangrijkste verboden concepten
REMOVE COMPLETELY
tradeable_setup
context_tradeable
context_tradeable_reason
invalid_rr
weak_trend
execution gating
conviction gating
grade gating
upstream urgency
upstream BUY logic
upstream SELL logic
7. Core Migration Goals
GOAL 1

Upstream layers mogen niets meer blokkeren.

GOAL 2

Decision Engine wordt enige allocation authority.

GOAL 3

Alle layers krijgen pure responsibilities.

GOAL 4

Alle hidden filtering wordt verwijderd.

8. Required Refactors
8.1 Scanner Refactor
File
scripts/core/scanner.py
REMOVE
Execution elimination

Verwijderen:

if rr < MIN_RR:
    return None
Regime elimination

Verwijderen:

if regime == "BEARISH":
    return None
Opportunity elimination

Scanner mag GEEN setups meer verwijderen op basis van:

RR
conviction
extension
breakout strength
regime
grade
execution viability
REMOVE
Grade-driven execution assumptions
grade
setup_grade

mogen NIET meer gebruikt worden als execution filters.

NEW RESPONSIBILITY

Scanner moet enkel:

✅ opportunities detecteren
✅ structure herkennen
✅ state labels genereren

Scanner mag NIET:

❌ alloceren
❌ executeren
❌ blokkeren
❌ reduceren

8.2 Validation Layer Refactor
File
scripts/core/build_validation_layer.py
REMOVE COMPLETELY
tradeable_setup
REMOVE
RR filtering
rr_invalid
Trend filtering
weak_trend
Execution gating

Verwijderen:

extension invalidation
breakout invalidation
volume invalidation
distance invalidation
REMOVE OUTPUT SEMANTICS
valid_setup

mag niet langer execution semantics bevatten.

NEW RESPONSIBILITY

Validation moet enkel:

✅ structure classification
✅ schema validation
✅ pattern classification
✅ market structure labeling

doen.

NIEUWE OUTPUTVOORBEELDEN

Toegelaten:

structure_type
trend_state
compression_state
breakout_state
volatility_state

Verboden:

tradeable
valid entry
executable
8.3 Context Layer Refactor
File
scripts/core/build_context_layer.py
REMOVE COMPLETELY
context_tradeable
context_tradeable_reason
REMOVE

Implicit execution semantics:

STRONG = tradeable
WEAK = reject
NEW RESPONSIBILITY

Context layer mag enkel:

✅ leadership classification
✅ participation classification
✅ sector leadership
✅ relative strength state

doen.

8.4 Decision Engine Refactor
File
scripts/core/decision_engine.py
GOAL

Decision Engine wordt:

✅ enige allocatieautoriteit

KEEP

Toegelaten:

BUY
SELL
WAIT
HOLD
TRIM
ADD

ALLEEN hier.

REMOVE
Hardcoded validation ideology

Verwijderen:

Bullish regime presteert zwak in validation
REMOVE
Embedded strategy assumptions

Verwijderen:

VCP elimination
hardcoded regime rejection
scanner ideology duplication
validation ideology duplication
NEW RESPONSIBILITY

Decision Engine beslist allocation:

OP BASIS VAN:

classification outputs
portfolio state
timing state
context state
8.5 Portfolio Refactor
Files
scripts/portfolio/evaluate_positions.py
scripts/portfolio/build_portfolio.py
REMOVE

Portfolio mag GEEN allocation decisions meer nemen:

SELL
TRIM
HOLD
NEW RESPONSIBILITY

Portfolio moet enkel modelleren:

✅ exposure_state
✅ drawdown_state
✅ risk_state
✅ persistence_state
✅ concentration_state

Portfolio output mag NIET:

❌ allocation semantics bevatten

8.6 Reporting Refactor
File
scripts/reporting/build_telegram_summary.py
REMOVE

Reporting mag GEEN:

urgency logic
execution interpretation
allocation prioritisation

meer bevatten.

Reporting mag enkel:

✅ formatteren
✅ presenteren
✅ structureren

9. Schema Purification
REMOVE FROM ALL CSV OUTPUTS
Forbidden Fields
tradeable_setup
context_tradeable
context_tradeable_reason
urgency
candidate_status
execution_state
conviction
10. Duplicate Logic Elimination

De volgende logica mag nog maar op ÉÉN plaats bestaan:

Logic	Owner
Allocation	Decision Engine
RR logic	Decision Engine
Regime allocation	Decision Engine
Execution timing	Timing Layer (later sprint)
Portfolio allocation	Decision Engine
11. Hard Constraints
DO NOT

❌ introduceren nieuwe features
❌ nieuwe filtering
❌ nieuwe rejection logic
❌ nieuwe execution shortcuts

DO

✅ preserve data richness
✅ preserve classifications
✅ preserve opportunities
✅ centralise allocation

12. Required Deliverables

Developer moet opleveren:

Updated Files
scanner.py
build_validation_layer.py
build_context_layer.py
decision_engine.py
evaluate_positions.py
build_telegram_summary.py
New Governance-Clean Schemas

Nieuwe CSV schemas zonder allocation leakage.

Migration Notes

Document:

docs/audits/sprint_0_governance_purification_findings.md

met:

removed governance violations
remaining risks
migration assumptions
unresolved dependencies
13. Mandatory Validation

Developer MOET aantonen dat:

Validation

geen:

tradeable_setup

meer bevat.

Context

geen:

context_tradeable

meer bevat.

Scanner

geen:

return None

meer gebruikt voor execution filtering.

Portfolio

geen allocation decisions meer bevat.

Reporting

geen execution interpretation meer bevat.

14. Required Audit Checks

Developer MOET grep-checks uitvoeren op:

tradeable
BUY NOW
SET LIMIT BUY
SET STOP BUY
invalid_rr
weak_trend
extended
conviction
urgency

en aantonen waar deze nog bestaan.

15. Sprint Completion Criteria

Sprint 0 is PAS klaar wanneer:

✅ upstream geen allocation meer doet
✅ upstream geen opportunities meer elimineert
✅ Decision Engine enige allocation authority is
✅ schemas governance-clean zijn
✅ layer boundaries zuiver zijn
✅ architecture drift verwijderd is

16. Final Technical Lead Directive

Deze sprint is:

🚨 NIET optioneel

De volledige institutionele architectuur hangt af van deze purification.

Nieuwe features ontwikkelen vóór deze migration afgerond is:

zal de architecture debt exponentieel vergroten.