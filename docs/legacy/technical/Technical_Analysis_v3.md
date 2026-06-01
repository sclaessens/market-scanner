TECHNICAL SPECIFICATION DOCUMENT (v3 — ARCHITECTURE CORRECTED)
Trading System — Institutional Decision Engine Architecture

POST-SPRINT-0 CERTIFICATION STATUS

Status: ACTIVE, GOVERNANCE-SYNCHRONIZED

Sprint 0 Governance Purification is certified COMPLETE. This document describes the active architecture only when read under the binding doctrine:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Authoritative references:

- AGENTS.md
- docs/archive/migration/sprint_0_governance_status.md
- docs/archive/audits/sprint_0_final_governance_audit.md

If any older wording in this document appears to imply upstream tradeability, hidden filtering, or allocation semantics outside Decision Engine, the authoritative governance documents prevail.

1. CORE TECHNICAL ARCHITECTURE (CORRECTED)
1.1 Fundamental Architectural Principle

Het systeem evolueert definitief van:

signal generator

naar:

institutional decision engine

De architectuur moet daarom:

classificatie scheiden van allocatie
vroege filtering minimaliseren
decision leakage vermijden
opportunity distributie behouden
probabilistische evaluatie mogelijk maken
1.2 Definitieve Layer Architecture
Scanner Layer         → detectie
Validation Layer      → technische structuurclassificatie
Context Layer         → relatieve sterkteclassificatie
Fundamental Layer     → kwaliteitsclassificatie
Watchlist Layer       → timingstatus
Portfolio Layer       → exposure & risicostatus
Decision Engine       → ENIGE allocatieautoriteit
Reporting Layer       → communicatie
1.3 Core Institutional Doctrine
CRUCIALE NIEUWE REGEL
classification upstream
allocation downstream
Betekenis

Vroege layers:

✅ classificeren
✅ structureren
✅ verrijken
✅ observeren

Maar:

❌ géén tradeability bepalen
❌ géén allocatiebeslissingen nemen
❌ géén kapitaalbeslissingen nemen

1.4 Decision Engine Authority
HARD RULE
Decision Engine = enige bron van waarheid

De Decision Engine is de enige layer die mag bepalen:

BUY
SELL
HOLD
TRIM
REMOVE
WAIT
conviction
allocation eligibility
capital priority
2. ARCHITECTURAL CORRECTIONS (CRITICAL)
2.1 Validation Layer Redefinition
OUDE DEFINITIE (FOUT)
VALID_SETUP = tradeable kwaliteit

Probleem:

Validation werd:

pre-decision filtering
trade gating
execution gating

Dit veroorzaakte:

opportunity collapse
edge truncation
over-filtering
NIEUWE DEFINITIE (CORRECT)
VALID_SETUP = technische structurele coherentie

Validation bepaalt uitsluitend:

“Is deze setup technisch coherent?”

Niet:

“Verdient deze setup kapitaal?”
2.2 Context Layer Redefinition
OUDE DEFINITIE (FOUT)
context_tradeable =
valid_setup AND strong_context

Probleem:

Context werd:

classificatie
filtering

Dit schendt separation of concerns.

NIEUWE DEFINITIE (CORRECT)

Context Layer doet uitsluitend:

relative strength classification

Context bepaalt:

leadership
relatieve sterkte
momentumpositie
cross-sectional ranking

Maar NOOIT:

tradeability
entry eligibility
allocatie
2.3 Tradeability Redefinition
BELANGRIJKSTE ARCHITECTURALE CORRECTIE

Tradeability wordt volledig verwijderd uit:

❌ Validation Layer
❌ Context Layer

NIEUWE DEFINITIE
Tradeability = capital allocation readiness

Tradeability hoort EXCLUSIEF thuis in:

Decision Engine
3. DATA ARCHITECTURE (CORRECTED)
3.1 validation_layer.csv (SIMPLIFIED)
Doel

Technische structuurclassificatie.

Schema
ticker
date
structure_state
structure_reason
setup_type
valid_setup
validation_reason
VERWIJDERD

❌ tradeable_setup
❌ allocation fields
❌ final_action

Waarom

Tradeability is geen validatieconcept.

3.2 context_strength.csv (SIMPLIFIED)
Doel

Momentum- en leadershipclassificatie.

Schema
ticker
date
rs_score
rs_percentile
rs_rank
rs_vs_market
rs_vs_sector
context_strength
context_reason
leadership_state
VERWIJDERD

❌ context_tradeable
❌ context_tradeable_reason

3.3 decision_output.csv (EXPANDED)
Nieuwe centrale autoriteit
final_decisions.csv
Schema
ticker
date
source_layer
setup_type
final_action
tradeability
conviction
allocation_priority
validation_state
context_strength
leadership_state
timing_state
portfolio_state
execution_style
decision_reason
entry
stop
target
rr
trigger_price
regime
close
ma20
ma50
high_20d
4. VALIDATION LAYER (REDEFINED)
4.1 Core Responsibility

Validation bepaalt uitsluitend:

technische coherentie
4.2 Wat Validation MAG doen

✅ structure checks
✅ data integrity checks
✅ structure state classification
✅ broken structure detecteren
✅ missing-data detection
✅ descriptive metadata

4.3 Wat Validation NIET MAG doen

❌ context interpreteren
❌ momentum evalueren
❌ entry chasing beoordelen
❌ conviction bepalen
❌ tradeability bepalen
❌ allocation beïnvloeden

4.4 VALID_SETUP (NIEUWE DEFINITIE)
VALID_SETUP = (
    structure_valid
    AND trend_structure_valid
    AND data_integrity_valid
)
4.5 ENTRY QUALITY (ARCHITECTURAL CORRECTION)
OUDE FOUT

Entry quality werd gebruikt als:

hard validation gate

Dat is architecturaal fout.

NIEUWE ROL

Entry quality wordt:

descriptive timing/structure metadata

Dus:

✅ loggen
✅ analyseren
✅ score verrijken

Maar:

❌ géén VALID_SETUP blokkeren
❌ géén execution instruction
❌ géén allocation gate

4.6 Validation Philosophy

Validation moet:

distributie behouden

Niet:

edge vroegtijdig elimineren
5. CONTEXT LAYER (REDEFINED)
5.1 Core Responsibility

Context bepaalt uitsluitend:

relative leadership
5.2 Institutionele Momentumdefinitie
OUDE DEFINITIE (FOUT)
momentum = outperforming benchmark
NIEUWE DEFINITIE (CORRECT)
momentum = cross-sectional leadership
5.3 Nieuwe RS Architectuur

Context moet evolueren van:

absolute threshold classification

naar:

cross-sectional ranking model
5.4 Nieuwe Core Metrics
Vereist
rs_rank
rangorde binnen universum
rs_percentile
percentiel binnen distributie
leadership_bucket

Bijvoorbeeld:

TOP_1
TOP_5
TOP_10
TOP_20
MID
WEAK
5.5 Context Classification (NIEUW)
OUDE FOUT
rs_20d > 0.25 = STRONG

Dit is institutioneel te simplistisch.

NIEUWE BENADERING
if rs_percentile >= 90:
    LEADING

elif rs_percentile >= 75:
    STRONG

elif rs_percentile >= 40:
    NEUTRAL

else:
    WEAK
5.6 Sector Dependency Correction
OUDE FOUT

LEADING afhankelijk van sectordata.

Probleem:

sectordata incomplete
→ classificatie instabiel

NIEUWE REGEL

Sector-relative strength:

✅ enrichment
❌ nooit dependency

6. FUNDAMENTAL LAYER
6.1 Core Responsibility

Fundamentals bepalen:

kwaliteit

Niet:

timing
6.2 HARD RULE

Fundamentals mogen NOOIT:

❌ entries triggeren
❌ validatie blokkeren
❌ context overrulen

7. WATCHLIST LAYER
7.1 Nieuwe Rol

Watchlist beheert:

timing state

Niet:

allocatie
7.2 Watchlist Mag

✅ READY
✅ EARLY
✅ EXTENDED
✅ PULLBACK
✅ BREAKOUT_PENDING

Maar:

❌ BUY
❌ SELL
❌ REMOVE

8. DECISION ENGINE (MAJOR REDEFINITION)
8.1 Nieuwe Institutionele Rol

De Decision Engine wordt:

centrale allocatieautoriteit
8.2 Nieuwe Verantwoordelijkheden
Decision Engine bepaalt:
tradeability
conviction
allocation priority
execution aggressiveness
portfolio interaction
conflict resolution
exposure balancing
8.3 Nieuwe Evaluatievolgorde
1. Portfolio state
2. Validation state
3. Context leadership
4. Fundamental quality
5. Watchlist timing
6. Risk constraints
7. Exposure constraints
8. Allocation eligibility
9. Conviction
10. Final action
8.4 Decision Philosophy

Decision Engine werkt:

probabilistisch

Niet:

binair
8.5 Nieuwe Conviction Architectuur

Conviction wordt downstream bepaald op basis van:

context leadership
setup quality
fundamental quality
portfolio state
market regime
execution quality
9. GOVERNANCE RULES (UPDATED)
9.1 Hard Rules That Stay

✅ Decision Engine authority
✅ deterministic output
✅ one decision per ticker
✅ separation of concerns
✅ no hidden decisions
✅ no cross-layer contamination

9.2 New Hard Rules
NIEUW
classification upstream
allocation downstream
NIEUW
No layer except Decision Engine may determine tradeability.
NIEUW
Validation may not reduce opportunity distribution excessively.
10. IMPLEMENTATION IMPACT
10.1 Wat Verdwijnt

❌ tradeable_setup
❌ context_tradeable
❌ entry-quality gating
❌ hard RS thresholds

10.2 Wat Toegevoegd Wordt

✅ cross-sectional RS
✅ percentile ranking
✅ leadership buckets
✅ downstream tradeability
✅ probabilistic decisioning

10.3 Wat Ongewijzigd Blijft

✅ scanner core
✅ deterministic pipeline
✅ logging discipline
✅ fail-fast infrastructure
✅ layer separation

11. STRATEGIC SYSTEM INSIGHT

De grootste architecturale les:

institutionele systemen filteren niet vroeg
institutionele systemen alloceren laat
12. FINAL TECHNICAL CONCLUSION

Technical Analysis v2 probeerde:

edge upstream op te lossen

Dat was architecturaal fout.

Technical Analysis v3 corrigeert dit fundamenteel:

Van:

filter-first architecture

Naar:

classification-first architecture

De nieuwe architectuur:

✅ behoudt distributie
✅ voorkomt premature edge destruction
✅ respecteert institutionele governance
✅ centraliseert allocatiebeslissingen correct
✅ maakt probabilistische decisioning mogelijk
✅ elimineert artificial bottlenecks

13. FINAL ARCHITECTURAL DOCTRINE
scanner detects
validation classifies structure
context classifies leadership
fundamentals classify quality
watchlist tracks timing
portfolio tracks exposure
decision engine allocates capital
reporting communicates outcomes
