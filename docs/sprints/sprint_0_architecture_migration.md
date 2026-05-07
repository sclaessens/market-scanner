SPRINT 0 — ARCHITECTURE MIGRATION & STABILISATION
Trading System — Institutional Decision Engine
Architecture-Corrected Migration Sprint
1. Executive Summary

Deze sprint vormt de belangrijkste architecturale migratiesprint van het volledige project.

De architecture audit heeft bevestigd dat de vorige architectuur fundamenteel verkeerde verantwoordelijkheden bevatte:

validation werd gebruikt als filtering engine
context werd gebruikt als pseudo-tradeability layer
upstream layers namen impliciete allocatiebeslissingen
opportunity distributie werd vroegtijdig vernietigd
de Decision Engine kreeg een gecollapste opportunity space

Deze sprint corrigeert deze problemen fundamenteel.

1.1 Grootste Architecturale Shift
OUDE ARCHITECTUUR
filter-first architecture
NIEUWE ARCHITECTUUR
classification-first architecture
1.2 Nieuwe Core Doctrine
CRUCIALE NIEUWE REGEL
classification upstream
allocation downstream

Zoals gedefinieerd in:

Technical Analysis v3
Functional Analysis v2
Decision Engine Design v2
Execution Roadmap v2
2. Sprint Objective

Het doel van Sprint 0 is:

de bestaande codebase compatibel maken met de nieuwe institutionele architectuur

zonder:

nieuwe edge logic
nieuwe filters
nieuwe decisioning
nieuwe optimization layers

toe te voegen.

2.1 Strategische Prioriteit

Deze sprint is:

CRITISCH

Geen enkele volgende sprint mag starten vóór Sprint 0 volledig afgerond is.

3. Strategic Context

De vorige architectuur creëerde artificiële bottlenecks:

Validation Layer

Werd gebruikt als:

pseudo-decision engine
Context Layer

Werd gebruikt als:

context_tradeable =
valid_setup AND strong_context

wat expliciet architecturaal afgekeurd werd.

Gevolg

Historical reconstruction toonde:

STRONG = 46
WEAK = 1

wat betekent dat:

classificatie aanwezig was
maar filtering architecturaal fout geplaatst was
4. Architectural Context

Deze sprint implementeert de nieuwe institutionele doctrine:

upstream layers classificeren
Decision Engine alloceert

Zoals expliciet gedefinieerd in de nieuwe functionele en technische architectuur.

5. Sprint Scope
5.1 Governance Cleanup
Verplicht verwijderen
tradeable_setup

❌ verboden buiten Decision Engine

context_tradeable

❌ verboden buiten Decision Engine

implicit allocation logic

Bijvoorbeeld:

hidden BUY logic
hidden conviction logic
hidden gating
pseudo tradeability
5.2 Validation Cleanup

Validation moet gereduceerd worden tot:

pure structure classification
Verplicht verwijderen

❌ entry-quality gating
❌ extension invalidation
❌ allocation simulation
❌ execution gating

VALID_SETUP nieuwe definitie
VALID_SETUP = (
    structure_valid
    AND trend_structure_valid
    AND data_integrity_valid
)

Zoals gedefinieerd in Technical Analysis v3.

5.3 Context Cleanup

Context moet gereduceerd worden tot:

pure leadership classification
Verplicht verwijderen

❌ context_tradeable
❌ hard filtering
❌ benchmark-only momentum
❌ allocation logic

Nieuwe momentum doctrine

Van:

benchmark-relative momentum

Naar:

cross-sectional leadership

5.4 Architecture Enforcement
Verplicht toevoegen
CI enforcement
grep governance
schema validation
layer-boundary validation
6. Explicit Non-Scope

Deze sprint mag NIET:

❌ nieuwe filters toevoegen
❌ thresholds optimaliseren
❌ nieuwe scoring bouwen
❌ conviction engine bouwen
❌ tradeability engine bouwen
❌ BUY/SELL logic introduceren
❌ probabilistische decisioning implementeren

Dat hoort pas thuis in Sprint 6.

7. Required Inputs
Architectuurdocumenten
Verplicht
Technical_Analysis_v3.md
Functional_Analysis_v2.md
decision_engine_design_v2.md
execution_roadmap_v2.md
Bestaande codebase
Verplicht reviewen
scripts/core/
scripts/watchlist/
scripts/portfolio/
scripts/reporting/
8. Required Outputs
8.1 validation_layer.csv
Nieuwe verplichte structuur
ticker
date
valid_setup
validation_reason
setup_type
structure_state
Verboden velden

❌ tradeable_setup
❌ conviction
❌ entry_quality_gate

8.2 context_strength.csv
Nieuwe verplichte structuur
ticker
date
rs_score
rs_percentile
rs_rank
context_strength
leadership_state
context_reason
Verboden velden

❌ context_tradeable
❌ allocation fields
❌ BUY logic

9. Data Contracts
HARD RULE

Geen enkele upstream layer mag:

tradeability
allocation eligibility
conviction
final actions

bevatten.

10. Governance Rules
HARD RULE
Decision Engine = enige allocatieautoriteit

HARD RULE
No upstream layer may determine tradeability.

HARD RULE
No classification layer may collapse opportunity distribution excessively.

11. Forbidden Logic
11.1 Verboden Buiten Decision Engine

❌ BUY
❌ SELL
❌ REMOVE
❌ TRIM
❌ tradeability
❌ conviction
❌ allocation eligibility

11.2 Verboden Validation Logic

❌ entry invalidation via extension
❌ execution-quality gating
❌ hidden conviction logic
❌ pseudo tradeability

11.3 Verboden Context Logic

❌ benchmark-only momentum
❌ context gating
❌ sector dependency blocking
❌ allocation filtering

12. Technical Requirements
12.1 CI Enforcement
Verplicht toevoegen
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py

Resultaat moet leeg zijn.

12.2 Logging Requirements
Validation distributions
Context distributions
Removed gating logs
Migration warnings
12.3 Schema Validation

Pipeline moet falen wanneer:

verboden velden aanwezig zijn
allocation leakage gedetecteerd wordt
schema contracts geschonden worden
13. Functional Requirements
Validation moet:

✅ structuur classificeren
✅ technische coherentie bepalen

Maar NOOIT:

❌ kapitaalbeslissingen simuleren

Context moet:

✅ leadership classificeren
✅ momentum distributie modelleren

Maar NOOIT:

❌ opportunities blokkeren

14. Validation Requirements
Verplicht aantonen
VALID_SETUP distributie

Voor en na migratie.

Context distributie

Voor en na migratie.

Allocation leakage audit

Moet leeg zijn.

15. Logging Requirements
Verplicht loggen
Removed fields
Removed logic
Legacy gating remnants
Distribution shifts
16. CI / Enforcement Requirements
Verplicht
Layer-boundary enforcement
Forbidden keyword scanning
Schema enforcement
Allocation leakage detection
17. Acceptance Criteria
Sprint is PAS geslaagd wanneer:

✅ tradeable_setup volledig verdwenen is
✅ context_tradeable volledig verdwenen is
✅ allocation leakage verdwenen is
✅ VALID_SETUP enkel structuur betekent
✅ Context enkel leadership betekent
✅ Pipeline end-to-end werkt
✅ CI enforcement actief is
✅ Geen hidden BUY logic meer bestaat
✅ Geen hidden filtering meer bestaat

18. Definition of Done
Verplicht

✅ Alle governance rules gerespecteerd
✅ Geen architectuurconflicten
✅ Geen hidden legacy logic
✅ Alle outputs voldoen aan nieuwe schemas
✅ Alle forbidden logic verwijderd
✅ Pipeline reproduceerbaar
✅ Logging aanwezig
✅ CI checks slagen
✅ Technical Lead review geslaagd
✅ Functional Analyst review geslaagd

19. Risks
Grootste risico
verborgen oude architectuurassumpties in de codebase

Bijvoorbeeld:

hidden gating
implicit filtering
hidden conviction
legacy thresholds
pseudo-tradeability
Verboden reactie

❌ snel nieuwe filtering toevoegen

Correcte reactie

✅ architectuur eerst zuiver maken
✅ classificatie stabiliseren
✅ allocatie pas later bouwen

20. Migration Notes

Tijdens deze sprint zal waarschijnlijk:

meer noise zichtbaar worden

Dat is EXPECTED.

En architecturaal correct.

Waarom?

Omdat de vorige architectuur:

opportunity distributie kunstmatig onderdrukte
21. Final Sprint Doctrine
remove hidden allocation
remove hidden filtering
preserve opportunity distribution
enforce classification-only upstream
prepare institutional allocation downstream