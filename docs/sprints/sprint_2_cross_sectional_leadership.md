SPRINT 2 — CROSS-SECTIONAL LEADERSHIP
Trading System — Institutional Decision Engine
Context Layer Reconstruction Sprint
1. Executive Summary

Sprint 2 herbouwt de volledige Context Layer volgens institutionele momentum-principes.

De architecture audit heeft bevestigd dat de vorige Context Layer architecturaal fout was omdat ze:

benchmark-relative momentum gebruikte
pseudo-tradeability bevatte
implicit filtering uitvoerde
sector dependency gebruikte als blocking condition
context gebruikte als allocatie-gate

Dit veroorzaakte:

distributiecompressie
zwakke leadershipmodellering
premature edge destruction
upstream allocation leakage

Sprint 2 corrigeert deze problemen fundamenteel.

1.1 Grootste Architecturale Correctie
OUDE CONTEXT ARCHITECTUUR
benchmark-relative filtering layer
NIEUWE CONTEXT ARCHITECTUUR
cross-sectional leadership classification layer
1.2 Nieuwe Context Doctrine

Context bepaalt uitsluitend:

relative leadership

Niet:

tradeability

Zoals expliciet gedefinieerd in:

Technical Analysis v3
Functional Analysis v2
Decision Engine Design v2
2. Sprint Objective

Het doel van Sprint 2 is:

een institutionele cross-sectional leadership layer bouwen

die:

✅ momentum distributie modelleert
✅ leadership classificeert
✅ relatieve sterkte rankt

Maar NOOIT:

❌ allocatie bepaalt
❌ opportunities filtert
❌ tradeability simuleert

2.1 Strategische Doelstelling

De Context Layer moet evolueren van:

absolute threshold momentum

naar:

cross-sectional leadership modelling
3. Strategic Context

De vorige Context Layer gebruikte:

rs_20d > 0.25 = STRONG

Dit modelleerde momentum als:

outperforming benchmark

Maar institutioneel momentum betekent:

outperforming the opportunity universe

Dat is fundamenteel verschillend.

3.1 Grootste Contextfout

De vorige architectuur bevatte:

context_tradeable =
valid_setup AND strong_context

wat expliciet architecturaal verboden werd.

3.2 Nieuwe Context Filosofie

Context moet:

leadership classificeren

Niet:

kapitaalwaardigheid bepalen
4. Architectural Context

Sprint 2 implementeert de doctrine:

momentum = cross-sectional leadership

Niet:

benchmark outperformance
4.1 Nieuwe Context Boundary

Context mag:

✅ leadership classificeren
✅ momentum ranken
✅ relatieve sterkte modelleren

Maar NOOIT:

❌ tradeability bepalen
❌ BUY/SELL beïnvloeden
❌ filtering uitvoeren
❌ allocation readiness bepalen

5. Sprint Scope
5.1 Nieuwe RS Architectuur
Verplicht bouwen
rs_rank
rs_percentile
leadership_bucket
leadership_state
cross_sectional_score
5.2 Nieuwe Leadership Buckets
Verplicht classificeren
TOP_1
TOP_5
TOP_10
TOP_20
MID
WEAK
LAGGING
5.3 Nieuwe Context States
Verplicht classificeren
LEADING
STRONG
NEUTRAL
WEAK
LAGGING
5.4 Momentum Distribution Infrastructure
Verplicht bouwen
percentile distributions
universe-relative ranking
cohort-relative positioning
leadership persistence tracking
5.5 Leadership Persistence
Verplicht toevoegen
rolling leadership persistence
leadership duration
consistency scoring
6. Explicit Non-Scope

Sprint 2 mag NIET:

❌ context_tradeable bouwen
❌ BUY/SELL logic introduceren
❌ tradeability bepalen
❌ conviction bepalen
❌ opportunities filteren
❌ allocation logic toevoegen
❌ hard momentum invalidation introduceren

7. Required Inputs
Verplichte documentatie
Technical_Analysis_v3.md
Functional_Analysis_v2.md
Decision Engine Design v2
execution_roadmap_v2.md
Vereiste datasets
scanner universe
OHLCV datasets
benchmark data
optional sector data
8. Required Outputs
8.1 context_strength.csv
Verplicht schema
ticker
date
rs_score
rs_rank
rs_percentile
leadership_bucket
leadership_state
context_strength
context_reason
leadership_persistence
8.2 Verboden Velden

❌ context_tradeable
❌ BUY/SELL fields
❌ conviction
❌ allocation_priority
❌ final_action

9. Data Contracts
HARD RULE

Context mag uitsluitend:

leadership metadata

produceren.

HARD RULE

Context mag NOOIT:

allocation metadata

produceren.

10. Governance Rules
HARD RULE
Context = leadership classification only
HARD RULE
No context logic may determine tradeability.
HARD RULE
No benchmark-only momentum architecture allowed.
HARD RULE
Sector-relative strength may enrich but never block classification.
HARD RULE
Context may not collapse opportunity distribution excessively.
11. Forbidden Logic
11.1 Verboden Context Logic

❌ context_tradeable
❌ hard filtering
❌ BUY/SELL logic
❌ pseudo allocation
❌ conviction scoring
❌ hidden gating

11.2 Verboden Momentum Modellen

❌ benchmark-only RS
❌ hardcoded binary momentum
❌ sector dependency blocking
❌ fixed momentum invalidation

11.3 Verboden Cross-Layer Logic

Context mag NOOIT:

❌ validation overrulen
❌ portfolio interpreteren
❌ timing interpreteren
❌ fundamentals interpreteren

12. Technical Requirements
12.1 build_context_layer.py
Verplicht herwerken

Bestand:

scripts/core/build_context_layer.py
12.2 Verplichte Functionaliteit
cross-sectional ranking
percentile calculation
leadership buckets
persistence tracking
deterministic outputs
fail-fast handling
12.3 Sector Dependency Correction
Nieuwe Regel

Sector data:

✅ enrichment allowed
❌ dependency forbidden

12.4 CI Enforcement
Verplicht toevoegen
grep -R "context_tradeable" scripts/core/build_context_layer.py
grep -R "BUY" scripts/core/build_context_layer.py
grep -R "SELL" scripts/core/build_context_layer.py

Resultaat moet leeg zijn.

13. Functional Requirements
Context moet:

✅ leadership modelleren
✅ momentum distributie classificeren
✅ relatieve sterkte ranken

Context mag NIET:

❌ opportunities blokkeren
❌ kapitaalwaardigheid bepalen
❌ allocation readiness bepalen

14. Validation Requirements
Verplicht aantonen
RS percentile distribution
Leadership distribution
Cohort distribution
Persistence distribution
Benchmark vs cross-sectional comparison
15. Logging Requirements
Verplicht loggen
percentile distributions
leadership shifts
persistence shifts
removed legacy logic
sector-data warnings
16. CI / Enforcement Requirements
Verplicht
schema enforcement
forbidden-field detection
forbidden-keyword scanning
deterministic output validation
17. Acceptance Criteria
Sprint is PAS geslaagd wanneer:

✅ context_tradeable volledig verdwenen is
✅ benchmark-only momentum verwijderd is
✅ cross-sectional ranking actief is
✅ leadership buckets correct functioneren
✅ sector dependency niet blocking is
✅ Geen allocation leakage aanwezig is
✅ context_strength.csv schema correct is
✅ Logging aanwezig is
✅ CI checks slagen

18. Definition of Done
Verplicht

✅ Alle governance rules gerespecteerd
✅ Geen hidden filtering
✅ Geen hidden tradeability
✅ Context volledig deterministic
✅ Outputs reproduceerbaar
✅ Leadership distributions valide
✅ Logging aanwezig
✅ CI enforcement actief
✅ Technical Lead review geslaagd
✅ Functional Analyst review geslaagd
✅ Quant Analyst review geslaagd

19. Risks
Grootste Risico
legacy benchmark-relative momentum assumptions

Bijvoorbeeld:

fixed RS thresholds
hidden binary momentum logic
implicit filtering
sector dependency remnants
Verboden Reactie

❌ snel opnieuw hard filtering toevoegen

Correcte Reactie

✅ distributie observeren
✅ leadership classificatie stabiliseren
✅ allocatie downstream oplossen

20. Migration Notes

Na Sprint 2 zal waarschijnlijk:

de momentum distributie breder worden

Dat is EXPECTED.

En institutioneel correct.

Waarom?

Omdat de Context Layer niet langer:

benchmark-only filtering

gebruikt.

21. Final Sprint Doctrine
classify leadership
model momentum distribution
eliminate benchmark-only thinking
preserve opportunity distribution
prepare downstream allocation
POST-SPRINT-0 GOVERNANCE INHERITANCE

Status: FUTURE SPRINT PLAN — ACTIVE ONLY UNDER CERTIFIED GOVERNANCE

This sprint plan must inherit Sprint 0 certification:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Context work may classify relative strength and leadership only. It may not reintroduce `context_tradeable`, tradeability, conviction, allocation priority, hidden filtering, or final actions.
