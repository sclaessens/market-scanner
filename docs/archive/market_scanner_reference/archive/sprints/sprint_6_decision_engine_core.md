SPRINT 6 — DECISION ENGINE CORE
Trading System — Institutional Decision Engine
Institutional Allocation Authority Sprint
1. Executive Summary

Sprint 6 bouwt de volledige institutionele Decision Engine Core.

Dit is de belangrijkste allocatiesprint van het volledige systeem.

Alle vorige sprints bouwden uitsluitend:

✅ classificatie
✅ observatie
✅ verrijking
✅ context

Sprint 6 is de eerste sprint die effectief:

kapitaalallocatie-intelligentie

mag introduceren.

1.1 Grootste Architecturale Shift
OUDE DECISION ENGINE
signal mapper
NIEUWE DECISION ENGINE
institutional allocation authority

Zoals expliciet gedefinieerd in:

Decision Engine Design v2
Functional Analysis v2
Technical Analysis v3
execution_roadmap_v2.md
1.2 Nieuwe Decision Doctrine

De Decision Engine bepaalt exclusief:

✅ tradeability
✅ conviction
✅ allocation priority
✅ execution aggressiveness
✅ capital allocation
✅ final actions

Geen enkele andere layer mag dit doen.

2. Sprint Objective

Het doel van Sprint 6 is:

de institutionele allocatie-engine bouwen

die:

✅ probabilistische evaluatie uitvoert
✅ opportunities relatief beoordeelt
✅ allocatieprioriteiten bepaalt
✅ portfolio pressure interpreteert
✅ conviction opbouwt
✅ finale acties bepaalt

2.1 Strategische Doelstelling

De Decision Engine moet evolueren van:

rule-based mapping

naar:

probabilistic institutional allocation
3. Strategic Context

Historisch werd de Decision Engine gevoed door:

reeds gefilterde opportunities
collapsed distributions
upstream pseudo tradeability
implicit allocation leakage

Daardoor werkte de engine eigenlijk als:

final signal formatter

In plaats van:

institutionele allocatieautoriteit

Sprint 6 corrigeert dit fundamenteel.

3.1 Grootste Architecturale Correctie

De vorige architectuur werkte:

upstream filtering
→ downstream mapping

De nieuwe architectuur werkt:

upstream classification
→ downstream interpretation
→ downstream allocation
3.2 Nieuwe Decision Filosofie

De Decision Engine moet:

opportunities interpreteren

Niet:

simpele regels uitvoeren
4. Architectural Context

Sprint 6 implementeert de doctrine:

classification upstream
allocation downstream

Zoals expliciet gedefinieerd in de architecture audit.

4.1 Nieuwe Decision Boundary

Decision Engine mag:

✅ tradeability bepalen
✅ conviction bepalen
✅ portfolio pressure interpreteren
✅ allocation priority bepalen
✅ execution aggressiveness bepalen
✅ final actions bepalen

4.2 Exclusieve Allocatiebevoegdheid
HARD RULE
Alle allocatie-intelligentie hoort exclusief thuis in de Decision Engine.
5. Sprint Scope
5.1 Tradeability Engine
Verplicht bouwen
tradeability_state
allocation_readiness
capital_eligibility
execution_feasibility
5.2 Conviction Engine
Verplicht bouwen
conviction_score
conviction_bucket
allocation_priority
opportunity_quality_score
5.3 Opportunity Ranking Engine
Verplicht bouwen
cross-opportunity ranking
relative prioritization
opportunity scoring
allocation queueing
5.4 Portfolio Interaction Engine
Verplicht bouwen
exposure balancing
concentration balancing
correlation balancing
liquidity-aware allocation
5.5 Execution Aggressiveness Engine
Verplicht bouwen
aggressive entry
passive entry
staged entry
reduced sizing
delayed execution
5.6 Final Action Engine
Verplicht bouwen
BUY
ACCUMULATE
PREPARE
WAIT
HOLD
TRIM
SELL
REMOVE
REVIEW
NO_ACTION
5.7 Decision Persistence Infrastructure
Verplicht toevoegen
conviction persistence
action persistence
probabilistic smoothing
escalation tracking
6. Explicit Non-Scope

Sprint 6 mag NIET:

❌ nieuwe structure classification bouwen
❌ nieuwe leadership classification bouwen
❌ nieuwe quality classification bouwen
❌ nieuwe timing classification bouwen
❌ nieuwe portfolio classification bouwen

Deze moeten reeds bestaan vóór Sprint 6 start.

7. Required Inputs
Verplichte documentatie
Decision Engine Design v2
Technical_Analysis_v3.md
Functional_Analysis_v2.md
execution_roadmap_v2.md
Vereiste datasets
validation_layer.csv
context_strength.csv
fundamental_profile.csv
watchlist_state.csv
portfolio_state.csv
8. Required Outputs
8.1 decision_output.csv
Verplicht schema
ticker
date
final_action
tradeability_state
conviction_score
conviction_bucket
allocation_priority
execution_style
decision_reason
validation_state
leadership_state
quality_state
timing_state
portfolio_state
8.2 Verplicht Toevoegen
decision_timestamp
decision_persistence
allocation_rank
execution_aggressiveness
9. Data Contracts
HARD RULE

Decision Engine mag ALLEEN lezen uit:

validation_layer.csv
context_strength.csv
fundamental_profile.csv
watchlist_state.csv
portfolio_state.csv
HARD RULE

Decision Engine mag de ENIGE producer zijn van:

allocation metadata
10. Governance Rules
HARD RULE
Decision Engine = exclusive allocation authority
HARD RULE
Tradeability may only exist inside the Decision Engine.
HARD RULE
Conviction may only exist inside the Decision Engine.
HARD RULE
All final actions must originate from the Decision Engine.
HARD RULE
Decision Engine must evaluate opportunities probabilistically, not binarily.
11. Forbidden Logic
11.1 Verboden Decision Behaviour

❌ deterministic hard filtering
❌ simplistic signal mapping
❌ binary-only decisioning
❌ hardcoded absolute gating

11.2 Verboden Architecture Behaviour

❌ upstream allocation leakage
❌ hidden filtering assumptions
❌ ignoring opportunity ranking
❌ collapsing distributions unnecessarily

11.3 Verboden Cross-Layer Behaviour

Decision Engine mag NOOIT:

❌ upstream layers herschrijven
❌ classification states overschrijven
❌ hidden upstream logic introduceren

12. Technical Requirements
12.1 decision_engine.py
Verplicht bouwen/herwerken

Bestand:

scripts/core/decision_engine.py
12.2 Verplichte Functionaliteit
probabilistic evaluation
opportunity ranking
conviction scoring
tradeability scoring
portfolio balancing
action generation
persistence tracking
deterministic outputs
fail-fast handling
12.3 Deterministic Decisioning
HARD RULE

Zelfde inputs moeten:

exact dezelfde outputs

produceren.

12.4 CI Enforcement
Verplicht toevoegen
grep -R "tradeable" scripts/ | grep -v decision_engine.py
grep -R "conviction" scripts/ | grep -v decision_engine.py
grep -R "allocation_priority" scripts/ | grep -v decision_engine.py

Resultaat moet leeg zijn.

13. Functional Requirements
Decision Engine moet:

✅ opportunities interpreteren
✅ relatieve allocatie bepalen
✅ conviction modelleren
✅ risk balancing uitvoeren
✅ probabilistische evaluatie uitvoeren

Decision Engine mag NIET:

❌ upstream classification wijzigen
❌ hidden filtering introduceren
❌ deterministic simplificatie gebruiken

14. Validation Requirements
Verplicht aantonen
Tradeability distribution
Conviction distribution
Allocation-priority distribution
Final-action distribution
Ranking distribution
Decision persistence distribution
15. Logging Requirements
Verplicht loggen
conviction shifts
allocation shifts
ranking shifts
action transitions
portfolio balancing decisions
probabilistic smoothing
16. CI / Enforcement Requirements
Verplicht
schema enforcement
allocation-authority enforcement
forbidden-field detection
deterministic-output validation
forbidden-keyword scanning
17. Acceptance Criteria
Sprint is PAS geslaagd wanneer:

✅ Decision Engine exclusieve allocatieautoriteit is
✅ tradeability uitsluitend downstream bestaat
✅ conviction uitsluitend downstream bestaat
✅ probabilistische evaluatie actief is
✅ opportunity ranking actief is
✅ Geen upstream allocation leakage meer bestaat
✅ decision_output.csv schema correct is
✅ Logging aanwezig is
✅ CI checks slagen

18. Definition of Done
Verplicht

✅ Alle governance rules gerespecteerd
✅ Geen hidden filtering
✅ Geen hidden upstream allocation
✅ Decision Engine volledig deterministic
✅ Outputs reproduceerbaar
✅ Conviction distributions valide
✅ Allocation distributions valide
✅ Logging aanwezig
✅ CI enforcement actief
✅ Technical Lead review geslaagd
✅ Functional Analyst review geslaagd
✅ Quant Analyst review geslaagd

19. Risks
Grootste Risico
legacy rule-based filtering assumptions

Bijvoorbeeld:

binary decisioning
hidden hard gating
simplistic signal mapping
upstream dependency remnants
Verboden Reactie

❌ opnieuw vroegtijdig filtering introduceren

Correcte Reactie

✅ probabilistische allocatie bouwen
✅ opportunity ranking gebruiken
✅ conviction downstream modelleren

20. Migration Notes

Na Sprint 6 zal waarschijnlijk:

de decision distributie rijker en complexer worden

Dat is EXPECTED.

En institutioneel correct.

Waarom?

Omdat de Decision Engine eindelijk:

echte allocatie-intelligentie

bevat.

21. Final Sprint Doctrine
interpret opportunities
rank opportunities
allocate probabilistically
centralize allocation authority
preserve upstream classification integrity
POST-SPRINT-0 GOVERNANCE INHERITANCE

Status: FUTURE SPRINT PLAN — ACTIVE ONLY UNDER CERTIFIED GOVERNANCE

This sprint plan must inherit Sprint 0 certification:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Decision Engine work is the only sprint category that may own final actions, tradeability, conviction, allocation priority, and capital allocation semantics.
