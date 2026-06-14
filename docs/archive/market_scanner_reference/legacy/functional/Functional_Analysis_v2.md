FUNCTIONAL ANALYSIS DOCUMENT (v2 — ARCHITECTURE CORRECTED)
Trading System — Institutional Decision Behaviour Framework

POST-SPRINT-0 CERTIFICATION STATUS

Status: ACTIVE, GOVERNANCE-SYNCHRONIZED

Sprint 0 Governance Purification is certified COMPLETE. The current functional doctrine is:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Upstream layers classify only. Reporting communicates only. Only Decision Engine may produce tradeability, conviction, allocation priority, and final action fields.

Authoritative references:

- AGENTS.md
- docs/archive/migration/sprint_0_governance_status.md
- docs/archive/audits/sprint_0_final_governance_audit.md

1. Executive Overview (REDEFINED)

Het systeem evolueert definitief van:

signal generator

naar:

institutional decision engine

De grootste functionele verandering:

het systeem beslist niet langer vroeg
het systeem classificeert vroeg en beslist laat

Dit is een fundamentele gedragswijziging.

1.1 Nieuwe Functionele Doctrine
CRUCIALE NIEUWE REGEL
upstream layers verzamelen opportunity space
decision engine alloceert kapitaal
1.2 Nieuwe Filosofie

Het systeem probeert NIET langer:

vroegtijdig slechte setups elimineren

Maar:

de opportunity distributie correct classificeren

en pas later:

kapitaalbeslissingen nemen
1.3 Belangrijkste Gedragsverandering

Van:

VALID_SETUP = trade candidate

Naar:

VALID_SETUP = technisch coherente structuur

Dit onderscheid is cruciaal.

2. System Scope & Behavioural Boundaries
2.1 Nieuwe Functionele Layer Boundaries
Layer	Functionele verantwoordelijkheid
Scanner	Ideeën detecteren
Validation	Structuur classificeren
Context	Leadership classificeren
Fundamentals	Kwaliteit classificeren
Watchlist	Timingstatus beheren
Portfolio	Risicostatus beheren
Decision Engine	Allocatiebeslissingen nemen
Reporting	Beslissingen communiceren
2.2 Nieuwe Hard Rule
Geen enkele upstream layer mag tradeability bepalen.

Tradeability hoort EXCLUSIEF thuis in:

Decision Engine
3. Core Pipeline Behaviour (REDEFINED)
3.1 Nieuwe Pipelinefilosofie

De pipeline werkt niet langer als:

progressive filtering chain

Maar als:

classification enrichment pipeline
3.2 Nieuwe Gedragsflow
scanner
→ validation classification
→ context classification
→ fundamental classification
→ timing state
→ portfolio state
→ decision engine allocation
→ reporting
3.3 Belangrijkste Functionele Correctie
OUDE FOUT

Vroege layers probeerden reeds te bepalen:

“verdient deze setup kapitaal?”
NIEUW MODEL

Vroege layers bepalen uitsluitend:

“wat is dit voor opportunity?”
4. Functional Definition per Layer (REDEFINED)
4.1 Scanner Layer
Functie

Scanner detecteert uitsluitend:

setups
momentumstructuren
technische patronen
Scanner MAG

✅ detecteren
✅ ranken
✅ structureren

Scanner MAG NIET

❌ valideren
❌ context interpreteren
❌ fundamentals interpreteren
❌ tradeability bepalen
❌ acties bepalen

4.2 Validation Layer (FUNDAMENTALLY REDEFINED)
4.2.1 Nieuwe Definitie
VALID_SETUP
VALID_SETUP = technisch coherente setup

Niet:

tradeable setup
4.2.2 Validation bepaalt uitsluitend

✅ structure coherence
✅ technical integrity
✅ broken structures
✅ structure_state classification

4.2.3 Validation bepaalt NIET

❌ conviction
❌ timing quality
❌ context quality
❌ allocation worthiness
❌ capital eligibility

4.2.4 Nieuwe VALID_SETUP Filosofie

Een setup mag:

technisch valide zijn
maar contextueel zwak
fundamenteel zwak
allocation-onwaardig

En toch:

VALID_SETUP = TRUE

Dat is institutioneel correct.

4.2.5 ENTRY QUALITY (HERGEDEFINIEERD)
OUDE FOUT

Entry quality werd gebruikt als:

hard invalidation logic
NIEUWE FUNCTIONELE DEFINITIE

Entry quality wordt:

descriptive timing/structure metadata
Entry quality mag:

✅ geobserveerd worden door downstream lagen
✅ als metadata gecommuniceerd worden
✅ later door Decision Engine geïnterpreteerd worden

Maar:

❌ VALID_SETUP niet blokkeren
❌ geen execution instruction genereren
❌ geen allocation gate vormen

4.2.6 Nieuwe Validation Gedragsregel

Validation moet:

distributie behouden

Niet:

opportunities vroeg elimineren
4.3 Context Layer (FUNDAMENTALLY REDEFINED)
4.3.1 Nieuwe Definitie

Context bepaalt:

relative leadership

Niet:

tradeability
4.3.2 Nieuwe Functionele Doelstelling

De Context Layer moet:

sterkte classificeren
leadership classificeren
momentum distributie modelleren

Maar NOOIT:

entries blokkeren
tradeability bepalen
acties triggeren
4.3.3 Nieuwe Momentumfilosofie
OUDE FOUT

Momentum werd behandeld als:

outperforming benchmark
NIEUWE DEFINITIE

Momentum wordt:

cross-sectional leadership
4.3.4 Nieuwe Contextstates
LEADING
STRONG
NEUTRAL
WEAK
UNKNOWN
4.3.5 Nieuwe Gedragsregel

Context bepaalt:

hoe sterk een opportunity is

Niet:

of ze kapitaal verdient
4.4 Fundamental Layer
4.4.1 Nieuwe Definitie

Fundamentals bepalen uitsluitend:

kwaliteit

Niet:

timing
4.4.2 Nieuwe Gedragsregel

Sterke fundamentals:

✅ classificeren kwaliteit
✅ verrijken downstream interpretatie
✅ mogen later door Decision Engine worden gebruikt

Maar:

❌ triggeren geen entries
❌ bepalen geen upstream conviction

4.4.3 Belangrijkste Functionele Correctie
OUDE FOUT

Fundamentals werden impliciet gebruikt als:

trade filter
NIEUWE DEFINITIE

Fundamentals worden:

allocation confidence modifier
4.5 Watchlist Layer
4.5.1 Nieuwe Rol

Watchlist beheert uitsluitend:

timing readiness
4.5.2 Nieuwe Watchliststates
EARLY
READY
CONFIRMED
EXTENDED
PULLBACK
BREAKOUT_PENDING
STALE
FAILED
4.5.3 Watchlist MAG NIET

❌ BUY bepalen
❌ REMOVE bepalen
❌ conviction bepalen
❌ tradeability bepalen

4.6 Portfolio Layer
4.6.1 Nieuwe Rol

Portfolio beheert:

risk context
4.6.2 Portfolio bepaalt

✅ exposure
✅ overlap
✅ correlation pressure
✅ portfolio heat
✅ concentration risk

4.6.3 Portfolio bepaalt NIET

❌ signal validity
❌ setup quality
❌ momentum quality

5. Decision Engine (MAJOR FUNCTIONAL REDEFINITION)
5.1 Nieuwe Institutionele Rol

Decision Engine wordt:

centrale allocatieautoriteit
5.2 Decision Engine is exclusief verantwoordelijk voor
Tradeability
Conviction
Capital allocation
Priority ranking
Exposure balancing
Conflict resolution
Final actions
5.3 Nieuwe Decision Filosofie

De Decision Engine werkt:

probabilistisch

Niet:

binair
5.4 Nieuwe Functionele Evaluatieflow
1. Validation state
2. Context leadership
3. Fundamental quality
4. Timing state
5. Portfolio exposure
6. Risk state
7. Opportunity ranking
8. Allocation eligibility
9. Conviction
10. Final action
5.5 Nieuwe Conviction Filosofie

Conviction wordt bepaald door:

context leadership
setup coherence
timing quality
portfolio risk
market regime
execution quality
fundamental quality
5.6 Nieuwe Tradeability Definitie
CRUCIAAL
TRADEABLE ≠ VALID
Nieuwe definitie

Tradeable betekent:

de setup verdient momenteel kapitaalallocatie

Dat vereist:

context
timing
exposure
risk budget
conviction
portfolio state
5.7 Nieuwe Action Definitions
BUY

Niet langer:

“valid + strong”

Maar:

hoogste allocatieprioriteit
WAIT

Nieuwe betekenis:

opportunity erkend
maar allocatie nog niet verantwoord
REMOVE

Nieuwe betekenis:

opportunity niet langer relevant

Niet:

“niet perfect genoeg”
REVIEW

Nieuwe betekenis:

conflicting classification states
6. State Transition Logic (REDEFINED)
6.1 Nieuwe Filosofie

State transitions moeten:

stabiliteit verhogen

Niet:

reactiviteit maximaliseren
6.2 Nieuwe Regels
Zwakke context

→ meer bevestiging vereist

Sterke leadership

→ minder confirmation latency

Hoge conviction

→ agressievere allocatie toegestaan

7. Interaction Rules (REDEFINED)
7.1 Nieuwe Interactie Filosofie

Layers mogen:

elkaar verrijken

Maar niet:

elkaar overrulen
7.2 Nieuwe Gedragsvoorbeelden
Sterke setup + zwakke context
recognized but low priority

Niet:

invalid
Zwakke fundamentals + sterke leadership
tradeable possible
but lower conviction
Sterke fundamentals + zwakke structuur
no allocation
8. Edge Cases (UPDATED)
8.1 EXTENDED Momentum
OUDE FOUT

EXTENDED setups werden:

hard invalidated
NIEUWE DEFINITIE

EXTENDED betekent:

higher execution risk

Niet:

invalid opportunity
8.2 High Momentum + Weak Fundamentals

Toegestaan:

Maar:

lagere conviction
lagere holding tolerance
agressievere exits
8.3 Strong Fundamentals + Weak Context

Geen allocatie.

Maar:

❌ niet invalid

9. Output Behaviour (REDEFINED)
9.1 Nieuwe Output Filosofie

Output moet:

allocatiebeslissingen communiceren

Niet:

filteringbeslissingen
9.2 Verplichte Outputvelden
Decision Engine output:

final_action
conviction
tradeability
allocation_priority
validation_state
context_strength
leadership_state
timing_state
portfolio_state
execution_style
decision_reason
10. Governance Rules (UPDATED)
10.1 Hard Rules That Stay

✅ Decision Engine authority
✅ one decision per ticker
✅ deterministic outputs
✅ no hidden decisions
✅ layer separation

10.2 Nieuwe Hard Rules
NIEUW
classification upstream
allocation downstream
NIEUW
No upstream layer may determine capital eligibility.
NIEUW
No classification layer may collapse opportunity distribution excessively.
11. Strategic Behavioural Shift

De grootste functionele verandering:

Van:

“welke setups zijn goed genoeg?”

Naar:

“welke opportunities verdienen allocatieprioriteit?”
12. Final Functional Conclusion

Functional Analysis v1 probeerde:

vroegtijdig edge te beschermen via filtering

Dat leidde tot:

over-filtering
premature invalidation
distributie collapse
architecturale bottlenecks

Functional Analysis v2 corrigeert dit fundamenteel:

Van:

filter-first behaviour

Naar:

classification-first behaviour

De nieuwe functionele architectuur:

✅ behoudt distributie
✅ respecteert institutionele decisioning
✅ elimineert premature gating
✅ centraliseert allocatiebeslissingen correct
✅ ondersteunt probabilistische evaluatie
✅ voorkomt artificial bottlenecks

13. Final Behavioural Doctrine
scanner detects
validation classifies structure
context classifies leadership
fundamentals classify quality
watchlist tracks timing
portfolio tracks exposure
decision engine allocates capital
reporting communicates decisions
