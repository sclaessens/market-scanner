Decision Engine Design v2
Institutional Allocation Authority Architecture

POST-SPRINT-0 CERTIFICATION STATUS

Status: ACTIVE, GOVERNANCE-SYNCHRONIZED

Sprint 0 Governance Purification is certified COMPLETE. This document is active only under the binding doctrine:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Current runtime Decision Engine output is `data/processed/final_decisions.csv`. Older references to `decision_output.csv` are conceptual and should be read as `final_decisions.csv`.

Authoritative references:

- AGENTS.md
- docs/sprints/sprint_0_governance_status.md
- docs/audits/sprint_0_final_governance_audit.md

1. Purpose (REDEFINED)

De Decision Engine is de centrale allocatieautoriteit van het volledige trading system.

De engine is verantwoordelijk voor:

opportunity interpretation
capital allocation
conviction modelling
exposure balancing
conflict resolution
probabilistische besluitvorming

De Decision Engine is NIET langer:

een simpele action mapper

maar:

het institutionele brein van het systeem
1.1 Grootste Architecturale Correctie
OUDE FOUT

Upstream layers bepaalden impliciet:

tradeability
filtering
opportunity elimination

Daardoor kreeg de Decision Engine:

een reeds gecollapste opportunity space
NIEUWE ARCHITECTUUR

Upstream layers:

✅ classificeren
✅ structureren
✅ verrijken

Decision Engine:

✅ beslist
✅ alloceert
✅ prioriteert
✅ interpreteert

1.2 Nieuwe Institutionele Doctrine
CRUCIALE NIEUWE REGEL
classification upstream
allocation downstream
2. Core Principles (UPDATED)
2.1 Single Source of Truth
HARD RULE
Decision Engine = enige bron van waarheid

Alleen de Decision Engine mag bepalen:

BUY
SELL
HOLD
TRIM
WAIT
REMOVE
conviction
tradeability
allocation priority
2.2 One Decision Per Ticker

Voor elke ticker:

exact één final_action

Conflicterende beslissingen zijn verboden.

2.3 Probabilistic Decisioning
OUDE FOUT

Vroegere architectuur werkte:

binair

Bijvoorbeeld:

valid / invalid
NIEUWE FILOSOFIE

De Decision Engine werkt:

probabilistisch

Bijvoorbeeld:

hogere conviction
lagere conviction
hogere allocatieprioriteit
hogere execution risk
lagere timingkwaliteit
2.4 Separation of Concerns
Layer	Verantwoordelijkheid
Scanner	Detectie
Validation	Structuurclassificatie
Context	Leadershipclassificatie
Fundamentals	Kwaliteitsclassificatie
Watchlist	Timingstatus
Portfolio	Risicostatus
Decision Engine	Allocatiebeslissing
Reporting	Communicatie
2.5 No Upstream Tradeability
NIEUWE HARD RULE

Geen enkele upstream layer mag:

❌ tradeability bepalen
❌ capital eligibility bepalen
❌ allocatie blokkeren

3. Input Architecture (REDEFINED)
3.1 Scanner Layer
Doel

Opportunity detectie.

Inputbestand
scanner_ranked.csv
Scanner levert

✅ setup_type
✅ setup_score
✅ momentum structure
✅ RR metadata
✅ breakout metadata

Scanner bepaalt NIET

❌ actions
❌ tradeability
❌ conviction

3.2 Validation Layer
Doel

Technische coherentieclassificatie.

Inputbestand
validation_layer.csv
Validation levert

✅ valid_setup
✅ structure_state
✅ validation_reason

Validation bepaalt NIET

❌ tradeability
❌ conviction
❌ timing quality
❌ allocation eligibility

3.3 Context Layer
Doel

Relative leadership classificatie.

Inputbestand
context_strength.csv
Context levert

✅ rs_rank
✅ rs_percentile
✅ leadership_state
✅ context_strength

Context bepaalt NIET

❌ tradeability
❌ actions
❌ filtering

3.4 Fundamental Layer
Doel

Kwaliteitsclassificatie.

Inputbestand
fundamental_profile.csv
Fundamentals leveren

✅ quality classification
✅ profitability quality
✅ balance-sheet quality

Fundamentals bepalen NIET

❌ timing
❌ entries
❌ tradeability

3.5 Watchlist Layer
Doel

Timing readiness.

Watchlist levert

✅ readiness
✅ confirmation state
✅ extension state
✅ pullback readiness

Watchlist bepaalt NIET

❌ BUY
❌ SELL
❌ REMOVE

3.6 Portfolio Layer
Doel

Risk & exposure context.

Portfolio levert

✅ exposure
✅ overlap
✅ heat
✅ concentration risk
✅ pnl state

4. New Decision Architecture
4.1 Belangrijkste Architecturale Shift
OUDE ARCHITECTUUR
upstream filtering
→ downstream mapping
NIEUWE ARCHITECTUUR
upstream classification
→ downstream interpretation
→ downstream allocation
4.2 Nieuwe Core Responsibilities

De Decision Engine bepaalt exclusief:

Tradeability
verdient deze opportunity momenteel kapitaal?
Conviction
hoe sterk is de allocatiecase?
Allocation Priority
hoe belangrijk is deze opportunity relatief?
Execution Aggressiveness

Bijvoorbeeld:

aggressive breakout entry
passive pullback entry
delayed entry
reduced sizing
Portfolio Interaction

Bijvoorbeeld:

overlap reduction
concentration limits
correlation pressure
regime exposure
5. Decision Evaluation Flow (MAJOR REWRITE)
5.1 Nieuwe Institutionele Evaluatievolgorde
1. Portfolio state
2. Validation state
3. Context leadership
4. Fundamental quality
5. Timing state
6. Risk state
7. Opportunity ranking
8. Allocation eligibility
9. Conviction
10. Final action
5.2 Waarom deze volgorde correct is

Institutioneel hoort:

allocatie later te gebeuren dan classificatie

Niet vroeger.

6. Tradeability Model (FULLY REDEFINED)
6.1 Grootste Correctie
OUDE FOUT
tradeable = valid_setup AND strong_context

Dat was architecturaal fout.

6.2 Nieuwe Definitie
NIEUWE DEFINITIE
Tradeability = capital allocation readiness
6.3 Tradeability Vereist
Validation coherence
Leadership quality
Timing readiness
Portfolio capacity
Risk acceptance
Conviction threshold
Exposure allowance
6.4 Belangrijk Functioneel Verschil

Een opportunity kan:

✅ VALID zijn
✅ STRONG zijn
✅ kwalitatief zijn

Maar toch:

NOT TRADEABLE

Bijvoorbeeld:

portfolio te geconcentreerd
sector exposure te hoog
regime ongunstig
timing te extended
conviction onvoldoende

Dit is institutioneel correct.

7. Conviction Architecture (NEW)
7.1 Nieuwe Conviction Filosofie

Conviction wordt downstream opgebouwd uit:

context leadership
setup coherence
execution quality
portfolio pressure
regime alignment
fundamental quality
timing quality
7.2 Conviction States
VERY_HIGH
HIGH
MEDIUM
LOW
VERY_LOW
7.3 Conviction is GEEN upstream filter

Belangrijk:

lage conviction ≠ invalid
8. Opportunity Ranking Engine (NEW)
8.1 Nieuwe Institutionele Component

De Decision Engine moet opportunities relatief evalueren.

Niet binair.

8.2 Ranking Criteria

Bijvoorbeeld:

RS percentile
setup quality
timing quality
regime alignment
portfolio fit
leadership persistence
8.3 Nieuwe Filosofie

Het systeem moet:

de beste opportunities prioriteren

Niet:

alles buiten perfecte criteria verwijderen
9. Action Model (REDEFINED)
9.1 Nieuwe Actions
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
9.2 Nieuwe Betekenissen
BUY

Hoogste allocatieprioriteit.

ACCUMULATE

Bestaande positie uitbreiden.

PREPARE

Opportunity erkend.

Timing nog niet optimaal.

WAIT

Opportunity blijft geldig.

Maar allocatie nog niet verantwoord.

REMOVE

Opportunity niet langer relevant.

Niet:

“niet perfect genoeg”
REVIEW

Conflicterende classificaties.

10. Portfolio Interaction Logic (NEW)
10.1 Portfolio Is Geen Override Meer
OUDE FOUT

Portfolio blokkeerde upstream opportunities volledig.

NIEUWE FILOSOFIE

Portfolio beïnvloedt:

sizing
conviction
aggressiveness
allocation priority

Maar vernietigt opportunities niet automatisch.

10.2 Nieuwe Portfolio Constraints

Bijvoorbeeld:

sector concentration
correlation heat
max momentum exposure
regime-adjusted exposure
liquidity risk
11. Output Architecture (REDEFINED)
11.1 Nieuwe Centrale Output
final_decisions.csv
11.2 Verplichte Velden
ticker
date
final_action
tradeability
conviction
allocation_priority
validation_state
context_strength
leadership_state
timing_state
fundamental_profile
portfolio_state
execution_style
decision_reason
11.3 Nieuwe Output Filosofie

Output moet:

allocatie-intelligentie communiceren

Niet:

filteringresultaten
12. Governance Rules (UPDATED)
12.1 Hard Rules That Stay

✅ Decision Engine authority
✅ deterministic outputs
✅ one decision per ticker
✅ separation of concerns
✅ no hidden decisions
✅ no cross-layer contamination

12.2 Nieuwe Hard Rules
NIEUW
No upstream layer may determine tradeability.
NIEUW
No classification layer may collapse opportunity distribution excessively.
NIEUW
Decision Engine owns all allocation decisions.
13. Strategic System Shift

De grootste verandering:

Van:

rule-based filtering engine

Naar:

institutional allocation framework
14. Final Architectural Conclusion

Decision Engine v1 was nog te veel:

deterministic signal mapper

De architectuur gaf:

te weinig opportunity space
te veel upstream filtering
te weinig probabilistische interpretatie

Decision Engine v2 corrigeert dit fundamenteel:

Van:

mapping downstream

Naar:

institutionele allocatieautoriteit

De nieuwe architectuur:

✅ behoudt distributie
✅ centraliseert allocatie correct
✅ ondersteunt probabilistische evaluatie
✅ voorkomt premature edge destruction
✅ respecteert separation of concerns
✅ elimineert artificial bottlenecks
✅ maakt echte institutionele decisioning mogelijk

15. Final Decision Doctrine
scanner detects opportunities
validation classifies structure
context classifies leadership
fundamentals classify quality
watchlist tracks timing
portfolio tracks exposure
decision engine allocates capital
reporting communicates decisions
