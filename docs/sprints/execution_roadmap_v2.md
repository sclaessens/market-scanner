SPRINT RESTRUCTURING DOCUMENT v2
Architecture-Corrected Delivery Roadmap
Trading System — Institutional Decision Engine

POST-SPRINT-0 CERTIFICATION STATUS

Status: ACTIVE ROADMAP, GOVERNANCE-SYNCHRONIZED

Sprint 0 Governance Purification is certified COMPLETE. This roadmap remains active only under the certified doctrine:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Future sprints must inherit the certified runtime schemas and layer responsibilities documented in:

- AGENTS.md
- docs/sprints/sprint_0_governance_status.md
- docs/audits/sprint_0_final_governance_audit.md

Any roadmap wording about filtering, conviction, allocation, tradeability, or final actions is permitted only for Decision Engine-owned work. Upstream sprint work must remain classification-only.

1. Executive Summary

De originele sprintstructuur was gebaseerd op een architectuur die:

te vroeg besliste
validation gebruikte als filtering engine
context gebruikte als pseudo-tradeability layer
opportunity distributie vernietigde vóór de Decision Engine

De architecture audit heeft bevestigd dat deze aanpak institutioneel fout was.

Daarom wordt de volledige roadmap herwerkt.

1.1 Grootste Strategische Shift
OUDE ROADMAP
filtering-first delivery
NIEUWE ROADMAP
classification-first delivery
1.2 Nieuwe Delivery Doctrine

De nieuwe sprintstructuur moet:

✅ distributie behouden
✅ classificatie upstream houden
✅ allocatie downstream centraliseren
✅ probabilistische decisioning ondersteunen
✅ premature edge destruction vermijden

Zoals expliciet gedefinieerd in:

Technical Analysis v3
Functional Analysis v2
Decision Engine Design v2
Execution Framework v2
2. New Delivery Strategy
2.1 Nieuwe Developmentfilosofie

Development gebeurt voortaan in drie grote fasen:

Fase	Doel
Fase A	Classification Infrastructure
Fase B	Allocation Infrastructure
Fase C	Stability & Communication
2.2 Nieuwe Sprinttypes
Sprinttype	Functie
Type A	Classification Sprint
Type B	Allocation Sprint
Type C	Stability Sprint
Type D	Reporting Sprint
3. Revised Delivery Roadmap
PHASE A — CLASSIFICATION INFRASTRUCTURE
Sprint 0 — Architecture Migration & Stabilisation
Doel

De bestaande codebase corrigeren zodat ze compatibel wordt met de nieuwe institutionele architectuur.

Status: COMPLETE and certified.

PRIORITEIT

CRITISCH — alle verdere development hangt hiervan af.

Scope
Governance cleanup

✅ verwijderen van tradeable_setup
✅ verwijderen van context_tradeable
✅ verwijderen van allocation leakage
✅ verwijderen van implicit filtering

Validation cleanup

✅ VALID_SETUP reduceren tot structure coherence
✅ entry quality verwijderen uit validation gating
✅ extension invalidation verwijderen

Context cleanup

✅ context reduceren tot pure classification
✅ benchmark-relative momentum verwijderen
✅ sector dependency neutraliseren

Architecture enforcement

✅ grep-enforcement
✅ CI governance
✅ layer-boundary validation
✅ schema cleanup

Verboden

❌ nieuwe filtering
❌ conviction logic
❌ nieuwe BUY/SELL logica
❌ nieuwe thresholds
❌ optimization tuning

Deliverables
Nieuwe outputs
validation_layer.csv
context_strength.csv

zonder tradeabilityvelden.

Nieuwe governance
classification upstream
allocation downstream
Nieuwe CI checks

Verbod op:

BUY buiten Decision Engine
tradeability buiten Decision Engine
allocation logic upstream
Definition of Done

✅ Geen allocation leakage meer
✅ Geen tradeability buiten Decision Engine
✅ Validation bevat enkel structuurclassificatie
✅ Context bevat enkel leadershipclassificatie
✅ Pipeline draait end-to-end
✅ Alle oude pseudo-tradeability verwijderd

Sprint 1 — Structure Classification Layer

Status: CERTIFIED COMPLETE. See `docs/sprints/sprint_1_closeout.md`. Sprint 2 may begin after Sprint 1 certification.
Doel

Institutionele structure classification bouwen.

Scope
Validation architecture

✅ structure_valid
✅ trend_structure_valid
✅ data_integrity_valid
✅ structure_state

Validation metadata

✅ validation_reason
✅ structure tags
✅ invalidation metadata

Logging

✅ structure distribution
✅ invalidation distribution
✅ setup distribution

Belangrijke Architecturale Regel

Validation mag:

✅ classificeren
✅ structureren

Maar NOOIT:

❌ allocatie simuleren
❌ context interpreteren
❌ timing blokkeren

Verboden

❌ tradeability
❌ conviction
❌ capital-worthiness
❌ execution gating

Deliverables
scripts/core/build_validation_layer.py
data/processed/validation_layer.csv
Definition of Done

✅ VALID_SETUP bevat geen tradeability meer
✅ entry quality beïnvloedt VALID_SETUP niet
✅ distribution collapse blijft beperkt
✅ logging aanwezig
✅ CI checks slagen

Sprint 2 — Cross-Sectional Leadership Layer
Doel

Institutioneel momentum correct modelleren.

Grootste Correctie

Van:

benchmark-relative momentum

Naar:

cross-sectional leadership
Scope
Nieuwe RS architectuur

✅ rs_rank
✅ rs_percentile
✅ leadership_bucket
✅ leadership_state
✅ percentile distribution

Momentum distribution

✅ cross-sectional ranking
✅ relative cohort positioning
✅ leadership persistence

Classification buckets
TOP_1
TOP_5
TOP_10
TOP_20
MID
WEAK
Verboden

❌ context_tradeable
❌ hard filtering
❌ BUY/SELL impact
❌ allocation logic

Deliverables
scripts/core/build_context_layer.py
data/processed/context_strength.csv
Definition of Done

✅ context_tradeable bestaat niet meer
✅ cross-sectional ranking actief
✅ leadership distribution valide
✅ benchmark-only momentum verwijderd
✅ sector dependency niet langer blocking

Sprint 3 — Fundamental Quality Classification
Doel

Institutionele kwaliteitsclassificatie toevoegen.

Scope
Quality metrics

✅ profitability quality
✅ balance-sheet quality
✅ earnings quality
✅ capital efficiency quality

Quality states

✅ HIGH_QUALITY
✅ MEDIUM_QUALITY
✅ LOW_QUALITY

Belangrijke Regel

Fundamentals mogen:

✅ conviction downstream beïnvloeden

Maar NOOIT:

❌ entries triggeren
❌ timing bepalen
❌ VALID_SETUP beïnvloeden

Deliverables
scripts/core/build_fundamental_layer.py
data/processed/fundamental_profile.csv
Sprint 4 — Timing State Layer
Doel

Timing readiness classificeren zonder allocatiebeslissingen.

Scope
Timing states

✅ READY
✅ EARLY
✅ EXTENDED
✅ PULLBACK
✅ BREAKOUT_PENDING

Timing metadata

✅ extension state
✅ pullback proximity
✅ breakout readiness
✅ timing quality

Grootste Correctie

EXTENDED betekent:

higher execution risk

Niet:

invalid opportunity
Verboden

❌ BUY
❌ SELL
❌ tradeability
❌ allocation eligibility

Deliverables
watchlist_state.csv
Sprint 5 — Portfolio Intelligence Layer
Doel

Portfolio pressure modelleren.

Scope
Exposure metrics

✅ concentration risk
✅ sector exposure
✅ correlation heat
✅ liquidity pressure
✅ momentum concentration

Portfolio states

✅ NORMAL
✅ CONCENTRATED
✅ OVEREXPOSED
✅ HIGH_CORRELATION

Belangrijke Regel

Portfolio mag:

✅ downstream conviction beïnvloeden

Maar NOOIT:

❌ upstream opportunities vernietigen

Deliverables
portfolio_state.csv
PHASE B — ALLOCATION INFRASTRUCTURE
Sprint 6 — Decision Engine Core
Doel

De institutionele allocatie-engine bouwen.

Grootste Strategische Shift

Van:

signal mapper

Naar:

institutional allocation authority
Scope
Tradeability engine

✅ capital allocation readiness
✅ probabilistic tradeability
✅ allocation gating downstream

Conviction engine

✅ conviction scoring
✅ allocation priority
✅ execution aggressiveness

Conflict resolution

✅ portfolio interaction
✅ exposure balancing
✅ risk balancing

Final actions

✅ BUY
✅ ACCUMULATE
✅ PREPARE
✅ WAIT
✅ HOLD
✅ TRIM
✅ SELL
✅ REMOVE
✅ REVIEW

Belangrijke Regel

ALLE allocatie-intelligentie hoort nu hier.

Deliverables
scripts/core/decision_engine.py
data/processed/decision_output.csv
Definition of Done

✅ Eén beslissing per ticker
✅ Geen allocation leakage upstream
✅ Conviction downstream gecentraliseerd
✅ Tradeability downstream gecentraliseerd
✅ Probabilistische evaluatie actief

Sprint 7 — Stability & Persistence Layer
Doel

Decision stability verhogen.

Scope
Stability systems

✅ persistence logic
✅ confirmation logic
✅ flip-flop reduction
✅ probabilistic smoothing

Behaviour stabilization

✅ reduced action churn
✅ stable conviction shifts
✅ controlled action escalation

Deliverables
stability_state.csv
PHASE C — COMMUNICATION & OBSERVABILITY
Sprint 8 — Reporting Layer
Doel

Institutionele communicatie-output bouwen.

Scope
Reporting

✅ Telegram reports
✅ dashboards
✅ allocation summaries
✅ conviction summaries
✅ portfolio summaries

Reporting principles

Reporting mag:

✅ communiceren
✅ structureren
✅ visualiseren

Maar NOOIT:

❌ interpreteren
❌ beslissen
❌ filtering toevoegen

Deliverables
telegram_message.txt
daily reports
allocation dashboards
4. Mandatory Sprint Gates
4.1 Architect Gate

Voor ELKE sprint:

Technisch Analist moet bevestigen:

✅ geen layer contamination
✅ correcte separation of concerns
✅ geen allocation leakage
✅ correcte data contracts

4.2 Functional Gate

Functioneel Analist moet bevestigen:

✅ correct gedrag
✅ correcte state transitions
✅ geen implicit decisions
✅ geen verborgen gating

4.3 Quant Gate

Financieel Analist moet bevestigen:

✅ momentum-theorie correct
✅ edge niet vernietigd
✅ distributie behouden
✅ classificatie institutioneel correct

4.4 Scrum Gate

Scrum Master moet bevestigen:

✅ sprint scope gerespecteerd
✅ geen forbidden logic toegevoegd
✅ Definition of Done gehaald

5. Critical Governance Rules
HARD RULE
No upstream layer may determine tradeability.
HARD RULE
No classification layer may collapse opportunity distribution excessively.
HARD RULE
Decision Engine owns all allocation decisions.
HARD RULE
classification first
allocation downstream
6. Final Scrum Master Conclusion

De vorige sprintstructuur probeerde:

edge vroegtijdig af te dwingen via filtering

De nieuwe sprintstructuur corrigeert dit fundamenteel.

Nieuwe deliverystrategie:

✅ behoudt opportunity distributie
✅ centraliseert allocatie downstream
✅ elimineert premature gating
✅ ondersteunt probabilistische decisioning
✅ voorkomt artificial bottlenecks
✅ respecteert institutionele governance
✅ maakt echte institutional-grade evolutie mogelijk

7. Final Delivery Doctrine
classify first
observe second
allocate later
stabilize downstream
communicate last
