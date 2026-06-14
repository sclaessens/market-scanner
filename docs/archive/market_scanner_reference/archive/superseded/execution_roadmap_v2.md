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
- docs/sprints/sprint_status_tracker.md
- docs/sprints/project_backlog.md

Any roadmap wording about filtering, conviction, allocation, tradeability, or final actions is permitted only for Decision Engine-owned work. Upstream sprint work must remain classification-only.

Operational sprint lifecycle status is maintained in `docs/sprints/sprint_status_tracker.md`. This roadmap defines doctrine, sequencing, and sprint intent; it is not the operational status tracker. Sprint phase transitions must be recorded in the tracker and supported by the relevant governance artifact.

Deferred improvements, optional corrections, technical debt, research questions, and future enhancement ideas are maintained in `docs/sprints/project_backlog.md`. Backlog items do not modify this roadmap, authorize implementation, or change sprint scope without formal governance.

Mandatory Backlog Reconciliation is active across the sprint lifecycle. Every future sprint audit, implementation audit, and sprint closeout must include a dedicated `Backlog Impact Assessment` section and must add any newly identified deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work to `docs/sprints/project_backlog.md` before sprint closure.

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

Status: CERTIFIED COMPLETE. See `docs/sprints/sprint_2_closeout.md`. Sprint 3 may begin after Sprint 2 certification.

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

Sprint 3 — Fundamental Quality Layer

Status: CERTIFIED COMPLETE / CLOSED. See `docs/sprints/sprint_3_closeout.md`. Sprint 4 may begin after Sprint 3 certification.

Doel

Institutionele kwaliteitsclassificatie voorbereiden als pure classification/enrichment layer.

Scope
Quality classification metadata

✅ profitability quality
✅ balance-sheet quality
✅ earnings quality
✅ capital efficiency quality
✅ cash-flow quality
✅ stability metrics
✅ quality-factor metadata
✅ sector-relative quality metadata

Governance-clean quality outputs

✅ quality_profile
✅ profitability_profile
✅ balance_sheet_profile
✅ earnings_quality_profile
✅ capital_efficiency_profile
✅ cashflow_profile
✅ quality_reason
✅ quality_metadata
✅ quality_classification
✅ quality_state

Belangrijke Regel

Fundamentals mogen:

✅ business quality classificeren
✅ financial quality classificeren
✅ descriptive metadata toevoegen
✅ upstream opportunity distribution behouden

Maar NOOIT:

❌ conviction creëren
❌ priority creëren
❌ ranking authority creëren
❌ scoring authority creëren
❌ tradeability bepalen
❌ allocation bepalen
❌ execution readiness bepalen
❌ entries triggeren
❌ timing bepalen
❌ VALID_SETUP beïnvloeden
❌ opportunities suppressen, verwijderen, herordenen, prioriteren, versmallen of gatekeepen

Decision Engine Protection

Fundamental metadata may be consumed by a future Decision Engine-owned sprint only if that sprint explicitly defines the interpretation. The Fundamental Layer itself must not carry actionability, conviction, priority, ranking, scoring-authority, allocation, or execution semantics.

Deliverables
scripts/core/build_fundamental_layer.py
tests/core/test_build_fundamental_layer.py
data/processed/fundamental_quality.csv
data/logs/fundamental_layer_log.csv
docs/sprints/sprint_3_developer_spec.md
docs/audits/sprint_3_implementation_audit.md
docs/sprints/sprint_3_closeout.md

Definition of Done

✅ Fundamental Layer implemented as classification/enrichment only
✅ ticker/date row-key governance enforced
✅ duplicate ticker/date rows fail fast
✅ missing fundamentals preserve rows
✅ descriptive quality metadata emitted
✅ forbidden allocation, tradeability, conviction, urgency, priority, ranking, scoring, execution readiness, and BUY/SELL semantics absent
✅ tests passed
✅ implementation audit passed
✅ Sprint 3 closeout certified

Sprint 4 — Timing State Layer

Status: CERTIFIED COMPLETE / CLOSED. See `docs/sprints/sprint_4_closeout.md`. Sprint 5 may begin only after Sprint 4 certification and explicit Sprint 5 preparation authorization.

Doel

Timing state descriptively classify and enrich without allocation decisions.

Scope
Timing states

✅ UNCLASSIFIED
✅ EXTENDED
✅ PULLBACK_OBSERVED
✅ NEUTRAL
✅ EXPANDING
✅ SOURCE_MISSING

Timing metadata

✅ extension state
✅ pullback state
✅ breakout state
✅ compression state
✅ participation state
✅ timing environment
✅ timing pattern state
✅ timing structure state
✅ deterministic audit metadata

Grootste Correctie

EXTENDED betekent:

descriptive extension condition observed

Niet:

invalid opportunity
Verboden

❌ BUY
❌ SELL
❌ tradeability
❌ allocation eligibility
❌ actionability
❌ urgency
❌ conviction
❌ readiness
❌ priority
❌ ranking
❌ scoring

Deliverables
scripts/core/build_timing_state_layer.py
tests/core/test_build_timing_state_layer.py
data/processed/timing_state_layer.csv
data/logs/timing_state_layer_log.csv
docs/sprints/sprint_4_closeout.md
Sprint 5 — Portfolio Intelligence Layer

Status: CERTIFIED COMPLETE / CLOSED. See `docs/sprints/sprint_5_closeout.md`. Sprint 6 may begin only after Sprint 5 certification and explicit Sprint 6 preparation authorization.

Doel

Portfolio awareness descriptively classify and enrich without allocation decisions.

Scope

Portfolio Intelligence metadata

✅ in_portfolio
✅ portfolio_position_state
✅ exposure_state
✅ diversification_state
✅ concentration_state
✅ overlap_state
✅ sector_exposure_state
✅ position_context_state
✅ portfolio_environment
✅ portfolio_metadata_status
✅ portfolio_metadata_reason
✅ source provenance and classification rationale

Belangrijke Regel

Portfolio Intelligence mag:

✅ descriptive portfolio-awareness metadata toevoegen
✅ upstream opportunity universe behouden
✅ audit-traceable logs schrijven

Maar NOOIT:

❌ downstream conviction beïnvloeden
❌ allocation bepalen
❌ execution bepalen
❌ tradeability bepalen
❌ ranking/scoring/priority creëren
❌ opportunities vernietigen, filteren, herordenen, suppressen of gatekeepen

Deliverables
scripts/core/build_portfolio_intelligence.py
tests/core/test_build_portfolio_intelligence.py
data/processed/portfolio_intelligence.csv
data/logs/portfolio_intelligence_log.csv
docs/sprints/sprint_5_closeout.md
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
✅ backlog impact assessment completed where the Architect Gate is an audit artifact

4.2 Functional Gate

Functioneel Analist moet bevestigen:

✅ correct gedrag
✅ correcte state transitions
✅ geen implicit decisions
✅ geen verborgen gating
✅ deferred functional follow-up captured in project_backlog.md where identified

4.3 Quant Gate

Financieel Analist moet bevestigen:

✅ momentum-theorie correct
✅ edge niet vernietigd
✅ distributie behouden
✅ classificatie institutioneel correct
✅ deferred quant/research follow-up captured in project_backlog.md where identified

4.4 Scrum Gate

Scrum Master moet bevestigen:

✅ sprint scope gerespecteerd
✅ geen forbidden logic toegevoegd
✅ Definition of Done gehaald
✅ mandatory backlog reconciliation completed before CLOSED status

4.5 Mandatory Backlog Reconciliation Gate

Every future sprint audit, implementation audit, and sprint closeout must contain a section named:

```text
Backlog Impact Assessment
```

The section must conclude exactly one of:

```text
Backlog impact assessment:
- No new backlog items identified.
```

or:

```text
Backlog impact assessment:
- New backlog items identified and added to project_backlog.md
```

No sprint may be considered fully closed until this assessment is complete and any identified backlog items have been added to `docs/sprints/project_backlog.md`.

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
