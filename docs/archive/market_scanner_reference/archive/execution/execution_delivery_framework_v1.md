Execution & Delivery Framework v1
Trading System — Market Scanner & Decision Engine
1. Purpose

Dit document definieert het operationele delivery framework voor het trading system project.

Het doel is om gecontroleerde ontwikkeling af te dwingen binnen de pipeline:

scanner → validation_layer → context_layer → watchlist → portfolio → decision engine → reporting

Dit framework voorkomt:

ad-hoc development
cross-layer logica
decision leakage
inconsistent gedrag
niet-valideerbare wijzigingen
2. Core Delivery Principles
2.1 Layer-Based Development

Elke sprint behandelt maximaal één functionele layer.

Niet toegestaan:

context en fundamentals tegelijk bouwen
confidence logic toevoegen tijdens context sprint
reporting wijzigen tijdens core layer development
decision logic toevoegen buiten decision engine

Elke layer heeft één verantwoordelijkheid.

2.2 Validation-First Development

Elke wijziging wordt behandeld als hypothese.

Elke feature moet:

expliciet gedefinieerd worden
technisch gespecificeerd worden
gelogd worden
gevalideerd worden
pas daarna geactiveerd worden

Geen enkele nieuwe regel wordt als succesvol beschouwd zonder meetbare impact.

2.3 Separation of Concerns

Verantwoordelijkheden:

Layer	Verantwoordelijkheid
Scanner	Ideeën detecteren
Validation Layer	Technische geldigheid bepalen
Context Layer	Relatieve sterkte bepalen
Watchlist	Timingstatus beheren
Portfolio	Risicocontext beheren
Decision Engine	Finale beslissing nemen
Reporting	Beslissingen presenteren

Hard rule:

Geen enkele module behalve de Decision Engine mag BUY, SELL, HOLD, TRIM of REMOVE bepalen.

2.4 Non-Disruptive Evolution

Nieuwe logica wordt modulair toegevoegd.

Bestaande werkende pipeline mag niet breken.

Elke sprint moet backward compatible blijven tenzij expliciet anders goedgekeurd.

3. Sprint Roadmap
Sprint 0 — Stabilisation
Doel

De bestaande codebase stabiliseren en decision leakage identificeren.

Scope
audit van scanner, watchlist, portfolio, reporting
data contract controle
pipeline baseline vastleggen
verborgen beslissingslogica detecteren
inconsistenties documenteren
Niet toegestaan
nieuwe strategieën
fundamentals
confidence logic
decision engine rewrite
reporting redesign
Output
stabiele baseline
audit findings
lijst met decision leakage
gekende risico’s
startpunt voor Sprint 1
Sprint 1 — Validation Layer
Doel

Technische geldigheid expliciet maken.

Scope
valid_setup
eerste versie van tradeable_setup
validation_layer.csv
validation logging
Niet toegestaan
context logic
fundamentals
confidence
BUY/SELL beslissingen
reporting wijzigingen
Output
data/processed/validation_layer.csv
scripts/core/build_validation_layer.py
logging van validation reasons
Sprint 2 — Context Layer
Doel

Relatieve sterkte toevoegen als tweede filterlaag.

Scope
context_strength
rs_20d
rs_vs_sector
update van tradeable_setup
Niet toegestaan
fundamentals
confidence
decision logic
reporting wijzigingen
position sizing
Output
scripts/core/build_context_layer.py
data/processed/context_strength.csv
context logging
aangepaste tradeable setup definitie
Sprint 3 — Fundamental Layer
Doel

Fundamental quality logging toevoegen zonder beslissingsimpact.

Scope
Piotroski Score
Earnings Yield
EV/EBITDA
ROIC
fundamental_profile
Niet toegestaan
entries triggeren
timing bepalen
BUY/SELL beïnvloeden
confidence activeren
Output
scripts/core/build_fundamental_layer.py
data/processed/fundamental_profile.csv
logging per fundamental profile
Sprint 4 — Decision Engine
Doel

Alle finale beslissingen centraliseren.

Scope
één beslissing per ticker
decision hierarchy
scanner/watchlist/portfolio/context/fundamentals als input
finale action output
Niet toegestaan
nieuwe indicators
nieuwe fundamentals
reporting redesign
position sizing
Output
scripts/core/decision_engine.py
data/processed/decision_output.csv
één finale actie per ticker
Sprint 5 — Confidence Layer
Doel

Confidence toevoegen aan bestaande beslissingen.

Scope
confidence levels
HIGH / MEDIUM / LOW
confidence reason
mapping op context en fundamentals
Niet toegestaan
nieuwe actions
position sizing
reporting redesign
nieuwe setupregels
Output
confidence kolommen in decision output
confidence logging
validation per confidence bucket
Sprint 6 — Stability Layer
Doel

Flip-flop gedrag verminderen.

Scope
confirmation rules
state persistence
minimum confirmation days
confluence rules
Niet toegestaan
nieuwe setupdetectie
fundamentals toevoegen
reporting redesign
entries buiten decision engine
Output
stability state
confirmation logging
minder instabiele action changes
Sprint 7 — Output Layer
Doel

Reporting structureren op basis van decision output.

Scope
Telegram output
daily report
action-first structuur
duidelijke categorieën
Niet toegestaan
nieuwe beslissingslogica
nieuwe filters
nieuwe confidence regels
portfolio logic wijzigen
Output
reporting gebaseerd op decision output
geen interpretatie in reporting
consistente eindcommunicatie
4. Feature Lifecycle

Elke feature doorloopt exact deze stappen:

4.1 Idea

Beschrijving van het probleem of de hypothese.

Vereist:

probleem
verwachte impact
betrokken layer
risico bij niet oplossen
4.2 Functional Definition

Exacte gedragsdefinitie.

Vereist:

input
output
toegestane states
verboden gedrag
edge cases
4.3 Technical Specification

Exacte implementatiedefinitie.

Vereist:

file names
script names
kolommen
datatypes
pseudo-code
pipeline positie
4.4 Logging

Elke feature moet traceerbaar zijn.

Vereist:

input count
output count
filtered count
reason codes
timestamp
run status
4.5 Validation

Elke feature moet meetbaar zijn.

Vereist:

winrate
average return
hit rate
drawdown indien beschikbaar
segmentatie per nieuwe classificatie
4.6 Activation

Een feature mag pas impact hebben wanneer:

logging correct is
validation uitgevoerd is
edge cases getest zijn
geen cross-layer impact bestaat
5. Governance Rules
5.1 Decision Engine Is Single Source of Truth

Alle finale acties komen uitsluitend uit de Decision Engine.

Verboden buiten Decision Engine:

BUY
SELL
HOLD
TRIM
REMOVE
PREPARE_BUY als finale actie
5.2 One Decision Per Ticker

Per run mag elke ticker exact één finale beslissing krijgen.

Conflicten zijn niet toegestaan.

5.3 No Cross-Layer Logic

Elke layer mag alleen zijn eigen verantwoordelijkheid uitvoeren.

Voorbeelden:

Context Layer mag geen fundamentals gebruiken
Fundamental Layer mag geen timing bepalen
Reporting mag geen actie interpreteren
Watchlist mag geen koopbeslissing nemen
5.4 No Implicit Decisions

Elke output moet expliciet verklaard worden met reason codes.

Geen verborgen defaults.

Geen stille fallback behalve waar expliciet gespecificeerd.

5.5 Deterministic Output

Bij identieke input moet output identiek zijn.

Niet toegestaan:

randomness
ongedocumenteerde state
tijdsafhankelijke logica zonder expliciete datuminput
6. Definition of Done Framework

Een sprint is pas klaar wanneer alle volgende punten waar zijn:

Alle afgesproken bestanden bestaan.
Alle data contracts zijn geïmplementeerd.
Alle verplichte kolommen bestaan.
Alle scripts draaien zonder handmatige tussenkomst.
Pipeline werkt end-to-end.
Logging is aanwezig.
Reason codes zijn aanwezig.
Edge cases zijn getest.
Geen verboden logic is toegevoegd.
Geen bestaande outputs zijn onbedoeld gebroken.
Unit tests slagen.
Nieuwe output is reproduceerbaar.
Scope blijft beperkt tot de sprint.
Audit document of sprint findings zijn bijgewerkt.
7. Risk Management
7.1 Overfitting

Risico:

Nieuwe regels lijken goed op beperkte data maar verslechteren later performance.

Mitigatie:

segmentatie verplicht
minimum sample size
validation-first
geen activatie zonder meting
7.2 Inconsistent Logic

Risico:

Meerdere modules interpreteren dezelfde ticker anders.

Mitigatie:

single source of truth
one decision per ticker
reason codes
centrale decision output
7.3 Pipeline Breaks

Risico:

Nieuwe layer breekt downstream scripts.

Mitigatie:

backward compatibility
schema checks
hard fail bij ontbrekende verplichte input
end-to-end run verplicht
7.4 Wrong Layer Implementation

Risico:

Logica wordt in verkeerde module gebouwd.

Mitigatie:

sprint scope hard bewaken
grep-tests op verboden actions
code review op separation of concerns
7.5 Silent Data Corruption

Risico:

Ontbrekende of foute data wordt stil verwerkt.

Mitigatie:

schema validation
duplicate checks
missing value policy
logging van unknown/missing records
8. Continuous Improvement Loop

Het systeem verbetert via vaste cyclus:

decision_output
→ validation_results
→ analysis
→ backlog
→ sprint planning
→ implementation
→ validation

Geen wijziging wordt structureel behouden zonder aantoonbare verbetering in:

winrate
drawdown
consistency
false positive reduction
output stability
9. Change Control

Elke wijziging moet vastgelegd worden met:

datum
sprint
betrokken layer
gewijzigd bestand
reden
verwachte impact
testresultaat
validatieresultaat

Geen wijziging zonder traceerbaarheid.

10. Final Operating Model

Dit framework legt de operationele discipline vast voor het volledige project.

De kernregel blijft:

scanner detects
validation validates
context qualifies
fundamentals assess quality
watchlist tracks timing
portfolio tracks risk
decision engine decides
reporting communicates

Elke afwijking van deze structuur creëert technisch risico.

Dit document is de operationele basis voor alle verdere sprints.

11. Pipeline Enforcement (NIEUW)
run_scan.py is de enige toegestane entrypoint voor pipeline execution.

De pipeline moet exact deze volgorde volgen:

1. scanner
2. validation_layer
3. context_layer
4. watchlist
5. portfolio
6. decision_engine
7. reporting

Hard rules:

geen script mag standalone in productie draaien
elke stap moet afhankelijk zijn van vorige output
ontbrekende input = hard fail
12. Data Ownership (NIEUW)
Bestand	Owner	Mag schrijven	Mag lezen
scanner_ranked.csv	Scanner	Scanner	Alle
validation_layer.csv	Validation Layer	Validation + Context (alleen tradeable)	Alle
context_strength.csv	Context Layer	Context	Alle
fundamental_profile.csv	Fundamental Layer	Fundamental	Decision
decision_output.csv	Decision Engine	Decision	Reporting

Hard rule:

👉 Slechts één owner per bestand (write authority)

13. Hard Enforcement Rules (NIEUW)

Verplicht in CI / code review:

grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "TRIM" scripts/ | grep -v decision_engine.py
grep -R "HOLD" scripts/ | grep -v decision_engine.py

Resultaat moet leeg zijn.

14. Fail Fast Policy (NIEUW)

Het systeem moet crashen bij:

ontbrekende kolommen
duplicate ticker + date
lege input bestanden (waar data verwacht wordt)
verkeerde datatypes

Niet toegestaan:

silent fallback
automatische defaults zonder logging
15. Backward Compatibility Contract (NIEUW)

Elke sprint moet garanderen:

bestaande CSV kolommen blijven bestaan
bestaande scripts blijven werken
bestaande outputs blijven leesbaar
16. Rollback Mechanisme (NIEUW)

Elke feature moet kunnen worden uitgeschakeld via:

config/settings.py

Voorbeeld:

ENABLE_CONTEXT_LAYER = True
17. Hard Definition of Done (VERBETERD)

Sprint is alleen klaar als:

Pipeline draait via run_scan.py
Alle outputs bestaan en zijn valide
Geen enkele BUY/SELL buiten decision engine bestaat (gecontroleerd via grep)
Data contracts gevalideerd zijn (schema check)
Duplicate detection actief is
Logging aanwezig is
Unit tests slagen
Output reproduceerbaar is bij identieke input
Geen bestaande kolommen verwijderd zijn
Feature kan gedeactiveerd worden via config