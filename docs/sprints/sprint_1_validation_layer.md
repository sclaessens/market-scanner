Sprint 1 — Validation Layer

1. Sprint Overview
Doel van Sprint 1
Sprint 1 heeft als doel het expliciet maken van technische validiteit binnen het systeem door de introductie van een Validation Layer.
Deze layer zorgt ervoor dat:
setup → validatie → (later) beslissing
in plaats van:
setup → actie
Dit is een fundamentele shift richting een institutionele decision engine zoals vereist door de projectstrategie en het delivery framework .

Link met Sprint 0 Findings
Sprint 0 heeft kritische structurele problemen blootgelegd:


decision leakage in watchlist en portfolio


meerdere decision sources


instabiele pipeline door feedback loops


inconsistente data contracts 


Belangrijkste implicatie:
👉 Het systeem mist een expliciete validatiestap tussen detectie en actie
De Validation Layer adresseert direct:


eliminatie van impliciete beslissingen


scheiding tussen “technisch geldig” en “actie vereist”


voorbereiding op centrale decision engine



Rol van de Validation Layer
Binnen de architectuur:
scanner → validation → watchlist → portfolio → decision engine → reporting
De Validation Layer:


bepaalt of een setup technisch geldig is


maakt validiteit expliciet en reproduceerbaar


voorkomt dat downstream lagen interpretatie moeten doen


fungeert als first gate before decision logic


👉 De layer bevat geen beslissingslogica

2. Scope Definition
In Scope
Sprint 1 implementeert:
1. VALID_SETUP definitie


Technische validiteit van een setup


Gebaseerd op scanner output en technische criteria


2. TRADEABLE_SETUP definitie


Eerste vorm van “bruikbaarheid”


Gebaseerd op VALID_SETUP (zonder fundamentals)


3. Validation Logging


Volledige traceability van validatiebeslissingen


Geen impliciete logica


4. Nieuwe output file
data/processed/validation_layer.csv

Out of Scope
Strikt uitgesloten:


Fundamentals (geen filtering, geen scoring)


Decision engine logica


Confidence levels


Output/logica in reporting aanpassen


Strategie-aanpassingen


👉 Dit volgt expliciet de layer-based development regel 

3. Functional Specification
3.1 VALID_SETUP
Definitie:
VALID_SETUP = TRUE wanneer alle technische setup-condities voldaan zijn
Gebaseerd op:


scanner output (setup_type, setup_grade, entry, stop, target)


technische structuur (trend, breakout, pullback)


Conform functionele analyse:
👉 VALID_SETUP = technische geldigheid
👉 Geen context, geen fundamentals 
Voorbeelden:
ScenarioVALID_SETUPBreakout boven 20D high met volumeTRUEPullback naar MA20 in uptrendTRUERange zonder duidelijke structuurFALSE

3.2 TRADEABLE_SETUP
Definitie:
TRADEABLE_SETUP = VALID_SETUP + minimale contextvoorwaarden
Tijdens Sprint 1:
TRADEABLE_SETUP = VALID_SETUP
Reden:


Context layer wordt pas in Sprint 2 geïntroduceerd


Geen premature logica toegestaan


Future-ready definitie (niet actief):
TRADEABLE_SETUP =    VALID_SETUP    AND context_strength OK    AND validation_passed
Conform technische analyse 

3.3 Validation Reason
Elke validatie moet expliciet verklaren:
validation_reason ∈ {    "valid_breakout",    "valid_pullback",    "invalid_structure",    "missing_data",    "no_setup"}
👉 Geen black-box beslissingen toegestaan

4. Technical Specification
4.1 Output Bestand
data/processed/validation_layer.csv

4.2 Kolommen
KolomTypeBeschrijvingtickerstringunieke identifierdatedatetimetimestamp van validatievalid_setupbooltechnische validiteittradeable_setupboolverhandelbaarheidvalidation_reasonstringexpliciete reden
Conform TSD specificatie 

4.3 Data Sources
Input komt uit:
data/processed/scanner_ranked.csv
Bevat:


setup_type


setup_grade


entry / stop / target


technische indicatoren



4.4 Build Mechanisme
Nieuw script:
scripts/core/build_validation_layer.py
Functionaliteit:


Lees scanner_ranked.csv


Evalueer VALID_SETUP


Bepaal TRADEABLE_SETUP


Genereer validation_reason


Schrijf naar validation_layer.csv



4.5 Pipeline Position
Nieuwe flow:
scanner   ↓validation_layer   ↓watchlist   ↓portfolio   ↓reporting
👉 Decision engine blijft onaangeraakt

5. Implementation Plan
Stap 1 — Validation Logic isoleren


Nieuwe module creëren


Geen wijzigingen in bestaande scripts



Stap 2 — Script toevoegen
Nieuw bestand:
scripts/core/build_validation_layer.py

Stap 3 — run_scan.py aanpassen
Toevoegen:
scanner → validation_layer → rest pipeline
Zonder bestaande flow te breken

Stap 4 — Data contracts definiëren
Voor:


scanner_ranked.csv


validation_layer.csv


👉 expliciete schema’s verplicht (Sprint 0 finding) 

Stap 5 — Logging toevoegen


elke validatie wordt gelogd


geen impliciete default values



Stap 6 — Backward Compatibility
Garanderen dat:


watchlist blijft werken zonder validation input


portfolio blijft werken


reporting unchanged blijft


👉 Validation Layer is additief, niet disruptief

6. Data Flow Integration
Nieuwe Pipeline
scanner   ↓validation_layer   ↓watchlist   ↓portfolio   ↓reporting

Belangrijke Regel


Watchlist gebruikt VALID_SETUP als input


Geen enkele module interpreteert validiteit zelf


👉 Centralisatie van technische waarheid

7. Validation Strategy
7.1 Test VALID_SETUP
Controle:


komt VALID overeen met visueel correcte setups?


false positives identificeren



7.2 Test TRADEABLE_SETUP
Tijdens Sprint 1:


moet identiek zijn aan VALID_SETUP



7.3 Impact Meting
Gebruik bestaande validation pipeline:


winrate


average return


hit rate


Segmentatie:
valid_setup = TRUE vs FALSE

7.4 Validation-First Principe
Elke regel:
hypothese → implementatie → meting
Conform delivery framework 

8. Risks & Mitigations
Risico 1 — Foute validatielogica
Impact:


slechte filtering


downstream fouten


Mitigatie:


eenvoudige regels


volledige logging


visuele verificatie



Risico 2 — Pipeline break
Impact:


systeem werkt niet end-to-end


Mitigatie:


additieve implementatie


backward compatibility



Risico 3 — Data inconsistentie
Impact:


fouten in downstream layers


Mitigatie:


expliciete data contracts


schema validatie



9. Definition of Done
Sprint 1 is voltooid wanneer:


validation_layer.csv bestaat


valid_setup correct berekend wordt


tradeable_setup correct berekend wordt


validation_reason volledig is


logging aanwezig is


pipeline end-to-end werkt


geen impact op bestaande decision logic


👉 Conform Definition of Done in framework 

10. Link met toekomstige sprints
Sprint 2 — Context Layer
Validation Layer wordt uitgebreid met:
TRADEABLE_SETUP =    VALID_SETUP    + context_strength

Sprint 3 — Fundamental Layer
Toevoeging:


quality filtering (logging only)


geen impact op validatie



Sprint 4 — Decision Engine
Validation Layer wordt input voor:
if VALID_SETUP and TRADEABLE_SETUP → evaluatie
👉 Cruciaal voor:


centrale besluitvorming


eliminatie decision leakage



Final Statement
Sprint 1 legt de eerste echte bouwsteen van een institutioneel systeem:
👉 expliciete validatie vóór elke beslissing
Zonder deze layer:
→ blijft het systeem reageren
Met deze layer:
→ begint het systeem denken in stappen
Dit is de noodzakelijke overgang van:
signal system
naar:
decision framework