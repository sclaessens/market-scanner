Rapport — Joseph Piotroski: F-Score als Fundamental Filter

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt het onderzoek van Joseph Piotroski (F-score model) naar concrete implementatie binnen onze trading applicatie.

De kernboodschap:

Fundamentals mogen geen signalen creëren, maar moeten slechte bedrijven elimineren

Piotroski is daarom geen “alpha generator” in ons systeem, maar een:

QUALITY FILTER

Dit document dient als input voor:

functioneel analist (regels & gedrag)
technisch analist (berekening & integratie)
2. Wie is Joseph Piotroski en waarom is hij relevant?

Joseph Piotroski is een accounting professor die het F-score model ontwikkelde om:

financieel sterke bedrijven te onderscheiden van zwakke bedrijven

Zijn onderzoek toont:

binnen goedkope aandelen (value stocks)
bedrijven met hoge F-score outperformen
bedrijven met lage F-score underperformen

Belangrijk inzicht:

Niet alle “goedkope” aandelen zijn goed → kwaliteit maakt het verschil

Voor ons systeem:

👉 Niet alle momentum aandelen zijn gelijk
👉 sommige zijn structureel zwak

3. De Piotroski F-score uitgelegd

De F-score bestaat uit 9 binaire signalen:

Score = 0 tot 9

Drie categorieën:

3.1 Winstgevendheid (Profitability)
+1 als ROA > 0
+1 als cashflow > 0
+1 als cashflow > net income
+1 als ROA stijgt
3.2 Leverage / Liquiditeit
+1 als leverage daalt
+1 als current ratio stijgt
+1 als geen nieuwe aandelen uitgegeven
3.3 Operationele efficiëntie
+1 als gross margin stijgt
+1 als asset turnover stijgt
Interpretatie
8–9 → zeer sterk
6–7 → goed
4–5 → neutraal
0–3 → zwak
4. Waarom Piotroski werkt

De kracht zit niet in complexiteit, maar in:

Combinatie van simpele maar relevante signalen

Het model detecteert:

improving fundamentals
financieel gezonde bedrijven
vermijden van “value traps”
5. Vertaling naar onze applicatie
5.1 Belangrijkste regel
Piotroski mag NOOIT een trade trigger zijn

Maar:

Piotroski bepaalt of we een aandeel vertrouwen
5.2 Rol in het systeem
Rol:
- FILTER

Gebruik:
- classificatie van aandelenkwaliteit

Impact:
- beïnvloedt confidence_score
- beïnvloedt setup_grade
5.3 Integratie in pipeline

Nieuwe flow:

scanner → fundamentals → watchlist → portfolio → decision

Fundamental layer:

build_piotroski_score.py

Output:

fundamental_profiles.csv
6. Fundamental Profiles (op basis van Piotroski)
BEST:
F-score 8–9

GOOD:
F-score 6–7

NEUTRAL:
F-score 4–5

WEAK:
F-score 0–3

UNKNOWN:
geen data
7. Interactie met technische signalen
7.1 Sterk momentum + hoge F-score
→ beste scenario
→ hoge confidence
→ A-grade mogelijk
7.2 Sterk momentum + lage F-score
→ trade toegestaan
→ maar:

- lagere confidence
- hogere failure kans
- snellere exit nodig
7.3 Zwak momentum + hoge F-score
→ geen trade

Belangrijk:

Goede fundamentals ≠ goed moment om te kopen
7.4 Zwak momentum + lage F-score
→ volledig negeren
8. Impact op bestaande modules
8.1 Scanner

Geen wijziging:

Scanner blijft momentum-driven
8.2 Scoring

Nieuwe logica:

A-grade + F-score laag → downgrade naar B
B-grade + F-score hoog → mogelijk upgrade
8.3 Watchlist

Nieuwe regel:

READY setups met lage F-score → extra voorzichtig
8.4 Decision Engine
HIGH F-score:
→ confidence omhoog

LOW F-score:
→ risk_flag
→ mogelijk kleinere positie (later)
9. Nieuwe velden in systeem

Toevoegen:

piotroski_score
fundamental_profile
quality_flag

Voorbeeld:

ASML
piotroski_score = 8
fundamental_profile = BEST
quality_flag = STRONG
10. Functionele vereisten
FR-PI-001:
Het systeem moet Piotroski F-score berekenen per aandeel.

FR-PI-002:
De score moet gebruikt worden als filter, niet als signaal.

FR-PI-003:
Aandelen met lage score mogen niet automatisch uitgesloten worden.

FR-PI-004:
Score moet invloed hebben op confidence.

FR-PI-005:
Ontbrekende data moet leiden tot UNKNOWN.

FR-PI-006:
Fundamentals mogen nooit een BUY trigger genereren.
11. Technische implementatie

Nieuwe module:

scripts/fundamental/piotroski.py

Functies:

compute_piotroski_score()
classify_fundamental_profile()

Input:

financial statements data

Output:

data/processed/fundamental_profiles.csv

Integratie:

score_setups.py
decision_engine.py
reporting
12. Validatie

Analyse:

- winrate per F-score bucket
- avg return per F-score
- drawdown vs F-score

Voorbeeld:

F-score 8–9 vs F-score 2–3

Doel:

Bevestigen dat hoge kwaliteit betere setups oplevert
13. Wat we NIET doen
- geen pure value strategie
- geen filtering op “goedkoop”
- geen uitsluiting van lage scores
- geen overcomplexe accounting metrics
14. Eindconclusie

De belangrijkste les van Piotroski:

Niet alle bedrijven zijn gelijk — kwaliteit bepaalt duurzaamheid

Voor onze applicatie:

Momentum bepaalt wanneer we kopen
Piotroski bepaalt of we het vertrouwen
Nieuwe rolverdeling
Momentum → timing
Trend → context
Fundamentals → kwaliteit
Piotroski → filter
Validation → bewijs
Decision → actie