Rapport — Ernest Chan: Practical Algorithmic Trading & Robustness

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt de werkwijze en principes van Ernest Chan naar concrete richtlijnen voor de ontwikkeling van onze trading applicatie.

De kernboodschap:

Een strategie bestaat pas als ze reproduceerbaar, testbaar en robuust is.

Waar eerdere rapporten (Asness, Moskowitz) focussen op wat werkt, focust Chan op:

Hoe je zeker weet dat het écht werkt

Dit document vormt dus de brug tussen theorie en implementatie.

2. Wie is Ernest Chan en waarom is hij cruciaal?

Ernest Chan is één van de weinige quant traders die:

hedge fund ervaring heeft
zelf strategieën bouwt
én deze vertaalt naar code

Zijn boeken (Algorithmic Trading, Machine Trading) focussen op:

backtesting
data integrity
overfitting vermijden
real-world implementatie

Voor onze applicatie is hij essentieel omdat:

Hij voorkomt dat we een “mooi systeem” bouwen dat niet werkt in realiteit
3. Kernprincipes uit Chan’s werk
3.1 Geen strategie zonder validatie

Chan’s belangrijkste regel:

If you can't backtest it, you don't have a strategy

Voor onze applicatie betekent dit:

elke regel
elke filter
elke metric

moet meetbaar zijn via:

- winrate
- average return
- drawdown
- hit ratio

👉 Dit sluit perfect aan op jouw bestaande validation pipeline.

3.2 Simpel > complex

Chan benadrukt:

Complexe modellen lijken beter, maar falen vaker

Waarom?

overfitting
afhankelijk van specifieke data
niet robuust

Voor onze applicatie:

Breakout + trend + volume
> beter dan 15 indicatoren combineren
3.3 Overfitting is het grootste gevaar

Overfitting ontstaat wanneer:

model perfect werkt op historische data
maar faalt in nieuwe data

Chan’s oplossing:

out-of-sample testing
walk-forward analysis
minimale parameters

Voor ons systeem:

thresholds.yaml = risicozone

👉 elke parameter moet kritisch bekeken worden

3.4 Data is belangrijker dan strategie

Chan benadrukt:

Garbage in = garbage out

Problemen:

missing data
survivorship bias
look-ahead bias

Voor onze applicatie:

data/raw + data/processed structuur is kritiek

👉 dit is al goed opgezet in jouw architectuur

3.5 Realistische simulatie

Backtests moeten rekening houden met:

slippage
execution delays
realistische entries

Voor jouw systeem:

entry ≠ perfecte prijs
4. Vertaling naar onze applicatie
4.1 Nieuwe kernregel
Elke feature moet bewezen worden via data
4.2 Validatie wordt een first-class component

Momenteel:

validate_scans.py bestaat

Na Chan integratie:

Validation wordt verplicht voor elke wijziging
4.3 Integratie in pipeline

Huidige flow:

scanner → watchlist → portfolio → decision → reporting

Nieuwe flow:

scanner → validation → watchlist → portfolio → decision → reporting
4.4 Feature lifecycle (zeer belangrijk)

Elke nieuwe feature moet dit pad volgen:

1. Idee (theorie)
2. Implementatie
3. Backtest
4. Analyse
5. Beslissing (houden / verwijderen)

👉 Geen shortcut toegestaan

5. Impact op bestaande modules
5.1 Scanner

Regel:

Geen nieuwe indicator zonder validatie

Voorbeeld:

RSI toevoegen?
→ eerst testen
→ dan pas integreren
5.2 Watchlist

Nieuwe regel:

READY status moet historisch winstgevend zijn

Dus:

READY ≠ logische setup
READY = bewezen winstgevende setup
5.3 Portfolio

Nieuwe focus:

Decision = data-driven

Niet:

"dit voelt goed"

Wel:

historisch gezien werkt dit
6. Nieuwe validatie metrics

Toevoegen aan validation_summary.csv:

winrate
avg_return
max_drawdown
profit_factor
expectancy
sharpe_ratio (optioneel)
7. Overfitting bescherming
7.1 Parameter discipline

Voorbeeld:

MA20 distance = 4.5%

Vraag:

Waarom 4.5 en niet 4.2?

Als je dat niet kan uitleggen:

→ overfitting risico
7.2 Regel
Minimaal aantal parameters
8. Data validatie regels

Functionele regels:

FR-CH-001:
Alle input data moet gecontroleerd worden op volledigheid.

FR-CH-002:
Missing data mag geen crashes veroorzaken.

FR-CH-003:
Look-ahead bias moet vermeden worden.

FR-CH-004:
Backtest mag enkel historische data gebruiken.

FR-CH-005:
Elke metric moet reproduceerbaar zijn.
9. Technische implementatie

Nieuwe structuur:

scripts/validation/
  run_backtest.py
  analyze_results.py
  compare_versions.py
9.1 Backtest engine (basis)

Input:

scanner_ranked.csv
historical prices

Output:

validation_results.csv
9.2 Analyse

Output:

validation_summary.csv
9.3 Vergelijking
version A vs version B
10. Concrete regels voor developers
Geen code zonder test
Geen feature zonder metric
Geen verbetering zonder bewijs
11. Wat we NIET doen
- geen machine learning zonder data volume
- geen complexe modellen zonder noodzaak
- geen “optimalisatie” zonder validatie
- geen indicator stacking
12. Impact op Decision Engine

Decision engine moet gebaseerd zijn op:

historische performance

Voorbeeld:

Als pullback setups:
winrate 55%
avg return +4%

→ vertrouwen verhogen
13. Belangrijkste risico’s die Chan oplost

Zonder Chan:

- overfitting
- random resultaten
- inconsistente beslissingen

Met Chan:

- robuuste strategie
- meetbare edge
- reproduceerbare resultaten
14. Eindconclusie

De belangrijkste les van Ernest Chan:

Een strategie zonder data is geen strategie

Voor onze applicatie betekent dit:

Scanner → ideeën
Validation → bewijs
Decision → actie
Nieuwe rolverdeling
Momentum → detectie
Fundamentals → context
Validation → waarheid
Decision → uitvoering