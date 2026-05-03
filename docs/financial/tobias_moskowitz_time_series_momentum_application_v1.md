Rapport — Tobias Moskowitz: Time Series Momentum

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt het onderzoek van Tobias Moskowitz (in samenwerking met o.a. Cliff Asness en Lasse Pedersen) rond Time Series Momentum (TSMOM) naar concrete ontwerpprincipes voor onze applicatie.

Het doel is:

momentum robuuster maken
inconsistent gedrag (flip-flop) verminderen
trenddetectie structureel verbeteren

De output van dit document dient als input voor:

functioneel analist (logica en regels)
technisch analist (implementatie en dataflow)
2. Kern van Time Series Momentum
Definitie

Time Series Momentum stelt:

De richting van een asset in het verleden voorspelt met statistische significantie de richting in de nabije toekomst.

Concreet:

Als een asset de afgelopen X maanden positief rendement heeft:
→ verhoogde kans dat de trend doorzet

En omgekeerd:

Negatieve trend → verhoogde kans op verdere daling
3. Belangrijkste inzichten uit het onderzoek
3.1 Momentum is universeel

TSMOM werkt over:

aandelen
obligaties
grondstoffen
valuta

Belangrijk inzicht:

Momentum is geen anomaly → het is een structureel marktfenomeen

Voor onze applicatie:

momentum is niet afhankelijk van één setup (breakout/pullback)
het is een diepere structurele eigenschap van markten
3.2 Trend werkt op meerdere tijdschalen

TSMOM toont dat momentum werkt op:

korte termijn (1–3 maanden)
middellange termijn (3–12 maanden)
langere termijn (>12 maanden)

Belangrijk:

Momentum is robuuster wanneer meerdere tijdframes dezelfde richting bevestigen

👉 Dit is cruciaal voor onze applicatie

3.3 Simpele signalen outperformen complexe modellen

TSMOM gebruikt extreem eenvoudige logica:

sign(momentum) = +1 of -1

Geen complexe indicatoren, geen overfitting.

Conclusie:

Simpel momentum > complexe indicatoren
3.4 Momentum is persistent, maar niet stabiel

Momentum werkt:

gemiddeld zeer goed
maar kent drawdowns
kan tijdelijk falen

Daarom:

Momentum moet gefilterd en gestabiliseerd worden

👉 Dit is exact waarom jouw systeem een Decision Stability Layer nodig heeft

4. Vertaling naar onze applicatie
4.1 Grootste verandering: Momentum ≠ enkel setup

Vandaag gebruikt jouw systeem momentum impliciet via:

breakout
pullback
MA-structuur

Na integratie van TSMOM wordt momentum expliciet een aparte laag:

Momentum Layer (nieuw)
→ bepaalt richting en consistentie
4.2 Introductie van de Trend Phase Layer

Op basis van Moskowitz definiëren we:

trend_phase:
- STRONG_UPTREND
- WEAK_UPTREND
- SIDEWAYS
- WEAK_DOWNTREND
- STRONG_DOWNTREND

Gebaseerd op:

- return_1m
- return_3m
- return_6m
- return_12m
Voorbeeld logica
Als:
1M > 0
3M > 0
6M > 0
12M > 0

→ STRONG_UPTREND
Als:
korte termijn positief
lange termijn negatief

→ WEAK_UPTREND (instabiel)
4.3 Momentum consistentie (zeer belangrijk)

Nieuw concept:

momentum_consistency_score

Meet:

Hoeveel tijdframes dezelfde richting tonen

Voorbeeld:

1M +, 3M +, 6M +, 12M +
→ score = 4/4 = sterk

1M +, 3M +, 6M -, 12M -
→ score = 2/4 = zwak

Gebruik:

Sterke consistentie → hogere confidence
Lage consistentie → lagere betrouwbaarheid
4.4 Impact op bestaande scanner

Belangrijk:

Scanner blijft setups detecteren

Maar:

Momentum layer bepaalt of setups betrouwbaar zijn
Voorbeeld
Scenario 1 — Breakout + sterke momentum consistentie
- breakout boven 20D high
- trend_phase = STRONG_UPTREND
- momentum_consistency = hoog

→ HIGH CONFIDENCE setup
→ A-grade mogelijk
Scenario 2 — Breakout + zwakke trend
- breakout
- trend_phase = SIDEWAYS
- inconsistent momentum

→ LAGE betrouwbaarheid
→ B of C setup
Scenario 3 — Pullback in sterke trend
- prijs nabij MA20
- STRONG_UPTREND
- momentum consistent

→ IDEALE setup

👉 Dit is exact waar TSMOM zijn kracht toont

5. Nieuwe velden voor het systeem

Toe te voegen aan scanner_ranked.csv:

trend_phase
momentum_1m
momentum_3m
momentum_6m
momentum_12m
momentum_consistency_score
momentum_direction

Voorbeeld:

NVDA
trend_phase = STRONG_UPTREND
momentum_1m = +12%
momentum_3m = +28%
momentum_6m = +45%
momentum_12m = +110%
momentum_consistency_score = 4
momentum_direction = UP
6. Impact op Watchlist Engine

Momenteel:

READY = dicht bij MA20

Na integratie:

READY = pullback + STRONG_UPTREND
Nieuwe regels
Pullback:
- STRONG_UPTREND → READY
- WEAK_UPTREND → WAIT
- SIDEWAYS → REJECTED

Breakout:
- STRONG_UPTREND → valide
- SIDEWAYS → twijfel
- DOWNTREND → reject
7. Impact op Decision Engine

De decision layer moet momentum gebruiken als stabilisator:

BUY NOW:
→ alleen bij sterke trend + consistente momentum

WAIT:
→ bij inconsistente momentum

REMOVE:
→ bij negatieve trend shift
8. Functionele vereisten
FR-MO-001:
Momentum moet gemeten worden over meerdere tijdframes.

FR-MO-002:
Trend phase moet expliciet bepaald worden.

FR-MO-003:
Momentum consistentie moet berekend worden.

FR-MO-004:
Technische setups mogen niet los staan van trend phase.

FR-MO-005:
Sterke setups in zwakke trend moeten gedegradeerd worden.

FR-MO-006:
Pullbacks mogen alleen READY worden in uptrend.

FR-MO-007:
Breakouts in downtrend zijn ongeldig.
9. Technische implementatie

Nieuwe module:

scripts/core/momentum.py

Functies:

compute_momentum_returns()
compute_trend_phase()
compute_momentum_consistency()

Input:

data/processed/{ticker}_indicators.csv

Output:

data/processed/{ticker}_momentum.csv

Integratie:

scan_market.py → voegt momentum data toe
score_setups.py → gebruikt momentum in scoring
evaluate_watchlist.py → gebruikt trend_phase
decision_engine.py → gebruikt consistency

Belangrijk:

Geen bestaande code herschrijven
→ enkel uitbreiden
10. Validatie

Analyse:

- winrate per trend_phase
- avg return per momentum_consistency
- breakout success vs trend_phase
- pullback success vs trend_phase

Voorbeeld:

Pullback + STRONG_UPTREND vs Pullback + SIDEWAYS

Doel:

Bewijzen dat TSMOM waarde toevoegt
11. Wat we NIET doen
- geen long/short strategie
- geen asset allocation model
- geen futures-based momentum
- geen leverage
- geen pure TSMOM strategie

Waarom:

Onze applicatie blijft:

stock-based
long-only
decision system
12. Eindconclusie

De belangrijkste les van Tobias Moskowitz:

Momentum is geen setup
Momentum is een structurele kracht die trends aandrijft

Voor onze applicatie betekent dit:

Momentum moet:
- expliciet gemodelleerd worden
- consistent gemeten worden
- gebruikt worden als stabilisator

Nieuwe rolverdeling:

Momentum → richting + consistentie
Trend phase → context
Setups → entry
Fundamentals → kwaliteit
Decision → actie