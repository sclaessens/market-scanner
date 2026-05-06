Rapport — Andreas Clenow: Momentum Ranking & Systematic Trend Following

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt de aanpak van Andreas Clenow (Stocks on the Move) naar concrete implementatie binnen onze trading applicatie.

De kernboodschap:

Niet alle momentum is gelijk — je moet momentum meten, filteren en rangschikken

Waar:

Moskowitz → momentum werkt
Asness → combineer factoren
Chan → valideer alles

voegt Clenow toe:

Selecteer de BESTE momentum aandelen via ranking

Dit document is cruciaal voor:

verbetering van score_setups.py
selectie van top setups
vermijden van ruis in de scanner
2. Wie is Andreas Clenow en waarom is hij relevant?

Andreas Clenow is hedge fund manager en auteur van Stocks on the Move. Zijn aanpak is:

volledig systematisch
gebaseerd op momentum
maar sterk gefilterd

Zijn edge komt niet van:

"momentum gebruiken"

Maar van:

De juiste momentum aandelen kiezen

Voor onze applicatie betekent dit:

👉 Scanner moet niet alles tonen
👉 Scanner moet de beste setups selecteren

3. Kernprincipes uit Clenow’s werk
3.1 Momentum moet gemeten worden (niet aangenomen)

Clenow gebruikt:

geannualiseerde regressie slope

In plaats van:

simpele return
of enkel MA-structuur

Dit meet:

Hoe sterk en consistent een trend is
3.2 Volatiliteit is cruciaal

Clenow corrigeert momentum voor volatiliteit:

Sterke trend + lage volatiliteit = beste setup

Niet:

Sterke trend + hoge volatiliteit

👉 Dit is een belangrijke verbetering voor jouw systeem

3.3 Ranking is essentieel

Clenow’s grootste bijdrage:

Niet alle setups gebruiken → alleen de top X%

Voorbeeld:

Top 20 momentum aandelen kopen

Voor onze applicatie:

Niet alle A/B setups tonen
→ alleen hoogste scores
3.4 Filters zijn noodzakelijk

Clenow gebruikt filters zoals:

minimum prijs
minimum volume
trend filters

Voor ons systeem:

👉 Dit zit al deels in thresholds.yaml

Maar moet strakker toegepast worden

3.5 Trend + momentum moeten samen gebruikt worden

Belangrijk inzicht:

Momentum zonder trend is zwak
Trend zonder momentum is traag

👉 Beide moeten gecombineerd worden

4. Vertaling naar onze applicatie
4.1 Grootste verandering: Ranking wordt centraal

Vandaag:

scanner → toont setups

Na Clenow:

scanner → rangschikt setups → toont beste
4.2 Nieuwe score structuur

Huidig:

score_total

Nieuw:

score_total =
  trend_score +
  momentum_score +
  volatility_score +
  position_score
4.3 Momentum score verbeteren

In plaats van:

distance_to_high

Toevoegen:

trend_strength (slope-based)
momentum_return (3m / 6m)
4.4 Volatility score (nieuw)
volatility_score =
  inverse van ATR / volatility

Interpretatie:

lage volatiliteit → hogere score
hoge volatiliteit → lagere score
4.5 Ranking mechanisme

Nieuwe regel:

Alle setups sorteren op score_total
→ alleen top N tonen

Bijvoorbeeld:

Top 10 = A setups
Next 20 = B setups
Rest = negeren
5. Impact op bestaande modules
5.1 Scanner

Nieuwe rol:

Scanner = generator + filter + ranker
5.2 score_setups.py (kritische wijziging)

Moet:

- scores normaliseren
- ranking toepassen
- top selectie maken
5.3 Watchlist

Nieuwe regel:

Alleen top-ranked aandelen op watchlist
5.4 Decision Engine

Nieuwe regel:

Alleen high-ranked setups → BUY NOW of PREPARE
6. Nieuwe velden in systeem

Toevoegen aan scanner_ranked.csv:

trend_strength
momentum_return_3m
momentum_return_6m
volatility_score
rank
percentile_rank

Voorbeeld:

NVDA
trend_strength = 0.85
momentum_6m = +45%
volatility_score = 0.70
rank = 3
percentile = 98%
7. Filtering regels
Min price
Min volume
Trend intact
Niet onder MA50
Niet extreem extended

👉 Deze moeten strikt toegepast worden vóór ranking

8. Functionele vereisten
FR-CL-001:
Het systeem moet momentum meten op basis van returns of trend.

FR-CL-002:
Het systeem moet volatiliteit meenemen in scoring.

FR-CL-003:
Alle setups moeten gerangschikt worden.

FR-CL-004:
Alleen top-ranked setups mogen getoond worden.

FR-CL-005:
Ranking moet consistent zijn over tijd.

FR-CL-006:
Filters moeten toegepast worden vóór ranking.
9. Technische implementatie

Aanpassingen in:

scripts/core/score_setups.py

Nieuwe functies:

compute_trend_strength()
compute_momentum_score()
compute_volatility_score()
rank_setups()

Flow:

scanner_candidates.csv
→ scoring
→ ranking
→ scanner_ranked.csv
10. Validatie

Analyse:

- top 10 vs rest performance
- rank vs return
- volatility vs winrate

Voorbeeld:

Top 5 ranked setups vs bottom 50%

Doel:

Bewijzen dat ranking waarde toevoegt
11. Wat we NIET doen
- geen complexe regressie modellen (optioneel later)
- geen overfitting van ranking
- geen te veel parameters
12. Eindconclusie

De belangrijkste les van Clenow:

Het gaat niet om momentum vinden, maar om het beste momentum selecteren
Nieuwe rolverdeling
Momentum → detectie
Trend → bevestiging
Volatiliteit → filtering
Ranking → selectie
Fundamentals → kwaliteit
Validation → bewijs
Decision → actie