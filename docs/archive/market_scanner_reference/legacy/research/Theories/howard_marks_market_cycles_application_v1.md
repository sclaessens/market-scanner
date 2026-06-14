Rapport — Howard Marks: Market Cycles, Risk & Context

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt de inzichten van Howard Marks (Oaktree Capital) naar concrete regels voor onze trading applicatie.

De kernboodschap:

Je kan de markt niet voorspellen, maar je kan wel begrijpen in welke fase je zit

Waar:

Asness → factoren
Moskowitz → momentum
Chan → validatie
Carver → risk & stabiliteit
Clenow → selectie

voegt Marks toe:

Context bepaalt hoe agressief of defensief je moet handelen

Dit document is cruciaal voor:

Context Strength Layer
Regime interpretatie
Decision Engine gedrag
2. Wie is Howard Marks en waarom is hij relevant?

Howard Marks is co-founder van Oaktree Capital en bekend om zijn memo’s over:

market cycles
risk perception
investor behavior

Hij is geen quant, maar zijn inzichten zijn essentieel omdat:

Hij uitlegt waarom dezelfde strategie anders presteert afhankelijk van de marktfase

Voor onze applicatie betekent dit:

👉 Dezelfde setup (bv. breakout) is niet altijd even goed
👉 Context bepaalt de kans op succes

3. Kernprincipes uit Marks’ werk
3.1 De markt beweegt in cycli

Marks stelt:

Markten bewegen niet lineair, maar in cycli

Typische fases:

- Bullish (optimisme)
- Late bull (overconfidence)
- Correctie
- Bearish (pessimisme)
- Recovery

Belangrijk:

De grootste fouten gebeuren aan de extremen van de cyclus
3.2 Risk ≠ volatiliteit

Marks definieert risico als:

De kans op permanent verlies

Niet:

Dagelijkse schommelingen

Voor onze applicatie:

👉 Volatiliteit (ATR) is niet voldoende
👉 Context moet risico bepalen

3.3 Second-level thinking

Marks maakt onderscheid:

First-level:
"Dit aandeel stijgt"

Second-level:
"Wat verwacht de markt al?"

Voor ons systeem:

Niet alleen kijken naar prijsactie
→ maar ook naar context (overextended, crowded, etc.)
3.4 Extremen zijn gevaarlijk

Marks benadrukt:

De grootste risico’s ontstaan wanneer iedereen hetzelfde denkt

Voorbeeld:

hype stocks
parabolische moves

Voor onze applicatie:

Sterk momentum ≠ altijd goed
3.5 Je moet je gedrag aanpassen aan de omgeving

Marks:

Je strategie hoeft niet te veranderen
Je agressiviteit wel

👉 Dit is een sleutelconcept

4. Vertaling naar onze applicatie
4.1 Introductie van Context Strength Layer

Nieuwe laag:

context_strength

Meet:

Hoe gunstig de marktomgeving is voor momentum
4.2 Context classificatie
STRONG:
- bull market
- trends werken goed

NEUTRAL:
- gemengd beeld

WEAK:
- choppy market
- veel false signals

EXTREME:
- overextended / hype / panic
4.3 Input voor context

Gebaseerd op:

- market_regime (QQQ / SPY)
- breadth (optioneel later)
- trend consistency
- volatility regime
- extension vs MA20

👉 sluit aan op bestaande market_regime.csv

5. Impact op Decision Engine
5.1 Zelfde setup, andere actie
Scenario 1 — STRONG context
→ agressief
→ BUY NOW sneller toegestaan
Scenario 2 — NEUTRAL context
→ selectief
→ alleen top setups
Scenario 3 — WEAK context
→ defensief
→ minder trades
→ hogere drempels
Scenario 4 — EXTREME context
→ voorzichtig
→ vermijden van chase gedrag
5.2 Integratie in beslissingen

Nieuwe logica:

confidence_score × context_multiplier

Voorbeeld:

Sterke setup + zwakke context
→ downgrade

Gemiddelde setup + sterke context
→ mogelijk upgrade
6. Impact op bestaande modules
6.1 Scanner

Geen wijziging:

Scanner blijft setups genereren
6.2 Watchlist

Nieuwe regel:

READY alleen als context niet WEAK is
6.3 Portfolio

Nieuwe rol:

Context bepaalt:
- trim sneller
- hold langer
7. Nieuwe velden in systeem

Toevoegen:

context_strength
context_regime
context_risk_flag
context_comment

Voorbeeld:

context_strength = WEAK
context_risk_flag = HIGH
context_comment = "choppy market, veel false breakouts"
8. Functionele vereisten
FR-HM-001:
Het systeem moet context_strength bepalen.

FR-HM-002:
Context moet beslissingen beïnvloeden.

FR-HM-003:
Sterke setups mogen afgezwakt worden in slechte context.

FR-HM-004:
Context mag geen trades creëren, enkel moduleren.

FR-HM-005:
Extreme markten moeten risk flags genereren.
9. Technische implementatie

Nieuwe module:

scripts/core/context.py

Functies:

compute_context_strength()
detect_extreme_conditions()
apply_context_adjustment()

Integratie:

decision_engine.py
evaluate_watchlist.py
portfolio_engine.py
10. Validatie

Analyse:

- performance per context
- winrate vs context_strength
- breakout success vs regime

Voorbeeld:

Bull vs Bear vs Neutral performance

Doel:

Bewijzen dat context waarde toevoegt
11. Wat we NIET doen
- geen macro voorspellingen
- geen economische forecasting
- geen discretionaire beslissingen
12. Eindconclusie

De belangrijkste les van Howard Marks:

Dezelfde actie is niet altijd juist — het hangt af van de context
Nieuwe rolverdeling
Momentum → richting
Trend → structuur
Ranking → selectie
Fundamentals → kwaliteit
Context → omgeving
Validation → bewijs
Decision → actie