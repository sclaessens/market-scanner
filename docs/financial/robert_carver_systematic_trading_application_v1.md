Rapport — Robert Carver: Systematic Trading, Risk & Position Sizing

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt de principes uit Systematic Trading van Robert Carver naar concrete richtlijnen voor onze applicatie.

Waar:

Asness → factoren (wat werkt)
Moskowitz → momentum (waarom trends werken)
Chan → validatie (bewijzen dat het werkt)

voegt Carver de ontbrekende laag toe:

Hoe je beslissingen stabiel maakt en risico beheert

Dit rapport is cruciaal voor:

Decision Stability Layer
Portfolio Engine
toekomstige position sizing
2. Wie is Robert Carver en waarom is hij relevant?

Robert Carver is een systematic trader met ervaring bij grote hedge funds (o.a. AHL / Man Group) en focust op:

robuuste trading systemen
risicogestuurde beslissingen
eenvoudige maar consistente modellen

Zijn kernfilosofie:

Niet de beste voorspelling winnen, maar de meest consistente uitvoering

Voor onze applicatie betekent dit:

minder focus op “perfecte entry”
meer focus op stabiele beslissingen over tijd
3. Kernprincipes uit Carver’s werk
3.1 Forecasts zijn onzeker

Carver vertrekt van:

Elke trading signal = een onzekere forecast

Dus:

een breakout is geen zekerheid
een pullback is geen zekerheid

Daarom:

Beslissingen mogen niet binair zijn (BUY / NOT BUY)

👉 Dit is een directe oplossing voor jouw huidige probleem:

flip-flop gedrag
inconsistente signalen
3.2 Schalen van signalen (forecast scaling)

Carver gebruikt geen binaire signalen, maar:

Continuous signals (bv. -20 tot +20)

Voorbeeld:

Sterke trend → +15
Zwakke trend → +5
Geen trend → 0

Voor onze applicatie:

score_total → moet geïnterpreteerd worden als sterkte, niet als label
3.3 Diversificatie van signalen

Carver combineert:

meerdere strategieën
meerdere signalen
meerdere timeframes

Belangrijk:

Niet vertrouwen op één type signaal

Voor ons systeem:

breakout
pullback
trend
momentum

→ moeten samen werken, niet apart

👉 Dit sluit perfect aan op jouw multi-layer aanpak

3.4 Position sizing is belangrijker dan entry

Carver’s belangrijkste inzicht:

Hoeveel je koopt is belangrijker dan wanneer je koopt

Waarom?

fouten gebeuren altijd
risk bepaalt survival

Voor onze applicatie:

Position sizing is nog niet geïmplementeerd → maar moet voorbereid worden

3.5 Risk is centraal, niet optioneel

Carver behandelt risico als:

de kern van het systeem, niet een bijzaak

Voorbeeld:

volatiliteit bepaalt positie
drawdown bepaalt gedrag

Voor ons systeem:

ATR en volatility moeten centrale rol krijgen
4. Vertaling naar onze applicatie
4.1 Decision Stability Layer (nieuw kerncomponent)

Op basis van Carver introduceren we:

decision_stability_score

Meet:

Hoe consistent een signaal is over tijd
Voorbeeld
Dag 1: BUY
Dag 2: WAIT
Dag 3: BUY

→ instabiel signaal → lage confidence
4.2 Van binaire naar graduele beslissingen

Vandaag:

READY / WAIT / REJECTED

Na Carver:

confidence_score (0–100)
Nieuwe structuur
- signal_strength
- momentum_consistency
- fundamental_quality
- volatility_risk

→ combineren tot confidence_score
4.3 Integratie in Decision Engine

Nieuwe logica:

HIGH confidence → BUY NOW
MEDIUM → SET ORDER
LOW → WAIT
VERY LOW → REMOVE

👉 Geen harde switches meer

4.4 Volatility als centrale factor

Carver gebruikt volatiliteit voor:

position sizing
risk control

Voor ons:

ATR wordt belangrijker dan nu

Gebruik:

extension_atr
risk_flag
volatility_bucket
5. Voorbereiding op position sizing

Hoewel nog niet direct geïmplementeerd, moeten we nu al voorbereiden:

Nieuwe velden:

volatility_score
position_risk_unit
recommended_position_size
Concept
Hoge volatiliteit → kleinere positie
Lage volatiliteit → grotere positie
6. Impact op bestaande modules
6.1 Scanner

Geen grote wijziging:

Scanner blijft signalen genereren

Maar:

score_total moet interpreteerbaar zijn als sterkte
6.2 Watchlist

Nieuwe regel:

READY alleen bij stabiel signaal
6.3 Portfolio

Nieuwe rol:

Portfolio = risk manager

Niet alleen:

HOLD / SELL

Maar ook:

hoe groot is de positie?
7. Nieuwe velden in het systeem

Toevoegen:

confidence_score
signal_stability
volatility_score
risk_bucket
8. Functionele vereisten
FR-RC-001:
Beslissingen mogen niet binair zijn.

FR-RC-002:
Elk signaal moet een confidence score krijgen.

FR-RC-003:
Stabiliteit van signalen moet gemeten worden.

FR-RC-004:
Volatiliteit moet meegenomen worden in beslissingen.

FR-RC-005:
Sterkere signalen moeten meer gewicht krijgen.

FR-RC-006:
Inconsistente signalen moeten gedegradeerd worden.
9. Technische implementatie

Nieuwe module:

scripts/core/risk_model.py

Functies:

compute_confidence_score()
compute_signal_stability()
compute_volatility_score()

Integratie:

decision_engine.py → gebruikt confidence_score
portfolio_engine.py → gebruikt volatility_score
10. Validatie

Analyse:

- winrate per confidence bucket
- return vs volatility
- stability vs performance

Voorbeeld:

HIGH confidence vs LOW confidence setups
11. Wat we NIET doen
- geen complexe optimalisatie
- geen dynamic leverage
- geen hedge fund level portfolio construction

Waarom:

Systeem moet simpel en robuust blijven
12. Eindconclusie

De belangrijkste les van Robert Carver:

Stabiliteit en risicobeheer bepalen je succes, niet je entry
Nieuwe rolverdeling
Momentum → richting
Trend → context
Fundamentals → kwaliteit
Validation → bewijs
Risk model → stabiliteit
Decision → actie