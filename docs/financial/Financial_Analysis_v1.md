FINANCIAL MODEL & THEORY INTEGRATION DOCUMENT
1. Executive Overview

De rol van financiële theorie binnen dit systeem is niet om signalen te genereren, maar om:

kwaliteit van setups te beoordelen
risico te moduleren
betrouwbaarheid van momentum te verhogen

Het systeem is fundamenteel een price-driven decision engine, gebouwd rond:

trend continuation
breakout behavior
pullback entries

Fundamentals worden toegevoegd als een Context Strength Layer binnen de bestaande pipeline:

scanner → watchlist → portfolio → decision engine → reporting

Deze laag:

beïnvloedt confidence
beïnvloedt position sizing (later)
beïnvloedt filtering van zwakke setups

Maar:

bepaalt NOOIT timing
vervangt NOOIT technische signalen

Dit is essentieel om de bestaande edge (momentum) te behouden.

Dit sluit aan bij de architectuur waarbij de core market engine verantwoordelijk blijft voor detectie en scoring, en extra lagen enkel context toevoegen .

2. Investment Philosophy Alignment
Positionering

Het systeem volgt expliciet deze hiërarchie:

Momentum → bepaalt timing
Fundamentals → bepalen kwaliteit
Interpretatie
Momentum = wanneer de markt beweegt
Fundamentals = of de beweging duurzaam is
Waarom deze combinatie werkt

Momentum strategieën werken omdat:

kapitaal stroomt naar winnaars
trends zichzelf versterken

Fundamentals verbeteren dit door:

zwakke bedrijven uit te filteren
drawdowns te beperken
false breakouts te reduceren
Wat NIET de bedoeling is
geen DCF-based beslissingen
geen intrinsieke waarde berekeningen
geen lange termijn holding bias
geen “goedkoop = koop” logica

Dit is geen value investing systeem, maar een:

momentum system met quality filters

3. Selection of Financial Concepts
3.1 Piotroski Score

Wat meet het?
Financiële gezondheid via 9 signalen:

winstgevendheid
leverage
liquiditeit
operationele efficiëntie

Waarom relevant?

identificeert bedrijven met verbeterende fundamentals
reduceert kans op “junk rallies”

Wanneer werkt het goed?

mid/long term trends
bull en neutral regimes

Wanneer minder goed?

hyper-growth stocks (lage score maar sterke prijsactie)
early breakouts

Rol binnen momentum systeem

filter tegen zwakke bedrijven
verhoogt betrouwbaarheid van setups

Waarom gekozen?

bewezen academisch model
eenvoudig te implementeren
binary interpretatie mogelijk

Waarom alternatieven niet gekozen zijn

Altman Z-score → meer bankruptcy focus, minder relevant
pure profitability ratios → missen dynamiek
3.2 Earnings Yield (E/P)

Wat meet het?

inverse van P/E
earnings vs prijs

Waarom relevant?

voorkomt extreme overvaluation
detecteert “crowded trades”

Sterktes

eenvoudig
breed beschikbaar
goed voor sanity check

Beperkingen

earnings manipulatie mogelijk
cyclicals vertekend

Relevantie voor momentum

voorkomt kopen van extreme blow-off tops
helpt bij risk classification

Waarom gekozen?

robuust
makkelijk te integreren

Waarom niet gekozen

P/E → minder stabiel bij losses
PEG → afhankelijk van forecast (onbetrouwbaar)
3.3 EV/EBITDA

Wat zegt deze ratio?

enterprise value vs operationele winst

Waarom nuttig?

capital structure neutraal
beter dan earnings bij groei bedrijven

Wanneer nuttig?

sectorvergelijkingen
sanity check valuation

Wanneer minder nuttig?

financials
bedrijven met lage EBITDA

Gebruik in systeem

detectie van extreme valuation
vergelijking binnen universe

Waarom gekozen?

industry standard
robuust voor screening
3.4 Return on Capital (ROIC)

Wat meet het?

efficiëntie van kapitaalgebruik

Waarom relevant?

identificeert kwaliteitsbedrijven
ondersteunt duurzame trends

Wanneer waardevol?

long trends
compounders

Beperkingen

backward-looking
sector afhankelijk

Waarom gekozen?

sterke economische betekenis
consistent met “quality momentum”
4. Role of Each Metric in the System
Piotroski Score
Rol:
- FILTER

Gebruik:
- minimum threshold (bv. ≥ 5)

Impact:
- zwakke bedrijven worden uitgesloten
Earnings Yield
Rol:
- CONTEXT

Gebruik:
- detectie extreme overvaluation

Impact:
- verlaagt confidence bij extreme values
EV/EBITDA
Rol:
- CONTEXT

Gebruik:
- sector-relative vergelijking

Impact:
- voorkomt irrationele entries
ROIC
Rol:
- FILTER / CONTEXT

Gebruik:
- high quality tagging

Impact:
- verhoogt vertrouwen in setups

Belangrijk:

GEEN enkele metric = CORE SIGNAL
5. Fundamental Profiles
BEST
hoge Piotroski
hoge ROIC
redelijke valuation

Gedrag:
sterke trend continuation, lage failure rate

GOOD
solide fundamentals
geen extreme waardering

Gedrag:
betrouwbare momentum

MOMENTUM
neutrale fundamentals
sterke prijsactie

Gedrag:
werkt, maar hogere volatiliteit

RISKY
zwakke fundamentals
slechte profitability

Gedrag:
meer false breakouts

UNKNOWN
ontbrekende data

Gedrag:
neutraal behandelen

ENTRY TIMING QUALITY (NIEUW)

Binnen momentum strategieën is niet alleen richting belangrijk,
maar ook timing van entry.

Observatie:

Late entries (sterk extended boven moving averages)
hebben:

- slechtere risk/reward
- hogere kans op pullbacks
- lagere expectancy

Implicatie voor systeem:

Entry quality moet expliciet gevalideerd worden
binnen de technische validation layer.

Belangrijk:

Dit is GEEN fundamentele evaluatie,
maar een technische optimalisatie van entry timing.

6. Interaction with Technical Signals
Sterk technisch + sterke fundamentals
hoogste confidence
volledige validatie
Sterk technisch + zwakke fundamentals
trade toegestaan
maar:
lagere conviction
mogelijk snellere exit
Zwak technisch + sterke fundamentals
GEEN trade

Fundamentals geven geen timing.

Zwak technisch + zwakke fundamentals
volledig negeren
7. Limitations & Risks

Fundamentals:

zijn traag
reageren op kwartaaldata
geven geen entry signal

Risico’s:

data lag
accounting noise
verkeerde interpretatie

Grootste risico:

overconfidence door “goede fundamentals”

8. Data Availability & Practical Constraints

Realistisch beschikbare data:

earnings
EBITDA
balance sheet metrics

Niet gebruiken:

complexe modellen (DCF)
forward estimates zonder betrouwbaarheid

Fallback:

missing data → UNKNOWN profiel
9. Validation Framework

Impact wordt gemeten via bestaande validation pipeline:

winrate
average return
drawdown
setup performance

Deze pipeline is al aanwezig in het systeem via validation scripts en analyse outputs .

Teststrategie:

splitsen per fundamental profile
vergelijken:
BEST vs RISKY
GOOD vs UNKNOWN

Beslissingsregel:

metric blijft alleen als:
winrate stijgt
drawdown daalt
consistency verbetert
10. Clear Do’s & Don’ts
DO:
- gebruik fundamentals als filter
- gebruik ze om confidence te moduleren
- gebruik ze om risico te verlagen

DON’T:
- gebruik fundamentals als entry trigger
- vervang momentum signalen niet
- bouw geen value strategy in een momentum systeem
- voeg geen metrics toe zonder validatie
Slotconclusie

Dit document definieert een strikte rolverdeling:

Momentum = driver
Fundamentals = stabilisator

De kracht van het systeem zit in:

duidelijke scheiding van verantwoordelijkheden
vermijden van indicator overload
meetbare impact via validation

Correct geïmplementeerd zal deze laag:

false positives verminderen
drawdowns beperken
consistentie verhogen

zonder de bestaande edge te compromitteren.