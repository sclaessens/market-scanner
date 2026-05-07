1. Executive Overview

Dit systeem is een systematische, momentum-gedreven decision engine waarin financiële theorie niet gebruikt wordt om signalen te genereren, maar om:

signalen te selecteren
signalen te valideren
signalen te stabiliseren
risico te moduleren

De architectuur blijft:

scanner → watchlist → portfolio → decision → reporting

Maar de interne logica evolueert naar een multi-layer factor framework waarin elke laag een strikt afgebakende rol heeft.

Kernprincipe

Momentum genereert kansen
Ranking selecteert kansen
Fundamentals beoordelen kwaliteit
Context bepaalt agressiviteit
Risk stabiliseert beslissingen
Validation bepaalt wat blijft bestaan

Deze structuur volgt direct uit:

factor hiërarchie (Asness)
implementatie discipline (Alpha Architect)
validatie noodzaak (Chan)
2. Investment Philosophy
2.1 Core Positionering

Het systeem is expliciet:

Momentum-first, factor-enhanced, context-aware

Dit betekent:

Momentum = enige driver van trades
Fundamentals = filters en confidence modifiers
Context = gedragsaanpassing
Risk = stabiliteit, niet voorspelling
2.2 Strikte Hiërarchie
Momentum → bepaalt WANNEER
Trend → bepaalt RICHTING
Ranking → bepaalt WELKE
Quality → bepaalt BETROUWBAARHEID
Value → bepaalt RISICO (extremen)
Context → bepaalt AGRESSIVITEIT
Risk → bepaalt STABILITEIT
Validation → bepaalt BESTAANSRECHT
Benchmark → bepaalt RELATIEVE EDGE

➡️ Dit voorkomt factor-conflicten en volgt AQR’s multi-factor logica zonder strategieverwarring

3. Integrated Factor Framework
3.1 Momentum (Moskowitz / TSMOM)

Wat meet het?
Trendpersistentie over meerdere tijdframes

Waarom?
Momentum is een structureel fenomeen in markten

Wanneer werkt het?
Trending markets, multi-timeframe alignment

Gebruik in systeem:

momentum_1m / 3m / 6m / 12m
momentum_consistency_score
trend_phase

➡️ Momentum bepaalt setup validiteit en timing

3.2 Trend (Trend Phase Layer)

Wat meet het?
Structurele richting van de markt

Waarom?
Momentum zonder trend = instabiel
Trend zonder momentum = traag

Gebruik:

STRONG_UPTREND → permissief
SIDEWAYS → restrictief
DOWNTREND → blokkeren

➡️ Trend bepaalt of setups überhaupt toegestaan zijn

3.3 Ranking (Clenow)

Wat meet het?
Relatieve sterkte van setups

Waarom?
Niet alle momentum is gelijk

Wanneer werkt het?
Altijd — selectie is cruciaal

Gebruik:

score_total =

trend_score
momentum_score
volatility_score
position_score

Top X% = actionable setups

➡️ Ranking bepaalt welke setups zichtbaar zijn

3.4 Quality (Piotroski + ROIC)

Wat meet het?
Financiële gezondheid en efficiëntie

Waarom?
Quality verhoogt duurzaamheid van trends

Gebruik:

Piotroski score
ROIC

➡️ Output:

BEST / GOOD / MOMENTUM / RISKY / UNKNOWN

➡️ Quality beïnvloedt confidence en downgrade/upgrade

3.5 Value (Earnings Yield / EV-EBITDA)

Wat meet het?
Waardering

Waarom?
Detecteert extremen, niet kansen

Gebruik:

valuation_flag
extreme overvaluation detection

➡️ Value beïnvloedt:

risk flags
position sizing (later)

➡️ Value bepaalt NOOIT een trade

3.6 Volatility (Carver / Quantpedia)

Wat meet het?
Risico en ruis

Waarom?
Lage volatiliteit = betere risk-adjusted returns

Gebruik:

ATR
volatility_score

➡️ Impact:

ranking
risk flags
later position sizing
3.7 Context (Howard Marks)

Wat meet het?
Marktomgeving en cyclus

Waarom?
Zelfde setup ≠ zelfde kans

Gebruik:

STRONG / NEUTRAL / WEAK / EXTREME

➡️ Context bepaalt:

aggressiveness
filtering
downgrade/upgrade
4. System Layer Mapping
Scanner

Rol:

detectie (momentum + setups)
filtering
ranking

Toevoegingen:

momentum layer
ranking layer
volatility integration
Watchlist

Rol:

timing engine

Nieuwe regels:

setup-aware logica (kritisch)
trend + momentum consistentie vereist
context filtering
Portfolio

Rol:

risk management (Carver)

Toevoegingen:

volatility-based beoordeling
stability checks
geen nieuwe BUY signals
Decision Engine

Rol:

centrale actie-logica

Nieuwe structuur:

combineert alle lagen
lost conflicten op

Prioriteit:
Portfolio > Watchlist > Scanner

5. Decision Logic Framework
5.1 Core formule
confidence_score =
  momentum_strength
+ trend_alignment
+ ranking_percentile
+ quality_adjustment
- volatility_penalty
± context_adjustment
5.2 Scenario’s
Sterk signaal + slechte context

→ downgrade
→ WAIT / kleinere positie

Sterk signaal + zwakke fundamentals

→ toegestaan
→ risk_flag
→ lagere confidence

Zwak signaal + sterke fundamentals

→ GEEN trade

Sterk signaal + sterke fundamentals + sterke context

→ HIGH CONFIDENCE
→ BUY NOW

5.3 Output mapping
Confidence	Actie
80–100	BUY NOW
60–80	SET ORDER
40–60	WAIT
<40	REMOVE
6. Fundamental Profiles

Gebaseerd op:

Piotroski
ROIC
Earnings Yield
Profielen
BEST → ideaal momentum profiel
GOOD → stabiel
MOMENTUM → prijs gedreven
RISKY → hoge failure kans
UNKNOWN → neutraal

➡️ Alleen impact op confidence, niet op entry

7. Risk & Stability Layer (Carver)
Kernconcepten
signal_strength ≠ zekerheid
stabiliteit belangrijker dan entry
Componenten
signal_stability
volatility_score
confidence_score

➡️ Beslissingen worden gradueel, niet binair

8. Validation & Benchmark Framework
Validation (Chan)

Elke component moet:

winrate
avg return
drawdown

bewijzen leveren

➡️ Geen validatie = geen feature

Benchmark (Quantpedia)

Vergelijk met:

momentum strategies
quality strategies
low volatility

➡️ Doel:

weten of systeem competitief is
Research discipline (Alpha Architect)

Elke feature:

formule
data
test
beslissing

➡️ Geen black box

9. Data & Implementation Constraints
Nodige data
OHLCV
fundamentals
market regime
Beperkingen
fundamentals = lagging
missing data → UNKNOWN
geen forward estimates
Fallback
geen data = neutraal behandelen
10. Do’s & Don’ts
DO
gebruik momentum als enige trigger
valideer elke wijziging
gebruik ranking voor selectie
gebruik context voor gedrag
gebruik fundamentals voor filtering
DON’T
geen BUY op basis van fundamentals
geen indicator stacking
geen overfitting
geen black-box modellen
geen strategie zonder benchmark
EINDCONCLUSIE

Dit systeem evolueert van:

→ losse indicatoren

naar:

→ geïntegreerd, factor-based decision system

Waar:

momentum = motor
ranking = selectie
fundamentals = kwaliteit
context = omgeving
risk = stabiliteit
validation = waarheid

De echte edge ontstaat niet uit één factor, maar uit:

de consistente interactie tussen alle lagen