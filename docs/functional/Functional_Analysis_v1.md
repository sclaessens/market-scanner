1. Executive Overview (UPDATED)

Het systeem evolueert van:

→ signal generator
naar
→ decision engine met kwaliteitsfiltering

Nieuwe kernprincipes:

Momentum bepaalt WANNEER
Fundamentals bepalen OF
Context bepaalt HOE

Belangrijke upgrade:

Het systeem maakt nu expliciet onderscheid tussen:
signaal validiteit
signaal kwaliteit
signaal betrouwbaarheid

Doel van deze versie:

False positives reduceren
Drawdowns beperken
Consistente beslissingen afdwingen
Kapitaal allocatie verbeteren

Zoals gedefinieerd door de PM: het systeem moet evolueren naar een institutioneel beslissingsframework

2. System Scope & Boundaries (UPDATED)

Toevoeging:

Het systeem maakt nu expliciet onderscheid tussen:

Signal Layer → scanner
Validation Layer → fundamentals + context
Decision Layer → finale actie

Nieuwe boundary:

Fundamentals mogen NOOIT timing bepalen
Fundamentals mogen NOOIT een trade triggeren
3. Core Pipeline Behaviour (UPDATED)

Pipeline blijft:

scanner → watchlist → portfolio → decision engine → reporting

Maar:

👉 Decision engine wordt nu de centrale autoriteit

Nieuwe regel:

Scanner = ideeën
Watchlist = timing
Portfolio = risico
Fundamentals = kwaliteit
Decision engine = ENIGE bron van waarheid
4. Functional Definition per Layer (UPDATED)
4.1 Trend Phase Layer (verfijnd)

Blijft grotendeels gelijk, maar:

Nieuwe regel:

EXTENDED fase vereist extra validatie via fundamentals
EARLY fase tolereert zwakkere fundamentals
4.2 Setup Interpretation Layer (belangrijke update)

Nieuwe interpretatie:

👉 Setup is nu technisch geldig, maar nog niet “tradeable”

Nieuwe concepten:

VALID_SETUP
TRADEABLE_SETUP

Definitie:

VALID_SETUP:
→ voldoet aan technische criteria

TRADEABLE_SETUP:
→ VALID_SETUP + context + fundamentals OK

4.3 Decision Stability Layer (strenger)

Toevoeging:

Status verandering vereist:
2 dagen bevestiging
OF sterke confluence (trend + context + fundamentals)

Nieuwe regel:

Hoe zwakker fundamentals → hoe meer bevestiging vereist
4.4 Context Strength Layer (uitgebreid)

Input:

rs_20d_pct
relative strength vs sector (NIEUW)

Nieuwe output:

STRONG / NEUTRAL / WEAK / LEADING

LEADING:
→ outperformt zowel markt als sector

Gedrag:

Alleen STRONG of LEADING → full conviction trades
WEAK → alleen toegestaan met uitzonderingen
4.5 Fundamental Quality Layer (VOLLEDIG HERWERKT)

Dit is de belangrijkste upgrade.

Zoals gedefinieerd door de financial analyst:

👉 Fundamentals = stabilisator, geen driver

Input
Piotroski Score
Earnings Yield
EV/EBITDA
ROIC (NIEUW)
Output

fundamental_profile ∈
{BEST, GOOD, MOMENTUM, RISKY, UNKNOWN}

Definitie

BEST:

Piotroski ≥ 7
ROIC hoog
geen extreme valuation

GOOD:

Piotroski ≥ 5
stabiele metrics

MOMENTUM:

neutrale fundamentals
sterke prijsactie

RISKY:

Piotroski ≤ 3
lage profitability

UNKNOWN:

data ontbreekt
Gedrag (CRUCIAAL)

BEST:
→ verhoogt confidence
→ lagere failure rate

GOOD:
→ standaard gedrag

MOMENTUM:
→ trade toegestaan
→ hogere volatiliteit

RISKY:
→ trade toegestaan
→ maar:

lagere conviction
strengere exit

UNKNOWN:
→ neutraal behandelen

HARD RULES
Fundamentals blokkeren GEEN trades volledig
EXCEPT:
EXTREME zwakte + zwakke context
Fundamentals bepalen:
confidence
risicoclassificatie
toekomstige position sizing
5. Decision Logic (VOLLEDIG HERWERKT)
NIEUW CONCEPT: CONFIDENCE LEVEL

Elke beslissing krijgt:

confidence ∈ {HIGH, MEDIUM, LOW}

BUY

Voorwaarden:

watchlist.status = READY (CONFIRMED)
VALID_SETUP = TRUE
context_strength ∈ {STRONG, LEADING}
trend_phase ≠ EXTENDED

Fundamental impact:

BEST → HIGH confidence
GOOD → MEDIUM
MOMENTUM → MEDIUM/LOW
RISKY → LOW

WAIT

Nieuwe regels:

VALID_SETUP maar niet TRADEABLE
onvoldoende confirmation
EXTENDED fase
REMOVE

Nieuwe regels:

technische invalidatie
OF combinatie:
WEAK trend
RISKY fundamentals
WEAK context
HOLD

Uitbreiding:

afhankelijk van fundamentals:
BEST → langer HOLD
RISKY → sneller exit
TRIM

Nieuwe regel:

EXTENDED + lage fundamental kwaliteit
→ agressiever trimmen
SELL

Uitbreiding:

SELL wordt getriggerd door:

technische breuk
OF:
trend_phase = WEAK
fundamentals = RISKY + momentum verzwakt
REVIEW

Nieuwe definitie:

conflict tussen:
sterke fundamentals
zwakke prijsactie
6. Interaction Rules (NIEUW — ZEER BELANGRIJK)

Sterk technisch + BEST fundamentals
→ HIGH confidence BUY

Sterk technisch + RISKY fundamentals
→ LOW confidence BUY
→ snellere exit

Zwak technisch + sterke fundamentals
→ GEEN trade

Sterk technisch + WEAK context
→ WAIT

👉 Dit komt rechtstreeks uit de financiële analyse

7. State Transitions (UPDATED)

Nieuwe toevoeging:

READY → BUY is afhankelijk van:

CONFIRMATION
CONFIDENCE ≥ MEDIUM
8. Edge Cases (UITGEBREID)

Nieuwe cases:

High momentum + slechte fundamentals
→ trade toegestaan maar:

kortere holding periode

Extreme valuation (lage earnings yield)
→ WAIT of TRIM

9. Output Behaviour (UPDATED)

Nieuwe verplichte velden per ticker:

actie
reden
trend_phase
context_strength
fundamental_profile
confidence_level

Telegram output wordt:

ACTIE NU
(alleen HIGH en MEDIUM confidence)

VOORBEREIDEN
(WAIT setups)

RISICO / VERWIJDEREN
(low quality / rejected)

PORTFOLIO

10. Governance Rules (NIEUW)
Geen enkele metric mag toegevoegd worden zonder validatie
Metrics blijven enkel indien:
winrate stijgt
drawdown daalt
consistentie stijgt
Fundamentals blijven:
filter
nooit driver
11. Slot (Strategische shift)

De grootste verandering is:

Van:
→ “Welke setups zijn er?”

Naar:
→ “Welke setups verdienen kapitaal?”