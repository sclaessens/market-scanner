Rapport — Joel Greenblatt: Magic Formula (ROIC + Earnings Yield)

Toepassing binnen onze Trading System Architectuur

1. Doel van dit rapport

Dit document vertaalt de “Magic Formula” van Joel Greenblatt naar een concrete, implementeerbare rol binnen onze trading applicatie.

De kernboodschap:

Gebruik eenvoudige, robuuste fundamentals om kwaliteit en waardering te beoordelen — zonder de timing van momentum te vervangen

De Magic Formula is voor ons geen strategie op zich, maar een:

COMBINED FUNDAMENTAL FILTER (Quality + Value)

Dit document dient als input voor:

functioneel analist (regels & gedrag)
technisch analist (berekening & integratie)
2. Wie is Joel Greenblatt en waarom is hij relevant?

Joel Greenblatt is een hedge fund manager en auteur van The Little Book That Still Beats the Market. Zijn Magic Formula is een van de meest bekende en praktisch toepasbare factorstrategieën.

Zijn aanpak:

combineer kwaliteit (ROIC)
met waardering (Earnings Yield)
rangschik aandelen op beide
koop de beste combinaties

Belangrijk inzicht:

Je hoeft geen complexe modellen te bouwen om betere beslissingen te nemen

Voor onze applicatie is dit waardevol omdat:

het perfect aansluit bij onze wens om simpel en robuust te blijven
het direct vertaald kan worden naar code
3. De Magic Formula uitgelegd

De formule gebruikt twee kernmetrics:

3.1 ROIC (Return on Invested Capital)
ROIC = EBIT / Invested Capital

Meet:

hoe efficiënt een bedrijf kapitaal gebruikt
hoeveel winst gegenereerd wordt per geïnvesteerde euro

Interpretatie:

Hoge ROIC → sterke business
Lage ROIC → inefficiënt bedrijf
3.2 Earnings Yield
Earnings Yield = EBIT / Enterprise Value

Meet:

hoeveel “winst” je krijgt per geïnvesteerde euro
inverse van waardering

Interpretatie:

Hoge earnings yield → relatief goedkoop
Lage earnings yield → relatief duur
3.3 Combinatie

De Magic Formula rangschikt:

1. Hoogste ROIC
2. Hoogste Earnings Yield
→ combineert rankings

Belangrijk:

Het gaat niet om absolute waarden, maar om relatieve ranking
4. Waarom deze aanpak werkt

De kracht zit in:

Kwaliteit + waardering tegelijk bekijken

Dit voorkomt:

dure slechte bedrijven
goedkope slechte bedrijven
hype zonder fundament

Voor ons systeem:

Het is de perfecte aanvulling op momentum
5. Vertaling naar onze applicatie
5.1 Belangrijkste regel
Magic Formula mag geen trade trigger zijn

Maar:

Magic Formula bepaalt kwaliteit + waarderingscontext
5.2 Opsplitsing binnen ons systeem

Wij gebruiken de formule NIET als één score, maar splitsen:

ROIC → quality filter
Earnings Yield → valuation check

👉 Dit is cruciaal om consistent te blijven met onze architectuur

5.3 Rol in het systeem
ROIC
Rol:
- FILTER / CONTEXT

Gebruik:
- classificatie van bedrijfskwaliteit

Impact:
- verhoogt of verlaagt confidence
Earnings Yield
Rol:
- CONTEXT

Gebruik:
- detectie over- of undervaluation

Impact:
- voegt risk flag toe
6. Fundamental Profiles (uitgebreid)

We combineren Piotroski + ROIC + Earnings Yield:

BEST:
- hoge ROIC
- sterke Piotroski
- redelijke valuation

GOOD:
- degelijke ROIC
- stabiele fundamentals

MOMENTUM:
- zwakke fundamentals
- sterke prijsactie

RISKY:
- lage ROIC
- slechte fundamentals

EXPENSIVE:
- sterke fundamentals
- extreem hoge waardering
7. Interactie met technische signalen
7.1 Sterk momentum + hoge ROIC + redelijke valuation
→ ideale setup
→ hoge confidence
→ A-grade
7.2 Sterk momentum + hoge ROIC + dure valuation
→ trade toegestaan
→ maar:

- risk_flag
- lagere position sizing (later)
7.3 Sterk momentum + lage ROIC
→ zwakke kwaliteit
→ hogere failure kans
→ downgrade mogelijk
7.4 Zwak momentum + goede fundamentals
→ geen trade

Belangrijk:

Fundamentals bepalen nooit timing
8. Impact op bestaande modules
8.1 Scanner

Geen wijziging:

Scanner blijft momentum-driven
8.2 Scoring

Nieuwe logica:

score_total + quality_adjustment + valuation_adjustment
8.3 Watchlist

Nieuwe regel:

READY setups met slechte ROIC → lagere prioriteit
8.4 Decision Engine
ROIC:
→ verhoogt confidence

Earnings Yield:
→ bepaalt risk_flag
9. Nieuwe velden in systeem

Toevoegen:

roic
earnings_yield
quality_score
valuation_score
valuation_flag

Voorbeeld:

ASML
roic = 28%
earnings_yield = 4.5%
quality_score = HIGH
valuation_flag = FAIR
10. Functionele vereisten
FR-JG-001:
Het systeem moet ROIC berekenen of ophalen.

FR-JG-002:
Het systeem moet earnings yield berekenen.

FR-JG-003:
ROIC moet gebruikt worden als quality filter.

FR-JG-004:
Earnings yield moet gebruikt worden als valuation context.

FR-JG-005:
Geen enkele metric mag een BUY trigger genereren.

FR-JG-006:
Extreem dure aandelen moeten een risk flag krijgen.

FR-JG-007:
Ontbrekende data → UNKNOWN.
11. Technische implementatie

Nieuwe module:

scripts/fundamental/magic_formula.py

Functies:

compute_roic()
compute_earnings_yield()
classify_quality()
classify_valuation()

Input:

financial statements
market cap
enterprise value

Output:

data/processed/fundamental_profiles.csv

Integratie:

score_setups.py
decision_engine.py
reporting
12. Validatie

Analyse:

- winrate vs ROIC buckets
- return vs earnings yield
- high ROIC vs low ROIC performance

Voorbeeld:

ROIC > 20% vs ROIC < 5%

Doel:

Bevestigen dat kwaliteit + value betere setups oplevert
13. Wat we NIET doen
- geen pure Magic Formula strategie
- geen ranking systeem als primaire selectie
- geen value investing systeem
- geen lange termijn holding bias
14. Eindconclusie

De belangrijkste les van Greenblatt:

Koop goede bedrijven tegen een redelijke prijs

Voor onze applicatie:

Momentum zegt wanneer
ROIC zegt hoe goed
Earnings Yield zegt hoe duur
Nieuwe rolverdeling
Momentum → timing
Trend → context
Piotroski → filter
ROIC → kwaliteit
Earnings Yield → valuation check
Validation → bewijs
Decision → actie