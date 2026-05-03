Rapport — Cliff Asness / AQR: Factor Investing als basis voor onze Financial Layer
1. Doel van dit rapport

Dit rapport vertaalt de werkwijze en onderzoeken van Cliff Asness en AQR Capital Management naar concrete principes voor onze trading scanner applicatie.

De kernboodschap is:

Momentum blijft de primaire motor van het systeem. Value en quality worden gebruikt als aanvullende factoren om de betrouwbaarheid van technische signalen te beoordelen, niet om timingbeslissingen over te nemen.

Dit sluit rechtstreeks aan bij onze huidige projectvisie: scanner → ideeën, watchlist → timing, portfolio → risk, decision → actie .

2. Wie is Cliff Asness en waarom is hij relevant?

Cliff Asness is medeoprichter van AQR Capital Management en één van de bekendste namen binnen systematisch factor investing. Zijn werk focust vooral op evidence-based factoren zoals momentum, value, quality, defensive investing en carry. AQR beschrijft deze factoren als dominante factoren binnen academische en praktische asset pricing modellen.

Voor ons project is Asness vooral relevant omdat hij niet vertrekt vanuit losse indicatoren, maar vanuit herhaalbare, testbare en economisch verklaarbare factoren. Dat is exact wat we nodig hebben om onze scanner te verbeteren zonder indicator overload te creëren.

3. Belangrijkste theoretische inzichten
3.1 Momentum is een bewezen factor

In het onderzoek “Value and Momentum Everywhere” tonen Asness, Moskowitz en Pedersen aan dat value- en momentum premies consistent voorkomen over meerdere markten en asset classes. Het onderzoek stelt ook dat value en momentum vaak negatief gecorreleerd zijn, waardoor ze elkaar kunnen aanvullen in een robuuster systeem.

Voor onze applicatie betekent dit:

Momentum blijft het core signal. Onze technische signalen zoals breakout, pullback, VCP, relatieve sterkte en trendstructuur blijven dus leidend.

Concreet gebruiken we momentum in:

scanner:
- trend_ok
- momentum_ok
- close > MA20 / MA50 / MA200
- distance_to_high
- breakout_strength
- relative strength
- setup_grade

De scanner mag dus blijven doen wat hij nu al doet: ideeën genereren op basis van prijsactie en momentum. Dat is consistent met onze roadmap waarin scanner intelligence draait rond pullbacks, breakouts, VCP en scoring .

3.2 Value werkt, maar mag momentum niet vervangen

Asness’ werk toont dat value een afzonderlijke premie heeft, maar ook dat value en momentum samen sterker kunnen zijn dan elk apart. Belangrijk: dat betekent niet dat een momentum systeem plots een value systeem moet worden.

Voor onze applicatie betekent dit:

Value wordt geen entry trigger. Een aandeel wordt niet gekocht omdat het goedkoop is. Een aandeel komt alleen in aanmerking als de technische setup sterk genoeg is.

Value gebruiken we enkel als sanity check:

Sterk momentum + redelijke waardering = hogere betrouwbaarheid
Sterk momentum + extreme waardering = verhoogd risico
Zwak momentum + goedkope waardering = geen trade

Dit voorkomt dat we een pure value strategie introduceren. Dat is belangrijk omdat onze bestaande edge momentum-based is.

3.3 Quality is de beste fundamentele overlay

AQR’s “Quality Minus Junk” definieert quality als kenmerken waarvoor beleggers rationeel meer zouden willen betalen: winstgevendheid, groei, veiligheid en goed management. Het onderzoek vindt dat high-quality stocks historisch sterke risk-adjusted returns hebben geleverd in de VS en internationaal.

Voor onze applicatie is dit waarschijnlijk de belangrijkste les.

Quality is beter bruikbaar dan pure value, omdat quality goed combineert met momentum. Een aandeel dat stijgt én financieel sterk is, heeft vaak een betere kans om zijn trend vol te houden dan een aandeel dat alleen stijgt op hype.

Daarom gebruiken we quality als factor overlay in de Fundamental Layer.

4. Wat wij concreet overnemen in onze applicatie
4.1 Factor hiërarchie

Onze applicatie krijgt deze vaste hiërarchie:

1. Momentum = timing en setup detectie
2. Trend = marktstructuur en richting
3. Quality = betrouwbaarheid van het bedrijf
4. Value = sanity check tegen extreme overwaardering
5. Decision stability = voorkomt flip-flop gedrag

Geen enkele fundamentele factor mag een core signal worden.

4.2 Momentum als core signal

Momentum blijft verantwoordelijk voor:

- kandidaatselectie
- setup detectie
- breakout / pullback / VCP classificatie
- entry timing
- watchlist status
- order type

Voorbeeld:

Aandeel X:
- close boven MA20, MA50, MA200
- dicht bij 20D high
- sterke relatieve sterkte
- volume bevestigt beweging

=> Momentum setup blijft geldig

Fundamentals mogen dit signaal niet creëren, alleen versterken of verzwakken.

4.3 Quality als confidence filter

Wij gebruiken quality om te bepalen of een technische setup meer of minder vertrouwen verdient.

Mogelijke input metrics:

- ROIC
- gross margin / operating margin
- earnings growth
- debt ratio
- Piotroski Score
- free cash flow positief / negatief

Output:

quality_profile:
- HIGH_QUALITY
- GOOD
- NEUTRAL
- LOW_QUALITY
- UNKNOWN

Gebruik in decision engine:

Sterk technisch signaal + HIGH_QUALITY:
- hogere confidence
- setup mag A blijven

Sterk technisch signaal + LOW_QUALITY:
- setup blijft mogelijk
- maar krijgt risk flag
- eventueel grade verlagen van A naar B
4.4 Value als sanity check

Value gebruiken we niet om goedkope aandelen te zoeken. We gebruiken value om extreme risico’s te detecteren.

Mogelijke metrics:

- earnings_yield
- EV/EBITDA
- price_to_sales

Gebruik:

Als valuation extreem duur is:
- geen automatische reject
- wel risk flag
- lagere confidence
- mogelijk kleinere positie in latere position sizing

Voorbeeld:

Momentum sterk
Quality goed
Valuation extreem duur

=> trade toegestaan, maar niet als “clean best setup”
5. Nieuwe velden voor onze applicatie

Voor de functioneel en technisch analist stel ik voor dat we later deze kolommen toevoegen aan scanner_ranked.csv of een aparte fundamental_profiles.csv.

ticker
fundamental_profile
quality_score
value_score
momentum_score
factor_confidence
valuation_risk_flag
quality_risk_flag
factor_comment

Voorbeeld:

NVDA
fundamental_profile = MOMENTUM_QUALITY
quality_score = 82
value_score = 35
factor_confidence = HIGH
valuation_risk_flag = EXPENSIVE_BUT_SUPPORTED
quality_risk_flag = NONE
6. Impact op bestaande decision engine

Onze roadmap zegt dat de decision layer moet evolueren naar één duidelijke beslissing per ticker, met prioriteit Portfolio > Watchlist > Scanner . De factorlaag moet daarin alleen een ondersteunende rol krijgen.

De decision engine mag dus geen beslissing nemen zoals:

BUY omdat Piotroski hoog is
BUY omdat EV/EBITDA laag is
BUY omdat earnings yield hoog is

Wel toegestaan:

BUY NOW omdat technische setup READY is
confidence verhoogd omdat fundamentals sterk zijn
risk flag toegevoegd omdat valuation extreem is
7. Regels voor interactie tussen technische signalen en factoren
Sterk technisch signaal + sterke quality

Actie:

- setup blijft geldig
- confidence omhoog
- hogere ranking
- mag A-grade blijven

Interpretatie:

Dit is het ideale profiel voor ons systeem.

Sterk technisch signaal + zwakke quality

Actie:

- setup niet automatisch verwijderen
- risk flag toevoegen
- grade eventueel verlagen
- sneller naar REVIEW bij zwakte

Interpretatie:

Momentum kan nog steeds werken, maar failure risk is hoger.

Zwak technisch signaal + sterke quality

Actie:

- geen BUY
- eventueel op watchlist
- wachten op technisch signaal

Interpretatie:

Goede bedrijven zijn niet automatisch goede trades.

Zwak technisch signaal + zwakke quality

Actie:

- negeren
- geen watchlist prioriteit
- geen scanner highlight

Interpretatie:

Geen edge.

8. Wat wij expliciet niet overnemen

Wij nemen niet over:

- pure long/short factor portfolios
- market neutral implementation
- short leg van quality-minus-junk
- volledige AQR multi-factor portfolio constructie
- complexe optimalisatie
- leverage
- factor timing als aparte strategie

Waarom niet?

Omdat onze applicatie geen institutioneel multi-asset long/short fonds is. Het is een practical decision system voor aandelen, met scanner, watchlist, portfolio en reporting. Onze architectuur is modulair opgebouwd rond scanner, watchlist, portfolio en decision/report layer .

9. Functionele vereisten voor de analist

De functioneel analist moet deze regels opnemen:

FR-FA-001:
Het systeem gebruikt momentum als primaire setup driver.

FR-FA-002:
Fundamentele factoren mogen nooit zelfstandig een BUY signaal genereren.

FR-FA-003:
Quality metrics mogen setup confidence verhogen of verlagen.

FR-FA-004:
Value metrics mogen alleen gebruikt worden als valuation sanity check.

FR-FA-005:
Een technisch zwakke setup mag nooit worden goedgekeurd op basis van fundamentals.

FR-FA-006:
Elke factor moet gevalideerd worden op winrate, average return en drawdown.

FR-FA-007:
Ontbrekende fundamentele data leidt tot UNKNOWN, niet tot REJECTED.
10. Technische implicaties

De technisch analist moet dit vertalen naar een aparte module, niet rechtstreeks in de scannerlogica.

Aanbevolen module:

scripts/fundamental/
  fetch_fundamentals.py
  build_factor_scores.py
  classify_fundamental_profile.py

Output:

data/processed/fundamental_profiles.csv

Daarna pas koppelen in:

scripts/core/score_setups.py
scripts/core/decision_engine.py
scripts/reporting/build_telegram_summary.py

Belangrijk: de bestaande scanner mag niet herschreven worden. De factorlaag moet als extra input naast de technische score komen.

11. Validatie

Elke factor moet apart getest worden.

Minimum analyse:

- winrate per fundamental_profile
- average return per profile
- max drawdown per profile
- target hit rate per profile
- stop hit rate per profile

Voorbeeld:

A-grade momentum + HIGH_QUALITY
vs
A-grade momentum + LOW_QUALITY

Alleen als HIGH_QUALITY structureel betere resultaten geeft, mag de factor zwaarder meewegen.

Dit sluit aan bij de bestaande roadmap waarin validation en edge moeten meten wat werkt en slechte setups moeten elimineren .

12. Eindconclusie

De belangrijkste les van Cliff Asness voor onze applicatie is:

Gebruik bewezen factoren, maar geef elke factor een duidelijke rol.

Voor ons betekent dit:

Momentum = core signal
Trend = structuur
Quality = betrouwbaarheid
Value = risico-check
Validation = bewijs

De applicatie mag dus geen verzameling losse indicatoren worden. Ze moet een strak factor-based decision system worden waarin momentum de motor blijft en fundamentals alleen helpen bepalen welke signalen betrouwbaarder zijn.