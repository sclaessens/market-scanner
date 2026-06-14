Sprint 0 — System Stabilisation & Alignment
Doel

De bestaande codebase klaarmaken zodat de nieuwe decision engine betrouwbaar gebouwd kan worden. Deze sprint voegt geen nieuwe strategieën toe. De focus ligt op audit, stabiliteit, consistente outputs en het verwijderen van verborgen beslissingslogica.

De decision engine wordt later de enige bron van waarheid voor acties, zoals BUY, WAIT, HOLD, TRIM en SELL .

Sprint Scope

Deze sprint controleert en stabiliseert:

scripts/core/
scripts/watchlist/
scripts/portfolio/
scripts/reporting/
data/processed/
data/watchlist/
data/portfolio/
reports/daily/
Niet toegestaan in Sprint 0

Tijdens Sprint 0 worden geen nieuwe features gebouwd.

Niet doen:

geen Fundamental Layer bouwen
geen nieuwe confidence logic toevoegen
geen nieuwe decision rules activeren
geen Telegram-output herschrijven
geen scannerstrategie wijzigen zonder auditreden
Werkvolgorde
1. End-to-end baseline run

Voer eerst de bestaande pipeline uit.

python scripts/run_scan.py

Controleer daarna:

data/processed/scanner_ranked.csv
data/watchlist/watchlist_status.csv
data/portfolio/portfolio_review.csv
reports/daily/telegram_message.txt

Doel:

vastleggen hoe het systeem vandaag werkt
bestaande fouten zichtbaar maken
baseline creëren vóór aanpassingen
2. Core scanner audit

Controleer:

scripts/core/scanner.py
scripts/core/regime.py
scripts/core/indicators.py

Auditvragen:

Geeft scanner alleen ideeën, of al impliciete acties?
Worden setup_type en setup_grade consistent gevuld?
Zijn entry, stop, target en rr puur informatief?
Zijn kolommen stabiel voor downstream scripts?
Wordt nergens direct BUY / SELL-logica bepaald?

Acceptatiecriteria:

scanner genereert setups
scanner bepaalt geen finale actie
scanner output is reproduceerbaar
scanner_ranked.csv bevat consistente kolommen
3. Watchlist audit

Controleer:

scripts/watchlist/evaluate_watchlist.py
scripts/watchlist/build_watchlist.py
scripts/watchlist/parse_watchlist_commands.py

Auditvragen:

Is watchlist_status alleen timing/status?
Worden WAIT, READY, REJECTED en EXPIRED consistent gebruikt?
Wordt setup_type correct meegenomen?
Wordt er al actie bepaald zoals BUY NOW of SET LIMIT BUY?
Is die actie tijdelijk toegestaan of moet die later naar decision_engine?

Belangrijke voorbereiding:

watchlist mag voorlopig status geven
finale actie moet later naar decision_engine

Acceptatiecriteria:

watchlist_active.csv is correct
watchlist_status.csv is stabiel
statuswaarden zijn consistent
geen verborgen finale beslissing buiten decision layer

De functionele analyse maakt duidelijk dat scanner ideeën levert, watchlist timing bepaalt, portfolio risico beheert en de decision engine de enige bron van waarheid moet worden .

4. Portfolio audit

Controleer:

scripts/portfolio/build_portfolio.py
scripts/portfolio/evaluate_positions.py
scripts/portfolio/parse_trade_commands.py

Auditvragen:

Worden posities correct opgebouwd uit transacties?
Zijn quantity, avg_cost, pnl_pct en status correct?
Geeft portfolio alleen HOLD/TRIM/SELL-context?
Kan dezelfde ticker tegelijk BUY en TRIM krijgen?

Acceptatiecriteria:

portfolio_positions.csv klopt
portfolio_review.csv bevat één beoordeling per open positie
portfolio-tickers krijgen prioriteit boven scanner-signalen
geen dubbele of conflicterende acties
5. Reporting audit

Controleer:

scripts/reporting/build_telegram_summary.py
scripts/reporting/build_daily_report.py

Auditvragen:

Komt Telegram-output uit meerdere bronnen zonder centrale prioriteit?
Worden scanner, watchlist en portfolio dubbel getoond?
Kan dezelfde ticker in meerdere secties verschijnen?
Is output action-first of data-first?

Acceptatiecriteria:

rapport toont geen conflicterende signalen
één ticker krijgt geen tegenstrijdige boodschap
output is bruikbaar als baseline voor latere Sprint 7
6. Data contract check

Controleer of deze bestanden bestaan en stabiele kolommen hebben:

data/processed/scanner_ranked.csv
data/processed/market_regime.csv
data/watchlist/watchlist_active.csv
data/watchlist/watchlist_status.csv
data/portfolio/portfolio_positions.csv
data/portfolio/portfolio_review.csv

Voor elk bestand vastleggen:

doel
inputbron
outputkolommen
welke scripts het bestand lezen
welke scripts het bestand schrijven

Dit voorkomt dat latere sprints breken wanneer nieuwe layers worden toegevoegd.

7. Decision leakage checklist

Zoek in de code naar directe acties:

grep -R "BUY" scripts/
grep -R "SELL" scripts/
grep -R "TRIM" scripts/
grep -R "HOLD" scripts/
grep -R "READY" scripts/

Beoordeel per resultaat:

Is dit status?
Is dit rapporttekst?
Is dit echte beslissingslogica?
Moet dit later naar decision_engine.py?

Doel:

alle verborgen decision logic identificeren
niet noodzakelijk meteen verwijderen
wel documenteren wat later verplaatst wordt
8. Sprint 0 outputdocument

Maak na de audit dit document:

docs/sprints/sprint_0_audit_findings.md

Inhoud:

gevonden issues
betrokken scripts
risico
aanbevolen fix
welke sprint dit oplost
Definition of Done

Sprint 0 is klaar wanneer:

de huidige pipeline end-to-end draait
alle kernscripts geaudit zijn
data outputs consistent zijn
verborgen decision logic geïdentificeerd is
conflicterende ticker-output gekend is
sprint_0_audit_findings.md is aangemaakt
Resultaat van Sprint 0

Na Sprint 0 is het systeem klaar om gecontroleerd verder te bouwen aan:

Sprint 1 — Validation Layer
Sprint 2 — Context Strength Layer
Sprint 3 — Fundamental Layer
Sprint 4 — Decision Engine
Sprint 5 — Confidence Integration
Sprint 6 — Decision Stability Layer
Sprint 7 — Output & Execution

Sprint 0 is dus geen vertraging. Het is de noodzakelijke fundering waarop de nieuwe versie veilig gebouwd wordt.