Sprint 0 — Audit Findings

System Stabilisation & Alignment

1. Executive Summary

Deze audit analyseert de huidige staat van het trading system met als doel:

stabiliteit te garanderen
inconsistenties bloot te leggen
decision leakage te identificeren
de basis te leggen voor een institutionele decision engine

Belangrijkste conclusie:

👉 Het systeem bevat sterke logica, maar:
👉 beslissingslogica is verspreid over meerdere modules

Dit vormt het grootste risico voor:

inconsistent gedrag
conflicterende signalen
onbetrouwbare output
2. System Overview (Current State)

Pipeline:

scanner → watchlist → portfolio → decision engine → reporting

Werkelijke implementatie:

scanner → watchlist (decision) → portfolio (decision) → decision_engine → reporting

👉 De decision engine is niet de enige bron van waarheid

3. Critical Findings
🔴 3.1 Decision Leakage — Watchlist

Betrokken scripts:

evaluate_watchlist.py
parse_watchlist_commands.py
update_watchlist_actions.py

Probleem:

Watchlist genereert acties zoals:

BUY NOW
SET LIMIT BUY
SET STOP BUY
REMOVE

👉 Dit zijn finale beslissingen, geen timing-status.

Impact:

dubbele decision logic
conflict met decision_engine
moeilijk te debuggen gedrag

Risico: 🔴 HIGH

Aanbevolen oplossing:

Watchlist reduceren tot pure state layer
Gebruik enkel:
READY
WAIT_PULLBACK
WAIT_BREAKOUT
WAIT
INVALID
MISSED

Oplossen in: Sprint 1

🔴 3.2 Decision Leakage — Portfolio

Betrokken script:

evaluate_positions.py

Probleem:

Portfolio genereert:

SELL
TRIM
HOLD
REVIEW

👉 Dit zijn echte beslissingen buiten de decision engine.

Impact:

portfolio kan scanner/watchlist overrulen
geen centrale besluitvorming
inconsistent gedrag per ticker

Risico: 🔴 HIGH

Aanbevolen oplossing:

Vervangen door:

EXIT_RISK
TRIM_CANDIDATE
WARNING
HEALTHY

👉 portfolio wordt risk context layer

Oplossen in: Sprint 1

🔴 3.3 Fake Decision Engine in Watchlist

Betrokken script:

update_watchlist_actions.py

Probleem:

Script schrijft:

SIGNAL
UNWATCH
source = "decision_engine" ❗

👉 Dit is een parallelle decision engine

Impact:

dubbele acties
verwarring over bron van waarheid
corruptie van watchlist state

Risico: 🔴 HIGH

Aanbevolen oplossing:

script volledig disablen
OF
verplaatsen naar echte decision engine

Oplossen in: Sprint 1

🔴 3.4 Multiple Decision Sources

Huidige situatie:

Module	Beslissingen
Watchlist	BUY
Portfolio	SELL
Decision Engine	BUY/SELL
Reporting	interpretatie

👉 Minstens 3 decision sources

Impact:

conflicterende output
geen deterministisch gedrag
moeilijk schaalbaar

Risico: 🔴 HIGH

Aanbevolen oplossing:

Decision engine = enige bron van waarheid

Oplossen in: Sprint 4 (volledig), voorbereiding in Sprint 1

🔴 3.5 Pipeline Feedback Loop

Betrokken script:

run_full_pipeline.py

Probleem:

evaluate_watchlist
→ update_watchlist_actions
→ build_watchlist
→ evaluate_watchlist opnieuw

👉 Cyclus zonder centrale controle

Impact:

instabiele outputs
state kan veranderen binnen één run
moeilijk reproduceerbaar

Risico: 🔴 HIGH

Aanbevolen oplossing:

pipeline lineair maken
één evaluatie per layer

Oplossen in: Sprint 1–2

🔴 3.6 Duplicate Portfolio Logic

Betrokken scripts:

build_portfolio.py
portfolio_manager.py

Probleem:

2 verschillende implementaties van:

position building

Impact:

data inconsistentie
moeilijk onderhoud
bugs moeilijk detecteerbaar

Risico: 🔴 HIGH

Aanbevolen oplossing:

één bron behouden
andere verwijderen of deprecaten

Oplossen in: Sprint 1

🟠 3.7 Monolithic Orchestration Script

Betrokken script:

run_scan.py

Probleem:

Script doet:

data fetch
scanning
portfolio
decision
reporting
telegram

👉 alles in één file

Impact:

moeilijk te testen
moeilijk te isoleren
lage modulariteit

Risico: 🟠 MEDIUM

Aanbevolen oplossing:

opsplitsen in orchestrator + modules

Oplossen in: Sprint 2+

🟠 3.8 Data Contract Inconsistency

Probleem:

Niet alle lagen gebruiken dezelfde kolommen:

scanner vs watchlist vs portfolio
impliciete afhankelijkheden

Impact:

breekbare pipeline
moeilijk uitbreidbaar

Risico: 🟠 MEDIUM

Aanbevolen oplossing:

expliciete schema-definitie per CSV

Oplossen in: Sprint 1

🟢 3.9 Reporting Layer Correct

Betrokken scripts:

build_telegram_summary.py
send_telegram.py

Observatie:

gebruikt final_decisions.csv
maakt geen eigen beslissingen

Impact:

correcte scheiding van verantwoordelijkheid

Risico: 🟢 LOW

🟢 3.10 Trade Command System Correct

Betrokken scripts:

parse_trade_commands.py
process_telegram_commands.py

Observatie:

logt enkel trades
geen decision logic

Risico: 🟢 LOW

4. Data Contract Overview
scanner_ranked.csv

Doel: setups genereren
Gebruikt door: watchlist, decision engine

watchlist_status.csv

Doel: timing bepalen
Probleem: bevat acties (moet state worden)

portfolio_review.csv

Doel: risico evalueren
Probleem: bevat beslissingen (moet context worden)

market_regime.csv

Doel: marktcontext
Status: correct

5. Decision Leakage Summary
Locatie	Type
Watchlist	BUY / LIMIT / STOP
Portfolio	SELL / TRIM
Watchlist update	SIGNAL / UNWATCH
Decision engine	finale actie

👉 Leakage confirmed op meerdere lagen

6. Risk Assessment
Risico	Level
Decision leakage	🔴 HIGH
Dubbele logica	🔴 HIGH
Pipeline instability	🔴 HIGH
Data inconsistency	🟠 MEDIUM
Orchestration complexity	🟠 MEDIUM
7. Recommended Actions (By Sprint)
Sprint 1 — Stabilisation Continuation
Watchlist → state only
Portfolio → risk only
Remove update_watchlist_actions
Remove duplicate portfolio logic
Data contracts definiëren
Sprint 2–3
pipeline structureren
context + validation layers
Sprint 4
decision engine centraliseren
alle acties migreren
8. Definition of Done — Sprint 0

✔ Pipeline draait end-to-end
✔ Alle modules geaudit
✔ Decision leakage geïdentificeerd
✔ Dataflows gedocumenteerd
✔ Risico’s in kaart gebracht
✔ Audit document aangemaakt

9. Final Conclusion

Sprint 0 bevestigt:

👉 Het systeem heeft een sterke basis
👉 Maar mist een centrale beslissingsstructuur

De volgende stap is niet optimalisatie van strategie, maar:

controle over beslissingen centraliseren
Status
Sprint: 0
Status: COMPLETE
Ready for: Sprint 1 — Validation Layer