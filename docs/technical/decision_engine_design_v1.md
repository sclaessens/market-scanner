Decision Engine Design v1
1. Purpose

De Decision Engine is de enige bron van waarheid voor alle tradingbeslissingen binnen het systeem.

Het doel van de engine is:

Alle beschikbare context samenbrengen
Eén consistente beslissing per ticker genereren
Conflicterende signalen elimineren
Deterministische, reproduceerbare output leveren

De Decision Engine vormt de kern van het systeem en vervangt alle verspreide beslissingslogica in andere modules.

2. Core Principles
2.1 Single Source of Truth

Alle BUY / SELL / HOLD / TRIM beslissingen worden uitsluitend gegenereerd door de Decision Engine.

Geen enkele andere module mag beslissingen nemen.

2.2 One Decision per Ticker

Voor elke ticker geldt:

exact één final_action per run

Conflicterende signalen zijn niet toegestaan.

2.3 Separation of Concerns
Layer	Verantwoordelijkheid
Scanner	Ideeën genereren
Watchlist	Timing bepalen
Portfolio	Risico evalueren
Regime	Marktcontext
Decision	Beslissing nemen
Reporting	Output presenteren
2.4 Deterministic Output

Bij identieke input moet de output identiek zijn.

Geen randomness, geen impliciete state.

3. Input Layers

De Decision Engine gebruikt vier inputbronnen.

3.1 Scanner (setup context)

Bron: scanner_ranked.csv

Bevat:

ticker
setup_type
setup_grade
score_total
entry
stop
target
rr
extension_atr
breakout_strength
rs_20d_pct

Betekenis:
→ “Is dit een kwalitatieve setup?”

3.2 Watchlist (timing context)

Bron: watchlist_status.csv

Bevat:

ticker
status
timing_state
trigger_type
trigger_price
urgency
reason_code

Toegestane states:

READY
WAIT_PULLBACK
WAIT_BREAKOUT
WAIT
INVALID
MISSED

Betekenis:
→ “Is dit het juiste moment?”

3.3 Portfolio (risk context)

Bron: portfolio_review.csv

Bevat:

ticker
position_state
risk_state
trend_state
pnl_pct
risk_flag

Toegestane risk states:

HEALTHY
WARNING
TRIM_CANDIDATE
EXIT_RISK

Betekenis:
→ “Moet risico verlaagd worden?”

3.4 Market Regime (context layer)

Bron: market_regime.csv

Toegestane waarden:

BULLISH
NEUTRAL
BEARISH
UNKNOWN

Betekenis:
→ “Hoe agressief mag het systeem zijn?”

4. Data Contracts

Alle inputbestanden moeten voldoen aan strikte schema’s.

Ontbrekende kolommen → record wordt genegeerd.

Geen impliciete defaults.

5. Decision Hierarchy

Beslissingen worden genomen volgens prioriteit:

1. Portfolio Risk
2. Existing Exposure
3. Watchlist Timing
4. Scanner Quality
5. Market Regime
5.1 Portfolio heeft absolute prioriteit

Als ticker in portfolio zit:

→ Scanner en Watchlist worden genegeerd
6. Action Model

De Decision Engine mag enkel deze acties produceren:

BUY
PREPARE_BUY
WAIT
HOLD
TRIM
SELL
REMOVE
REVIEW
NO_ACTION
6.1 Betekenis van acties
Action	Betekenis
BUY	Kapitaal inzetten
PREPARE_BUY	Setup klaar, wacht op trigger
WAIT	Geen actie nodig
HOLD	Positie behouden
TRIM	Positie verkleinen
SELL	Positie sluiten
REMOVE	Van watchlist verwijderen
REVIEW	Manuele controle nodig
NO_ACTION	Geen relevante actie
7. Decision Mapping
7.1 Portfolio → Actions
Risk State	Action
EXIT_RISK	SELL
TRIM_CANDIDATE	TRIM
WARNING	REVIEW
HEALTHY	HOLD
7.2 Watchlist → Actions
Timing State	Action
READY	BUY
WAIT_PULLBACK	PREPARE_BUY
WAIT_BREAKOUT	PREPARE_BUY
WAIT	WAIT
INVALID	REMOVE
MISSED	WAIT
7.3 Scanner → Actions
Setup Grade	Action
A	NO_ACTION (candidate)
B/C	NO_ACTION

Scanner genereert nooit directe acties.

8. Confidence Model
HIGH
Watchlist READY
Setup grade A
Regime BULLISH/NEUTRAL
RR ≥ 2
MEDIUM
WAIT_PULLBACK / WAIT_BREAKOUT
Setup grade A/B
Regime ≠ BEARISH
LOW
Zwakke setup
BEARISH regime
Incomplete data
9. Output Contract

Bestand: final_decisions.csv

Kolommen:

ticker
date
source_layer
final_action
execution_style
confidence
setup_type
setup_grade
entry
stop
target
rr
trigger_type
trigger_price
risk_state
regime
reason_code
reason
10. Execution Style
Style	Betekenis
MARKET	Directe aankoop
LIMIT	Wacht op pullback
STOP	Breakout entry
NONE	Geen uitvoering
11. Reporting Mapping

Decision Engine → Reporting:

Engine Action	Telegram
BUY	🔥 ACTIE NU
PREPARE_BUY	📌 SET LIMIT / STOP BUY
SELL	❌ SELL
TRIM	⚖️ TRIM
HOLD	💼 HOLD
12. Non-Goals (v1)

Niet inbegrepen:

Fundamental scoring
Machine learning
Position sizing
Risk budgeting
Multi-asset support
13. Future Extensions
Fundamental Layer integratie
Multi-factor scoring
Confidence weighting
Portfolio allocation logic
Execution engine (orders)
🔥 Final Statement

De Decision Engine is het hart van het systeem.

Alle intelligentie moet hier geconcentreerd worden.

Elke afwijking van deze structuur leidt tot:

inconsistente beslissingen
onbetrouwbare output
verlies van controle
Status
Version: v1
Status: Approved for implementation
Next step: Sprint 1 — Decision Isolation