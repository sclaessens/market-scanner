Research Document — Entry Quality Validation Layer
Sectie 1 — Executive Summary

De kernbevinding is duidelijk: momentum werkt, maar late entries verslechteren de risk/reward. Professionele systemen kopen niet zomaar “sterkte”; ze kopen gecontroleerde sterkte: een trend/breakout die valide is, maar nog niet parabolisch of te ver verwijderd van een rationeel referentiepunt zoals MA20, breakout level of ATR-band.

Academisch momentumonderzoek ondersteunt vooral intermediate momentum: 3–12 maanden relatieve sterkte, vaak met uitsluiting van de meest recente maand om kortetermijnruis en reversal-effecten te vermijden. Jegadeesh & Titman documenteerden significante winnaars-minus-verliezers rendementen over 3–12 maanden, terwijl Quantpedia momentum definieert als 12-maands rendement met skip van de meest recente maand.

Voor ons systeem betekent dit:

Grootste fout vandaag: validatie is nog te binair: A setup = valid, zonder nuance rond extension, breakout proximity of volume. Dat probleem is ook expliciet vastgesteld in Sprint 1: geen onderscheid tussen sterke/zwakke breakouts, geen controle op extension en geen volume-confirmatie.

Belangrijkste upgrade: de Validation Layer moet geen “meer signalen” zoeken, maar slechte entries verwijderen.

Sectie 2 — Research Findings
2.1 AQR / Moskowitz — Time-Series Momentum

AQR/Moskowitz toont dat time-series momentum robuust bestaat over meerdere asset classes, met positieve return persistence over 1 tot 12 maanden en gedeeltelijke reversal op langere horizons.

Vertaling:

Momentum zelf is valide, maar het onderzoek ondersteunt trend persistence, niet blind najagen van de laatste candle. Late-stage overextension is niet hetzelfde als duurzame trendkwaliteit.

2.2 Jegadeesh & Titman / Quantpedia — Cross-Sectional Momentum

De klassieke equity momentumregel is 3–12 maanden relatieve sterkte, vaak met skip van de meest recente maand. Dat is belangrijk: de meest recente beweging bevat vaak microstructure noise, kortetermijnreversal en overreactie.

Vertaling:

Een breakout direct na een extreme korte sprint moet strenger beoordeeld worden dan een breakout uit compressie.

2.3 Barroso & Santa-Clara / Daniel & Moskowitz — Momentum Crashes

Momentum heeft sterke gemiddelde returns, maar ook crashrisico. Barroso & Santa-Clara tonen dat momentumrisico tijdsvariabel en voorspelbaar is; volatiliteitsbeheer reduceert crashes en verhoogt Sharpe. Daniel & Moskowitz tonen dat momentum crashes vooral ontstaan na marktdalingen, in hoge volatiliteit en tijdens rebounds.

Vertaling:

Volatility expansion en ATR-extension zijn geen detailmetingen. Ze zijn noodzakelijke risicofilters.

2.4 Moving Average Distance Research

Onderzoek naar Moving Average Distance toont dat de afstand tussen korte en lange moving averages voorspellende waarde heeft boven klassieke momentum- en 52-week-high signalen.

Vertaling:

Afstand tot MA20/MA50/MA200 mag niet alleen visueel gebruikt worden. Het moet expliciet gelogd en gevalideerd worden.

2.5 Clenow — Stocks on the Move

Clenow gebruikt momentum-ranking, trendfilters, ATR en volatility-adjusted sizing. Zijn aanpak vermijdt het idee dat “hoogste momentum = automatisch beste koop”. ATR wordt gebruikt om risico te normaliseren en entries niet los te zien van volatiliteit.

Vertaling:

Gebruik extension_atr, distance_ma20_pct en volatility_expansion als entry-quality metrics.

2.6 Turtle Trading

Turtle-systemen kopen breakouts, maar mechanisch: N-day high entries, ATR-gebaseerde stops, en pyramiding pas na verdere bevestiging, typisch +0.5 ATR increments.

Vertaling:

Een breakout-entry hoort dicht bij de breakout-trigger. Als prijs al meerdere ATR’s boven het breakout level staat, is het geen initiële breakout-entry meer maar een late chase.

Sectie 3 — Entry Quality Framework
Goede entry

Een goede breakout-entry heeft:

Breakout proximity
Prijs ligt dicht bij het breakout level.
Controlled extension
Prijs is niet extreem ver boven MA20.
Volatility confirmation
Volume/volatiliteit bevestigt de move, maar zonder blow-off karakter.
Trend intact
MA20 > MA50 of prijs boven relevante moving averages.
Risk/reward nog intact
Stopafstand is niet zo groot dat RR instort.
Slechte entry

Een slechte entry is:

Breakout al ver voorbij trigger.
Close meerdere ATR’s boven MA20.
Grote candle na sterke meerdaagse run.
Volume spike zonder consolidatie.
Entry vereist brede stop, waardoor RR verslechtert.
Sectie 4 — Concrete Rules
Rule 1 — Breakout proximity
distance_to_breakout_pct = (close - breakout_level) / breakout_level * 100

Aanbevolen ranges:

0% tot 2%      = valid_breakout_entry
2% tot 4%      = late_but_acceptable
> 4%           = extended_breakout_reject

Voor liquid large caps kan 2–3% redelijk zijn. Voor volatiele growth stocks moet dit beter via ATR gebeuren.

Rule 2 — ATR extension vanaf breakout level
breakout_extension_atr = (close - breakout_level) / atr14

Aanbevolen ranges:

0.0 tot 0.75 ATR     = ideal
0.75 tot 1.5 ATR     = acceptable
1.5 tot 2.0 ATR      = late
> 2.0 ATR            = reject_chase

Institutionele logica: Turtle-achtige systemen voegen pas toe in stappen van ongeveer 0.5 ATR na initiële entry; een eerste entry >2 ATR boven trigger is dus geen zuivere breakout-entry meer.

Rule 3 — Extension vanaf MA20
extension_atr = (close - ma20) / atr14
distance_ma20_pct = (close - ma20) / ma20 * 100

Aanbevolen ranges:

extension_atr <= 1.0       = healthy
1.0 tot 2.0                = stretched
2.0 tot 3.0                = extended
> 3.0                      = parabolic / reject

Voor distance_ma20_pct:

0% tot 5%       = healthy
5% tot 10%      = stretched
> 10%           = extended, alleen nog watchlist/pullback

Belangrijk: ATR is beter dan pure procentafstand, omdat het rekening houdt met volatiliteit.

Rule 4 — Volume confirmation
volume_ratio = volume / avg_volume_20

Aanbevolen ranges:

1.2 tot 2.5     = healthy confirmation
2.5 tot 4.0     = aggressive / possible exhaustion
> 4.0           = blowoff_risk
< 1.0           = weak_breakout

Volume is bevestiging, maar extreme volume spikes na een grote prijsrun kunnen juist exhaustion signaleren.

Rule 5 — Candle expansion / range expansion
range_atr = (high - low) / atr14
close_position = (close - low) / (high - low)

Aanbevolen:

range_atr <= 1.5             = normal
1.5 tot 2.5                  = expansion
> 2.5                        = exhaustion risk

close_position >= 0.6        = acceptable
close_position < 0.5         = weak close / failed breakout risk
Sectie 5 — Mapping naar ons systeem

De Validation Layer moet volgens jullie architectuur technische geldigheid bepalen, expliciet loggen en geen decision logic bevatten. Dat past bij de bestaande afspraak dat de Validation Layer valid_setup, tradeable_setup en validation_reason schrijft naar validation_layer.csv.

Te behouden

Behouden:

scanner → validation_layer → context_layer → watchlist → portfolio → decision_engine → reporting

Het delivery framework vereist layer-based development, validation-first werken en verbiedt decision logic buiten de Decision Engine.

Toe te voegen kolommen in validation_layer.csv
ticker
date
valid_setup
tradeable_setup
validation_reason

entry_quality_score
distance_to_breakout_pct
breakout_extension_atr
extension_atr
distance_ma20_pct
volume_ratio
range_atr
entry_quality_label
entry_quality_reason
Nieuwe validation reasons
valid_breakout_clean
valid_breakout_late_but_acceptable
invalid_breakout_chase
invalid_overextended_ma20
invalid_weak_volume
invalid_blowoff_volume
invalid_range_expansion
valid_pullback_controlled
invalid_structure
missing_data
Concrete validatie voor BREAKOUT
valid_breakout = (
    setup_type == "BREAKOUT"
    and setup_grade in ["A", "B"]
    and distance_to_breakout_pct >= 0
    and distance_to_breakout_pct <= 4
    and breakout_extension_atr <= 2.0
    and extension_atr <= 3.0
    and volume_ratio >= 1.2
    and volume_ratio <= 4.0
    and range_atr <= 2.5
)

Strengere variant voor A-quality:

clean_breakout = (
    distance_to_breakout_pct <= 2
    and breakout_extension_atr <= 1.5
    and extension_atr <= 2.0
    and 1.2 <= volume_ratio <= 2.5
)
Concrete validatie voor PULLBACK
valid_pullback = (
    setup_type == "PULLBACK"
    and close >= ma50
    and abs(close - ma20) / ma20 <= 0.03
    and extension_atr <= 1.0
)
Concrete validatie voor VCP

VCP niet automatisch rejecten. Sprint 1 stelde expliciet vast dat VCP volledig uitsluiten zonder analyse mogelijk edge kost.

valid_vcp = (
    setup_type == "VCP"
    and range_atr <= 1.2
    and distance_to_breakout_pct <= 2.5
    and volume_ratio >= 0.8
)
Sectie 6 — Risks & Trade-offs
1. Overfiltering

Te strenge thresholds reduceren false positives, maar missen sterke trends. Dit probleem zien jullie al: Sprint 2 stelde vast dat Validation de bottleneck is en sterke context-setups blokkeert.

Mitigatie:

reject alleen extreme chase
label twijfelgevallen als late_but_acceptable
log alles voor validatie
2. Underfiltering

Te soepele regels laten parabolische breakouts door.

Mitigatie:

hard reject:
breakout_extension_atr > 2.0
OF extension_atr > 3.0
OF distance_to_breakout_pct > 4%
3. Overfitting

Thresholds mogen niet “perfect” gemaakt worden op 10 trades. Het delivery framework vereist dat features pas geactiveerd worden na logging, validation en meetbare impact.

Minimum:

sample_size >= 30 per setup_type
liefst >= 100 voor definitieve thresholds
4. Regime-afhankelijkheid

Volatiliteit en momentum crashes zijn regimegevoelig. Maar regime hoort niet in deze Validation Layer als decision/context logic. Validation mag alleen technische entry quality loggen.

Eindconclusie

De evidence-based upgrade is:

Niet:
A setup = valid

Maar:
technische setup + entry proximity + ATR extension + volume quality + range control = valid entry

Aanbevolen hard rules voor de volgende refinement:

distance_to_breakout_pct <= 4%
breakout_extension_atr <= 2.0
extension_atr <= 3.0
volume_ratio tussen 1.2 en 4.0
range_atr <= 2.5
POST-SPRINT-0 DOCUMENT STATUS

Status: ARCHIVAL / PRE-SPRINT-0 RESEARCH NOTE

This document preserves historical entry-quality research and contains outdated schema assumptions such as `tradeable_setup`. It must not be used as active implementation guidance.

Current binding governance:

- AGENTS.md
- docs/archive/migration/sprint_0_governance_status.md
- docs/archive/audits/sprint_0_final_governance_audit.md

Certified doctrine:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

Entry quality is descriptive metadata only and must not determine upstream eligibility, tradeability, or allocation.
