TECHNICAL SPECIFICATION DOCUMENT (v2 — PRODUCTION READY)

Trading System: Scanner → Decision Engine Pipeline

1. CORE TECHNICAL ARCHITECTURE (CORRECTED)
Layer Separation (HARD RULE)
Signal Layer        → scanner
Validation Layer    → technische geldigheid
Context Layer       → relatieve sterkte
Fundamental Layer   → kwaliteit
Decision Layer      → ENIGE bron van waarheid

👉 Decision Engine is centrale autoriteit
👉 Geen enkele andere layer mag acties bepalen

2. DATA ARCHITECTURE (CURRENT STATE)
2.1 validation_layer.csv
ticker
date
valid_setup
tradeable_setup
validation_reason
2.2 context_strength.csv
ticker
date
rs_20d
rs_vs_sector
context_strength
context_reason
context_tradeable
context_tradeable_reason

Values:

WEAK / NEUTRAL / STRONG / LEADING / UNKNOWN
2.3 future layers (NOT ACTIVE YET)
fundamental_profile.csv
decision_output.csv

👉 niet gebruikt in huidige pipeline

3. CORE CONCEPTS
3.1 VALID vs TRADEABLE
valid_setup = technical_validity
context_tradeable = (
    valid_setup
    AND context_strength in ["STRONG", "LEADING"]
)

👉 tradeable wordt NIET geschreven naar validation_layer
👉 context layer bepaalt tradeability

3.2 HARD RULES
# NIET toegestaan in validation:
assert no_context_usage
assert no_fundamental_usage
assert no_decision_logic
4. VALIDATION LAYER (v3 — ENTRY QUALITY AWARE)

4.1 Doel

VALIDATION = technische geldigheid + entry structure kwaliteit

De Validation Layer bepaalt:

1. Is de setup technisch correct?
2. Is de entry structureel valide (niet te laat / niet te extended)?

👉 Belangrijk onderscheid:

VALIDATION bepaalt GEEN:
- relatieve sterkte (context layer)
- fundamentals (fundamental layer)
- beslissingen (decision engine)

VALIDATION bepaalt WEL:
- of een entry technisch verantwoord is

---

4.2 VALID_SETUP definitie

VALID_SETUP = (
    setup_structure_valid
    AND rr >= 1.8
    AND trend_ok
    AND entry_quality_ok
)

---

4.3 ENTRY QUALITY CONSTRAINT (NIEUW)

Doel:

Voorkomen van late entries in momentum setups.

Probleem:

Momentum setups die te ver extended zijn hebben:
- slechtere risk/reward
- hogere failure rate
- grotere drawdowns

Definitie:

Een setup kan technisch correct zijn,
maar alsnog invalid zijn indien de entry te ver verwijderd is van een optimale entry zone.

---

4.4 Setup-type validatie

BREAKOUT

valid_breakout = (
    distance_high <= 0.08
    AND volume_ratio >= 1.3
    AND close > ma20
    AND close > ma50

    # ENTRY QUALITY (NIEUW)
    AND extension_atr <= 2.5
    AND distance_high <= 0.03
)

Interpretatie:

- volume bevestigt breakout kwaliteit
- extension_atr voorkomt late entries
- distance_high voorkomt chasing boven breakout

---

PULLBACK

(valid_pullback blijft ongewijzigd)

valid_pullback = (
    -0.08 <= distance_ma20 <= 0.03
    AND close > ma50
)

---

VCP

(valid_vcp blijft ongewijzigd)

valid_vcp = (
    contraction_detected
    AND near_high
    AND trend_aligned
)

---

4.5 RR constraint (ongewijzigd)

if rr is None or rr < 1.8:
    valid_setup = False

---

4.6 validation_reason (STRICT ENUM)

GEEN wijzigingen

valid_breakout
valid_pullback
valid_vcp
invalid_rr
invalid_structure
weak_trend
missing_data
no_setup

---

4.7 tradeable_setup (ongewijzigd)

tradeable_setup = valid_setup
5. CONTEXT LAYER (Sprint 2 — CURRENT)
5.1 Classification
if rs_20d is None:
    context_strength = "UNKNOWN"

elif abs(rs_20d) <= 0.25:
    context_strength = "NEUTRAL"

elif rs_20d < -0.25:
    context_strength = "WEAK"

elif rs_20d > 0.25:
    if rs_vs_sector > 0.25:
        context_strength = "LEADING"
    else:
        context_strength = "STRONG"
5.2 Tradeable logic
context_tradeable = (
    valid_setup
    AND context_strength in ["STRONG", "LEADING"]
)
6. DECISION ENGINE (CURRENT STATE)
6.1 Pipeline order
scanner
→ validation
→ context
→ watchlist
→ portfolio
→ decision_engine
→ reporting
6.2 Current limitation
Decision engine gebruikt context nog NIET actief

👉 komt in latere sprint

7. GOVERNANCE (CRITICAL)
7.1 Layer isolation
assert validation_layer_independent
assert context_layer_independent
assert decision_engine_single_source_of_truth
7.2 No cross contamination
validation_layer MUST NOT:
    - use context
    - use fundamentals
    - define actions
8. IMPLEMENTATION IMPACT
8.1 Wat veranderd is
validation logic (v2)
context layer toegevoegd
scanner uitgebreid met sector + date
8.2 Wat NIET veranderd is
scanner core logic
reporting logic
decision engine gedrag
9. STRATEGIC INSIGHT

Het systeem werkt nu volgens:

setup → validation → context → (later decision)

Niet meer:

setup → direct actie
10. CRITICAL SYSTEM LEARNING
Validation bepaalt toegang
Context bepaalt kwaliteit
Decision bepaalt actie

👉 scheiding van verantwoordelijkheden is essentieel

11. NEXT PHASE

Volgende implementatie:

Decision Engine Upgrade

Waar:

context effectief gebruikt wordt
tradeable setups acties genereren
12. CONCLUSION

Deze versie corrigeert:

verkeerde layer-definities
over-simplistische validatie
ontbrekende context integratie

En brengt het systeem naar:

institutionele architectuurstandaard