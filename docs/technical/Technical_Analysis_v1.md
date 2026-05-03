TECHNICAL SPECIFICATION DOCUMENT

Trading System Upgrade: Signal → Decision Engine

1. CORE TECHNICAL SHIFT (CRUCIAAL)
Nieuwe architecturale realiteit
Signal Layer        → scanner
Validation Layer    → context + fundamentals
Decision Layer      → ENIGE bron van waarheid

👉 Decision engine wordt centrale autoriteit
👉 Geen enkele module mag nog zelfstandig acties bepalen

Zoals expliciet vereist door PM & Financial Analysis

2. NIEUWE DATA ARCHITECTUUR (UITBREIDING)
2.1 Nieuwe verplichte bestanden
data/processed/validation_layer.csv
column	type
ticker	string
date	datetime
valid_setup	bool
tradeable_setup	bool
validation_reason	string
data/processed/context_strength.csv (UPDATED)
column	type
ticker	string
date	datetime
rs_20d	float
rs_vs_sector	float
context_strength	string

Nieuwe waarden:

WEAK / NEUTRAL / STRONG / LEADING
data/processed/fundamental_profile.csv (NIEUW)
column	type
ticker	string
piotroski_score	int
earnings_yield	float
ev_ebitda	float
roic	float
fundamental_profile	string
data/processed/decision_output.csv (VERVANGT final_decisions.csv)
column	type
ticker	string
action	string
confidence	string
trend_phase	string
context_strength	string
fundamental_profile	string
validation_state	string
reason	string
3. NIEUWE CORE CONCEPTEN (TECHNISCH AFDWINGEN)
3.1 VALID vs TRADEABLE SETUP
valid_setup = technical_conditions_met

tradeable_setup = (
    valid_setup
    AND context_strength in ["STRONG", "LEADING"]
    AND validation_passed
)

👉 FUNDAMENTALS zitten NIET in validatie
👉 alleen in confidence

(zoals expliciet gedefinieerd)

3.2 CONFIDENCE ENGINE (NIEUW — CORE)
Definitie
def compute_confidence(fundamental_profile, context_strength, trend_phase):

    if context_strength in ["STRONG", "LEADING"]:
        if fundamental_profile == "BEST":
            return "HIGH"
        elif fundamental_profile == "GOOD":
            return "MEDIUM"
        elif fundamental_profile == "MOMENTUM":
            return "MEDIUM"
        elif fundamental_profile == "RISKY":
            return "LOW"

    return "LOW"
3.3 HARD RULES (MOET IN CODE)
# FUNDAMENTALS NOOIT:
# - entry triggeren
# - setup blokkeren

# EXCEPTIE:
if context_strength == "WEAK" and fundamental_profile == "RISKY":
    tradeable_setup = False

Bron: Financial + Functional analyse

4. LAYER IMPLEMENTATIONS (UPGRADE)
4.1 Trend Phase Layer (UPDATED)

Nieuwe toevoeging:

extended_phase = (close - ma20) / atr14 > threshold_extended

Nieuwe regel:

if trend_phase == "EXTENDED":
    require_extra_validation = True
4.2 Context Strength Layer (UPDATED)
if rs_20d > 0 and rs_vs_sector > 0:
    context_strength = "LEADING"
elif rs_20d > 0:
    context_strength = "STRONG"
elif rs_20d < 0:
    context_strength = "WEAK"
else:
    context_strength = "NEUTRAL"
4.3 Fundamental Layer (VOLLEDIG NIEUW GEDRAG)
Mapping
if piotroski >= 7 and roic_high:
    profile = "BEST"

elif piotroski >= 5:
    profile = "GOOD"

elif piotroski <= 3:
    profile = "RISKY"

elif data_missing:
    profile = "UNKNOWN"

else:
    profile = "MOMENTUM"

👉 EXACT conform financial model

4.4 Decision Stability Layer (UPDATED)

Nieuwe regel:

if state_change:
    require_confirmation_days = 2

OF:

if strong_confluence:
    bypass_confirmation = True
5. DECISION ENGINE (VOLLEDIG HERWERKT)
5.1 Evaluatie volgorde (NIEUW)
1. Portfolio check
2. VALID_SETUP bepalen
3. Context check
4. Tradeable check
5. Stability check
6. Fundamental profile
7. Confidence berekenen
8. Final action bepalen
5.2 Action Logic (EXACT)
BUY
if tradeable_setup
and watchlist_status == "READY"
and trend_phase != "EXTENDED":
    action = "BUY"
WAIT
if valid_setup and not tradeable_setup:
    action = "WAIT"
REMOVE
if technical_invalid:
    action = "REMOVE"

elif context_strength == "WEAK" and fundamental_profile == "RISKY":
    action = "REMOVE"
HOLD / TRIM / SELL
if in_portfolio:

    if trend_ok:
        action = "HOLD"

    elif extended and fundamental_profile == "RISKY":
        action = "TRIM"

    elif trend_break:
        action = "SELL"
6. INTERACTION ENGINE (NIEUW — KRITISCH)
# STRONG + BEST
→ HIGH confidence

# STRONG + RISKY
→ LOW confidence

# WEAK + BEST
→ WAIT

# WEAK + RISKY
→ REMOVE

Dit is rechtstreeks vertaald naar code uit de analyse

7. OUTPUT LOGIC (HERWERKT)
Telegram / reporting input
if confidence in ["HIGH", "MEDIUM"]:
    section = "ACTIE NU"

elif action == "WAIT":
    section = "VOORBEREIDEN"

elif action in ["REMOVE", "SELL"]:
    section = "RISICO"
8. STATE MANAGEMENT (UITBREIDING)

Nieuwe state velden:

validation_state
confidence
fundamental_profile

State wordt opgeslagen in:

data/processed/decision_state.csv
9. VALIDATION INTEGRATIE (CRUCIAAL)

Nieuwe logging:

log:
    fundamental_profile
    confidence
    context_strength

Nieuwe analyse:

winrate per:
- fundamental_profile
- confidence
- context_strength

👉 metrics blijven alleen als performance stijgt

10. GOVERNANCE (TECHNISCH AFDWINGEN)
Hard constraints in code:
assert fundamentals_do_not_trigger_entries
assert decision_engine_is_single_source_of_truth
assert one_decision_per_ticker
11. IMPLEMENTATIE IMPACT

Wat verandert NIET:

scanner
watchlist logging
portfolio logging

Wat verandert WEL:

decision_engine.py (major rewrite)
nieuwe build scripts:
build_validation_layer.py
build_fundamental_profile.py
reporting logic
12. CONCLUSIE (TECHNISCH)

De upgrade is geen uitbreiding van features.
Het is een herstructurering van besluitvorming:

Van:

if setup → actie

Naar:

if setup → valideren → kwalificeren → confidence → actie

👉 Dit is exact wat nodig is om:

false positives te reduceren
drawdowns te beperken
consistentie te verhogen

Zoals vereist door alle drie de rollen (PM, FA, FSD)