Sprint: Sprint 1 Refinement — Validation Layer Upgrade (v2)

Owner: Scrum Master
Status: READY FOR DEVELOPMENT
Depends on:

Sprint 1 Audit Findings
Sprint 2 Audit Findings
Technical Analysis v2 (updated)
1. 🎯 Sprint Objective

De Validation Layer upgraden van een:

simplistische grade-based filter

naar een:

setup-aware, rule-based technical validation layer

Doel:

Over-filtering elimineren
Alignment met Context Layer verbeteren
VALID_SETUP correct definiëren
Tradeable pipeline voorbereiden
2. 📦 Scope
In Scope

✔ build_validation_layer.py herschrijven (v2)
✔ VALID_SETUP logica upgraden
✔ validation_reason uitbreiden
✔ setup-type validatie implementeren
✔ RR filtering aanpassen

Out of Scope (HARD EXCLUSIONS)

❌ Context Layer wijzigen
❌ Decision Engine aanpassen
❌ Reporting wijzigen
❌ Fundamentals integreren
❌ Tradeable logic wijzigen buiten valid_setup

👉 Schending = sprint failure

3. 📁 Te wijzigen bestanden
Core
scripts/core/build_validation_layer.py
Data output
data/processed/validation_layer.csv
data/logs/validation_layer_log.csv (indien bestaat → uitbreiden)
Tests
tests/core/test_build_validation_layer.py
4. 📥 Input Data (VERPLICHT)
data/processed/scanner_ranked.csv

Verplichte kolommen:

ticker
date
primary_setup
rr
close
ma20
ma50
high_20d
volume_ratio
extension_atr

Ontbrekend → HARD FAIL

5. 📤 Output Contract (NIET WIJZIGEN)
validation_layer.csv

Kolommen:

ticker
date
valid_setup
tradeable_setup
validation_reason
6. 🧠 VALIDATION LOGIC (EXACT IMPLEMENTEREN)
6.1 Base Rule
valid_setup = False
6.2 RR Constraint (GLOBAL)
if rr is None or rr < 1.8:
    valid_setup = False
    reason = "invalid_rr"
6.3 Trend Constraint
if close <= ma50:
    valid_setup = False
    reason = "weak_trend"
6.4 Setup-Type Logic
BREAKOUT
valid_setup = (
    distance_high <= 0.08
    and volume_ratio >= 1.1
    and close > ma20
)
reason = "valid_breakout"
PULLBACK
valid_setup = (
    -0.08 <= distance_ma20 <= 0.03
    and close > ma50
)
reason = "valid_pullback"
VCP
valid_setup = (
    contraction_detected
    and near_high
    and trend_aligned
)
reason = "valid_vcp"
6.5 Missing Data
if any required field missing:
    valid_setup = False
    reason = "missing_data"
7. 🚫 HARD RULES (MUST BE ENFORCED)
assert no_context_usage
assert no_fundamental_usage
assert no_decision_logic
8. 🔁 tradeable_setup (UNCHANGED)
tradeable_setup = valid_setup

👉 NIET aanpassen

9. 🧾 validation_reason ENUM (STRICT)

Toegestane waarden:

valid_breakout
valid_pullback
valid_vcp
invalid_rr
invalid_structure
weak_trend
missing_data
no_setup

Andere waarden → FAIL

10. 📊 Logging (VERPLICHT)

Logbestand:

data/logs/validation_layer_log.csv

Minimale velden:

run_date
total_rows
valid_count
invalid_count
breakout_valid_count
pullback_valid_count
vcp_valid_count
invalid_rr_count
weak_trend_count
missing_data_count
11. 🧪 Test Requirements

Test file:

tests/core/test_build_validation_layer.py

Minimale tests:

✔ valid_breakout correct
✔ valid_pullback correct
✔ valid_vcp correct
✔ invalid_rr correct
✔ weak_trend correct
✔ missing_data correct
✔ duplicate ticker/date → fail
✔ missing columns → fail
✔ empty scanner → fail
✔ validation output schema correct
✔ scanner file blijft unchanged

12. 📏 Acceptance Criteria

Sprint is PAS klaar als:

✔ unit tests 100% pass
✔ run_scan werkt end-to-end
✔ validation_layer.csv correct output geeft
✔ context overlap stijgt (manueel check)
✔ geen pipeline break
13. ⚠️ Failure Conditions

Sprint faalt indien:

developer logica interpreteert
output schema verandert
context wordt gebruikt
validation te complex wordt
tests ontbreken of falen
14. 📈 Expected Outcome

Voorheen:

VALID_SETUP ≈ alleen A setups

Na sprint:

VALID_SETUP = technisch correcte setups (A + sterke B)

Impact:

valid setups ↑
overlap met context ↑
tradeable setups ↑
15. 🚀 Definition of Done

✔ Validation v2 volledig geïmplementeerd
✔ Geen architectuurbreuk
✔ Data contract behouden
✔ Pipeline stabiel
✔ Tests volledig groen

🔥 Final Note (Scrum Master)

Deze sprint is kritisch voor edge-creatie.

Fout in deze laag =
→ slechte trades
→ slechte data
→ slechte beslissingen downstream

Correcte implementatie =
→ foundation van het volledige systeem