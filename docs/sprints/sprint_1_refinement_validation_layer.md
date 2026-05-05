Sprint 1 Refinement — Validation Layer Upgrade

Validation Layer v2 — Context-Aware Filtering

1. Executive Summary

Na Sprint 2 werd duidelijk:

STRONG context setups → worden weggefilterd door validation

👉 De huidige validation layer is:

te simplistisch
te restrictief
niet aligned met context
Belangrijkste conclusie:

👉 Validation is de bottleneck van het systeem
👉 Context detecteert kwaliteit, maar wordt niet benut
👉 Edge wordt momenteel vernietigd door over-filtering

2. Doel van deze Sprint

De Validation Layer upgraden van:

"binary filter (A vs rest)"

naar:

"multi-dimensional quality filter"
3. Scope Definition
In Scope

✔ VALID_SETUP verfijnen
✔ TRADEABLE voorbereiden (maar niet beslissen)
✔ validation_reason uitbreiden
✔ filtering minder destructief maken

Out of Scope

❌ Context Layer wijzigen
❌ Decision Engine aanpassen
❌ Reporting aanpassen
❌ Fundamentals toevoegen

👉 Strict layer separation blijft gelden

4. Core Probleem (v1)

Huidige logica:

valid_setup = grade == "A"

Probleem:

geen nuance
geen setup-type differentiatie
geen kwaliteit binnen setups
context wordt genegeerd
5. Nieuwe Validation Filosofie

Validation moet:

technische geldigheid bepalen
NIET kwaliteit volledig beslissen

👉 belangrijke shift:

van:
hard filter

naar:
gecontroleerde filtering
6. Nieuwe VALID_SETUP Definitie (v2)
Basisregel blijft:
VALID_SETUP = technische structuur is correct
Nieuwe logica:
VALID_SETUP = (
    setup_structure_valid
    AND rr >= 1.8
    AND trend_ok
)
Belangrijk:

👉 grade ≠ validiteit
👉 grade = ranking, niet filtering

7. Setup-specifieke validatie
7.1 BREAKOUT
valid_breakout = (
    distance_high <= 0.08
    AND volume_ratio >= 1.1
    AND close > ma20
)
7.2 PULLBACK
valid_pullback = (
    -0.08 <= distance_ma20 <= 0.03
    AND close > ma50
    AND trend_ok
)
7.3 VCP
valid_vcp = (
    contraction aanwezig
    AND near_high
    AND trend_aligned
)
8. RR (Risk Reward) Logica

Nieuwe regel:

if rr < 1.8:
    valid_setup = False

👉 vroeger:

rr < 2 → volledig gefilterd

👉 nu:

meer setups toegelaten
9. Validation Reason v2

Nieuwe granulariteit:

valid_breakout
valid_pullback
valid_vcp
invalid_rr
invalid_structure
weak_trend
extended_setup
missing_data

👉 geen black-box filtering meer

10. TRADEABLE_SETUP (voorbereiding)

Nog NIET context-aware maken
(komt pas via context layer)

Maar:

tradeable_setup = valid_setup

👉 blijft identiek voor deze sprint

11. Expected Behaviour Change
Voorheen:
A setups → valid
B setups → weg
Nu:
Sterke B setups → blijven
Zwakkere setups → eruit
12. Impact op jouw huidige output

Voor jouw case:

ALL, MS, CVS → STRONG context

👉 worden nu:

VALID_SETUP = True (als structuur klopt)

👉 dus later:

TRADEABLE mogelijk
13. Validation Strategy

Na implementatie meten:

valid_setup count ↑
tradeable overlap ↑
winrate per setup type
14. Risks
🔴 Risk 1 — Te los filter

Impact:

meer false positives

Mitigatie:

RR check
trend check
🟠 Risk 2 — Overlap met context

Impact:

dubbel filter

Mitigatie:

👉 context blijft aparte layer

15. Definition of Done

✔ valid_setup niet langer enkel A
✔ setup-type logica aanwezig
✔ RR filtering correct
✔ validation_reason uitgebreid
✔ pipeline blijft werken
✔ context overlap stijgt

16. Belangrijkste Insight
Validation moet NIET perfect zijn
Validation moet selectief zijn

👉 echte filtering gebeurt later via:

VALIDATION + CONTEXT + DECISION
17. Final Conclusion

Sprint 1 v1:

te streng → edge verloren

Sprint 1 refinement:

gebalanceerd → edge mogelijk