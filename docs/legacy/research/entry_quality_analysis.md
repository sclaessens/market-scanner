ENTRY QUALITY RESEARCH DOCUMENT

Author: Senior Quant Researcher
Scope: Entry Quality Metrics — Empirical Validation
Status: Data-driven analysis (phase 1 complete); ARCHIVAL / PRE-SPRINT-0 terminology

POST-SPRINT-0 GOVERNANCE NOTE

This research remains useful evidence that entry-quality gates were harmful, but it uses pre-Sprint-0 terms such as `entry_quality_flag` and filtering. Current runtime governance treats entry quality as descriptive metadata only.

Do not implement entry-quality filtering outside Decision Engine. Current doctrine:

classification upstream
allocation downstream
Decision Engine = ONLY allocation authority

1. 🎯 OBJECTIVE

Doel van deze analyse:

Bepalen of Entry Quality metrics:
→ edge toevoegen
→ filtering verbeteren
→ of gebruikt moeten worden in decision logic

Belangrijke constraint:

GEEN strategie bouwen
GEEN aannames zonder data
2. 📊 DATASET OVERVIEW

Na deduplicatie:

N = 47 trades

Verdeling:

ok                   44 (93.6%)
overextended_ma20     3 (6.4%)

👉 Eerste observatie:

extreem skewed dataset → filtering bijna niet actief
3. 🔬 CORE FINDINGS
3.1 Entry Quality Flag Performance
entry_quality_flag = True → slechtere performance
entry_quality_flag = False → betere performance
Metric	True	False
Return 10d	0.35%	2.38%
Max gain	2.20%	6.46%
Stop hit	18.2%	0%
🔴 Conclusie:
Entry Quality flag werkt contraproductief

👉 Filter verwijdert momenteel betere trades

3.2 Extension Analysis (CRUCIAAL)
Extension (ATR)	Return
< 0	-0.73% ❌
0–1	0.10%
1–2	1.00%
2–3	6.03% 🔥
🔥 Key Insight:
Performance stijgt met extension

👉 Dit is typisch momentum gedrag:

sterke trends blijven doorgaan
“overextended” ≠ slecht
vaak juist acceleration phase
3.3 Volume Analysis
Volume Ratio	Return
< 1	0.42%
1–2	0.43%
🟡 Conclusie:
Volume heeft beperkte differentiatie
→ secundaire metric
4. 🧠 THEORETICAL ALIGNMENT

Deze resultaten zijn volledig consistent met:

Momentum Research (empirisch bewezen):
Trends persist (Jegadeesh & Titman)
Leaders blijven outperformen
Breakouts werken vaak NA initial extension

👉 Belangrijk:

Mean reversion logica ≠ momentum logica
5. ❌ WHY THE CURRENT MODEL FAILS
Huidige assumptie:
extended = slechte entry
Realiteit:
extended = sterk momentum = betere performance
Root cause:
Entry Quality gebruikt mean-reversion bias
in een momentum systeem

👉 Dit is een fundamentele mismatch

6. 🔴 CRITICAL DESIGN ERROR

Huidige implementatie:

if extension_atr > threshold:
    flag = False
Probleem:
→ binary filtering
→ informatieverlies
→ verkeerde signaalinterpretatie
7. 🎯 CORRECT INTERPRETATION

Entry Quality metrics zijn:

GEEN filters
GEEN validatiecriteria

Maar:

FEATURES
Nieuwe interpretatie:
Metric	Betekenis
extension_atr	momentum strength
distance_ma20	pullback vs breakout
volume_ratio	confirmation
range_atr	volatility regime
8. 🚀 RECOMMENDED SYSTEM DESIGN
8.1 Entry Quality = Feature Layer

Niet:

valid / invalid

Wel:

continuous variables
8.2 Gebruik in Decision Engine (later)

Voorbeeld:

HIGH extension + STRONG context → agressieve trade
LOW extension + pullback → conservatieve trade
HIGH extension + WEAK context → risk
8.3 Geen filtering op:
extension_atr
distance_to_breakout
9. 📊 WHAT ACTUALLY CREATES EDGE

Uit deze analyse:

Edge zit NIET in individuele metrics

Maar in:

setup_type × extension × context × trend
10. ⚠️ LIMITATIONS

Belangrijk:

N = 47 → klein
False groep = 3 → extreem klein

👉 Dus:

richting is duidelijk
statistische zekerheid = beperkt
11. 🧭 FINAL CONCLUSIONS
1. Entry Quality flag
→ FAIL
→ mag NIET gebruikt worden als filter
2. Metrics
→ waardevol
→ moeten behouden blijven
3. Extension
→ positief gecorreleerd met performance
→ belangrijke momentum indicator
4. Conceptuele fout
Mean reversion logic toegepast op momentum systeem
12. 🔥 STRATEGIC INSIGHT

Dit is de belangrijkste takeaway:

“Overextended vermijden” is fout

→ je moet begrijpen wanneer het momentum is
13. 🚀 NEXT STEPS
Phase 2 — Advanced Analysis

Segmenteren op:

setup_type × extension_atr
context_strength × extension
trend_phase × extension
Phase 3 — Decision Engine Integration

Niet:

filtering

Wel:

confidence modulation
risk classification
position sizing (later)
14. FINAL STATEMENT
Entry Quality is geen filter

Entry Quality is informatie

Correct gebruikt:

→ verhoogt edge
→ verbetert beslissingen
→ maakt systeem institutioneel

Fout gebruikt:

→ vernietigt momentum edge
