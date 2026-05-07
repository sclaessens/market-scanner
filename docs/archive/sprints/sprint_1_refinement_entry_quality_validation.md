0. Governance & Alignment (CRUCIAAL)

Deze sprint volgt strikt:

Validation-first development
Eén layer per sprint
Geen decision logic buiten Decision Engine

👉 Conform Execution Framework

Hard Rules
Validation = ALLEEN technische validiteit
GEEN context logic
GEEN fundamentals
GEEN decision logic
GEEN interpretatie

👉 Deze sprint refinet uitsluitend de Validation Layer

1. Impact Analyse
1.1 Impact op Validation Layer

Probleem:

Validatie is te simplistisch (A-only filtering)

Oplossing:

👉 Entry quality wordt toegevoegd als technische validiteitsconstraint

Belangrijk:

GEEN graduele labels
GEEN ranking
ENKEL boolean validatie
1.2 Impact op Context Layer

Context gebruikt:

context_tradeable = valid_setup AND strong context

Probleem:

Validation blokkeert sterke context setups

Impact:

👉 Meer VALID setups → betere overlap
👉 GEEN wijziging in context layer

1.3 Impact op Data Contracts (CRUCIAAL)
🔒 Validation Layer blijft ONGEWIJZIGD
data/processed/validation_layer.csv

ticker
date
valid_setup
tradeable_setup
validation_reason

👉 Conform TSD

✅ Nieuw bestand
data/processed/entry_quality_metrics.csv
Data Contract — Entry Quality Metrics
ticker: string
date: datetime

distance_to_breakout_pct: float64
breakout_extension_atr: float64
extension_atr: float64
distance_ma20_pct: float64
volume_ratio: float64
range_atr: float64

entry_quality_flag: boolean
entry_quality_reason: string
Data Governance
PRIMARY KEY: (ticker, date)
UNIQUE: ticker + date

OWNER: Validation Layer
WRITE: build_validation_layer.py ONLY
READ: downstream allowed
Join Rules
INNER JOIN met validation_layer.csv op (ticker, date)

ASSERT:
len(metrics) == len(validation_layer)
Precision Rules
float type: float64
rounding: 4 decimalen
NaN: NIET toegestaan (behalve expliciet behandeld)
boolean: strict True/False
2. Scope Definition
IN SCOPE
Entry quality metrics (logging)
Entry quality validatie (boolean)
Validation refinement
OUT OF SCOPE
Decision logic ❌
Context logic ❌
Fundamentals ❌
Reporting ❌
3. Functional Definition
3.1 Entry Quality

Entry quality bepaalt:

👉 of een setup technisch valide instapbaar is

3.2 Entry Quality Output
entry_quality_flag ∈ {True, False}
3.3 Entry Quality Reason
entry_quality_reason ∈ {
    "ok",
    "too_far_from_breakout",
    "overextended_atr",
    "overextended_ma20",
    "weak_volume",
    "excessive_volume",
    "range_expansion",
    "invalid_structure",
    "missing_data"
}

👉 Logging only — GEEN decision

4. Technical Specification
4.1 Fail-Fast (CRUCIAAL)

ALTIJD eerst uitvoeren:

if atr14 <= 0:
    HARD FAIL

if breakout_level is None:
    entry_quality_flag = False
    reason = "missing_data"

if ma20 is None:
    entry_quality_flag = False

if volume is None or avg_volume_20 == 0:
    entry_quality_flag = False

if high == low:
    range_atr = 0
4.2 Formules
distance_to_breakout_pct = (close - breakout_level) / breakout_level * 100

breakout_extension_atr = (close - breakout_level) / atr14

extension_atr = (close - ma20) / atr14

distance_ma20_pct = (close - ma20) / ma20 * 100

volume_ratio = volume / avg_volume_20

range_atr = (high - low) / atr14
4.3 Extra Edge Case Handling
if breakout_extension_atr < 0:
    entry_quality_flag = False

if volume_ratio < 0:
    HARD FAIL

if volume_ratio > 10:
    entry_quality_flag = False
4.4 Config (GEEN HARDCODE)
ENTRY_QUALITY_CONFIG = {
    "max_distance_breakout_pct": 3.0,
    "max_breakout_extension_atr": 2.0,
    "max_extension_atr": 2.5,
    "min_volume_ratio": 1.2,
    "max_volume_ratio": 4.0,
    "max_range_atr": 2.5
}
Config Validation
assert 0 < min_volume_ratio < max_volume_ratio
assert max_extension_atr > 0
assert max_distance_breakout_pct > 0
4.5 Entry Quality Logic (DETERMINISTIC)
entry_quality_flag = True
reason = "ok"

if distance_to_breakout_pct > config.max_distance_breakout_pct:
    entry_quality_flag = False
    reason = "too_far_from_breakout"

elif breakout_extension_atr > config.max_breakout_extension_atr:
    entry_quality_flag = False
    reason = "overextended_atr"

elif extension_atr > config.max_extension_atr:
    entry_quality_flag = False
    reason = "overextended_ma20"

elif volume_ratio < config.min_volume_ratio:
    entry_quality_flag = False
    reason = "weak_volume"

elif volume_ratio > config.max_volume_ratio:
    entry_quality_flag = False
    reason = "excessive_volume"

elif range_atr > config.max_range_atr:
    entry_quality_flag = False
    reason = "range_expansion"
5. VALID_SETUP (FASE-GESTUURD)
Fase 1 — Logging Only
valid_setup = (
    structure_valid
    AND rr >= 1.8
    AND trend_ok
)

👉 entry_quality_flag wordt NIET gebruikt

Fase 4 — Activatie
valid_setup = (
    structure_valid
    AND rr >= 1.8
    AND trend_ok
    AND entry_quality_flag == True
)
6. VALIDATION REASONS (STRICT — NIET WIJZIGEN)
valid_breakout
valid_pullback
valid_vcp
invalid_rr
invalid_structure
weak_trend
missing_data
no_setup

👉 Entry quality zit NOOIT in validation_reason

7. Logging Requirements
7.1 Per ticker
alle metrics
entry_quality_flag
entry_quality_reason
7.2 Aggregation Logging
total_rows
valid_setups_before
valid_setups_after
rejected_by_entry_quality

avg_extension_atr
avg_volume_ratio
median_range_atr
8. Performance Requirements
Gebruik pandas vectorization
GEEN row loops

Performance:
< 2 seconden per 1000 tickers
9. Pipeline Safety (CRUCIAAL)
assert len(entry_quality_metrics) == len(validation_layer)

assert no duplicate (ticker, date)

assert every validation row aanwezig in metrics
10. Validation Plan
Segmentatie
entry_quality_flag TRUE/FALSE
extension_atr buckets
volume_ratio buckets
Metrics
winrate
average return
hit rate
drawdown
Minimum Sample
≥ 30 per setup
target ≥ 100
11. Activation Strategy
Fase 1 — Logging Only
GEEN impact op valid_setup
Fase 2 — Analyse
performance meten
Fase 3 — Validatie
edge bewijzen
Fase 4 — Activatie
entry_quality_flag integreren
12. Risks & Mitigations
Risico	Mitigatie
Overfiltering	config thresholds
Underfiltering	hard caps
Overfitting	minimum sample
Pipeline break	aparte metrics file
Data corruption	fail-fast
13. Definition of Done

Sprint is klaar als:

validation_layer.csv ONGEWIJZIGD
entry_quality_metrics.csv bestaat
alle metrics correct
fail-fast actief
logging volledig
pipeline via run_scan.py werkt
GEEN decision logic aanwezig
output reproduceerbaar
assertions actief
14. Final Statement

Nieuwe structuur:

Validation = technische geldigheid
Entry Quality = entry validiteit
Context = relatieve sterkte