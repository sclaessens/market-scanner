1. Doel

Sprint 2 introduceert een aparte Context Strength Layer tussen validation_layer en downstream modules.

Pipeline:

scanner
→ validation_layer
→ context_layer
→ watchlist
→ portfolio
→ decision_engine
→ reporting

Evaluatieflow:

VALID_SETUP
→ CONTEXT_STRENGTH
→ CONTEXT_TRADEABLE

Belangrijk:

Deze layer bepaalt geen acties
Decision Engine blijft de enige bron van waarheid
2. Scope
In scope

Sprint 2 lost uitsluitend:

Geen context awareness
tradeable_setup = valid_setup (ontkoppeling)

Nieuwe output:

data/processed/context_strength.csv

Nieuw script:

scripts/core/build_context_layer.py
Out of scope

Niet toegestaan:

BUY / SELL / HOLD / TRIM / REMOVE
fundamentals
confidence logic
decision engine logic
reporting wijzigingen
position sizing
setup filtering
trend phase logic
3. Architectuur (GEFIXT)
Data ownership (hard rule)
Bestand	Owner	Write
scanner_ranked.csv	Scanner	Scanner
validation_layer.csv	Validation Layer	Validation ONLY
context_strength.csv	Context Layer	Context ONLY

👉 Context Layer mag NOOIT validation_layer.csv wijzigen.

4. Data Contracts
4.1 Input — scanner_ranked.csv

Pad:

data/processed/scanner_ranked.csv

Kolommen:

ticker: string
date: datetime
rs_20d_pct: float
sector: string

Validatie:

ticker niet leeg
ticker uniek per date
rs_20d_pct numeriek
sector verplicht
ontbrekende kolom → hard fail
4.2 Input — validation_layer.csv

Pad:

data/processed/validation_layer.csv

Kolommen:

ticker: string
date: datetime
valid_setup: bool
tradeable_setup: bool
validation_reason: string

Validatie:

ticker + date match scanner
valid_setup boolean
validation_layer wordt niet gewijzigd
4.3 Input — sector_relative_strength.csv

Pad:

data/processed/sector_relative_strength.csv

Kolommen:

sector: string
date: datetime
sector_rs_20d_pct: float

Regels:

sector + date uniek
sector wordt genormaliseerd:
sector = sector.strip().upper()
ontbrekende sector → UNKNOWN context
ontbrekende benchmark → rs_vs_sector = NaN
4.4 Output — context_strength.csv

Pad:

data/processed/context_strength.csv

Schema:

ticker: string
date: datetime
rs_20d: float
rs_vs_sector: float
context_strength: string
context_reason: string
context_tradeable: bool
context_tradeable_reason: string

Toegestane values:

WEAK, NEUTRAL, STRONG, LEADING, UNKNOWN
5. Exacte Definities
5.1 rs_20d
rs_20d = scanner_ranked.rs_20d_pct

Ontbrekend → hard fail

5.2 rs_vs_sector
rs_vs_sector = rs_20d - sector_rs_20d_pct
5.3 Neutral band
NEUTRAL_BAND = 0.25
6. Context Classification
def classify_context(rs_20d, rs_vs_sector):
    if rs_20d is None or is_nan(rs_20d):
        return "UNKNOWN", "missing_rs_20d"

    if abs(rs_20d) <= 0.25:
        return "NEUTRAL", "neutral_rs"

    if rs_20d < -0.25:
        return "WEAK", "negative_rs"

    if rs_20d > 0.25:
        if rs_vs_sector is not None and not is_nan(rs_vs_sector) and rs_vs_sector > 0.25:
            return "LEADING", "market_and_sector_outperformance"
        return "STRONG", "market_outperformance"

    return "UNKNOWN", "fallback"
7. Tradeable Logic (GEFIXT)
context_tradeable = (
    valid_setup is True
    and context_strength in ["STRONG", "LEADING"]
)

👉 Dit wordt NIET teruggeschreven naar validation_layer.csv

8. Script Specification

Bestand:

scripts/core/build_context_layer.py
Flow
Load scanner_ranked.csv
Load validation_layer.csv
Load sector_relative_strength.csv
Schema validation
Normaliseer sector
JOIN datasets
merge on: ticker, date
type: inner join
Row count check:
assert len(df_after) == len(validation_layer)
Bereken rs_20d
Bereken rs_vs_sector
Classificeer context
Bereken context_tradeable
Schrijf context_strength.csv
Logging
Niet toegestaan
validation_layer wijzigen
scanner wijzigen
BUY/SELL bepalen
reporting aanpassen
9. Fail Fast Policy
Situatie	Gedrag
scanner leeg	❌ crash
validation leeg	❌ crash
ontbrekende kolom	❌ crash
duplicate ticker/date	❌ crash
sector data ontbreekt	✅ toegestaan
10. Edge Cases
Case	Gedrag
rs_20d NaN	UNKNOWN
sector ontbreekt	UNKNOWN
sector benchmark ontbreekt	max STRONG
valid_setup = False	context_tradeable = False
context WEAK	False
context NEUTRAL	False
context UNKNOWN	False
11. Logging

Bestand:

data/logs/context_layer_log.csv

Kolommen:

run_date
total_rows
valid_setups
tradeable_setups_before
tradeable_setups_after
weak_count
neutral_count
strong_count
leading_count
unknown_count
missing_sector_count
12. Validation Strategy

Segmentatie:

VALID_SETUP
WEAK
NEUTRAL
STRONG
LEADING
CONTEXT_TRADEABLE

Metrics:

winrate
average_return
hit_rate
drawdown
sample_size

Minimum:

sample_size >= 30
13. Tests
tests/core/test_build_context_layer.py

Test cases:

LEADING correct
STRONG correct
WEAK correct
NEUTRAL correct
UNKNOWN correct
context_tradeable correct
duplicates → fail
missing column → fail
14. Hard Definition of Done

Sprint 2 is alleen klaar als:

Script bestaat
context_strength.csv bestaat
Schema exact correct is
validation_layer.csv NIET gewijzigd wordt
context_tradeable correct berekend wordt
logging aanwezig is
pipeline werkt via run_scan.py
fail-fast actief is
join correct is
unit tests slagen
output reproduceerbaar is
geen BUY/SELL logic aanwezig is
