# BL121 — Scanner/provider boundary review for remaining core scanner modules

Status: completed
Sprint type: review-only / scanner-provider boundary review
Archive action performed: no

## Scope

BL121 reviewed the scanner/provider-adjacent script-era core modules:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/indicators.py`

BL121 did not archive, move, delete, fail-close, or modify runtime code.

## Out of scope

This sprint did not touch:

* archive moves
* runtime behavior
* `scripts/core/decision_engine.py`
* Decision Engine authority
* live scanner execution
* yfinance execution
* SEC/EDGAR integrations
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* scan validation runtime
* trade command parser
* portfolio command processing
* logging/validation/bootstrap modules

## Background

BL120 reviewed the remaining active `scripts/core/` tree after BL119 archived the fail-closed portfolio intelligence module.

BL120 classified:

* `scripts/core/data_fetcher.py` as scanner/provider review required;
* `scripts/core/scanner.py` as scanner/provider review required;
* `scripts/core/indicators.py` as scanner-adjacent pure-logic review required.

BL121 inspected these three modules statically and did not execute provider code.

## Commands executed

### Branch setup

```bash id="pyy9me"
git checkout main
git pull origin main
git status
git checkout -b bl121-scanner-provider-boundary-review-remaining-core-scanner-modules
git status
```

### Existence and active import check

```bash id="cmfce3"
test -f scripts/core/data_fetcher.py
test -f scripts/core/scanner.py
test -f scripts/core/indicators.py

grep -RInE \
  "scripts\.core\.(data_fetcher|scanner|indicators)|from scripts\.core import (data_fetcher|scanner|indicators)|from scripts\.core|import scripts\.core" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* the three target files exist;
* no active import references `scripts.core.data_fetcher`;
* no active import references `scripts.core.scanner`;
* no active import references `scripts.core.indicators`;
* the only positive active `scripts.core` import remains:

```text id="z4qjr2"
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Other matches are static string assertions preventing script-era imports.

### Canonical scanner boundary references

```bash id="ky08hj"
grep -RInE \
  "scripts/core/data_fetcher.py|scripts/core/scanner.py|scripts/core/indicators.py|data_fetcher.py|scanner.py|indicators.py" \
  src/market_scanner tests docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* `src/market_scanner/scanner/scanner_boundary.py` statically lists:

  * `scripts/core/data_fetcher.py`
  * `scripts/core/scanner.py`
* historical audit and backlog references also classify these files as scanner/provider cleanup targets;
* `scripts/core/indicators.py` appears as scanner-adjacent historical/pure-logic candidate.

### Provider/network markers

```bash id="ugsdxy"
grep -nE \
  "yfinance|yf\.|Ticker|download|requests|urllib|http|session|provider|market data|info|history|os\.environ|dotenv|API|token|credential" \
  scripts/core/data_fetcher.py scripts/core/scanner.py scripts/core/indicators.py
```

Result summary:

`data_fetcher.py` contains:

* `import yfinance as yf`;
* `yf.download(...)`;
* `yf.Ticker(...)`;
* `ticker_obj.history(...)`.

`scanner.py` contains:

* `import yfinance as yf`;
* `yf.Ticker(ticker).info`;
* sector lookup from provider metadata.

`indicators.py` contains no provider/network marker in this scan.

### Runtime / entrypoint / write-risk markers

```bash id="gq8ahv"
grep -nE \
  "if __name__|def main|argparse|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs|open[(]|json\.dump|pickle|csv\.writer" \
  scripts/core/data_fetcher.py scripts/core/scanner.py scripts/core/indicators.py
```

Result summary:

* `data_fetcher.py` uses `TICKERS_FILE.open(...)`;
* no direct `to_csv`, `mkdir`, `read_csv`, `def main`, or `if __name__` markers were found for the three target files in this scan;
* `scanner.py` and `indicators.py` did not report write-risk markers in this command.

### Function overview

```bash id="j7p3d6"
grep -nE \
  "^(def |class |[A-Z_]+ = )" \
  scripts/core/data_fetcher.py scripts/core/scanner.py scripts/core/indicators.py
```

Result summary:

`data_fetcher.py` exposes:

* `load_tickers()`
* `_normalize_columns(...)`
* `fetch_ohlcv_data(...)`

`scanner.py` exposes scanner logic including:

* `get_sector(...)`
* `_current_scan_date(...)`
* `_has_required_columns(...)`
* scoring helpers
* `is_liquid_leader(...)`
* `detect_vcp(...)`
* `build_tradeplan(...)`
* `scan_ticker(...)`
* `rank_setups(...)`

`indicators.py` exposes:

* `MIN_REQUIRED_ROWS`
* `add_indicators(...)`

### Static source inspection

Static inspection confirmed:

`data_fetcher.py`:

* imports yfinance;
* reads configured ticker file;
* normalizes OHLCV columns;
* fetches OHLCV through `yf.download(...)`;
* falls back to `yf.Ticker(...).history(...)`.

`scanner.py`:

* imports yfinance;
* retrieves sector through `yf.Ticker(ticker).info`;
* contains scanner classification/scoring/trade-plan fields;
* contains scanner setup ranking logic.

`indicators.py`:

* imports pandas only;
* validates required OHLCV columns;
* computes MA20, MA50, MA200;
* computes ATR14;
* computes 20-day high/low and average volume;
* returns a transformed DataFrame;
* has no provider/network/write marker in BL121 output.

## Validation

Operator visibility:

```text id="i5smd9"
8 passed
```

Full suite:

```text id="tvsszj"
667 passed
```

## Classification

| File                           | Classification                                  | Reason                                                                                                                                                     |
| ------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/core/data_fetcher.py` | `SCANNER_PROVIDER_BLOCKED`                      | yfinance download/history calls and ticker-file dependency. Not archive-ready.                                                                             |
| `scripts/core/scanner.py`      | `SCANNER_PROVIDER_AND_SCORING_BLOCKED`          | yfinance sector lookup plus scanner/scoring/trade-plan semantics. Not archive-ready.                                                                       |
| `scripts/core/indicators.py`   | `PURE_HELPER_ARCHIVE_READINESS_REVIEW_REQUIRED` | Pure pandas indicator helper; no active import, no provider marker, no write marker found in BL121. Needs focused archive-readiness review before archive. |

## Decision

BL121 does not approve a scanner/provider archive or fail-close sprint.

Decision:

```text id="h8a8iz"
BL122 archive sprint: NOT APPROVED
BL122 scanner/provider fail-close sprint: NOT APPROVED
BL122 indicators archive-readiness review: APPROVED
```

## Required next sprint

Approved next sprint:

```text id="ac27hp"
BL122 — Archive-readiness review for script-era indicators helper
```

BL122 must be review-only.

Recommended BL122 scope:

* `scripts/core/indicators.py`

BL122 must verify:

* no active imports from `src`, `tests`, `.github`, or `scripts`;
* no runtime entrypoint;
* no provider/network calls;
* no production write behavior;
* whether canonical scanner/tests already cover the required indicator behavior or whether parity is missing.

No archive action may occur in BL122.

## Final BL121 result

```text id="ejyo22"
BL122 indicators archive-readiness review: APPROVED
```
