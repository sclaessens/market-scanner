# RESET-10G — V2 Telegram Renderer Design and Synthetic Example Closeout

## Purpose

RESET-10G added a minimal v2 Telegram renderer design and a synthetic before/after example for the approved portfolio-first compact Telegram UX.

This sprint proves that the v2 Telegram message shape can be rendered from explicit in-memory records without implementing Telegram delivery, real report generation, real data loading, or Decision Engine behavior.

## Renderer Files Added

- `src/market_scanner/reporting/telegram_renderer.py`

## Tests Added

- `tests/unit/test_v2_telegram_renderer.py`

The tests cover:

- approved section order;
- portfolio-first rendering;
- compact ETF row rendering;
- explicit empty-state rendering;
- pullback and breakout threshold direction display;
- compact data-status display;
- explicit missing profit/loss and target values;
- absence of forbidden authority language;
- absence of legacy debug labels near the top of the message;
- no file creation on import or render;
- no `reports/daily/telegram_message.txt` creation;
- no legacy `scripts` imports.

## Synthetic Example Document Added

- `docs/resets/reset_10g_v2_telegram_renderer_synthetic_before_after_example.md`

## Scope Confirmation

No legacy reporting scripts were modified.

No legacy Telegram scripts were modified.

No legacy tests were modified.

No files were modified under:

- `scripts/`
- `data/processed/`
- `data/portfolio/`
- `data/watchlist/`
- `data/logs/`
- `reports/`
- `.github/workflows/`

No CSV contents were changed.

No report artifact was generated.

No `reports/daily/telegram_message.txt` file was created.

No Telegram message was sent.

No production pipeline, Telegram script, SEC diagnostic, provider call, broker call, network call, Telegram API call, or live data call was run.

## Validation Commands and Results

- `git diff --check` passed.
- `git status --short` showed only RESET-10G new files before staging.
- `git diff --stat` and `git diff --name-only` were empty before staging because the changed files were new and untracked.
- `git status --short --untracked-files=all` showed only:
  - `docs/resets/reset_10g_v2_telegram_renderer_design_and_synthetic_example_closeout.md`
  - `docs/resets/reset_10g_v2_telegram_renderer_synthetic_before_after_example.md`
  - `src/market_scanner/reporting/telegram_renderer.py`
  - `tests/unit/test_v2_telegram_renderer.py`
- `git diff -- tests scripts src data reports .github/workflows` showed no diff before staging because the changed files were new and untracked.
- `.venv/bin/python -m pytest tests/unit/test_v2_telegram_renderer.py` passed: 13 tests passed.
- `.venv/bin/python -m pytest` passed: 526 tests passed.
- Explicit guardrail checks found no diffs under `scripts/`, `data/processed/`, `data/portfolio/`, `data/watchlist/`, `data/logs/`, `reports/`, or `.github/workflows/`.
- Existing-test guardrail found no modified pre-existing test files. The only test file added was `tests/unit/test_v2_telegram_renderer.py`.
- `test ! -f reports/daily/telegram_message.txt` returned nonzero because an ignored local `reports/daily/telegram_message.txt` file already exists in the working tree from before RESET-10G. `git ls-files reports/daily/telegram_message.txt` returned no tracked file, `git status --short reports/daily/telegram_message.txt` showed no change, and `git diff -- reports/daily/telegram_message.txt` showed no diff. RESET-10G did not create, track, modify, or write that file.

## Recommended Next Action

RESET-10H — V2 Reporting Input Aggregation Contract.

If upstream portfolio display ownership needs to be formalized first, run RESET-10C — Portfolio Source-of-Truth Contract before RESET-10H.
