# ME-UNI08 Backlog Entry - Professional Swing Universe CLI Flag

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

## Goal

Add a first-class CLI flag for selecting the approved editable Professional Swing Universe in the local cached-source batch dry-run command.

## Outcome

The cached-source batch dry-run command now supports:

```text
--professional-swing-universe
```

The flag routes through the existing ME-UNI07 runtime-input builder and ME-UNI06 loader/validation behavior.

## Conflict Rule

The new flag is mutually exclusive with other ticker-input modes.

Combining `--professional-swing-universe` with a custom `--canonical-ticker-universe <path>` fails closed with the existing argparse mutual-exclusion error.

## Tests

ME-UNI08 added command tests for:

* flag acceptance;
* default Professional Swing Universe path/config resolution;
* custom canonical universe path preservation;
* conflict behavior;
* CLI help text.

## Boundaries

ME-UNI08 did not introduce provider calls, SEC or EDGAR live calls, yfinance calls, source refresh, source-support classification, Telegram or email delivery, reporting output, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Recommended Sprint

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

ME-SR05 should classify actual source support for Professional Swing Universe rows before broad supported-universe cached-source scanning.
