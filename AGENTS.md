# Market Scanner — Institutional AI Governance

## Core Doctrine

classification upstream  
allocation downstream

Decision Engine = ONLY allocation authority.

## Repository Language Governance

All repository content must remain English-only.

This applies to documentation, markdown files, sprint documents, audit documents, technical specifications, functional specifications, code comments, logging messages, test names, test descriptions, developer-facing text, CSV column names, generated reports, configuration descriptions, CI output messages, and governance documents.

Dutch is allowed only in direct chat communication with the user. It must not be introduced into repository files, generated artifacts, tests, logs, reports, configuration, or developer-facing output.

Future sprints, audits, migrations, refactors, tests, reports, and governance updates inherit this language standard.

## Hard Rules

No layer outside `scripts/core/decision_engine.py` may:

- determine tradeability
- determine conviction
- execute BUY logic
- execute SELL logic
- execute REMOVE logic
- execute allocation gating
- execute urgency logic

## Layer Responsibilities

| Layer | Responsibility |
|---|---|
| Scanner | discovery |
| Validation | structure classification |
| Context | leadership classification |
| Fundamentals | quality classification |
| Watchlist | timing-state tracking |
| Portfolio | exposure/risk-state modelling |
| Decision Engine | allocation decisions |
| Reporting | communication only |

## Forbidden Concepts

Forbidden outside Decision Engine:

- tradeable_setup
- context_tradeable
- conviction
- BUY NOW
- SET LIMIT BUY
- SET STOP BUY
- urgency
- execution gating
- hidden filtering

## Architecture Rules

Validation may NOT:

- invalidate extended momentum
- simulate execution quality
- determine allocation eligibility

Context may NOT:

- block opportunities
- determine tradeability
- simulate allocation

Portfolio may NOT:

- determine BUY/SELL
- destroy upstream opportunities

Reporting may NOT:

- interpret execution urgency
- prioritize allocation
- inject decision logic

## Required Development Behaviour

Preserve:

- opportunity distribution
- classification richness
- deterministic outputs
- separation of concerns

Do NOT:

- add new strategy logic
- add new filters
- optimize thresholds
- redesign architecture
- inject hidden assumptions

## Mandatory Checks Before PR

Run:

```bash
pytest
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
