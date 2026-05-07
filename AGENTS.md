# Market Scanner — Institutional AI Governance

## Core Doctrine

classification upstream  
allocation downstream

Decision Engine = ONLY allocation authority.

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
