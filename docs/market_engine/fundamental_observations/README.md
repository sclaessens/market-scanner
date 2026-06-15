# Fundamental Observations

This directory contains Market Engine documentation for the `ME-FO` Fundamental Observation job family.

Fundamental Observation jobs consume approved Source Context output and emit source-grounded, non-decision fundamental observations.

They must not create:

- analysis review;
- recommendation review;
- portfolio review;
- delivery output;
- Telegram behavior;
- Decision Engine behavior;
- BUY / SELL / HOLD semantics;
- scores, rankings, urgency, conviction, tradeability, allocation, or execution advice.

Current documents:

- `me_fo01_fundamental_observation_contract.md` — Fundamental Observation contract from SEC CompanyFacts Source Context.
