# Derived Observations

This directory contains Market Engine documentation for the `ME-DO` Derived Observation job family.

Derived Observation jobs consume approved upstream observations and emit computed, source-grounded, non-decision derived observations.

They may compute explicitly approved derived values, but they must not create:

* analysis review;
* recommendation review;
* portfolio review;
* delivery output;
* Telegram behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD semantics;
* scores, rankings, urgency, conviction, tradeability, allocation, or execution advice.

Current documents:

* `me_do01_derived_cash_generation_observations.md` — First derived cash-generation observation layer from SEC CompanyFacts Fundamental Observations.
