# Analysis Review

This directory contains Market Engine documentation for the `ME-AR` Analysis Review job family.

Analysis Review jobs consume approved upstream observations and produce non-recommendation analytical review output.

They may interpret observed source and derived conditions, but they must not create:

- recommendation review;
- portfolio review;
- delivery output;
- Telegram behavior;
- Decision Engine behavior;
- BUY / SELL / HOLD semantics;
- scores, rankings, urgency, conviction, tradeability, allocation, position sizing, or execution advice.

Current documents:

- `me_ar01_analysis_review_contract.md` — Analysis Review contract from Fundamental and Derived Observations.