<<'EOF'
# V2 One-Ticker Real-Source Data Capture Approval

Status: ACTIVE
Reset stage: RESET-10L-BL18

## Purpose

This document approves a single controlled real-source data capture target for
the next sprint.

The selected ticker is:

```text
NVDA

The selected company is NVIDIA Corporation.

This approval exists to move the project from synthetic-only proof toward one
controlled real-data test. The goal is to let the application start learning from
real fundamentals and reveal real defects in source capture, normalization,
persistence, readiness, and later analysis behavior.

This is an approval/governance artifact only. It does not execute a capture,
write production data, call providers, call SEC or EDGAR, call brokers, access
the network, generate reports, send Telegram messages, run the production
pipeline, update portfolio/watchlist data, or invoke the Decision Engine.

Approval Decision

Decision: approve a one-ticker controlled real-source capture attempt for NVDA in
the next sprint.

Approved next action:

RESET-10L-BL19 — Execute One-Ticker Real-Source Persistence Smoke

The next sprint may perform a manual or explicitly invoked one-ticker real-source
capture for NVDA and may persist only the approved redacted or governed output
specified by the sprint guardrails.

Approved Scope

Approved ticker:

NVDA

Approved source category:

official/regulatory or primary-source-shaped fundamentals data

Approved capture mode:

manual-only or explicitly invoked local smoke execution

Approved persistence mode for the next sprint:

controlled one-ticker smoke only

Approved review focus:

can real NVDA source data be captured or manually supplied through the existing
provider/source boundary;
can raw source evidence be preserved;
can normalized program-ready fundamentals be produced;
can source-data readiness remain neutral;
can missing values remain explicit;
can the persistence boundary reject forbidden production paths;
can the project identify real defects before broadening scope.
Not Approved

This approval does not authorize:

broad provider integration;
automatic scheduled provider execution;
multi-ticker capture;
production pipeline execution;
report generation;
Telegram delivery;
portfolio or watchlist updates;
Decision Engine investment action;
production data writes beyond the separately approved BL19 smoke scope;
BUY, SELL, HOLD, allocation, conviction, urgency, target-price, scoring,
tradeability, or recommendation behavior.
Capture Constraints for BL19

BL19 must stay limited to:

one ticker: NVDA;
one source family;
one controlled local execution path;
no committed credentials;
no committed raw live payload unless separately redacted and explicitly
approved;
no broad production persistence;
no pipeline execution;
no reports;
no Telegram artifacts;
no portfolio/watchlist updates;
no Decision Engine investment behavior.

If a real source call is made locally, the operator must ensure that credentials,
raw response bodies, private tokens, API keys, and unredacted live payloads are
not committed.

Allowed BL19 Outputs

BL19 may commit only governance-safe outputs such as:

a redacted smoke execution summary;
a controlled source-data readiness summary;
a minimal governed fixture or evidence summary if explicitly redacted;
backlog updates;
tests or code only if separately required by the BL19 implementation prompt and
still within approved guardrails.

BL19 must not commit:

credentials;
API keys;
raw unredacted live payloads;
generated reports;
Telegram output;
production portfolio/watchlist data changes;
Decision Engine recommendations;
BUY, SELL, HOLD, allocation, conviction, urgency, target-price, scoring,
tradeability, or recommendation output.
Why NVDA

NVDA is selected because it is a highly relevant listed company for the user's
investment workflow and is suitable for testing whether real fundamentals can
move through the new source-data architecture.

This selection is not an investment recommendation. It is a technical test target
for the application.

Success Criteria for BL19

BL19 should be considered successful if it can prove, for NVDA only, that:

the source capture path can be executed manually or explicitly;
provenance can be recorded or summarized;
source timestamps and retrieval timestamps can be tracked or summarized;
raw source evidence can remain separate from normalized fundamentals;
normalized fundamentals can preserve explicit missing-value states;
neutral readiness can be emitted;
no missing values are converted to zero;
no reports, Telegram artifacts, pipeline runs, portfolio updates, watchlist
updates, or Decision Engine behavior are triggered;
any committed output is redacted and governance-safe.
Failure Criteria for BL19

BL19 should fail closed if:

credentials would need to be committed;
raw live payloads cannot be safely handled;
provenance cannot be established;
source data cannot be traced;
missing values are hidden or zero-filled;
production paths would be modified without separate approval;
reports or Telegram output would be generated;
pipeline execution would be required;
Decision Engine investment behavior would be triggered.
Next Step

Proceed to:

RESET-10L-BL19 — Execute One-Ticker Real-Source Persistence Smoke

BL19 should perform the first controlled NVDA real-source persistence smoke and
record the result without broadening scope beyond one ticker and one source
family.
EOF