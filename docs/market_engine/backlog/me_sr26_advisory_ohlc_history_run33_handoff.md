# ME-SR26 — Advisory OHLC History and RUN33 Handoff

Sprint ID: ME-SR26
Status: IMPLEMENTED — POST-MERGE CANARY REQUIRED

## Outcome

ME-SR26 adds a non-canonical, advisory-only route for bounded full-universe
daily OHLC history and reruns the unchanged RUN30 technical setup and ranking
semantics only over validated current history. It emits a checksum-bound,
conditional RUN33 input handoff; it does not execute RUN33.

## Acceptance contract

- Exactly 952 canonical identities must reconcile.
- Each series is one currency, source mapping, unadjusted OHLC basis, and
  adjustment policy, with exact positive decimal strings and explicit volume.
- Minimum history is 252 sessions: 200 indicator warm-up sessions plus a
  52-session safety margin. At most 420 sessions are retained.
- Missing, invalid, insufficient, stale, identity-blocked, and
  adjustment-policy-blocked histories remain distinct.
- At least 80% of otherwise-valid series exactly one session late blocks the
  run as widespread provider-session lag. Individual fallbacks are capped at
  25; no retry wave exists.
- SR25 price and the latest history close must match identity, session,
  currency, source semantics, and exact decimal value.
- DATA11 approval remains pending for ASH, BIO, and CI, so the RUN33 handoff is
  conditional and all candidates remain ineligible.

## Required next action

After merge, one reviewed `workflow_dispatch` canary must acquire all 952
histories at the new 09:30 UTC window. Review the manifest status distribution,
provider-lag evidence, checksum replay, screening count, top-25 gap, and
workflow artifact. Do not publish, advance `market-data`, or execute RUN33.
