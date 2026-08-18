# ME-SR26 Current Technical Evidence Contract

Current technical authority requires this chain:

```text
canonical universe + history policy
  -> bar-derived semantic replay with internal time authority
  -> unchanged technical calculations under a checksum-bound policy
  -> new checksum-bound ranking
  -> exact SR25 close reconciliation
  -> validated DATA11 execution proof and downstream after-state
  -> private validated RUN33 handoff context
```

Every arrow is path-loaded and checksum-bound. Mappings, projections, a forged
digest, an old RUN30 ranking, stale canonical history, or an unvalidated
portfolio snapshot cannot enter the chain. A recent price cannot repair
missing or insufficient technical history. Pending fundamental evidence
cannot become approved through the handoff.

The reachable handoff states are pending approval, pending downstream refresh,
invalid downstream authority, and ready for RUN33. Ready requires a canonical
approval validation whose decision identity and checksum match the private
execution proof, plus a fully reconciled downstream after-state. Loose JSON or
caller mappings never carry authority.

This chain performs classification only. The Decision Engine remains the sole
decision authority under repository governance.
