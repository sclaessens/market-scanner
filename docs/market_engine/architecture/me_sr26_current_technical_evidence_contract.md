# ME-SR26 Current Technical Evidence Contract

## Canonical production authority

Public production callers select evidence artifact roots only. They cannot select the canonical universe, the ME-SR26 history or screening policies, the SR25 price policy used by RUN33, the provider, freshness time, or source provenance. Production resolves those inputs internally from `DEFAULT_UNIVERSE_SNAPSHOT`, `DEFAULT_POLICY_PATH`, `DEFAULT_SCREENING_POLICY`, `DEFAULT_PRICE_POLICY_PATH`, the approved provider adapter, the system UTC clock, and the executing repository's Git `HEAD`.

Custom universe, policy, provider, time, and source-SHA inputs exist only on private deterministic implementation helpers. They are test seams and are not production authority APIs.

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

The production history builder selects its provider internally. Public history,
screening, and RUN33 authority APIs expose no provider or freshness-time
override. Each screening or handoff operation captures one internal canonical
UTC time and applies it coherently to history and SR25 freshness. Artifact paths
remain caller-selectable; evidence freshness does not.

The reachable handoff states are pending approval, pending downstream refresh,
invalid downstream authority, and ready for RUN33. Ready requires a canonical
approval validation whose decision identity and checksum match the private
execution proof, plus a fully reconciled downstream after-state. Loose JSON or
caller mappings never carry authority.

This chain performs classification only. The Decision Engine remains the sole
decision authority under repository governance.
