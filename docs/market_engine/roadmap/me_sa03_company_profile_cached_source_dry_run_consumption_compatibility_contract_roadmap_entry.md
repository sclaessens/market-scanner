# ME-SA03 — Company Profile Cached-Source Dry-Run Consumption Compatibility Contract Roadmap Entry

Sprint ID: ME-SA03  
Status: COMPLETED BY ME-SA03  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26

## Roadmap position

ME-SA03 follows ME-RUN26.

ME-RUN26 showed that ME-SA02 `company_profile` packages can pass acquisition and staging validation while the current `cached_source_snapshot` dry-run path still expects SEC CompanyFacts metadata.

## Completed outcome

ME-SA03 defines the compatibility contract for `company_profile` cached-source dry-run consumption.

The contract covers:

- source-family identity;
- contextual-only profile role;
- manifest expectations;
- staging/import gate rules;
- dry-run eligibility;
- output visibility;
- compatibility matrix;
- proposed failure states;
- auditability requirements;
- future implementation acceptance criteria.

## Next candidate

```text
ME-SA04 — Implement company_profile cached-source dry-run consumption compatibility gate
```

ME-SA04 should implement deterministic local validation and preserve existing cached-source dry-run behavior.

## Safety boundary

ME-SA03 is docs-only and does not change runtime code, test code, provider behavior, network behavior, production writes, delivery behavior, portfolio/watchlist behavior, or downstream authority boundaries.