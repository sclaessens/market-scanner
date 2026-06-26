# ME-SA03 — Company Profile Cached-Source Dry-Run Consumption Compatibility Contract Backlog Entry

Sprint ID: ME-SA03  
Status: COMPLETED BY ME-SA03  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26

## Goal

Define the `company_profile` cached-source dry-run consumption compatibility contract after ME-RUN26 showed that ME-SA02 `company_profile` packages pass acquisition and staging validation but are not yet semantically consumable by the existing `cached_source_snapshot` dry-run path.

## Scope

Documentation-only compatibility contract.

ME-SA03 covers:

- `company_profile` source-family identity;
- snapshot expectations;
- manifest compatibility requirements;
- staging/import gate semantics;
- dry-run eligibility rules;
- required output behavior;
- compatibility matrix;
- failure states;
- auditability requirements;
- future implementation acceptance criteria.

## Outcome

ME-SA03 defined that `company_profile` may be consumed only as contextual company metadata after successful staging/import validation and only when the dry-run path can keep the source local, deterministic, visible, and fail-closed.

The contract explicitly preserves the distinction between:

1. structurally invalid cached-source packages;
2. structurally valid packages that are not semantically consumable by dry-run;
3. structurally valid and semantically compatible `company_profile` packages.

## Implemented documentation

```text
docs/market_engine/audits/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract.md
docs/market_engine/backlog/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract_roadmap_entry.md
```

## Safety boundary

ME-SA03 did not change runtime code, tests, providers, network behavior, production writes, delivery behavior, portfolio/watchlist behavior, or downstream authority boundaries.

## Validation

Runtime tests were not run by ChatGPT because ME-SA03 is docs-only and changed no runtime/test files.

Recommended local post-merge validation:

```text
git diff --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```

## Follow-up candidate

```text
ME-SA04 — Implement company_profile cached-source dry-run consumption compatibility gate
```

ME-SA04 may modify runtime code and tests only after ME-SA03 is accepted. ME-SA04 must preserve existing SEC CompanyFacts cached-source dry-run behavior while adding explicit `company_profile` compatibility gating.