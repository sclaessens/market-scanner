# ME-SA02 - Bounded automated cached-source acquisition job backlog entry

Status: COMPLETED BY ME-SA02

Job family: ME-SA / Source Acquisition

## Summary

ME-SA02 implements the first bounded automated cached-source acquisition job according to the ME-SA01 contract.

Implemented:

* request validation for the automated acquisition request format;
* bounded explicit ticker-list support;
* initial `company_profile` source-family support;
* deterministic fake adapter behavior for tests;
* per-ticker cached-source snapshot package writing;
* result payload writing;
* manifest/payload hash and size recording;
* provenance, freshness, validation state, and usable flag preservation;
* fail-closed request validation and per-entry failure statuses;
* staging-validator compatibility tests.

## Validation

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_acquisition/test_automated_cached_source_acquisition.py -q
12 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py -q
19 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
492 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1159 passed

git diff --check
passed
```

## Implemented documentation

```text
docs/market_engine/source_data/me_sa02_bounded_automated_cached_source_acquisition_job_implementation.md
docs/market_engine/audits/me_sa02_bounded_automated_cached_source_acquisition_job_audit.md
docs/market_engine/backlog/me_sa02_bounded_automated_cached_source_acquisition_job_backlog_entry.md
docs/market_engine/roadmap/me_sa02_bounded_automated_cached_source_acquisition_job_roadmap_entry.md
```

## Next Sprint

```text
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
```

ME-RUN26 should execute the ME-SA02 job output through staging validation and the existing local dry-run route, recording any dry-run source-family incompatibility as an explicit blocked reason.
