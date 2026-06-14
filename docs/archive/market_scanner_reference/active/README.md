# Active Documentation

This directory contains the current, normative project documentation for `market-scanner`.

Documents in this area describe the active project truth: product direction, architecture, data contracts, pipeline contracts, portfolio/reporting governance, testing strategy, and operating model.

Execution summaries, smoke results, migration reviews, cleanup evidence, and historical sprint artifacts do not belong in this directory. Those documents belong in `docs/audits/` or `docs/archive/`.

## Structure

- `project/` — product vision, charter, roadmap, backlog, operating model, roles, repository structure
- `architecture/` — current architecture and canonical runtime boundaries
- `data/` — data architecture, contracts, lifecycle, source strategy, provider and fundamentals policy
- `pipeline/` — pipeline and Decision Engine contracts
- `portfolio/` — portfolio source of truth and financial analysis documentation
- `reporting/` — reporting contracts, input aggregation, Telegram UX
- `governance/` — active governance policies
- `testing/` — testing strategy and active controlled smoke-test policy

## Placement Rules

New documentation must not be placed directly in `docs/active/`, except for this `README.md`.

New active documentation must be placed in the appropriate subdirectory:

- `project/` — product direction, roadmap, backlog, operating model, roles, repository structure
- `architecture/` — current architecture and canonical runtime boundaries
- `data/` — data architecture, contracts, lifecycle, source strategy, provider and fundamentals policy
- `pipeline/` — pipeline and Decision Engine contracts
- `portfolio/` — portfolio source of truth and financial analysis documentation
- `reporting/` — reporting contracts, input aggregation, Telegram UX
- `governance/` — active governance policies
- `testing/` — testing strategy and active controlled smoke-test policy

Execution summaries, smoke results, migration reviews, cleanup evidence, and historical execution artifacts must be placed in `docs/audits/`.

Superseded, retired, or historical-only documentation must be placed in `docs/archive/`.

Temporary reusable structures, formats, or document skeletons must be placed in `docs/templates/`.
