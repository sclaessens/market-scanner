# ME-DS01 - Dispatch Station Governor Report Output Contract Audit

Sprint ID: ME-DS01

Status: COMPLETED DOCS-ONLY CONTRACT

Job family: ME-DS / Dispatch Station

Date: 2026-07-05

Branch: `me-ds01-dispatch-station-governor-report-output-contract`

## Purpose

ME-DS01 defines the first Dispatch Station output contract after the completed Governor evaluation chain.

The contract establishes:

```text
market-engine-dispatch-station-governor-report-v1
```

and positions Dispatch Station as an output packaging and presentation boundary only.

## Inputs reviewed

ME-DS01 is grounded in the existing architecture and Governor boundaries:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

The contract preserves the existing Governor output envelope and the later Governor additions for factor scoring, recommendation-state mapping, buy-zone explanation, and position-management explanation.

## Contract decisions

The sprint defines:

* one canonical versioned Dispatch Station report contract;
* approved Governor-only upstream input boundaries;
* explicit `blocked`, `partial_report`, `descriptive_report`, and `evaluation_report` states;
* canonical report identity and subject fields;
* ordered report sections;
* factor, recommendation, buy-zone, and position-management rendering boundaries;
* mandatory blocker, missing-evidence, risk, limitation, provenance, and authority presentation;
* `canonical_json`, `operator_markdown`, and `compact_preview` rendering profiles;
* deterministic rendering requirements;
* fail-closed rules;
* cross-format semantic equivalence requirements;
* explicit ME-DS02 implementation acceptance criteria.

## Key governance decision

Dispatch Station is not allowed to improve, reinterpret, score, rank, or operationalize Governor output.

In particular, it cannot:

```text
calculate Governor scores
create weighted totals
create overall scores or rank
change recommendation meaning
invent price levels
invent targets or stops
create urgency or conviction
create tradeability semantics
create allocation or position sizing
create order instructions
send Telegram or email
publish production reports
mutate portfolio or watchlist state
call providers or fetch live prices
route broker orders
make Decision Engine decisions
```

## Output-profile boundary

The contract deliberately distinguishes rendering from delivery.

Allowed contract profiles:

```text
canonical_json
operator_markdown
compact_preview
```

`compact_preview` is channel-neutral. It may later feed a Telegram/email/dashboard preview adapter, but ME-DS01 authorizes no transport, destination, recipient, scheduling, notification, or production publishing behavior.

## Fail-closed behavior

The contract requires `blocked` output for malformed or unsafe report input, including unknown contracts, missing evaluation identity, contradictory recommendation eligibility/state, unreferenced numeric price levels, unsupported authority flags, missing material provenance, and renderer inability to preserve material blockers or limitations.

Missing evidence must remain explicit and must never be converted to zero or favorable text.

## Cross-format equivalence

The canonical JSON report is the semantic reference.

Markdown and compact preview representations must not:

* strengthen recommendation language;
* change numeric price levels;
* hide critical blockers while surfacing favorable output;
* change authority flags;
* infer holdings from absent portfolio context.

## Authority verification

ME-DS01 is documentation-only.

No changes are authorized for:

```text
src/
tests/
scripts/
providers
network calls
live data
runtime report generation
Telegram/email delivery
production publishing
portfolio mutation
watchlist mutation
broker behavior
order generation
scheduler behavior
UI behavior
Decision Engine authority
```

## Acceptance result

PASS.

ME-DS01 defines an implementable Dispatch Station report contract while preserving the separation between Governor investment evaluation, report rendering, delivery, and Decision Engine authority.

## Next sprint

```text
ME-DS02 - Implement local non-production Governor report artifact
```
