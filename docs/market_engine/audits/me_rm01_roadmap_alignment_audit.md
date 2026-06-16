# ME-RM01 — Roadmap Alignment Audit

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Sprint name: ME-RM01 — Align Market Engine roadmap and sprint sequence

Status: COMPLETED BY ME-RM01

## Purpose

ME-RM01 aligns the Market Engine roadmap and backlog after ME-RR02 and inserts Setup Detection as a required future layer before Portfolio Review.

This sprint is documentation-only.

## Files Changed

Created:

- `docs/market_engine/roadmap/market_engine_roadmap.md`
- `docs/market_engine/audits/me_rm01_roadmap_alignment_audit.md`

Updated:

- `docs/market_engine/backlog/market_engine_backlog.md`

## Reason For Sprint Insertion

The current backlog identified Portfolio Review as the next follow-up after Recommendation Review.

The project identified Setup Detection as a missing architectural layer between Derived Observations and downstream Analysis Review / Recommendation Review / Portfolio Review.

Setup Detection must be explicitly added before Portfolio Review so the project does not skip pattern/setup detection and jump too quickly from Recommendation Review to Portfolio Review.

Completed Analysis Review and Recommendation Review work remains valid. Future Analysis Review and Recommendation Review sprints can extend existing layers to consume Setup Detection output.

## Completed Chain Before ME-RM01

Completed chain:

- ME-SR01 — Source Refresh
- ME-SC01 — Source Context contract
- ME-SC02 — Source Context implementation
- ME-FO01 — Fundamental Observation contract
- ME-FO02 — Fundamental Observation implementation
- ME-DO01 — Derived Observation implementation
- ME-AR01 — Analysis Review contract
- ME-AR02 — Analysis Review implementation
- ME-RR01 — Recommendation Review contract
- ME-RR02 — Recommendation Review implementation

## Roadmap Sequence After ME-RM01

Recommended and planned sequence:

1. ME-SD01 — Define Setup Detection contract
2. ME-SD02 — Implement first Setup Detection layer
3. ME-AR03 — Extend Analysis Review contract for Setup Detection input
4. ME-AR04 — Implement Analysis Review consumption of Setup Detection
5. ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review
6. ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior
7. ME-PR01 — Define Portfolio Review contract from Recommendation Review
8. ME-PR02 — Implement Portfolio Review
9. ME-DE01 — Define Decision Engine handoff contract
10. ME-DE02 — Implement controlled Decision Engine handoff
11. ME-DL01 — Define Delivery / Reporting contract
12. ME-DL02 — Implement controlled Delivery / Reporting output

## Backlog Updates Made

Backlog updates:

- added the governance rule that future logical next sprints must be preserved in the backlog and roadmap as soon as they are identified;
- added the governance rule that inserted sprints require a documented reason;
- added `ME-SD` as the Setup Detection job family;
- added `ME-RM01` as completed;
- changed the next recommended sprint to `ME-SD01`;
- moved Portfolio Review after Setup Detection-aware Analysis Review and Recommendation Review work;
- recorded future planned sprint sections through Delivery / Reporting.

## Governance Rule Added

New governance rule:

Future sprints must be explicitly preserved in the backlog and roadmap as soon as they are identified as logical next steps. A sprint may only be inserted ahead of the planned sequence when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires it. When such a sprint is inserted, the reason for insertion must be documented in the backlog and roadmap.

## Boundaries Preserved

Confirmed:

- no Python runtime code changed;
- no tests changed;
- no provider code changed;
- no data files changed;
- no portfolio files changed;
- no Telegram/reporting files changed;
- no Decision Engine code changed;
- no existing runtime contracts were changed.

## Validation Performed

Validation commands:

```bash
git diff --check
grep -n "ME-SD01\|ME-SD02\|ME-PR01\|ME-DE01\|ME-DL01\|Status: RECOMMENDED NEXT" docs/market_engine/backlog/market_engine_backlog.md
grep -n "ME-SD01\|ME-SD02\|ME-PR01\|ME-DE01\|ME-DL01" docs/market_engine/roadmap/market_engine_roadmap.md
git status --short
```

Validation result:

- documentation-only changes;
- exactly one `Status: RECOMMENDED NEXT` remains in the backlog;
- `ME-SD01` is the recommended next sprint;
- future sprint sequence is present in both roadmap and backlog.

## Conclusion

ME-RM01 aligns the Market Engine roadmap and backlog with the required Setup Detection layer. The project should proceed next with:

```text
ME-SD01 — Define Setup Detection contract
```
