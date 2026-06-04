# V2 Python File Creation Policy

Status: ACTIVE
Reset stage: RESET-10M

## Purpose

This document defines a binding policy for creating or modifying Python files in
the v2 market-scanner codebase.

The purpose is to prevent a repeat of the previous application pattern where new
Python files were created too quickly, causing code to become scattered,
duplicated, tightly coupled, and difficult to reason about.

The default expectation for future coding work is:

```text
Update existing Python files first.
Create new Python files only when clearly justified.
```

This policy applies to all Codex/developer implementation sprints from BL20
onward and to any future sprint that touches Python code.

## Core Rule

A new `.py` file may not be created by default.

Before creating a new Python file, the implementer must inspect the existing
relevant modules and determine whether the change belongs in an existing file.

A new Python file is allowed only if all of the following conditions are true:

1. The existing relevant modules were inspected.
2. The change cannot be cleanly implemented inside an existing module.
3. The new file has one clearly separated responsibility.
4. The new file does not duplicate an existing module.
5. The new file does not bypass an existing architectural boundary.
6. The new file does not create a parallel one-off flow.
7. The PR description includes a Python file creation justification.

If a new Python file is created without this justification, the sprint is not
complete.

## Required PR Justification

Any PR that creates a new `.py` file must include a section named:

```text
Python file creation justification
```

That section must answer:

- which existing modules were inspected;
- why the change does not belong in those existing modules;
- what single responsibility the new file owns;
- how the new file fits the approved architecture;
- why the new file does not duplicate or bypass existing code;
- what tests prove the new boundary is necessary and safe.

## Forbidden Patterns

The following patterns are forbidden unless explicitly approved in a sprint
brief:

- ticker-specific Python files;
- temporary implementation modules;
- quick-test modules committed to the repository;
- duplicate helper modules;
- parallel v2 modules that bypass approved boundaries;
- new orchestration files when an existing orchestration boundary exists;
- new analysis files created only for a single experiment;
- new persistence/provider/reporting/decision files that duplicate existing
  responsibilities.

Examples of names that should normally be rejected:

```text
nvda_analysis.py
quick_real_test.py
run_temp_analysis.py
helper_new.py
real_data_v2.py
analysis_new.py
provider_temp.py
persistence_test_runner.py
```

Temporary local scripts may be used by the operator if needed, but they must not
be committed unless explicitly approved.

## Existing-File-First Workflow

Every coding sprint that may touch Python should follow this workflow:

1. Inspect relevant existing modules.
2. Identify the existing architectural boundary.
3. Prefer the smallest safe change inside the existing boundary.
4. Add or update tests around that existing boundary.
5. Create a new Python file only if the existing boundary would become unclear,
   overloaded, or architecturally incorrect.
6. Document the reason in the PR.

## Responsibility Boundaries

Future work should preserve the current v2 separation of concerns.

Provider/source code should remain in provider/source modules.

Persistence code should remain in persistence modules.

Smoke execution code should remain in smoke-specific modules.

Analysis review code should use the approved analysis or decision boundaries
rather than creating one-off ticker-specific files.

Reporting and Telegram behavior must not be introduced in unrelated modules.

Decision Engine behavior must not be bypassed by helper files or ad hoc analysis
scripts.

## BL20-Specific Rule

For `RESET-10L-BL20 — Run First One-Ticker Real Fundamental Analysis`, Codex must
not create a new ticker-specific Python file for NVDA.

Forbidden examples for BL20 include:

```text
nvda_analysis.py
nvda_real_analysis.py
run_nvda_analysis.py
first_real_analysis.py
```

BL20 must first inspect and use existing provider, normalization, persistence,
analysis, and Decision Engine boundaries.

If BL20 genuinely requires a new Python module, it must provide a formal Python
file creation justification in the PR and explain why existing modules cannot be
used safely.

## Cleanup Connection

This policy supports the wider reset goal of reducing legacy Python sprawl.

After the first real analysis run, the project should perform a Python usage and
legacy cleanup review to identify:

- Python files still required by the new v2 flow;
- Python files that are legacy-only;
- Python files that should be archived;
- Python files that can be removed after certified replacement;
- duplicate responsibilities that should be consolidated.

Cleanup must be done through a separate approved sprint. This policy does not
authorize file deletion.

## Guardrails

This policy does not authorize:

- code changes;
- file deletion;
- moving Python files;
- creating new runtime behavior;
- production pipeline changes;
- reports;
- Telegram behavior;
- Decision Engine changes;
- investment recommendations.

It only defines governance rules for future Python file creation.

## Required Prompt Language

Future Codex prompts that may touch Python code must include:

```text
Python file creation policy:
Default to updating existing Python files. Do not create a new `.py` file unless
existing relevant modules have been inspected and a new file is clearly necessary.
If a new Python file is created, include a Python file creation justification in
the PR. One-off ticker-specific Python files, temporary helper modules, duplicate
modules, and parallel bypass modules are forbidden unless explicitly approved.
```

## Status

This policy is binding for BL20 and all subsequent implementation sprints unless
superseded by a later approved governance document.
