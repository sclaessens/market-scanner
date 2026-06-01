# Reporting Contract

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the v2 reporting boundary.

## Core Rule

Reporting communicates Decision Engine outputs. Reporting does not decide.

## Allowed Reporting Behavior

Reporting may:

- format final decisions;
- group final decisions;
- summarize final decisions;
- truncate presentation when the source traceability remains available;
- prepare dashboard data;
- prepare Telegram or other communication messages;
- expose freshness and provenance status;
- show review-required and insufficient-data states.

## Forbidden Reporting Behavior

Reporting must not:

- create allocation priority;
- reinterpret urgency;
- suppress rows without explicit representation rules;
- change final action semantics;
- convert REVIEW into BUY, SELL, HOLD, or other action;
- hide source-data insufficiency;
- create independent conviction or ranking;
- use presentation order as hidden allocation logic.

## Reporting Inputs

v2 Reporting may consume final Decision Engine outputs and approved optional stability or persistence metadata. It must not consume upstream artifacts in a way that bypasses the Decision Engine.

## Reporting Outputs

Reporting outputs are generated communication artifacts. They are not source-of-truth inputs for future runs unless explicitly approved under a separate contract.

## Telegram Boundary

Telegram delivery is a transport surface. It must not alter reporting semantics or decision semantics. Network/API behavior should remain isolated from reporting logic.

## Testing Expectations

v2 reporting tests must prove that reporting preserves source final decisions, represents omitted or truncated content explicitly, and never creates allocation authority.
