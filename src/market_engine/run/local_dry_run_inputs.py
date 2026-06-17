from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION = (
    "market-engine-local-dry-run-input-fixture-v1"
)
LOCAL_DRY_RUN_INPUT_FIXTURE_STAGE_PAYLOADS_FIELD = "stage_payloads"
LOCAL_DRY_RUN_INPUT_FIXTURE_NON_PRODUCTION_FIELD = "non_production_fixture"


class LocalDryRunInputError(ValueError):
    pass


def load_market_engine_local_dry_run_input(
    path: str | Path,
    *,
    input_mode: str,
) -> Mapping[str, Any]:
    """Load approved local dry-run stage payloads from a local JSON file.

    The loader only reads caller-supplied local JSON. It does not fetch, enrich,
    normalize, repair, or execute upstream stages.
    """

    payload_path = Path(path)
    raw_payload = _read_json_object(payload_path)

    if input_mode == "local_snapshot_fixture":
        return _stage_payloads_from_local_snapshot_fixture(raw_payload)

    if input_mode == "explicit_in_memory_payload":
        return _stage_payloads_from_explicit_payload(raw_payload)

    raise LocalDryRunInputError(
        f"Local dry-run JSON input is not supported for input mode: {input_mode}"
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as payload_file:
            payload = json.load(payload_file)
    except OSError as exc:
        raise LocalDryRunInputError(f"Unable to read stage payload JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LocalDryRunInputError(f"Stage payload JSON is invalid: {path}") from exc

    if not isinstance(payload, Mapping):
        raise LocalDryRunInputError(
            "Stage payload JSON must contain an object at the top level."
        )
    return payload


def _stage_payloads_from_local_snapshot_fixture(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    observed_version = payload.get("dry_run_input_fixture_format_version")
    if observed_version != MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION:
        raise LocalDryRunInputError(
            "Local snapshot fixture JSON must use "
            f"{MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION}."
        )

    if payload.get(LOCAL_DRY_RUN_INPUT_FIXTURE_NON_PRODUCTION_FIELD) is not True:
        raise LocalDryRunInputError(
            "Local snapshot fixture JSON must explicitly set non_production_fixture=true."
        )

    fixture_input_mode = payload.get("input_mode")
    if fixture_input_mode not in (None, "local_snapshot_fixture"):
        raise LocalDryRunInputError(
            "Local snapshot fixture input_mode must be local_snapshot_fixture when present."
        )

    stage_payloads = payload.get(LOCAL_DRY_RUN_INPUT_FIXTURE_STAGE_PAYLOADS_FIELD)
    if not isinstance(stage_payloads, Mapping):
        raise LocalDryRunInputError(
            "Local snapshot fixture JSON must contain a stage_payloads object."
        )
    return stage_payloads


def _stage_payloads_from_explicit_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    observed_version = payload.get("dry_run_input_fixture_format_version")
    if observed_version == MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION:
        stage_payloads = payload.get(LOCAL_DRY_RUN_INPUT_FIXTURE_STAGE_PAYLOADS_FIELD)
        if not isinstance(stage_payloads, Mapping):
            raise LocalDryRunInputError(
                "Local dry-run fixture JSON must contain a stage_payloads object."
            )
        return stage_payloads

    return payload
