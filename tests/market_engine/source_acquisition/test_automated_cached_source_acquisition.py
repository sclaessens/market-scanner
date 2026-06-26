from __future__ import annotations

import hashlib
import json
from pathlib import Path

from market_engine.source_acquisition.automated_cached_source_acquisition import (
    REQUEST_FORMAT,
    RESULT_FORMAT,
    AutomatedCachedSourceAcquisitionError,
    DeterministicFakeCompanyProfileAdapter,
    run_automated_cached_source_acquisition,
)
from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    build_cached_source_snapshot_staging_validation,
)


REQUESTED_AT = "2026-06-26T12:00:00Z"
GENERATED_AT = "2026-06-26T12:01:00Z"


def test_acquisition_writes_company_profile_snapshot_packages(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, tickers=("NVDA", "AMD", "ASML"))

    result = run_automated_cached_source_acquisition(request)

    assert result["result_format"] == RESULT_FORMAT
    assert result["summary"] == {
        "requested_ticker_count": 3,
        "requested_source_family_count": 1,
        "entry_count": 3,
        "completed_count": 3,
        "completed_with_limitations_count": 0,
        "blocked_count": 0,
        "rejected_count": 0,
        "provider_error_count": 0,
        "unsupported_count": 0,
        "stale_count": 0,
        "invalid_manifest_count": 0,
    }
    assert result["safety"] == {
        "provider_calls_performed": False,
        "network_used": False,
        "telegram_sent": False,
        "portfolio_written": False,
        "watchlist_written": False,
        "broker_action_performed": False,
        "production_write_performed": False,
    }
    assert result["next_step"] == {
        "recommended_action": "run_existing_import_staging_validation",
        "import_candidate_root": str(tmp_path / "acquisition"),
        "dry_run_candidate": True,
        "blocked_reason": None,
    }
    assert (tmp_path / "acquisition" / "acquisition_result.json").exists()
    assert [entry["ticker"] for entry in result["entries"]] == ["NVDA", "AMD", "ASML"]
    for entry in result["entries"]:
        assert entry["status"] == "completed"
        assert entry["source_family"] == "company_profile"
        assert entry["issues"] == ()
        assert entry["provenance"]["adapter_id"] == "fake_company_profile_adapter"
        assert entry["provenance"]["adapter_version"] == "test-v1"
        assert entry["provenance"]["request_metadata"]["network_used"] is False
        assert entry["freshness"]["state"] == "current"
        manifest = json.loads(Path(entry["manifest_path"]).read_text(encoding="utf-8"))
        payload_path = Path(entry["payload_paths"][0])
        assert manifest["ticker"] == entry["ticker"]
        assert manifest["source_family"] == "company_profile"
        assert manifest["validation_status"] == "passed"
        assert manifest["staleness_status"] == "fresh"
        assert manifest["usable_for_cached_source_dry_run"] is True
        assert manifest["local_payload_sha256"] == _sha256(payload_path)
        assert manifest["local_payload_size_bytes"] == payload_path.stat().st_size


def test_acquisition_output_is_accepted_by_existing_staging_validator(
    tmp_path: Path,
) -> None:
    run_automated_cached_source_acquisition(_request(tmp_path, tickers=("NVDA",)))

    report = build_cached_source_snapshot_staging_validation(
        staging_root=tmp_path / "acquisition",
        validated_at="2026-06-26T12:02:00Z",
    )

    assert report["counts"]["accepted_entries"] == 1
    assert report["counts"]["rejected_entries"] == 0
    entry = report["entries"][0]
    assert entry["ticker"] == "NVDA"
    assert entry["source_family"] == "company_profile"
    assert entry["staging_validation_status"] == "accepted"
    assert entry["issues"] == ()


def test_invalid_request_format_fails_closed(tmp_path: Path) -> None:
    request = {**_request(tmp_path), "request_format": "unsupported"}

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert exc.reason == "automated cached-source acquisition request is invalid"
        assert "request_format_invalid" in exc.issues
    else:
        raise AssertionError("expected invalid request format to fail")


def test_empty_ticker_list_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path, tickers=())

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert "tickers_empty" in exc.issues
    else:
        raise AssertionError("expected empty ticker list to fail")


def test_malformed_and_duplicate_tickers_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path, tickers=("NVDA", "NVDA", "bad ticker"))

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert "ticker_duplicate" in exc.issues
        assert "ticker_invalid" in exc.issues
    else:
        raise AssertionError("expected malformed ticker list to fail")


def test_unsupported_ticker_is_reported_without_artifacts(tmp_path: Path) -> None:
    request = _request(tmp_path, tickers=("TSM",))

    result = run_automated_cached_source_acquisition(request)

    assert result["summary"]["unsupported_count"] == 1
    assert result["next_step"]["dry_run_candidate"] is False
    assert result["next_step"]["blocked_reason"] == "no_usable_acquisition_entries"
    entry = result["entries"][0]
    assert entry["ticker"] == "TSM"
    assert entry["status"] == "unsupported"
    assert entry["issues"] == ("unsupported_ticker",)
    assert entry["manifest_path"] is None
    assert not (tmp_path / "acquisition" / "TSM").exists()


def test_unsupported_source_family_fails_request_validation(tmp_path: Path) -> None:
    request = _request(tmp_path, source_families=("sec_companyfacts",))

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert "source_family_unsupported" in exc.issues
    else:
        raise AssertionError("expected unsupported source family to fail")


def test_safety_flags_must_remain_false(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["safety_flags"] = {
        **request["safety_flags"],
        "allow_network": True,
    }

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert "allow_network_must_be_false" in exc.issues
    else:
        raise AssertionError("expected unsafe request to fail")


def test_fake_adapter_failure_becomes_provider_error(tmp_path: Path) -> None:
    adapter = DeterministicFakeCompanyProfileAdapter(failing_tickers=("AMD",))

    result = run_automated_cached_source_acquisition(
        _request(tmp_path, tickers=("NVDA", "AMD")),
        company_profile_adapter=adapter,
    )

    assert result["summary"]["completed_count"] == 1
    assert result["summary"]["provider_error_count"] == 1
    assert result["entries"][0]["ticker"] == "NVDA"
    assert result["entries"][0]["status"] == "completed"
    assert result["entries"][1]["ticker"] == "AMD"
    assert result["entries"][1]["status"] == "provider_error"
    assert result["entries"][1]["issues"] == ("adapter_error",)
    assert not (tmp_path / "acquisition" / "AMD").exists()


def test_existing_snapshot_path_blocks_without_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "acquisition" / "NVDA" / "company_profile"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = run_automated_cached_source_acquisition(
        _request(tmp_path, tickers=("NVDA",)),
    )

    assert result["summary"]["blocked_count"] == 1
    entry = result["entries"][0]
    assert entry["status"] == "blocked"
    assert entry["issues"] == ("snapshot_path_already_exists",)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_destination_root_rejects_production_data_path(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["destination_root"] = "data/market_engine/cached_source_snapshots"

    try:
        run_automated_cached_source_acquisition(request)
    except AutomatedCachedSourceAcquisitionError as exc:
        assert "destination_root_must_be_non_production" in exc.issues
    else:
        raise AssertionError("expected production data destination to fail")


def test_module_does_not_import_provider_or_delivery_modules() -> None:
    import market_engine.source_acquisition.automated_cached_source_acquisition as module

    assert "SecCompanyFactsProvider" not in module.__dict__
    assert "requests" not in module.__dict__
    assert "yfinance" not in module.__dict__
    assert "telegram" not in module.__dict__


def _request(
    tmp_path: Path,
    *,
    tickers: tuple[str, ...] = ("NVDA",),
    source_families: tuple[str, ...] = ("company_profile",),
) -> dict[str, object]:
    return {
        "request_format": REQUEST_FORMAT,
        "request_id": "me-sa02-local-20260626T120000Z",
        "requested_at": REQUESTED_AT,
        "generated_at": GENERATED_AT,
        "run_mode": "dry_run",
        "ticker_source": {
            "mode": "explicit_list",
            "source_id": "operator_bounded_list",
        },
        "tickers": list(tickers),
        "source_families": list(source_families),
        "destination_root": str(tmp_path / "acquisition"),
        "freshness_policy": {
            "default_max_age_days": 7,
            "per_source_family": {
                "company_profile": {
                    "max_age_days": 30,
                    "source_timestamp_required": False,
                }
            },
        },
        "provider_policy": {
            "approved_adapters": [
                {
                    "adapter_id": "fake_company_profile_adapter",
                    "adapter_version": "test-v1",
                    "source_families": ["company_profile"],
                    "allowed_run_modes": ["dry_run", "local_non_production"],
                    "provider_name": "deterministic_fake_provider",
                    "canonical_source_identity": "fake://company_profile",
                    "network_required": False,
                    "rate_limit_policy": "not_applicable",
                    "error_policy": "fail_closed",
                }
            ],
            "allow_hidden_fallback": False,
            "allow_silent_substitution": False,
            "allow_fabricated_data": False,
        },
        "safety_flags": {
            "allow_provider_calls": False,
            "allow_network": False,
            "allow_production_writes": False,
            "allow_telegram_send": False,
            "allow_portfolio_writes": False,
            "allow_watchlist_writes": False,
            "allow_broker_actions": False,
        },
        "operator_context": {
            "requested_by": "operator",
            "purpose": "bounded local acquisition contract validation",
            "notes": [],
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
