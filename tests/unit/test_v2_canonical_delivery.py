import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.delivery.delivery_boundary import (
    ALLOWED_DELIVERY_CHANNELS,
    BLOCKED_BEHAVIOR_CODES,
    BLOCKED_EXECUTION_CODES,
    BLOCKED_FINAL_STATE_CODES,
    DELIVERY_CANONICAL_OWNER,
    LEGACY_DELIVERY_AUTHORITIES,
    build_delivery_plan,
    build_delivery_policy,
    build_telegram_delivery_plan,
)
from market_scanner.delivery import delivery_boundary


FORBIDDEN_OUTPUT_TERMS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "urgency",
    "scoring",
    "target-price",
    "tradeability",
    "recommendation",
}

BLOCKED_POLICY_FIELDS = {
    "blocked_execution_codes",
    "blocked_final_state_codes",
    "blocked_behavior_codes",
}

ARCHIVED_LEGACY_RUNTIME_DIR = Path("archive") / "legacy_runtime" / "scripts"


def _flatten_values(value):
    if is_dataclass(value):
        yield from _flatten_values(asdict(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in BLOCKED_POLICY_FIELDS:
                continue
            yield str(key)
            yield from _flatten_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield str(value)


def test_delivery_plan_is_deterministic():
    assert build_delivery_plan() == build_delivery_plan()
    assert build_telegram_delivery_plan() == build_telegram_delivery_plan()
    assert build_delivery_policy() == build_delivery_policy()


def test_delivery_plan_exposes_canonical_owner_and_stage_order():
    plan = build_delivery_plan()

    assert plan.canonical_owner == DELIVERY_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "telegram_delivery_planning",
        "delivery_execution_policy_block",
    )
    assert plan.legacy_delivery_authorities == LEGACY_DELIVERY_AUTHORITIES
    assert plan.migration_status == "canonical_delivery_boundary_established"


def test_delivery_plan_forbids_side_effects_by_default():
    for stage in build_delivery_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.network_calls_allowed is False
        assert stage.credentials_allowed is False
        assert stage.production_data_writes_allowed is False
        assert stage.report_file_writes_allowed is False
        assert stage.daily_message_file_writes_allowed is False
        assert stage.telegram_sending_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False
        assert stage.execution_quality_outputs_allowed is False
        assert stage.report_generation_allowed is False
        assert stage.message_composition_allowed is False
        assert stage.production_pipeline_allowed is False


def test_delivery_plan_exposes_delivery_planning_channels_only():
    plan = build_delivery_plan()

    assert plan.delivery_policy.allowed_delivery_channels == ALLOWED_DELIVERY_CHANNELS
    assert plan.delivery_policy.allowed_delivery_channels == (
        "telegram_planned",
        "operator_review_delivery",
        "dry_run_delivery",
        "manual_delivery_review",
    )
    assert plan.delivery_policy.delivery_planning_only is True
    assert plan.delivery_policy.requires_upstream_message_or_report_artifact is True
    assert plan.delivery_policy.credentials_allowed is False
    assert plan.delivery_policy.network_calls_allowed is False
    assert plan.delivery_policy.final_outcomes_allowed is False

    blocked = (
        set(BLOCKED_EXECUTION_CODES)
        | set(BLOCKED_FINAL_STATE_CODES)
        | set(BLOCKED_BEHAVIOR_CODES)
    )
    assert set(ALLOWED_DELIVERY_CHANNELS).isdisjoint(blocked)


def test_delivery_plan_explicitly_blocks_network_credentials_and_delivery():
    plan = build_delivery_plan()

    assert plan.delivery_policy.blocked_execution_codes == BLOCKED_EXECUTION_CODES
    assert set(BLOCKED_EXECUTION_CODES) == {
        "telegram_send",
        "telegram_api_call",
        "telegram_bot_post",
        "network_post",
        "network_get",
        "credential_read",
        "production_notification",
        "email_send",
        "write_delivery_artifact",
        "write_reports_daily_telegram_message_txt",
    }
    for stage in plan.stages:
        assert stage.blocked_execution_codes == BLOCKED_EXECUTION_CODES


def test_delivery_plan_explicitly_blocks_investment_semantics():
    plan = build_delivery_plan()

    assert plan.delivery_policy.blocked_final_state_codes == (
        BLOCKED_FINAL_STATE_CODES
    )
    assert plan.delivery_policy.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES
    assert set(BLOCKED_FINAL_STATE_CODES) == {
        "buy",
        "sell",
        "hold",
        "allocate",
        "increase_position",
        "reduce_position",
        "target_price",
        "tradeable",
        "not_tradeable",
        "recommendation",
    }
    assert set(BLOCKED_BEHAVIOR_CODES) == {
        "allocation",
        "conviction",
        "urgency",
        "scoring",
        "target-price",
        "tradeability",
        "recommendation",
    }
    for stage in plan.stages:
        assert stage.blocked_final_state_codes == BLOCKED_FINAL_STATE_CODES
        assert stage.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES


def test_delivery_boundary_import_and_plan_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.delivery.delivery_boundary")
    build_delivery_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_delivery_boundary_does_not_import_legacy_network_or_credential_modules():
    legacy_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name == "requests"
        or module_name.startswith("requests.")
        or module_name == "dotenv"
        or module_name.startswith("dotenv.")
    }

    importlib.import_module("market_scanner.delivery.delivery_boundary")
    build_delivery_plan()

    legacy_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name == "requests"
        or module_name.startswith("requests.")
        or module_name == "dotenv"
        or module_name.startswith("dotenv.")
    }

    assert legacy_modules_after == legacy_modules_before


def test_delivery_boundary_source_has_no_network_credential_or_telegram_execution():
    source = Path(delivery_boundary.__file__).read_text(encoding="utf-8")

    assert "requests" not in source
    assert "load_dotenv" not in source
    assert "getenv" not in source
    assert "TELEGRAM_BOT_TOKEN" not in source
    assert "TELEGRAM_CHAT_ID" not in source
    assert ".post(" not in source
    assert ".get(" not in source


def test_delivery_plan_contains_no_investment_behavior_outside_blocked_policy():
    output_text = " ".join(_flatten_values(build_delivery_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_delivery_files_are_not_expanded_to_import_delivery_boundary():
    legacy_sources = (
        (ARCHIVED_LEGACY_RUNTIME_DIR / "run_scan.py").read_text(encoding="utf-8"),
        (ARCHIVED_LEGACY_RUNTIME_DIR / "run_full_pipeline.py").read_text(
            encoding="utf-8"
        ),
        Path("scripts/reporting/send_telegram.py").read_text(encoding="utf-8"),
        Path("scripts/telegram/process_telegram_commands.py").read_text(
            encoding="utf-8"
        ),
        Path("scripts/reporting/build_reporting_layer.py").read_text(encoding="utf-8"),
        Path("scripts/reporting/build_telegram_summary.py").read_text(encoding="utf-8"),
        Path("src/market_scanner/reporting/telegram_renderer.py").read_text(
            encoding="utf-8"
        ),
    )

    assert not (Path("scripts") / "run_scan.py").exists()
    assert not (Path("scripts") / "run_full_pipeline.py").exists()

    for source in legacy_sources:
        assert "market_scanner.delivery.delivery_boundary" not in source
        assert "market_scanner.delivery.delivery_contracts" not in source
        assert "from market_scanner.delivery import" not in source
        assert "market_scanner.app" not in source
