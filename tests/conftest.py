from __future__ import annotations

from pathlib import Path


_TEST_ROOT = Path(__file__).parent

_HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS = (
    "core/test_decision_engine.py",
)

collect_ignore = [
    str(_TEST_ROOT / blocker)
    for blocker in _HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS
]
