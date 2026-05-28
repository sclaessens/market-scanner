from __future__ import annotations

from scripts.core import build_fundamental_analysis as core_analysis
from scripts.core import build_fundamental_layer as core_quality
from scripts.core import build_fundamental_metrics as core_metrics
from scripts.core import build_fundamentals_history_intake as core_history
from scripts.fundamentals import build_analysis
from scripts.fundamentals import build_history_intake
from scripts.fundamentals import build_metrics
from scripts.fundamentals import build_quality
from scripts import run_scan


def test_new_fundamentals_import_paths_expose_expected_builders() -> None:
    assert callable(build_history_intake.validate_fundamentals_history)
    assert callable(build_metrics.build_fundamental_metrics)
    assert callable(build_quality.build_fundamental_layer)
    assert callable(build_analysis.build_fundamental_analysis)
    assert callable(build_history_intake.main)
    assert callable(build_metrics.main)
    assert callable(build_quality.main)
    assert callable(build_analysis.main)


def test_legacy_core_import_paths_remain_compatible() -> None:
    assert core_history.validate_fundamentals_history is build_history_intake.validate_fundamentals_history
    assert core_metrics.build_fundamental_metrics is build_metrics.build_fundamental_metrics
    assert core_quality.build_fundamental_layer is build_quality.build_fundamental_layer
    assert core_analysis.build_fundamental_analysis is build_analysis.build_fundamental_analysis
    assert core_history.main is build_history_intake.main
    assert core_metrics.main is build_metrics.main
    assert core_quality.main is build_quality.main
    assert core_analysis.main is build_analysis.main


def test_run_scan_uses_new_fundamentals_namespace() -> None:
    assert run_scan.validate_fundamentals_history is build_history_intake.validate_fundamentals_history
    assert run_scan.build_fundamental_metrics is build_metrics.build_fundamental_metrics
    assert run_scan.build_fundamental_layer is build_quality.build_fundamental_layer
    assert run_scan.build_fundamental_analysis is build_analysis.build_fundamental_analysis
