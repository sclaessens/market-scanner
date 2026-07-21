from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from market_engine.data import primary_source_metric_derivation as derivation
from market_engine.data import operator_source_approval as direct_approval
from market_engine.data.validated_fundamental_metric_sourcing import _load_and_validate_operator_import


def _catalog() -> dict[str, object]:
    return json.loads(
        Path("config/market_engine/data10_fundamental_metric_formula_catalog.json").read_text(encoding="utf-8")
    )


def _fact(
    fact_id: str,
    concept: str,
    value: float,
    *,
    ticker: str = "AAA",
    framework: str = "us_gaap",
    period_type: str = "duration",
    period_start: str | None = "2026-04-01",
    period_end: str = "2026-06-30",
    fiscal_period: str = "Q2",
    unit: str = "USD",
    currency: str = "USD",
    scale: int = 0,
) -> dict[str, object]:
    return {
        "fact_id": fact_id,
        "ticker": ticker,
        "instrument_id": f"equity:{ticker.lower()}",
        "company_identity": f"{ticker} Corporation",
        "accounting_framework": framework,
        "canonical_concept": concept,
        "raw_source_concept": f"{framework}:{concept}",
        "value": value,
        "unit": unit,
        "currency": currency,
        "scale": scale,
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "fiscal_year": 2026,
        "fiscal_period": fiscal_period,
        "source_name": "official-primary-source",
        "source_reference": f"https://primary.example/{ticker}/filing",
        "source_document_checksum": "a" * 64,
        "source_publication_date": "2026-07-01",
        "observed_at": "2026-07-01T09:00:00Z",
        "acquired_at": "2026-07-01T10:00:00Z",
        "parser_version": "fixture-parser-v1",
        "source_approval_reference": "direct-source-approval-1",
        "canonical_mapping_approval_reference": "mapping-approval-1",
    }


def _request(
    formula_id: str,
    *,
    ticker: str = "AAA",
    numerator: list[str] | None = None,
    denominator: list[str] | None = None,
    components: list[str] | None = None,
    required_components: list[str] | None = None,
) -> dict[str, object]:
    return {
        "request_id": f"{ticker.lower()}-{formula_id}",
        "ticker": ticker,
        "canonical_metric": formula_id,
        "formula_id": formula_id,
        "formula_version": "2.0.0",
        "fiscal_year": 2026,
        "fiscal_period": "Q2",
        "numerator_fact_ids": numerator or [],
        "denominator_fact_ids": denominator or [],
        "component_fact_ids": components or [],
        "required_component_concepts": required_components or [],
        "applicability": {
            "status": "applicable",
            "approval_reference": "applicability-approval-1",
        },
    }


def _package(
    facts: list[dict[str, object]],
    requests: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": derivation.FACT_PACKAGE_SCHEMA_VERSION,
        "package_id": "generic-primary-facts-2026-q2",
        "derivation_timestamp": "2026-07-19T18:00:00Z",
        "derivation_approval_reference": "derivation-approval-1",
        "facts": facts,
        "derivation_requests": requests,
    }


def _margin_package(*, ticker: str = "AAA", framework: str = "us_gaap") -> dict[str, object]:
    return _package(
        [
            _fact("revenue", "revenue", 100, ticker=ticker, framework=framework),
            _fact("gross-profit", "gross_profit", 40, ticker=ticker, framework=framework),
        ],
        [_request("gross_margin", ticker=ticker, numerator=["gross-profit"], denominator=["revenue"])],
    )


def _derive(package: dict[str, object]):
    return derivation.derive_primary_source_metrics(package, _catalog())


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("framework", "ticker"), [("us_gaap", "AAA"), ("ifrs", "BBB")])
def test_generic_duration_formula_supports_us_gaap_ifrs_and_second_ticker(
    framework: str, ticker: str
) -> None:
    package, validation = _derive(_margin_package(ticker=ticker, framework=framework))

    assert validation["validation_status"] == "passed"
    assert package is not None
    result = package["derivations"][0]
    assert result["ticker"] == ticker
    assert result["calculation_result"] == 0.4
    assert result["evidence_type"] == "derived"
    assert result["formula"]["canonical_expression"] == "gross_profit / revenue"


def test_operating_margin_uses_explicit_numerator_and_denominator() -> None:
    package = _package(
        [_fact("revenue", "revenue", 250), _fact("operating", "operating_income", 50)],
        [_request("operating_margin", numerator=["operating"], denominator=["revenue"])],
    )
    derived, _ = _derive(package)

    assert derived["derivations"][0]["calculation_result"] == 0.2
    assert derived["derivations"][0]["numerator_fact_ids"] == ["operating"]
    assert derived["derivations"][0]["denominator_fact_ids"] == ["revenue"]


def test_explicit_debt_components_are_summed_before_equity_division() -> None:
    instant = {"period_type": "instant", "period_start": None, "period_end": "2026-06-30"}
    package = _package(
        [
            _fact("commercial", "commercial_paper", 10, **instant),
            _fact("current", "current_term_debt", 15, **instant),
            _fact("noncurrent", "noncurrent_term_debt", 25, **instant),
            _fact("equity", "total_equity", 100, **instant),
        ],
        [
            _request(
                "debt_to_equity",
                denominator=["equity"],
                components=["commercial", "current", "noncurrent"],
                required_components=["commercial_paper", "current_term_debt", "noncurrent_term_debt"],
            )
        ],
    )
    derived, _ = _derive(package)

    result = derived["derivations"][0]
    assert result["calculation_result"] == 0.5
    assert result["normalized_numerator"] == "50"
    assert result["canonical_metric"] == "debt_to_equity"


@pytest.mark.parametrize(
    ("formula_id", "numerator_concept"),
    [
        ("gross_margin", "operating_income"),
        ("operating_margin", "gross_profit"),
    ],
)
def test_margin_formula_rejects_substituted_numerator_concepts(
    formula_id: str, numerator_concept: str
) -> None:
    package = _package(
        [_fact("numerator", numerator_concept, 40), _fact("revenue", "revenue", 100)],
        [_request(formula_id, numerator=["numerator"], denominator=["revenue"])],
    )

    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "FORMULA_NUMERATOR_CONCEPT_MISMATCH" in validation["reason_codes"]


def test_formula_rejects_wrong_denominator_and_role_swap() -> None:
    wrong_denominator = _package(
        [_fact("gross", "gross_profit", 40), _fact("equity", "total_equity", 100)],
        [_request("gross_margin", numerator=["gross"], denominator=["equity"])],
    )
    swapped = _package(
        [_fact("gross", "gross_profit", 40), _fact("revenue", "revenue", 100)],
        [_request("gross_margin", numerator=["revenue"], denominator=["gross"])],
    )

    _, wrong_validation = _derive(wrong_denominator)
    _, swapped_validation = _derive(swapped)

    assert "FORMULA_DENOMINATOR_CONCEPT_MISMATCH" in wrong_validation["reason_codes"]
    assert "FORMULA_NUMERATOR_CONCEPT_MISMATCH" in swapped_validation["reason_codes"]
    assert "FORMULA_DENOMINATOR_CONCEPT_MISMATCH" in swapped_validation["reason_codes"]


def test_debt_formula_rejects_non_debt_concept_total_liabilities_and_overlap() -> None:
    instant = {"period_type": "instant", "period_start": None, "period_end": "2026-06-30"}
    unsupported = _package(
        [_fact("operating", "operating_income", 10, **instant), _fact("equity", "total_equity", 100, **instant)],
        [
            _request(
                "debt_to_equity",
                denominator=["equity"],
                components=["operating"],
                required_components=["operating_income"],
            )
        ],
    )
    total_liabilities = copy.deepcopy(unsupported)
    total_liabilities["facts"][0]["canonical_concept"] = "total_liabilities"
    overlap = _package(
        [
            _fact("total-debt", "total_interest_bearing_debt", 50, **instant),
            _fact("commercial", "commercial_paper", 10, **instant),
            _fact("equity", "total_equity", 100, **instant),
        ],
        [
            _request(
                "debt_to_equity",
                denominator=["equity"],
                components=["total-debt", "commercial"],
                required_components=["total_interest_bearing_debt", "commercial_paper"],
            )
        ],
    )

    _, unsupported_validation = _derive(unsupported)
    rejected, liabilities_validation = _derive(total_liabilities)
    _, overlap_validation = _derive(overlap)

    assert "FORMULA_COMPONENT_CONCEPT_NOT_ALLOWED" in unsupported_validation["reason_codes"]
    assert rejected is None
    assert "CANONICAL_CONCEPT_UNSUPPORTED" in liabilities_validation["reason_codes"]
    assert "FORMULA_COMPONENT_OVERLAP" in overlap_validation["reason_codes"]


def test_formula_rejects_unexpected_extra_operand() -> None:
    package = _margin_package()
    package["facts"].append(_fact("extra", "commercial_paper", 1))
    package["derivation_requests"][0]["component_fact_ids"] = ["extra"]

    _, validation = _derive(package)

    assert "FORMULA_OPERAND_SET_INVALID" in validation["reason_codes"]


def test_expression_text_cannot_mask_or_change_machine_readable_semantics() -> None:
    wrong_operands = _catalog()
    wrong_operands["formulas"][0]["operand_contract"]["numerator"]["required_canonical_concepts"] = [
        "operating_income"
    ]
    forged_expression = _catalog()
    forged_expression["formulas"][0]["canonical_expression"] = "operating_income / revenue"

    wrong_package, wrong_validation = derivation.derive_primary_source_metrics(
        _margin_package(), wrong_operands
    )
    forged_package, forged_validation = derivation.derive_primary_source_metrics(
        _margin_package(), forged_expression
    )

    assert wrong_package is None
    assert "FORMULA_SEMANTIC_CONTRACT_MISMATCH" in wrong_validation["reason_codes"]
    assert forged_package is None
    assert "FORMULA_CANONICAL_EXPRESSION_MISMATCH" in forged_validation["reason_codes"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda facts: facts[1].update(period_start=None, period_type="instant"), "PERIOD_TYPE_MISMATCH"),
        (lambda facts: facts[1].update(period_start="2026-01-01"), "PERIOD_START_MISMATCH"),
        (lambda facts: facts[1].update(period_end="2026-06-29"), "PERIOD_END_MISMATCH"),
        (lambda facts: facts[1].update(fiscal_period="Q1"), "FISCAL_PERIOD_MISMATCH"),
        (lambda facts: facts[1].update(currency="EUR"), "CURRENCY_MISMATCH"),
        (lambda facts: facts[1].update(unit="EUR"), "UNIT_MISMATCH"),
        (lambda facts: facts[1].update(scale=3), "SCALE_MISMATCH"),
    ],
)
def test_period_unit_currency_and_scale_mismatches_block(mutation, reason: str) -> None:
    package = _margin_package()
    mutation(package["facts"])
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert reason in validation["reason_codes"]


@pytest.mark.parametrize("revenue", [0, -1])
def test_non_positive_revenue_blocks_margin(revenue: float) -> None:
    package = _margin_package()
    package["facts"][0]["value"] = revenue
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "REVENUE_NOT_POSITIVE" in validation["reason_codes"]


@pytest.mark.parametrize("equity", [0, -1])
def test_non_positive_equity_blocks_debt_ratio(equity: float) -> None:
    instant = {"period_type": "instant", "period_start": None}
    package = _package(
        [_fact("debt", "commercial_paper", 10, **instant), _fact("equity", "total_equity", equity, **instant)],
        [_request("debt_to_equity", denominator=["equity"], components=["debt"], required_components=["commercial_paper"])],
    )
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "DENOMINATOR_NOT_POSITIVE" in validation["reason_codes"]


def test_negative_debt_component_blocks() -> None:
    instant = {"period_type": "instant", "period_start": None}
    package = _package(
        [_fact("debt", "commercial_paper", -1, **instant), _fact("equity", "total_equity", 100, **instant)],
        [_request("debt_to_equity", denominator=["equity"], components=["debt"], required_components=["commercial_paper"])],
    )
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "NEGATIVE_DEBT_COMPONENT" in validation["reason_codes"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("numerator_fact_ids", "NUMERATOR_MISSING"),
        ("denominator_fact_ids", "DENOMINATOR_MISSING"),
    ],
)
def test_missing_numerator_or_denominator_blocks(field: str, reason: str) -> None:
    package = _margin_package()
    package["derivation_requests"][0][field] = ["not-present"]
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert reason in validation["reason_codes"]


def test_missing_debt_component_is_not_treated_as_zero() -> None:
    instant = {"period_type": "instant", "period_start": None}
    package = _package(
        [_fact("commercial", "commercial_paper", 10, **instant), _fact("equity", "total_equity", 100, **instant)],
        [
            _request(
                "debt_to_equity",
                denominator=["equity"],
                components=["commercial"],
                required_components=["commercial_paper", "current_term_debt"],
            )
        ],
    )
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "DEBT_COMPONENT_MISSING" in validation["reason_codes"]


@pytest.mark.parametrize(("different", "reason"), [(False, "DUPLICATE_FACT"), (True, "CONFLICTING_FACT")])
def test_duplicate_and_conflicting_facts_fail_closed(different: bool, reason: str) -> None:
    package = _margin_package()
    duplicate = copy.deepcopy(package["facts"][0])
    duplicate["fact_id"] = "second-revenue"
    if different:
        duplicate["value"] = 101
    package["facts"].append(duplicate)
    derived, validation = _derive(package)

    assert derived is None
    assert reason in validation["reason_codes"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fact_values_are_rejected(value: float) -> None:
    package = _margin_package()
    package["facts"][0]["value"] = value
    derived, validation = _derive(package)

    assert derived is None
    assert "FACT_VALUE_NOT_FINITE" in validation["reason_codes"]


def test_unknown_formula_and_framework_fail_controlled() -> None:
    unknown_formula = _margin_package()
    unknown_formula["derivation_requests"][0]["formula_id"] = "unknown"
    derived, validation = _derive(unknown_formula)
    assert derived["derivations"][0]["reason_codes"] == ["FORMULA_ID_UNSUPPORTED"]

    unknown_framework = _margin_package(framework="unknown")
    derived, validation = _derive(unknown_framework)
    assert derived is None
    assert "ACCOUNTING_FRAMEWORK_UNSUPPORTED" in validation["reason_codes"]


def test_not_applicable_requires_explicit_approved_applicability_evidence() -> None:
    package = _margin_package()
    package["derivation_requests"][0]["applicability"] = {
        "status": "not_applicable",
        "approval_reference": "sector-applicability-review-1",
    }
    derived, validation = _derive(package)

    assert derived["derivations"][0]["status"] == "blocked"
    assert "FORMULA_NOT_APPLICABLE" in validation["reason_codes"]


def test_identical_input_and_reversed_fact_order_produce_identical_output() -> None:
    first = _margin_package()
    second = copy.deepcopy(first)
    second["facts"] = list(reversed(second["facts"]))

    assert _derive(first) == _derive(second)


def test_ticker_name_does_not_change_formula_selection_or_result() -> None:
    first, _ = _derive(_margin_package(ticker="AAA"))
    second, _ = _derive(_margin_package(ticker="CCC"))

    first_result = first["derivations"][0]
    second_result = second["derivations"][0]
    assert first_result["formula"] == second_result["formula"]
    assert first_result["calculation_result"] == second_result["calculation_result"]


def test_malformed_request_and_json_return_controlled_failures(tmp_path: Path) -> None:
    package = _margin_package()
    package["derivation_requests"] = [["invalid"]]
    derived, validation = _derive(package)
    assert derived["derivations"][0]["reason_codes"] == ["DERIVATION_REQUEST_INVALID"]

    malformed = tmp_path / "bad.json"
    malformed.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(derivation.PrimarySourceMetricDerivationError, match="strict JSON input is invalid"):
        derivation.load_strict_json(malformed)


def _direct_package(ticker: str = "AAA") -> dict[str, object]:
    return {
        "schema_version": "market-engine-data07-operator-fundamental-metrics-v1",
        "package_schema_version": direct_approval.INPUT_SCHEMA_VERSION,
        "package_id": "direct-package",
        "records": [
            {
                "ticker": ticker,
                "instrument_id": f"equity:{ticker.lower()}",
                "company_name": f"{ticker} Corporation",
                "provider_symbol": ticker,
                "provider": "official-source",
                "source_date": "2026-07-01",
                "reporting_period": "2026-Q2",
                "period_type": "quarter",
                "period_start": "2026-04-01",
                "period_end": "2026-06-30",
                "fiscal_year": 2026,
                "fiscal_period": "Q2",
                "source_reference": "https://primary.example/filing",
                "parser_version": "direct-parser-v1",
                "snapshot_id": "direct-package",
                "acquired_at": "2026-07-01T10:00:00Z",
                "observed_at": "2026-07-01T09:00:00Z",
                "metrics": {
                    "revenue_growth_yoy": {
                        "value": 10.0,
                        "unit": "percent",
                        "reporting_period": "2026-Q2",
                        "raw_source_field": "reported_revenue_growth",
                    }
                },
            }
        ],
    }


def test_direct_and_derived_evidence_remain_distinguishable_in_data07_v2(tmp_path: Path) -> None:
    derived, _ = _derive(_margin_package())
    governed = derivation.build_data07_governed_package(
        _direct_package(),
        derived,
        package_id="governed-package",
        direct_approval_reference="direct-approval",
        direct_approval_checksum="b" * 64,
        direct_package_checksum="c" * 64,
    )
    path = _write(tmp_path / "governed.json", governed)
    records, validation = _load_and_validate_operator_import(
        path,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": {"instrument_id": "equity:aaa"}},
        as_of=date(2026, 7, 19),
        allowed_tickers={"AAA"},
    )

    assert validation["validation_status"] == "passed"
    lineage = {row["canonical_metric"]: row for row in records[0]["metric_lineage"]}
    assert lineage["revenue_growth_yoy"]["evidence_type"] == "direct"
    assert lineage["revenue_growth_yoy"]["direct_lineage"] is not None
    assert lineage["gross_margin"]["evidence_type"] == "derived"
    assert lineage["gross_margin"]["derivation_lineage"]["calculation_checksum"]


@pytest.mark.parametrize("field", ["instrument_id", "company_name", "fiscal_year", "fiscal_period"])
def test_direct_and_derived_merge_rejects_identity_mismatch(field: str) -> None:
    derived, _ = _derive(_margin_package())
    direct = _direct_package()
    direct["records"][0][field] = "mismatch"

    with pytest.raises(
        derivation.PrimarySourceMetricDerivationError,
        match="instrument/company/fiscal identity must match",
    ):
        derivation.build_data07_governed_package(
            direct,
            derived,
            package_id="governed-package",
            direct_approval_reference="direct-approval",
            direct_approval_checksum="b" * 64,
            direct_package_checksum="c" * 64,
        )


def _direct_approval_fixture(tmp_path: Path, package_path: Path) -> tuple[Path, Path]:
    input_path = _write(
        tmp_path / "direct-input.json",
        {"schema_version": direct_approval.INPUT_SCHEMA_VERSION, "package_id": "direct-package"},
    )
    report_path = _write(
        tmp_path / "direct-report.json",
        {
            "schema_version": direct_approval.REPORT_SCHEMA_VERSION,
            "validator_version": direct_approval.VALIDATOR_VERSION,
            "package_id": "direct-package",
            "status": "accepted",
            "downstream_consumability": "structurally_valid_for_explicit_source_approval_review",
            "input_sha256": _sha(input_path),
        },
    )
    source_path = tmp_path / "source.html"
    source_path.write_text("official source", encoding="utf-8")
    decision = {
        "schema_version": direct_approval.DECISION_SCHEMA_VERSION,
        "decision_id": "direct-approval",
        "decision": "approved",
        "scope": direct_approval.APPROVED_SCOPE,
        "approved_tickers": ["AAA"],
        "reviewer_roles": list(direct_approval.REQUIRED_REVIEWER_ROLES),
        "package_id": "direct-package",
        "artifact_bindings": {
            "input_path": input_path.as_posix(),
            "input_sha256": _sha(input_path),
            "package_sha256": _sha(package_path),
            "validation_report_path": report_path.as_posix(),
            "validation_report_sha256": _sha(report_path),
        },
        "source_documents": [{"relative_path": source_path.name, "sha256": _sha(source_path)}],
        "reviews": {name: {"status": "approved"} for name in direct_approval.REQUIRED_REVIEW_DIMENSIONS},
        "approved_metrics": ["revenue_growth_yoy"],
        "explicitly_missing_metrics": [
            "debt_to_equity",
            "eps_growth_yoy",
            "gross_margin",
            "operating_margin",
        ],
    }
    return _write(tmp_path / "direct-approval.json", decision), source_path


def _approval_fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, object]]:
    fact_package = _margin_package()
    catalog = _catalog()
    derived, _ = derivation.derive_primary_source_metrics(fact_package, catalog)
    direct = _direct_package()
    paths = {
        "fact_package": _write(tmp_path / "facts.json", fact_package),
        "formula_catalog": _write(tmp_path / "catalog.json", catalog),
        "derived_package": _write(tmp_path / "derived.json", derived),
        "derivation_validation": _write(
            tmp_path / "derivation-validation.json",
            derivation.derive_primary_source_metrics(fact_package, catalog)[1],
        ),
        "direct_package": _write(tmp_path / "direct.json", direct),
    }
    paths["direct_approval"], source = _direct_approval_fixture(tmp_path, paths["direct_package"])
    governed = derivation.build_data07_governed_package(
        direct,
        derived,
        package_id="governed-package",
        direct_approval_reference="direct-approval",
        direct_approval_checksum=_sha(paths["direct_approval"]),
        direct_package_checksum=_sha(paths["direct_package"]),
    )
    paths["governed_package"] = _write(tmp_path / "governed.json", governed)
    decision = {
        "schema_version": derivation.DERIVATION_APPROVAL_DECISION_SCHEMA_VERSION,
        "decision_id": "derivation-approval-1",
        "decision": "approved",
        "scope": derivation.APPROVED_SCOPE,
        "governed_package_id": "governed-package",
        "reviewer_roles": list(derivation.REQUIRED_REVIEWER_ROLES),
        "reviews": {name: {"status": "approved"} for name in derivation.REQUIRED_REVIEW_DIMENSIONS},
        "artifact_bindings": {
            **{f"{name}_path": path.as_posix() for name, path in paths.items()},
            **{f"{name}_sha256": _sha(path) for name, path in paths.items()},
        },
        "source_documents": [{"relative_path": source.name, "sha256": _sha(source)}],
        "approved_tickers": ["AAA"],
        "approved_derived_metrics": ["gross_margin"],
        "explicitly_blocked_metrics": [],
        "approved_calculation_checksums": [derived["derivations"][0]["calculation_checksum"]],
    }
    return paths, decision


def test_checksum_bound_derivation_approval_accepts_and_tamper_blocks(tmp_path: Path) -> None:
    paths, decision = _approval_fixture(tmp_path)
    decision_path = _write(tmp_path / "decision.json", decision)

    accepted = derivation.validate_derivation_approval_decision(
        decision_path, paths["governed_package"], source_document_root=tmp_path
    )
    assert accepted["validation_status"] == "approved"

    decision["artifact_bindings"]["derived_package_sha256"] = "0" * 64
    _write(decision_path, decision)
    blocked = derivation.validate_derivation_approval_decision(
        decision_path, paths["governed_package"], source_document_root=tmp_path
    )
    assert blocked["validation_status"] == "blocked"
    assert "DERIVATION_ARTIFACT_CHECKSUM_MISMATCH" in blocked["reason_codes"]


def test_approval_candidate_binds_replay_artifacts_without_granting_approval(tmp_path: Path) -> None:
    paths, _ = _approval_fixture(tmp_path)
    candidate_path = tmp_path / "approval-candidate.json"

    candidate = derivation.persist_derivation_approval_candidate(
        decision_id="derivation-approval-1",
        fact_package_path=paths["fact_package"],
        formula_catalog_path=paths["formula_catalog"],
        derived_package_path=paths["derived_package"],
        derivation_validation_path=paths["derivation_validation"],
        direct_package_path=paths["direct_package"],
        direct_approval_path=paths["direct_approval"],
        governed_package_path=paths["governed_package"],
        source_document_template_path=paths["direct_approval"],
        output_path=candidate_path,
    )

    assert candidate["decision"] == "pending"
    assert {review["status"] for review in candidate["reviews"].values()} == {"pending"}
    assert candidate["artifact_bindings"]["derivation_validation_sha256"] == _sha(
        paths["derivation_validation"]
    )
    result = derivation.validate_derivation_approval_decision(
        candidate_path, paths["governed_package"], source_document_root=tmp_path
    )
    assert result["validation_status"] == "blocked"
    assert "DERIVATION_APPROVAL_BLOCKED" in result["reason_codes"]


def test_derivation_approval_rejects_a_different_governed_package_path(tmp_path: Path) -> None:
    paths, decision = _approval_fixture(tmp_path)
    decision_path = _write(tmp_path / "decision.json", decision)
    copied_package = tmp_path / "copied-governed.json"
    copied_package.write_bytes(paths["governed_package"].read_bytes())

    blocked = derivation.validate_derivation_approval_decision(
        decision_path, copied_package, source_document_root=tmp_path
    )

    assert blocked["validation_status"] == "blocked"
    assert "DERIVATION_GOVERNED_PACKAGE_PATH_MISMATCH" in blocked["reason_codes"]


def _rebind(decision: dict[str, object], stem: str, path: Path) -> None:
    decision["artifact_bindings"][f"{stem}_sha256"] = _sha(path)


def _update_direct_approval_for_package(paths: dict[str, Path], decision: dict[str, object]) -> None:
    approval = json.loads(paths["direct_approval"].read_text(encoding="utf-8"))
    package = json.loads(paths["direct_package"].read_text(encoding="utf-8"))
    approval["artifact_bindings"]["package_sha256"] = _sha(paths["direct_package"])
    approval["approved_metrics"] = sorted(package["records"][0]["metrics"])
    approval["explicitly_missing_metrics"] = sorted(
        {"revenue_growth_yoy", "eps_growth_yoy", "gross_margin", "operating_margin", "debt_to_equity"}
        - set(approval["approved_metrics"])
    )
    _write(paths["direct_approval"], approval)
    _rebind(decision, "direct_package", paths["direct_package"])
    _rebind(decision, "direct_approval", paths["direct_approval"])


def _tamper_reconciled_fixture(
    case: str, paths: dict[str, Path], decision: dict[str, object]
) -> None:
    if case in {"derived_content", "derived_governed_value", "calculation_checksum", "canonical_replay"}:
        derived = json.loads(paths["derived_package"].read_text(encoding="utf-8"))
        row = derived["derivations"][0]
        if case == "derived_content":
            row["limitations"].append("tampered but internally rechecksummed")
            calculation = dict(row)
            calculation.pop("calculation_checksum")
            row["calculation_checksum"] = derivation._canonical_checksum(calculation)
            decision["approved_calculation_checksums"] = [row["calculation_checksum"]]
        elif case == "derived_governed_value":
            row["calculation_result"] = 0.41
            calculation = dict(row)
            calculation.pop("calculation_checksum")
            row["calculation_checksum"] = derivation._canonical_checksum(calculation)
            decision["approved_calculation_checksums"] = [row["calculation_checksum"]]
        elif case == "calculation_checksum":
            row["calculation_checksum"] = "f" * 64
            decision["approved_calculation_checksums"] = ["f" * 64]
        else:
            derived["boundary"] = "tampered canonical replay boundary"
        _write(paths["derived_package"], derived)
        _rebind(decision, "derived_package", paths["derived_package"])
    elif case in {"governed_content", "governed_calculation_checksum", "identity_mismatch"}:
        governed = json.loads(paths["governed_package"].read_text(encoding="utf-8"))
        if case == "governed_content":
            governed["records"][0]["metrics"]["gross_margin"]["value"] = 0.41
        elif case == "governed_calculation_checksum":
            governed["records"][0]["metrics"]["gross_margin"]["derivation_lineage"][
                "calculation_checksum"
            ] = "e" * 64
        else:
            governed["records"][0]["company_name"] = "Different Corporation"
        _write(paths["governed_package"], governed)
        _rebind(decision, "governed_package", paths["governed_package"])
    elif case == "formula_catalog":
        catalog = json.loads(paths["formula_catalog"].read_text(encoding="utf-8"))
        catalog["catalog_id"] = "different-catalog-id"
        _write(paths["formula_catalog"], catalog)
        _rebind(decision, "formula_catalog", paths["formula_catalog"])
    elif case == "fact_package":
        facts = json.loads(paths["fact_package"].read_text(encoding="utf-8"))
        facts["facts"][0]["value"] = 101
        _write(paths["fact_package"], facts)
        _rebind(decision, "fact_package", paths["fact_package"])
    elif case == "derivation_validation":
        validation = json.loads(paths["derivation_validation"].read_text(encoding="utf-8"))
        validation["boundary"] = "tampered validation boundary"
        _write(paths["derivation_validation"], validation)
        _rebind(decision, "derivation_validation", paths["derivation_validation"])
    elif case == "direct_package_without_approval":
        direct = json.loads(paths["direct_package"].read_text(encoding="utf-8"))
        direct["records"][0]["metrics"]["revenue_growth_yoy"]["value"] = 11.0
        _write(paths["direct_package"], direct)
        _rebind(decision, "direct_package", paths["direct_package"])
    elif case == "invalid_direct_approval":
        approval = json.loads(paths["direct_approval"].read_text(encoding="utf-8"))
        approval["decision"] = "rejected"
        _write(paths["direct_approval"], approval)
        _rebind(decision, "direct_approval", paths["direct_approval"])
    elif case == "approval_package_id":
        decision["governed_package_id"] = "different-governed-package"
    elif case == "fiscal_mismatch":
        direct = json.loads(paths["direct_package"].read_text(encoding="utf-8"))
        direct["records"][0]["fiscal_period"] = "Q3"
        _write(paths["direct_package"], direct)
        _update_direct_approval_for_package(paths, decision)
    elif case == "duplicate_metric":
        direct = json.loads(paths["direct_package"].read_text(encoding="utf-8"))
        direct["records"][0]["metrics"]["gross_margin"] = {
            "value": 40.0,
            "unit": "percent",
            "reporting_period": "2026-Q2",
            "raw_source_field": "reported_gross_margin",
        }
        _write(paths["direct_package"], direct)
        _update_direct_approval_for_package(paths, decision)
    else:
        raise AssertionError(f"unknown tamper case: {case}")


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("derived_content", "DERIVED_PACKAGE_REPLAY_MISMATCH"),
        ("governed_content", "GOVERNED_PACKAGE_REPLAY_MISMATCH"),
        ("derived_governed_value", "DERIVED_PACKAGE_REPLAY_MISMATCH"),
        ("governed_calculation_checksum", "GOVERNED_PACKAGE_REPLAY_MISMATCH"),
        ("formula_catalog", "DERIVED_FORMULA_CATALOG_ID_MISMATCH"),
        ("fact_package", "DERIVED_FACT_PACKAGE_CHECKSUM_MISMATCH"),
        ("derivation_validation", "DERIVATION_VALIDATION_REPLAY_MISMATCH"),
        ("direct_package_without_approval", "DIRECT_SOURCE_APPROVAL_REVALIDATION_FAILED"),
        ("invalid_direct_approval", "DIRECT_SOURCE_APPROVAL_REVALIDATION_FAILED"),
        ("approval_package_id", "DERIVATION_APPROVAL_PACKAGE_ID_MISMATCH"),
        ("identity_mismatch", "GOVERNED_PACKAGE_REPLAY_MISMATCH"),
        ("fiscal_mismatch", "GOVERNED_PACKAGE_RECONSTRUCTION_FAILED"),
        ("duplicate_metric", "GOVERNED_PACKAGE_RECONSTRUCTION_FAILED"),
        ("calculation_checksum", "CALCULATION_CHECKSUM_REPLAY_MISMATCH"),
        ("canonical_replay", "DERIVED_PACKAGE_REPLAY_MISMATCH"),
    ],
)
def test_cross_artifact_tampering_with_refreshed_file_checksums_is_blocked(
    tmp_path: Path, case: str, reason: str
) -> None:
    paths, decision = _approval_fixture(tmp_path)
    _tamper_reconciled_fixture(case, paths, decision)
    decision_path = _write(tmp_path / "decision.json", decision)

    result = derivation.validate_derivation_approval_decision(
        decision_path, paths["governed_package"], source_document_root=tmp_path
    )

    assert result["validation_status"] == "blocked"
    assert reason in result["reason_codes"]


def test_malformed_bound_artifact_is_controlled_and_blocks_reconciliation(tmp_path: Path) -> None:
    paths, decision = _approval_fixture(tmp_path)
    paths["fact_package"].write_text("{", encoding="utf-8")
    _rebind(decision, "fact_package", paths["fact_package"])
    decision_path = _write(tmp_path / "decision.json", decision)

    result = derivation.validate_derivation_approval_decision(
        decision_path, paths["governed_package"], source_document_root=tmp_path
    )

    assert result["validation_status"] == "blocked"
    assert "BOUND_ARTIFACT_MALFORMED" in result["reason_codes"]


def test_failed_derivation_and_failed_approval_never_reach_data07(tmp_path: Path) -> None:
    paths, decision = _approval_fixture(tmp_path)
    decision_path = _write(tmp_path / "decision.json", decision)
    calls = {"data07": 0}

    def runner(**_kwargs):
        calls["data07"] += 1
        raise AssertionError("DATA07, raw snapshot, DATA06, and RUN31 must not execute")

    failed_facts = json.loads(paths["fact_package"].read_text(encoding="utf-8"))
    failed_facts["facts"][0]["value"] = 0
    _write(paths["fact_package"], failed_facts)
    failed = derivation.execute_approved_derivation_import(
        fact_package_path=paths["fact_package"],
        formula_catalog_path=paths["formula_catalog"],
        derived_package_path=paths["derived_package"],
        governed_package_path=paths["governed_package"],
        approval_decision_path=decision_path,
        source_document_root=tmp_path,
        data07_runner=runner,
        data07_kwargs={},
    )
    assert failed["status"] == "blocked"
    assert failed["data07_executed"] is False
    assert calls == {"data07": 0}

    paths, decision = _approval_fixture(tmp_path / "approval")
    decision["decision"] = "blocked"
    decision_path = _write(tmp_path / "approval" / "decision.json", decision)
    failed = derivation.execute_approved_derivation_import(
        fact_package_path=paths["fact_package"],
        formula_catalog_path=paths["formula_catalog"],
        derived_package_path=paths["derived_package"],
        governed_package_path=paths["governed_package"],
        approval_decision_path=decision_path,
        source_document_root=tmp_path / "approval",
        data07_runner=runner,
        data07_kwargs={},
    )
    assert failed["status"] == "blocked"
    assert failed["data07_executed"] is False
    assert calls == {"data07": 0}
