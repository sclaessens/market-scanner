from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Sequence


CACHED_SOURCE_COVERAGE_CONTRACT_VERSION = (
    "market-engine-supported-universe-cached-source-coverage-v1"
)


class CachedSourceCoverageError(ValueError):
    """Raised when generic coverage input cannot be classified safely."""


class SourceFamily(StrEnum):
    COMPANY_PROFILE = "company_profile"
    FUNDAMENTAL_FACTS = "fundamental_facts"
    PRICE_HISTORY = "price_history"
    SETUP_DETECTION_INPUT = "setup_detection_input"
    PORTFOLIO_CONTEXT = "portfolio_context"
    RECOMMENDATION_REVIEW_INPUT = "recommendation_review_input"
    DECISION_ENGINE_HANDOFF_INPUT = "decision_engine_handoff_input"


class CoverageStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    UNPROVENANCED = "unprovenanced"
    INVALID_MANIFEST = "invalid_manifest"
    MISSING_SNAPSHOT = "missing_snapshot"
    DESCRIPTIVE_ONLY = "descriptive_only"
    BLOCKED = "blocked"
    NOT_CONSUMABLE = "not_consumable"
    NOT_REQUIRED = "not_required"


class ReadinessStatus(StrEnum):
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    DESCRIPTIVE_ONLY = "descriptive_only"
    ANALYSIS_READY = "analysis_ready"
    RECOMMENDATION_REVIEW_READY = "recommendation_review_ready"
    ACTIONABLE = "actionable"
    DE_READY = "de_ready"
    BLOCKED = "blocked"


class TargetCapability(StrEnum):
    DESCRIPTIVE_ANALYSIS = "descriptive_analysis"
    ANALYSIS = "analysis"
    RECOMMENDATION_REVIEW = "recommendation_review"
    ACTIONABLE_REVIEW = "actionable_review"
    DECISION_ENGINE_HANDOFF = "decision_engine_handoff"


class RequirementMode(StrEnum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"


class BlockerCode(StrEnum):
    UNSUPPORTED_UNIVERSE = "unsupported_universe"
    UNSUPPORTED_SOURCE_FAMILY = "unsupported_source_family"
    MISSING_CACHED_SOURCE_SNAPSHOT = "missing_cached_source_snapshot"
    INVALID_MANIFEST = "invalid_manifest"
    MISSING_PROVENANCE = "missing_provenance"
    STALE_SOURCE = "stale_source"
    SOURCE_NOT_CONSUMABLE = "source_not_consumable"
    SOURCE_FAMILY_INCOMPLETE = "source_family_incomplete"
    COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE = (
        "company_profile_only_context_non_actionable"
    )
    MISSING_FUNDAMENTAL_EVIDENCE = "missing_fundamental_evidence"
    MISSING_SETUP_OR_PRICE_CONTEXT = "missing_setup_or_price_context"
    BLOCKED_MISSING_PORTFOLIO_CONTEXT = "blocked_missing_portfolio_context"
    RECOMMENDATION_REVIEW_BLOCKED = "recommendation_review_blocked"
    ACTIONABLE_CONTRACT_NOT_APPROVED = "actionable_contract_not_approved"
    DECISION_ENGINE_HANDOFF_BLOCKED = "decision_engine_handoff_blocked"


@dataclass(frozen=True)
class SourceFamilyRequirement:
    source_family: SourceFamily
    mode: RequirementMode = RequirementMode.REQUIRED
    require_manifest: bool = True
    require_provenance: bool = True
    require_freshness: bool = True
    require_consumability: bool = True
    require_completeness: bool = True


@dataclass(frozen=True)
class SourceFamilyEvidence:
    source_family: SourceFamily
    support_status: CoverageStatus = CoverageStatus.SUPPORTED
    availability_status: CoverageStatus = CoverageStatus.AVAILABLE
    manifest_status: CoverageStatus = CoverageStatus.ACCEPTED
    provenance_status: CoverageStatus = CoverageStatus.ACCEPTED
    freshness_status: CoverageStatus = CoverageStatus.ACCEPTED
    consumability_status: CoverageStatus = CoverageStatus.ACCEPTED
    completeness_status: CoverageStatus = CoverageStatus.ACCEPTED
    evidence_reference: str | None = None


@dataclass(frozen=True)
class CachedSourceCoverageInput:
    ticker: str
    universe_supported: bool
    target_capability: TargetCapability
    requirements: tuple[SourceFamilyRequirement, ...] = ()
    source_evidence: tuple[SourceFamilyEvidence, ...] = ()
    additional_blockers: tuple[BlockerCode, ...] = ()


@dataclass(frozen=True)
class CoverageBlocker:
    code: BlockerCode
    pipeline_stage: TargetCapability
    source_family: SourceFamily | None = None


@dataclass(frozen=True)
class SourceFamilyCoverageResult:
    source_family: SourceFamily
    required: bool
    coverage_status: CoverageStatus
    requirement_satisfied: bool
    blockers: tuple[CoverageBlocker, ...]
    evidence_reference: str | None


@dataclass(frozen=True)
class CachedSourceCoverageClassification:
    contract_version: str
    ticker: str
    target_capability: TargetCapability
    coverage_status: CoverageStatus
    readiness_status: ReadinessStatus
    source_family_results: tuple[SourceFamilyCoverageResult, ...]
    blockers: tuple[CoverageBlocker, ...]
    actionable: bool
    de_ready: bool
    recommendation_review_allowed: bool
    decision_engine_handoff_allowed: bool
    notes: tuple[str, ...]
    classification_boundary: str


@dataclass(frozen=True)
class ClassificationCount:
    status: str
    count: int


@dataclass(frozen=True)
class CachedSourceCoverageBatchClassification:
    contract_version: str
    classifications: tuple[CachedSourceCoverageClassification, ...]
    coverage_counts: tuple[ClassificationCount, ...]
    readiness_counts: tuple[ClassificationCount, ...]
    actionable_count: int
    de_ready_count: int
    classification_boundary: str


_CAPABILITY_REQUIRED_FAMILIES: dict[
    TargetCapability,
    tuple[SourceFamily, ...],
] = {
    TargetCapability.DESCRIPTIVE_ANALYSIS: (SourceFamily.COMPANY_PROFILE,),
    TargetCapability.ANALYSIS: (SourceFamily.FUNDAMENTAL_FACTS,),
    TargetCapability.RECOMMENDATION_REVIEW: (
        SourceFamily.FUNDAMENTAL_FACTS,
        SourceFamily.PRICE_HISTORY,
        SourceFamily.SETUP_DETECTION_INPUT,
        SourceFamily.RECOMMENDATION_REVIEW_INPUT,
    ),
    TargetCapability.ACTIONABLE_REVIEW: (
        SourceFamily.FUNDAMENTAL_FACTS,
        SourceFamily.PRICE_HISTORY,
        SourceFamily.SETUP_DETECTION_INPUT,
        SourceFamily.RECOMMENDATION_REVIEW_INPUT,
    ),
    TargetCapability.DECISION_ENGINE_HANDOFF: (
        SourceFamily.FUNDAMENTAL_FACTS,
        SourceFamily.PRICE_HISTORY,
        SourceFamily.SETUP_DETECTION_INPUT,
        SourceFamily.PORTFOLIO_CONTEXT,
        SourceFamily.RECOMMENDATION_REVIEW_INPUT,
        SourceFamily.DECISION_ENGINE_HANDOFF_INPUT,
    ),
}

_HARD_BLOCKING_COVERAGE_STATUSES = frozenset(
    {
        CoverageStatus.UNSUPPORTED,
        CoverageStatus.REJECTED,
        CoverageStatus.INVALID_MANIFEST,
        CoverageStatus.UNPROVENANCED,
        CoverageStatus.STALE,
        CoverageStatus.NOT_CONSUMABLE,
    }
)


def classify_cached_source_coverage(
    coverage_input: CachedSourceCoverageInput,
) -> CachedSourceCoverageClassification:
    _validate_input(coverage_input)
    requirements = _requirements_for_input(coverage_input)
    evidence_by_family = {
        evidence.source_family: evidence
        for evidence in coverage_input.source_evidence
    }
    results = tuple(
        _classify_source_family(
            requirement=requirement,
            evidence=evidence_by_family.get(requirement.source_family),
            target_capability=coverage_input.target_capability,
        )
        for requirement in requirements
    )
    unrequired_results = tuple(
        SourceFamilyCoverageResult(
            source_family=family,
            required=False,
            coverage_status=CoverageStatus.NOT_REQUIRED,
            requirement_satisfied=True,
            blockers=(),
            evidence_reference=evidence_by_family[family].evidence_reference,
        )
        for family in sorted(
            set(evidence_by_family) - {item.source_family for item in requirements},
            key=lambda item: item.value,
        )
    )
    all_results = results + unrequired_results
    blockers = _classification_blockers(
        coverage_input=coverage_input,
        results=results,
    )
    readiness_status = _readiness_status(
        coverage_input=coverage_input,
        required_results=results,
        blockers=blockers,
    )
    coverage_status = _aggregate_coverage_status(
        universe_supported=coverage_input.universe_supported,
        required_results=results,
    )
    if readiness_status is ReadinessStatus.DESCRIPTIVE_ONLY:
        coverage_status = CoverageStatus.DESCRIPTIVE_ONLY
    recommendation_review_allowed = readiness_status in {
        ReadinessStatus.RECOMMENDATION_REVIEW_READY,
        ReadinessStatus.ACTIONABLE,
        ReadinessStatus.DE_READY,
    }
    actionable = readiness_status in {
        ReadinessStatus.ACTIONABLE,
        ReadinessStatus.DE_READY,
    }
    de_ready = readiness_status is ReadinessStatus.DE_READY
    return CachedSourceCoverageClassification(
        contract_version=CACHED_SOURCE_COVERAGE_CONTRACT_VERSION,
        ticker=coverage_input.ticker,
        target_capability=coverage_input.target_capability,
        coverage_status=coverage_status,
        readiness_status=readiness_status,
        source_family_results=all_results,
        blockers=blockers,
        actionable=actionable,
        de_ready=de_ready,
        recommendation_review_allowed=recommendation_review_allowed,
        decision_engine_handoff_allowed=de_ready,
        notes=(
            "Coverage classification is deterministic and ticker-independent.",
            f"Target capability: {coverage_input.target_capability.value}.",
            f"Coverage status: {coverage_status.value}.",
            f"Readiness status: {readiness_status.value}.",
        ),
        classification_boundary=(
            "Coverage and readiness classification only; no recommendation, "
            "allocation, order, execution, provider, portfolio, delivery, or "
            "Decision Engine authority is invoked."
        ),
    )


def classify_cached_source_coverage_batch(
    coverage_inputs: Sequence[CachedSourceCoverageInput],
) -> CachedSourceCoverageBatchClassification:
    inputs = tuple(coverage_inputs)
    if not inputs:
        raise CachedSourceCoverageError("coverage batch must not be empty")
    tickers = [item.ticker for item in inputs]
    if len(tickers) != len(set(tickers)):
        raise CachedSourceCoverageError("coverage batch ticker values must be unique")
    classifications = tuple(
        classify_cached_source_coverage(item)
        for item in inputs
    )
    return CachedSourceCoverageBatchClassification(
        contract_version=CACHED_SOURCE_COVERAGE_CONTRACT_VERSION,
        classifications=classifications,
        coverage_counts=_counts(
            item.coverage_status.value for item in classifications
        ),
        readiness_counts=_counts(
            item.readiness_status.value for item in classifications
        ),
        actionable_count=sum(item.actionable for item in classifications),
        de_ready_count=sum(item.de_ready for item in classifications),
        classification_boundary=(
            "Batch coverage summary only; ticker identity is preserved as data "
            "and does not select classification behavior."
        ),
    )


def to_plain_dict(
    value: CachedSourceCoverageClassification
    | CachedSourceCoverageBatchClassification,
) -> dict[str, Any]:
    return asdict(value)


def _validate_input(coverage_input: CachedSourceCoverageInput) -> None:
    if not coverage_input.ticker or coverage_input.ticker != coverage_input.ticker.strip():
        raise CachedSourceCoverageError(
            "coverage input ticker must be non-empty and without padding"
        )
    requirement_families = [
        item.source_family for item in coverage_input.requirements
    ]
    if len(requirement_families) != len(set(requirement_families)):
        raise CachedSourceCoverageError(
            "coverage input requirements must use unique source families"
        )
    evidence_families = [
        item.source_family for item in coverage_input.source_evidence
    ]
    if len(evidence_families) != len(set(evidence_families)):
        raise CachedSourceCoverageError(
            "coverage input evidence must use unique source families"
        )
    for evidence in coverage_input.source_evidence:
        _validate_evidence(evidence)
def _validate_evidence(evidence: SourceFamilyEvidence) -> None:
    allowed = {
        "support_status": {
            CoverageStatus.SUPPORTED,
            CoverageStatus.UNSUPPORTED,
            CoverageStatus.NOT_REQUIRED,
        },
        "availability_status": {
            CoverageStatus.AVAILABLE,
            CoverageStatus.UNAVAILABLE,
            CoverageStatus.MISSING_SNAPSHOT,
            CoverageStatus.NOT_REQUIRED,
        },
        "manifest_status": {
            CoverageStatus.ACCEPTED,
            CoverageStatus.REJECTED,
            CoverageStatus.INVALID_MANIFEST,
            CoverageStatus.NOT_REQUIRED,
        },
        "provenance_status": {
            CoverageStatus.ACCEPTED,
            CoverageStatus.UNPROVENANCED,
            CoverageStatus.NOT_REQUIRED,
        },
        "freshness_status": {
            CoverageStatus.ACCEPTED,
            CoverageStatus.STALE,
            CoverageStatus.NOT_REQUIRED,
        },
        "consumability_status": {
            CoverageStatus.ACCEPTED,
            CoverageStatus.NOT_CONSUMABLE,
            CoverageStatus.NOT_REQUIRED,
        },
        "completeness_status": {
            CoverageStatus.ACCEPTED,
            CoverageStatus.PARTIAL,
            CoverageStatus.NOT_REQUIRED,
        },
    }
    for field_name, allowed_values in allowed.items():
        if getattr(evidence, field_name) not in allowed_values:
            raise CachedSourceCoverageError(
                f"{field_name} has an invalid coverage status"
            )


def _requirements_for_input(
    coverage_input: CachedSourceCoverageInput,
) -> tuple[SourceFamilyRequirement, ...]:
    requirements = {
        item.source_family: item
        for item in coverage_input.requirements
    }
    for family in _CAPABILITY_REQUIRED_FAMILIES[
        coverage_input.target_capability
    ]:
        requirements.setdefault(
            family,
            SourceFamilyRequirement(source_family=family),
        )
    return tuple(
        requirements[family]
        for family in sorted(requirements, key=lambda item: item.value)
    )


def _classify_source_family(
    *,
    requirement: SourceFamilyRequirement,
    evidence: SourceFamilyEvidence | None,
    target_capability: TargetCapability,
) -> SourceFamilyCoverageResult:
    if requirement.mode is RequirementMode.NOT_REQUIRED:
        return SourceFamilyCoverageResult(
            source_family=requirement.source_family,
            required=False,
            coverage_status=CoverageStatus.NOT_REQUIRED,
            requirement_satisfied=True,
            blockers=(),
            evidence_reference=evidence.evidence_reference if evidence else None,
        )
    if evidence is None:
        return _family_failure(
            requirement=requirement,
            target_capability=target_capability,
            status=CoverageStatus.MISSING_SNAPSHOT,
            blocker=BlockerCode.MISSING_CACHED_SOURCE_SNAPSHOT,
        )
    checks = (
        (
            evidence.support_status is not CoverageStatus.SUPPORTED,
            CoverageStatus.UNSUPPORTED,
            BlockerCode.UNSUPPORTED_SOURCE_FAMILY,
        ),
        (
            evidence.availability_status is not CoverageStatus.AVAILABLE,
            CoverageStatus.MISSING_SNAPSHOT,
            BlockerCode.MISSING_CACHED_SOURCE_SNAPSHOT,
        ),
        (
            requirement.require_manifest
            and evidence.manifest_status is CoverageStatus.INVALID_MANIFEST,
            CoverageStatus.INVALID_MANIFEST,
            BlockerCode.INVALID_MANIFEST,
        ),
        (
            requirement.require_manifest
            and evidence.manifest_status is CoverageStatus.REJECTED,
            CoverageStatus.REJECTED,
            BlockerCode.INVALID_MANIFEST,
        ),
        (
            requirement.require_manifest
            and evidence.manifest_status is not CoverageStatus.ACCEPTED,
            CoverageStatus.INVALID_MANIFEST,
            BlockerCode.INVALID_MANIFEST,
        ),
        (
            requirement.require_provenance
            and evidence.provenance_status is not CoverageStatus.ACCEPTED,
            CoverageStatus.UNPROVENANCED,
            BlockerCode.MISSING_PROVENANCE,
        ),
        (
            requirement.require_freshness
            and evidence.freshness_status is not CoverageStatus.ACCEPTED,
            CoverageStatus.STALE,
            BlockerCode.STALE_SOURCE,
        ),
        (
            requirement.require_consumability
            and evidence.consumability_status is not CoverageStatus.ACCEPTED,
            CoverageStatus.NOT_CONSUMABLE,
            BlockerCode.SOURCE_NOT_CONSUMABLE,
        ),
        (
            requirement.require_completeness
            and evidence.completeness_status is not CoverageStatus.ACCEPTED,
            CoverageStatus.PARTIAL,
            BlockerCode.SOURCE_FAMILY_INCOMPLETE,
        ),
    )
    for failed, status, blocker in checks:
        if failed:
            return _family_failure(
                requirement=requirement,
                target_capability=target_capability,
                status=status,
                blocker=blocker,
                evidence_reference=evidence.evidence_reference,
            )
    return SourceFamilyCoverageResult(
        source_family=requirement.source_family,
        required=True,
        coverage_status=CoverageStatus.ACCEPTED,
        requirement_satisfied=True,
        blockers=(),
        evidence_reference=evidence.evidence_reference,
    )


def _family_failure(
    *,
    requirement: SourceFamilyRequirement,
    target_capability: TargetCapability,
    status: CoverageStatus,
    blocker: BlockerCode,
    evidence_reference: str | None = None,
) -> SourceFamilyCoverageResult:
    return SourceFamilyCoverageResult(
        source_family=requirement.source_family,
        required=True,
        coverage_status=status,
        requirement_satisfied=False,
        blockers=(
            CoverageBlocker(
                code=blocker,
                pipeline_stage=target_capability,
                source_family=requirement.source_family,
            ),
        ),
        evidence_reference=evidence_reference,
    )


def _classification_blockers(
    *,
    coverage_input: CachedSourceCoverageInput,
    results: tuple[SourceFamilyCoverageResult, ...],
) -> tuple[CoverageBlocker, ...]:
    blockers: list[CoverageBlocker] = []
    if not coverage_input.universe_supported:
        blockers.append(
            CoverageBlocker(
                code=BlockerCode.UNSUPPORTED_UNIVERSE,
                pipeline_stage=coverage_input.target_capability,
            )
        )
    for result in results:
        blockers.extend(result.blockers)
        if not result.requirement_satisfied:
            capability_blocker = _capability_blocker(result.source_family)
            if capability_blocker is not None:
                blockers.append(
                    CoverageBlocker(
                        code=capability_blocker,
                        pipeline_stage=coverage_input.target_capability,
                        source_family=result.source_family,
                    )
                )
    accepted = {
        result.source_family
        for result in results
        if result.requirement_satisfied
        and result.coverage_status is CoverageStatus.ACCEPTED
    }
    analytical = accepted - {SourceFamily.COMPANY_PROFILE}
    profile_available = any(
        evidence.source_family is SourceFamily.COMPANY_PROFILE
        and _evidence_is_accepted(evidence)
        for evidence in coverage_input.source_evidence
    )
    if profile_available and not analytical:
        blockers.append(
            CoverageBlocker(
                code=BlockerCode.COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE,
                pipeline_stage=coverage_input.target_capability,
                source_family=SourceFamily.COMPANY_PROFILE,
            )
        )
    blockers.extend(
        CoverageBlocker(
            code=code,
            pipeline_stage=coverage_input.target_capability,
        )
        for code in coverage_input.additional_blockers
    )
    source_gates_pass = (
        coverage_input.universe_supported
        and all(result.requirement_satisfied for result in results)
    )
    if (
        source_gates_pass
        and coverage_input.target_capability
        in {
            TargetCapability.ACTIONABLE_REVIEW,
            TargetCapability.DECISION_ENGINE_HANDOFF,
        }
    ):
        blockers.append(
            CoverageBlocker(
                code=BlockerCode.ACTIONABLE_CONTRACT_NOT_APPROVED,
                pipeline_stage=coverage_input.target_capability,
            )
        )
    if (
        source_gates_pass
        and coverage_input.target_capability
        is TargetCapability.DECISION_ENGINE_HANDOFF
    ):
        blockers.append(
            CoverageBlocker(
                code=BlockerCode.DECISION_ENGINE_HANDOFF_BLOCKED,
                pipeline_stage=coverage_input.target_capability,
                source_family=SourceFamily.DECISION_ENGINE_HANDOFF_INPUT,
            )
        )
    return _deduplicated_blockers(blockers)


def _capability_blocker(
    source_family: SourceFamily,
) -> BlockerCode | None:
    if source_family is SourceFamily.FUNDAMENTAL_FACTS:
        return BlockerCode.MISSING_FUNDAMENTAL_EVIDENCE
    if source_family in {
        SourceFamily.PRICE_HISTORY,
        SourceFamily.SETUP_DETECTION_INPUT,
    }:
        return BlockerCode.MISSING_SETUP_OR_PRICE_CONTEXT
    if source_family is SourceFamily.PORTFOLIO_CONTEXT:
        return BlockerCode.BLOCKED_MISSING_PORTFOLIO_CONTEXT
    if source_family is SourceFamily.RECOMMENDATION_REVIEW_INPUT:
        return BlockerCode.RECOMMENDATION_REVIEW_BLOCKED
    if source_family is SourceFamily.DECISION_ENGINE_HANDOFF_INPUT:
        return BlockerCode.DECISION_ENGINE_HANDOFF_BLOCKED
    return None


def _aggregate_coverage_status(
    *,
    universe_supported: bool,
    required_results: tuple[SourceFamilyCoverageResult, ...],
) -> CoverageStatus:
    if not universe_supported:
        return CoverageStatus.UNSUPPORTED
    if not required_results:
        return CoverageStatus.BLOCKED
    statuses = tuple(result.coverage_status for result in required_results)
    if all(result.requirement_satisfied for result in required_results):
        return CoverageStatus.ACCEPTED
    for status in (
        CoverageStatus.INVALID_MANIFEST,
        CoverageStatus.REJECTED,
        CoverageStatus.UNPROVENANCED,
        CoverageStatus.STALE,
        CoverageStatus.NOT_CONSUMABLE,
    ):
        if status in statuses:
            return status
    if any(result.requirement_satisfied for result in required_results):
        return CoverageStatus.PARTIAL
    for status in (
        CoverageStatus.MISSING_SNAPSHOT,
        CoverageStatus.UNSUPPORTED,
        CoverageStatus.PARTIAL,
    ):
        if status in statuses:
            return status
    return CoverageStatus.BLOCKED


def _readiness_status(
    *,
    coverage_input: CachedSourceCoverageInput,
    required_results: tuple[SourceFamilyCoverageResult, ...],
    blockers: tuple[CoverageBlocker, ...],
) -> ReadinessStatus:
    if not coverage_input.universe_supported:
        return ReadinessStatus.BLOCKED
    statuses = {result.coverage_status for result in required_results}
    if statuses & _HARD_BLOCKING_COVERAGE_STATUSES:
        return ReadinessStatus.BLOCKED
    if coverage_input.additional_blockers:
        return ReadinessStatus.BLOCKED
    accepted = {
        result.source_family
        for result in required_results
        if result.requirement_satisfied
        and result.coverage_status is CoverageStatus.ACCEPTED
    }
    analytical = accepted - {SourceFamily.COMPANY_PROFILE}
    profile_available = any(
        evidence.source_family is SourceFamily.COMPANY_PROFILE
        and _evidence_is_accepted(evidence)
        for evidence in coverage_input.source_evidence
    )
    if profile_available and not analytical:
        return ReadinessStatus.DESCRIPTIVE_ONLY
    unsatisfied = [
        result for result in required_results
        if not result.requirement_satisfied
    ]
    if unsatisfied:
        if accepted:
            return ReadinessStatus.PARTIAL
        return ReadinessStatus.UNAVAILABLE
    target = coverage_input.target_capability
    if target is TargetCapability.DESCRIPTIVE_ANALYSIS:
        return ReadinessStatus.DESCRIPTIVE_ONLY
    if target is TargetCapability.ANALYSIS:
        return ReadinessStatus.ANALYSIS_READY
    if target is TargetCapability.RECOMMENDATION_REVIEW:
        return ReadinessStatus.RECOMMENDATION_REVIEW_READY
    if target is TargetCapability.ACTIONABLE_REVIEW:
        return ReadinessStatus.RECOMMENDATION_REVIEW_READY
    return ReadinessStatus.RECOMMENDATION_REVIEW_READY


def _deduplicated_blockers(
    blockers: Sequence[CoverageBlocker],
) -> tuple[CoverageBlocker, ...]:
    result: list[CoverageBlocker] = []
    seen: set[tuple[BlockerCode, TargetCapability, SourceFamily | None]] = set()
    for blocker in blockers:
        key = (
            blocker.code,
            blocker.pipeline_stage,
            blocker.source_family,
        )
        if key not in seen:
            seen.add(key)
            result.append(blocker)
    return tuple(result)


def _evidence_is_accepted(evidence: SourceFamilyEvidence) -> bool:
    return (
        evidence.support_status is CoverageStatus.SUPPORTED
        and evidence.availability_status is CoverageStatus.AVAILABLE
        and evidence.manifest_status is CoverageStatus.ACCEPTED
        and evidence.provenance_status is CoverageStatus.ACCEPTED
        and evidence.freshness_status is CoverageStatus.ACCEPTED
        and evidence.consumability_status is CoverageStatus.ACCEPTED
        and evidence.completeness_status is CoverageStatus.ACCEPTED
    )


def _counts(statuses: Sequence[str]) -> tuple[ClassificationCount, ...]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return tuple(
        ClassificationCount(status=status, count=counts[status])
        for status in sorted(counts)
    )
