from __future__ import annotations

from dataclasses import dataclass, field

from market_engine.source_intake.provider_boundary import (
    InvalidTickerError,
    ProviderSourceResponse,
    ProviderUnavailableError,
    SourceProvider,
    UnsupportedTickerError,
)


@dataclass(frozen=True)
class FakeProviderScenario:
    fields: dict[str, object] | None = None
    raw_evidence: object | None = None
    raw_evidence_summary: str | None = None
    missing_source: bool = False
    unsupported: bool = False
    invalid: bool = False
    provider_error: bool = False


@dataclass
class FakeSourceProvider(SourceProvider):
    scenarios: dict[str, FakeProviderScenario] = field(default_factory=dict)
    name_value: str = "fake-source-provider"

    @property
    def name(self) -> str:
        return self.name_value

    def fetch_source(self, ticker: str) -> ProviderSourceResponse | None:
        scenario = self.scenarios.get(ticker)
        if scenario is None or scenario.missing_source:
            return None
        if scenario.unsupported:
            raise UnsupportedTickerError(f"{ticker} is not supported by {self.name}")
        if scenario.invalid:
            raise InvalidTickerError(f"{ticker} is invalid for {self.name}")
        if scenario.provider_error:
            raise ProviderUnavailableError(f"{self.name} failed while fetching {ticker}")
        return ProviderSourceResponse(
            ticker=ticker,
            fields=dict(scenario.fields or {}),
            raw_evidence=scenario.raw_evidence,
            raw_evidence_summary=scenario.raw_evidence_summary,
        )
