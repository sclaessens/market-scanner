from __future__ import annotations

import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_engine.source_intake.provider_boundary import (
    InvalidTickerError,
    ProviderSourceResponse,
    ProviderUnavailableError,
    SourceProvider,
    UnsupportedTickerError,
)


SEC_COMPANYFACTS_PROVIDER_NAME = "SEC_COMPANYFACTS"
SEC_COMPANYFACTS_REQUIRED_FIELDS = (
    "revenue",
    "net_income",
    "operating_cash_flow",
    "capital_expenditures",
)

SMOKE_TICKER_CIKS = {
    "NVDA": "0001045810",
    "AMD": "0000002488",
    "META": "0001326801",
    "COST": "0000909832",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "AVGO": "0001730168",
}

SEC_FACT_ALIASES = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
}

JsonFetcher = Callable[[str], dict[str, Any] | None]


class SecCompanyFactsHttpError(ProviderUnavailableError):
    """Raised for controlled SEC CompanyFacts HTTP failures."""


class SecCompanyFactsJsonParseError(ProviderUnavailableError):
    """Raised for controlled SEC CompanyFacts JSON parse failures."""


class SecCompanyFactsNetworkError(ProviderUnavailableError):
    """Raised for controlled SEC CompanyFacts network failures."""


class SecCompanyFactsProvider(SourceProvider):
    def __init__(
        self,
        ticker_to_cik: dict[str, str] | None = None,
        fetch_json: JsonFetcher | None = None,
    ) -> None:
        self._ticker_to_cik = {
            ticker.upper(): _normalize_cik(cik)
            for ticker, cik in (ticker_to_cik or SMOKE_TICKER_CIKS).items()
        }
        self._fetch_json = fetch_json or _fetch_json_from_sec

    @property
    def name(self) -> str:
        return SEC_COMPANYFACTS_PROVIDER_NAME

    def fetch_source(self, ticker: str) -> ProviderSourceResponse | None:
        normalized_ticker = _normalize_ticker(ticker)
        cik = self._ticker_to_cik.get(normalized_ticker)
        if cik is None:
            raise UnsupportedTickerError(f"{normalized_ticker} has no bounded smoke CIK mapping")

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            payload = self._fetch_json(url)
        except HTTPError as error:
            raise SecCompanyFactsHttpError(
                f"SEC CompanyFacts HTTP error for {normalized_ticker}: status={error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise SecCompanyFactsNetworkError(
                f"SEC CompanyFacts network error for {normalized_ticker}: {error}"
            ) from error
        except ValueError as error:
            raise SecCompanyFactsJsonParseError(
                f"SEC CompanyFacts JSON parse error for {normalized_ticker}"
            ) from error

        if not payload:
            return None

        fields = _extract_required_fields(payload)
        return ProviderSourceResponse(
            ticker=normalized_ticker,
            fields=fields,
            raw_evidence=payload,
            raw_evidence_summary=f"SEC CompanyFacts CIK{cik}",
        )


def _normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized or any(character.isspace() for character in normalized):
        raise InvalidTickerError(f"{ticker!r} is not a valid bounded smoke ticker")
    return normalized


def _normalize_cik(cik: str) -> str:
    digits = "".join(character for character in str(cik) if character.isdigit())
    if not digits:
        raise ValueError("CIK mapping must contain digits")
    return digits.zfill(10)


def _fetch_json_from_sec(url: str) -> dict[str, Any] | None:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "market-engine-source-intake-smoke contact@example.com",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_required_fields(payload: dict[str, Any]) -> dict[str, Any]:
    us_gaap_facts = payload.get("facts", {}).get("us-gaap", {})
    return {
        field_name: _latest_numeric_fact(us_gaap_facts, aliases)
        for field_name, aliases in SEC_FACT_ALIASES.items()
    }


def _latest_numeric_fact(us_gaap_facts: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    for alias in aliases:
        fact = us_gaap_facts.get(alias)
        units = fact.get("units", {}) if isinstance(fact, dict) else {}
        values = units.get("USD", [])
        numeric_values = [
            value
            for value in values
            if isinstance(value, dict)
            and value.get("val") is not None
            and isinstance(value.get("end"), str)
        ]
        if numeric_values:
            latest = max(numeric_values, key=lambda value: value["end"])
            return latest.get("val")
    return None
