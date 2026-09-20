"""Small client for the weather station HTTP API."""

from __future__ import annotations

import json
import os
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class WeatherApiError(RuntimeError):
    """Raised when the weather API cannot provide a valid response."""


class WeatherApi:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or os.getenv("WEATHER_API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, str] | None = None) -> object:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(f"{self.base_url}/{path.lstrip('/')}{query}", headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise WeatherApiError(f"Weather API request failed: {error}") from error

    def get_series(self) -> list[dict]:
        response = self._get("series/")
        derived_response = self._get("series/derived")
        if not isinstance(response, list) or not isinstance(derived_response, list):
            raise WeatherApiError("The series response has an unexpected format")
        return response + derived_response

    def get_latest(self) -> list[dict]:
        response = self._get("weather/latest")
        if not isinstance(response, list):
            raise WeatherApiError("The latest weather response has an unexpected format")
        return response

    def get_measurements(self, series_short: str, start_date: date, end_date: date) -> list[dict]:
        response = self._get(
            f"weather/{series_short}",
            {"start_date": f"{start_date.isoformat()}T00:00:00", "end_date": f"{end_date.isoformat()}T23:59:59"},
        )
        if not isinstance(response, list):
            raise WeatherApiError("The measurements response has an unexpected format")
        return response
