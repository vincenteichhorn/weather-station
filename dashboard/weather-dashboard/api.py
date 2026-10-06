"""Small client for the weather station HTTP API."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
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

    def get_measurements(
        self,
        series_short: str,
        start_date: date | datetime,
        end_date: date | datetime,
        density: str = "raw",
    ) -> list[dict]:
        def format_date(value: date | datetime, end: bool = False) -> str:
            if isinstance(value, datetime):
                return value.isoformat()
            return f"{value.isoformat()}T{'23:59:59' if end else '00:00:00'}"

        response = self._get(
            f"weather/{series_short}",
            {
                "start_date": format_date(start_date),
                "end_date": format_date(end_date, end=True),
                "density": density,
            },
        )
        if not isinstance(response, list):
            raise WeatherApiError("The measurements response has an unexpected format")
        return response

    def get_forecast(self) -> dict[str, list[dict]]:
        response = self._get("weather/forecast")
        if not isinstance(response, dict):
            raise WeatherApiError("The forecast response has an unexpected format")
        return response

    def get_historical_forecast(self, reference_time: str | None = None) -> dict[str, list[dict]]:
        params = {"reference_time": reference_time} if reference_time else None
        response = self._get("weather/forecast/historical", params)
        if not isinstance(response, dict):
            raise WeatherApiError("The historical forecast response has an unexpected format")
        return response
