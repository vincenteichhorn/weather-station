import os
from datetime import datetime

import requests


def get_latest_weather():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/weather/")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch latest weather data: {response.status_code} - {response.text}"
        )


def get_last_week_weather():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/weather/report/")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch last week's weather data: {response.status_code} - {response.text}"
        )


def get_weather_series():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/series/")
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch weather series: {response.status_code} - {response.text}")


def get_series_measurements(series_short, start_date, end_date):
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(
        f"{api_url}/weather/{series_short}",
        params={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
    )
    if response.status_code == 200:
        return response.json()
    raise Exception(
        f"Failed to fetch series '{series_short}': {response.status_code} - {response.text}"
    )


def get_forecast():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/weather/forecast")
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch forecast: {response.status_code} - {response.text}")


def get_sunrise_sunset():
    location = os.getenv("LOCATION")
    if not location:
        raise ValueError("LOCATION is not set in the environment variables.")
    try:
        latitude, longitude = (float(coordinate.strip()) for coordinate in location.split(","))
    except ValueError as error:
        raise ValueError(
            "LOCATION must contain latitude and longitude separated by a comma."
        ) from error

    response = requests.get(
        "https://api.sunrise-sunset.org/v2",
        params={"lat": latitude, "lng": longitude, "formatted": 0},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("results", payload)

    def to_local_time(value):
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        )

    return {
        "sunrise": to_local_time(result["sunrise"]),
        "sunset": to_local_time(result["sunset"]),
        "civil_twilight_begin": to_local_time(result["civil_twilight_begin"]),
        "civil_twilight_end": to_local_time(result["civil_twilight_end"]),
    }
