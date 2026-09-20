import os
import requests


def get_latest_weather():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/weather/")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch latest weather data: {response.status_code} - {response.text}")


def get_last_week_weather():
    api_url = os.getenv("WEATHER_API_URL")
    if not api_url:
        raise ValueError("WEATHER_API_URL is not set in the environment variables.")
    response = requests.get(f"{api_url}/weather/report/")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch last week's weather data: {response.status_code} - {response.text}")
