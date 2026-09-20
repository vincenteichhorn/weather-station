from datetime import datetime

from fastapi import APIRouter, Request
from endpoints import health, series, weather

from models import MeasurementEntry, SeriesEntry, SuccessResponse, WeatherEntry, HealthInfo

router = APIRouter()


@router.get("/", response_model=HealthInfo)
async def read_health(request: Request):
    """Endpoint to check the health of the API."""
    db = request.app.mongodb
    return await health.get_health(db)


@router.get("/series/", response_model=list[SeriesEntry])
async def read_series(request: Request):
    db = request.app.mongodb
    return await series.get_series(db)


@router.get("/weather/", response_model=list[MeasurementEntry])
async def read_weather(request: Request):
    db = request.app.mongodb
    return await weather.get_latest(db)


@router.get("/weather/latest", response_model=list[MeasurementEntry])
async def read_latest_weather(request: Request):
    db = request.app.mongodb
    return await weather.get_latest(db)


@router.get("/weather/report", response_model=dict[str, list[MeasurementEntry]])
async def read_weather_report(request: Request, start_date: datetime = None, end_date: datetime = None):
    db = request.app.mongodb
    return await weather.get_report(db, start_date, end_date)


@router.get("/weather/{series_short}", response_model=list[MeasurementEntry])
async def read_weather_by_series(request: Request, series_short: str, start_date: datetime = None, end_date: datetime = None):
    db = request.app.mongodb
    return await weather.get_series_measurements(db, series_short, start_date, end_date)


@router.post("/weather/add", response_model=SuccessResponse)
async def add_weather_entry(request: Request):
    db = request.app.mongodb
    form_data = await request.form()
    station_log = dict(form_data)
    await weather.add_weather_entry(db, station_log)
    return {
        "message": "Weather entry added successfully.",
        "timestamp": datetime.now(),
    }
