"""HTTP routes for the weather station API."""

from datetime import datetime

from fastapi import APIRouter, Request
from endpoints import health, series, weather

from models import MeasurementEntry, SeriesEntry, SuccessResponse, HealthInfo

router = APIRouter()


@router.get("/", response_model=HealthInfo)
async def read_health(request: Request):
    """Return the health of the API and its database connection."""
    db = request.app.mongodb
    return await health.get_health(db)


@router.get("/series/", response_model=list[SeriesEntry])
async def read_series(request: Request):
    """Return all persisted measurement series."""
    db = request.app.mongodb
    return await series.get_series(db)


@router.get("/series/derived", response_model=list[SeriesEntry])
async def read_derived_series():
    """Return all virtual series calculated from stored measurements."""
    return await series.get_derived_series()


@router.get("/weather/", response_model=list[MeasurementEntry])
async def read_weather(request: Request):
    """Return the latest value for every available series."""
    db = request.app.mongodb
    return await weather.get_latest(db)


@router.get("/weather/latest", response_model=list[MeasurementEntry])
async def read_latest_weather(request: Request):
    """Return the latest value for every available series."""
    db = request.app.mongodb
    return await weather.get_latest(db)


@router.get("/weather/report", response_model=dict[str, list[MeasurementEntry]])
async def read_weather_report(request: Request, start_date: datetime = None, end_date: datetime = None):
    """Return minimum, maximum, and average values for each series."""
    db = request.app.mongodb
    return await weather.get_report(db, start_date, end_date)


@router.get("/weather/{series_short}", response_model=list[MeasurementEntry])
async def read_weather_by_series(request: Request, series_short: str, start_date: datetime = None, end_date: datetime = None):
    """Return measurements for one series and optional date range."""
    db = request.app.mongodb
    return await weather.get_series_measurements(db, series_short, start_date, end_date)


@router.post("/weather/add", response_model=SuccessResponse)
async def add_weather_entry(request: Request):
    """Store the submitted station form data as weather measurements."""
    db = request.app.mongodb
    form_data = await request.form()
    station_log = dict(form_data)
    await weather.add_weather_entry(db, station_log)
    return {
        "message": "Weather entry added successfully.",
        "timestamp": datetime.now(),
    }
