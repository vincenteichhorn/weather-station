from fastapi import APIRouter, Request
from endpoints import health, series, weather

from models import SeriesEntry, SuccessResponse, WeatherEntry, HealthInfo

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


@router.get("/weather/", response_model=WeatherEntry)
async def read_weather(request: Request):
    db = request.app.mongodb
    return await weather.get_latest(db)


@router.post("/weather/add", response_model=SuccessResponse)
async def add_weather_entry(request: Request):
    db = request.app.mongodb
    form_data = await request.form()
    station_log = dict(form_data)
    print(f"Received weather entry: {station_log}")
    return await weather.add_weather_entry(db, station_log)
