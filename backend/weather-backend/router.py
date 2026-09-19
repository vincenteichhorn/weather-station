from fastapi import APIRouter, Request
from endpoints import health, series, weather

router = APIRouter()


@router.get("/")
def read_health(request: Request):
    """Endpoint to check the health of the API."""
    db = request.app.mongodb
    return health.get_health(db)


@router.get("/series/")
def read_series(request: Request):
    """Endpoint to retrieve a list of series."""
    db = request.app.mongodb
    return series.get_series(db)


@router.get("/weather/")
def read_weather(request: Request):
    db = request.app.mongodb
    """Endpoint to retrieve current weather."""
    return weather.get_latest(db)
