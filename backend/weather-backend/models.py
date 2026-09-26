"""Pydantic response and persistence models for the weather API."""

from datetime import datetime
from typing_extensions import Annotated

from pydantic import BaseModel, BeforeValidator

from utils import oid_to_str


class SuccessResponse(BaseModel):
    """Response returned after a write operation succeeds."""

    message: str
    timestamp: datetime = datetime.now()


class HealthInfo(BaseModel):
    """Status of the API and its MongoDB connection."""

    api_status: str
    db_status: str
    exception: str | None


class SeriesEntry(BaseModel):
    """Metadata describing a measured or derived weather series."""

    name: str
    unit: str
    shorts: list[str]
    disabled: bool


class WeatherEntry(BaseModel):
    """Raw weather value as stored in MongoDB."""

    series: Annotated[str, BeforeValidator(oid_to_str)]
    date: datetime
    value: float


class MeasurementEntry(BaseModel):
    """Weather value enriched with the series name and unit."""

    name: str
    unit: str
    date: datetime
    value: float
