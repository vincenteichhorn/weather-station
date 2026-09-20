from datetime import datetime
from typing_extensions import Annotated

from pydantic import BaseModel, BeforeValidator

from utils import oid_to_str


class SuccessResponse(BaseModel):
    """Pydantic model for a successful response."""

    message: str
    timestamp: datetime = datetime.now()


class HealthInfo(BaseModel):
    """Pydantic model for health information."""

    api_status: str
    db_status: str
    exception: str | None


class SeriesEntry(BaseModel):
    """Pydantic model for a series entry."""

    name: str
    unit: str
    shorts: list[str]
    disabled: bool


class WeatherEntry(BaseModel):
    """Pydantic model for a weather entry."""

    series: Annotated[str, BeforeValidator(oid_to_str)]
    date: datetime
    value: float
