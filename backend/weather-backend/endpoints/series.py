"""Operations for measured and derived weather series."""

from motor.motor_asyncio import AsyncIOMotorDatabase

from derived_metrics import DERIVED_METRICS
from models import SeriesEntry


async def get_series(db: AsyncIOMotorDatabase) -> list:
    """Retrieve persisted series metadata from MongoDB.

    Args:
        db: Database containing the ``series`` collection.

    Returns:
        Series documents without their internal MongoDB identifiers.
    """
    series_collection = db["series"]
    series_list = []
    async for series in series_collection.find(projection={"_id": False}):
        series_list.append(series)
    return series_list


async def get_derived_series() -> list[SeriesEntry]:
    """Return derived metrics as virtual, non-disabled series entries."""
    return [SeriesEntry(name=metric.name, unit=metric.unit, shorts=list(metric.shorts), disabled=False) for metric in DERIVED_METRICS]
