from datetime import datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from models import SeriesEntry, WeatherEntry


async def get_latest(db: AsyncIOMotorDatabase) -> dict:
    """Retrieve the latest weather data from the weather collection."""
    weather_collection = db["weather"]
    latest_weather = await weather_collection.find_one(sort=[("date", -1)], projection={"_id": False})
    return latest_weather


async def add_weather_entry(db: AsyncIOMotorDatabase, weather_entry: dict) -> dict:
    """Add a new weather entry to the weather collection."""
    series: dict[ObjectId, SeriesEntry] = {}
    for series_entry in await db["series"].find().to_list(length=None):
        series[series_entry["_id"]] = SeriesEntry(**series_entry)

    current_time = datetime.now()
    print(series)
    for series_id, series_entry in series.items():
        for short in series_entry.shorts:
            if short in weather_entry:
                new_entry: WeatherEntry = {"series": series_id, "date": current_time, "value": weather_entry[short]}
                weather_collection = db["weather_test"]
                await weather_collection.insert_one(new_entry)

    return {
        "message": "Weather entry added successfully.",
        "timestamp": current_time,
    }
