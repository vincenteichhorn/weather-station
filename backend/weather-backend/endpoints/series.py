from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_series(db: AsyncIOMotorDatabase) -> list:
    """Retrieve the series info from the series collection."""
    series_collection = db["series"]
    series_list = []
    async for series in series_collection.find(projection={"_id": False}):
        series_list.append(series)
    return series_list
