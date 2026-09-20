from datetime import datetime, timedelta

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from models import MeasurementEntry, SeriesEntry, WeatherEntry


async def get_latest(db: AsyncIOMotorDatabase) -> list[MeasurementEntry]:
    """Retrieve the latest weather data from the weather collection."""
    series: dict[ObjectId, SeriesEntry] = {}
    for series_entry in await db["series"].find().to_list(length=None):
        series[series_entry["_id"]] = SeriesEntry(**series_entry)

    latest_weather: list[MeasurementEntry] = []
    raw_measurements: dict[str, float] = {}
    for series_id, series_entry in series.items():
        latest_entry = await db["weather"].find_one({"series": series_id}, sort=[("date", -1)], projection={"_id": False})
        measurement_entry = MeasurementEntry(name=series_entry.name, unit=series_entry.unit, **latest_entry) if latest_entry else None
        if latest_entry:
            latest_weather.append(measurement_entry)
        if "temp" in series_entry.shorts:
            raw_measurements["temp"] = float(latest_entry["value"]) if latest_entry else None
        if "hum" in series_entry.shorts:
            raw_measurements["hum"] = float(latest_entry["value"]) if latest_entry else None
        if "bar" in series_entry.shorts:
            raw_measurements["bar"] = float(latest_entry["value"]) if latest_entry else None

    if "temp" in raw_measurements and "hum" in raw_measurements:
        temp = raw_measurements["temp"]
        hum = raw_measurements["hum"]
        dew_point = temp - ((100 - hum) / 5)
        latest_weather.append(MeasurementEntry(name="Taupunkt", unit="°C", date=datetime.now(), value=dew_point))

    if "temp" in raw_measurements and "hum" in raw_measurements:
        temp = raw_measurements["temp"]
        hum = raw_measurements["hum"]
        temp_kelvin = temp + 273.15
        vapor_pressure = 6.112 * (10 ** ((7.5 * temp) / (237.7 + temp))) * (hum / 100)
        absolute_humidity = (vapor_pressure * 100) / (461.5 * temp_kelvin) * 1000
        latest_weather.append(MeasurementEntry(name="Absolute Feuchtigkeit", unit="g/m³", date=datetime.now(), value=absolute_humidity))

    if "temp" in raw_measurements and "hum" in raw_measurements:
        temp = raw_measurements["temp"]
        hum = raw_measurements["hum"]
        heat_index = (
            -8.784695
            + 1.61139411 * temp
            + 2.338549 * hum
            - 0.14611605 * temp * hum
            - 0.012308094 * (temp**2)
            - 0.016424828 * (hum**2)
            + 0.002211732 * (temp**2) * hum
            + 0.00072546 * temp * (hum**2)
            - 0.000003582 * (temp**2) * (hum**2)
        )
        latest_weather.append(MeasurementEntry(name="Gefühlte Temperatur", unit="°C", date=datetime.now(), value=heat_index))

    return latest_weather


async def get_report(db: AsyncIOMotorDatabase, start_date: datetime = None, end_date: datetime = None) -> list[MeasurementEntry]:
    """Retrieve weather data within a specified date range from the weather collection.
    Build a list of MeasurementEntry objects for each series reporting min, max, and average values within the date range.
    """
    series: dict[ObjectId, SeriesEntry] = {}
    for series_entry in await db["series"].find().to_list(length=None):
        series[series_entry["_id"]] = SeriesEntry(**series_entry)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()

    report: dict[str, list[MeasurementEntry]] = {}
    for series_id, series_entry in series.items():
        pipeline = [
            {"$match": {"series": series_id, "date": {"$gte": start_date, "$lte": end_date}}},
            {
                "$group": {
                    "_id": None,
                    "min_value": {"$min": "$value"},
                    "max_value": {"$max": "$value"},
                    "avg_value": {"$avg": "$value"},
                }
            },
        ]
        result = await db["weather"].aggregate(pipeline).to_list(length=1)
        if result:
            min_entry = MeasurementEntry(name="Minimum", unit=series_entry.unit, date=start_date, value=result[0]["min_value"])
            max_entry = MeasurementEntry(name="Maximum", unit=series_entry.unit, date=start_date, value=result[0]["max_value"])
            avg_entry = MeasurementEntry(name="Durchschnitt", unit=series_entry.unit, date=start_date, value=result[0]["avg_value"])
            report[series_entry.name] = [min_entry, max_entry, avg_entry]

    return report


async def add_weather_entry(db: AsyncIOMotorDatabase, weather_entry: dict):
    """Add a new weather entry to the weather collection."""
    series: dict[ObjectId, SeriesEntry] = {}
    for series_entry in await db["series"].find().to_list(length=None):
        series[series_entry["_id"]] = SeriesEntry(**series_entry)

    current_time = datetime.now()
    for series_id, series_entry in series.items():
        for short in series_entry.shorts:
            if short in weather_entry:
                new_entry: WeatherEntry = {"series": series_id, "date": current_time, "value": weather_entry[short]}
                weather_collection = db["weather"]
                await weather_collection.insert_one(new_entry)
