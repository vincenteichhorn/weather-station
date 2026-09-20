"""Database operations for raw and derived weather measurements."""

from datetime import datetime, timedelta

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from derived_metrics import DERIVED_METRICS, DERIVED_METRICS_BY_SHORT, DerivedMetric
from models import MeasurementEntry, SeriesEntry, WeatherEntry


def _get_raw_measurements(series: dict[ObjectId, SeriesEntry], measurements: dict[ObjectId, dict]) -> dict[str, float]:
    """Extract temperature, humidity, and pressure values from latest data."""
    raw_measurements = {}
    for series_id, series_entry in series.items():
        measurement = measurements.get(series_id)
        if measurement:
            for short in ("temp", "hum", "bar"):
                if short in series_entry.shorts:
                    raw_measurements[short] = float(measurement["value"])
    return raw_measurements


def _calculate_derived_measurements(raw_measurements: dict[str, float], date: datetime) -> list[MeasurementEntry]:
    """Calculate all derived metrics when temperature and humidity are present."""
    if "temp" not in raw_measurements or "hum" not in raw_measurements:
        return []

    return [
        MeasurementEntry(name=metric.name, unit=metric.unit, date=date, value=metric.calculate(raw_measurements["temp"], raw_measurements["hum"]))
        for metric in DERIVED_METRICS
    ]


def _find_derived_metric(series_short: str) -> DerivedMetric | None:
    """Resolve a derived metric by one of its short names."""
    return DERIVED_METRICS_BY_SHORT.get(series_short)


async def get_latest(db: AsyncIOMotorDatabase) -> list[MeasurementEntry]:
    """Retrieve the latest raw and derived measurement for each series.

    Args:
        db: Database containing series metadata and weather measurements.

    Returns:
        Latest measurements, including virtual values calculated from the
        latest temperature and humidity readings.
    """
    series: dict[ObjectId, SeriesEntry] = {}
    for series_entry in await db["series"].find().to_list(length=None):
        series[series_entry["_id"]] = SeriesEntry(**series_entry)

    latest_weather: list[MeasurementEntry] = []
    latest_measurements: dict[ObjectId, dict] = {}
    for series_id, series_entry in series.items():
        latest_entry = await db["weather"].find_one({"series": series_id}, sort=[("date", -1)], projection={"_id": False})
        if latest_entry:
            latest_measurements[series_id] = latest_entry
            latest_weather.append(MeasurementEntry(name=series_entry.name, unit=series_entry.unit, **latest_entry))

    latest_weather.extend(_calculate_derived_measurements(_get_raw_measurements(series, latest_measurements), datetime.now()))

    return latest_weather


async def get_report(db: AsyncIOMotorDatabase, start_date: datetime = None, end_date: datetime = None) -> list[MeasurementEntry]:
    """Build min/max/average reports for all series in a date range.

    If no range is supplied, the previous seven days ending at the current
    time are used.
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


async def get_series_measurements(
    db: AsyncIOMotorDatabase, series_short: str, start_date: datetime = None, end_date: datetime = None
) -> list[MeasurementEntry]:
    """Retrieve raw or derived measurements for one series.

    If no range is supplied, the previous seven days ending at the current
    time are used. Unknown series names return an empty list.
    """
    derived_metric = _find_derived_metric(series_short)
    if derived_metric:
        return await _get_derived_series_measurements(db, derived_metric, start_date, end_date)

    series_entry = await db["series"].find_one({"shorts": series_short})
    if not series_entry:
        return []

    series_id = series_entry["_id"]
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()

    measurements_cursor = db["weather"].find({"series": series_id, "date": {"$gte": start_date, "$lte": end_date}}, projection={"_id": False}).sort("date", 1)

    measurements: list[MeasurementEntry] = []
    async for measurement in measurements_cursor:
        measurements.append(MeasurementEntry(name=series_entry["name"], unit=series_entry["unit"], **measurement))

    return measurements


async def _get_derived_series_measurements(
    db: AsyncIOMotorDatabase, metric: DerivedMetric, start_date: datetime = None, end_date: datetime = None
) -> list[MeasurementEntry]:
    """Calculate one derived metric from matching temperature and humidity values."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()

    source_series = await db["series"].find({"shorts": {"$in": ["temp", "hum"]}}).to_list(length=None)
    measurements_by_date: dict[datetime, dict[str, float]] = {}
    for source_series_entry in source_series:
        source_short = next(short for short in ("temp", "hum") if short in source_series_entry["shorts"])
        cursor = db["weather"].find(
            {
                "series": source_series_entry["_id"],
                "date": {"$gte": start_date, "$lte": end_date},
            },
            projection={"_id": False},
        )
        async for measurement in cursor:
            measurements_by_date.setdefault(measurement["date"], {})[source_short] = float(measurement["value"])

    return [
        MeasurementEntry(
            name=metric.name,
            unit=metric.unit,
            date=date,
            value=metric.calculate(values["temp"], values["hum"]),
        )
        for date, values in sorted(measurements_by_date.items())
        if "temp" in values and "hum" in values
    ]


async def add_weather_entry(db: AsyncIOMotorDatabase, weather_entry: dict):
    """Persist submitted station values for every matching series.

    Args:
        db: Database containing series metadata and the weather collection.
        weather_entry: Mapping from series short names to measured values.
    """
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
