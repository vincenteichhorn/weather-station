"""Database operations for raw and derived weather measurements."""

from datetime import datetime, timedelta

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
import numpy as np
import pandas as pd
import torch

from derived_metrics import DERIVED_METRICS, DERIVED_METRICS_BY_SHORT, DerivedMetric
from weather_forecast.datasets import WeatherDataset
from models import ForecastEntry, MeasurementEntry, SeriesEntry, WeatherEntry

VALID_DENSITIES = {"raw", "daily", "weekly", "monthly"}


def _validate_density(density: str) -> str:
    if density not in VALID_DENSITIES:
        raise ValueError(f"Unknown density '{density}'. Use raw, daily, weekly, or monthly.")
    return density


def _bucket_expression(density: str) -> dict:
    if density == "daily":
        return {"$dateTrunc": {"date": "$date", "unit": "day"}}
    if density == "weekly":
        return {"$dateTrunc": {"date": "$date", "unit": "week", "startOfWeek": "monday"}}
    return {"$dateTrunc": {"date": "$date", "unit": "month"}}


def _get_raw_measurements(
    series: dict[ObjectId, SeriesEntry], measurements: dict[ObjectId, dict]
) -> dict[str, float]:
    """Extract temperature, humidity, and pressure values from latest data."""
    raw_measurements = {}
    for series_id, series_entry in series.items():
        measurement = measurements.get(series_id)
        if measurement:
            for short in ("temp", "hum", "bar"):
                if short in series_entry.shorts:
                    raw_measurements[short] = float(measurement["value"])
    return raw_measurements


def _calculate_derived_measurements(
    raw_measurements: dict[str, float], date: datetime
) -> list[MeasurementEntry]:
    """Calculate all derived metrics when temperature and humidity are present."""
    if "temp" not in raw_measurements or "hum" not in raw_measurements:
        return []

    return [
        MeasurementEntry(
            name=metric.name,
            unit=metric.unit,
            date=date,
            value=metric.calculate(raw_measurements["temp"], raw_measurements["hum"]),
        )
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
        latest_entry = await db["weather"].find_one(
            {"series": series_id}, sort=[("date", -1)], projection={"_id": False}
        )
        if latest_entry:
            latest_measurements[series_id] = latest_entry
            latest_weather.append(
                MeasurementEntry(name=series_entry.name, unit=series_entry.unit, **latest_entry)
            )

    latest_weather.extend(
        _calculate_derived_measurements(
            _get_raw_measurements(series, latest_measurements), datetime.now()
        )
    )

    return latest_weather


async def get_report(
    db: AsyncIOMotorDatabase, start_date: datetime = None, end_date: datetime = None
) -> list[MeasurementEntry]:
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
            {
                "$match": {
                    "series": series_id,
                    "date": {"$gte": start_date, "$lte": end_date},
                }
            },
            {
                "$set": {
                    "numeric_value": {
                        "$convert": {
                            "input": "$value",
                            "to": "double",
                            "onError": None,
                            "onNull": None,
                        }
                    }
                }
            },
            {"$match": {"numeric_value": {"$ne": None}}},
            {
                "$group": {
                    "_id": None,
                    "min_value": {"$min": "$numeric_value"},
                    "max_value": {"$max": "$numeric_value"},
                    "avg_value": {"$avg": "$numeric_value"},
                }
            },
        ]
        result = await db["weather"].aggregate(pipeline).to_list(length=1)
        if result:
            min_entry = MeasurementEntry(
                name="Minimum",
                unit=series_entry.unit,
                date=start_date,
                value=result[0]["min_value"],
            )
            max_entry = MeasurementEntry(
                name="Maximum",
                unit=series_entry.unit,
                date=start_date,
                value=result[0]["max_value"],
            )
            avg_entry = MeasurementEntry(
                name="Durchschnitt",
                unit=series_entry.unit,
                date=start_date,
                value=result[0]["avg_value"],
            )
            report[series_entry.name] = [min_entry, max_entry, avg_entry]

    return report


async def get_series_measurements(
    db: AsyncIOMotorDatabase,
    series_short: str,
    start_date: datetime = None,
    end_date: datetime = None,
    density: str = "raw",
) -> list[MeasurementEntry]:
    """Retrieve raw or derived measurements for one series.

    If no range is supplied, the previous seven days ending at the current
    time are used. Unknown series names return an empty list.
    """
    density = _validate_density(density)
    derived_metric = _find_derived_metric(series_short)
    if derived_metric:
        return await _get_derived_series_measurements(
            db, derived_metric, start_date, end_date, density
        )

    series_entry = await db["series"].find_one({"shorts": series_short})
    if not series_entry:
        return []

    series_id = series_entry["_id"]
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()

    if density == "raw":
        measurements_cursor = (
            db["weather"]
            .find(
                {"series": series_id, "date": {"$gte": start_date, "$lte": end_date}},
                projection={"_id": False},
            )
            .sort("date", 1)
        )
        return [
            MeasurementEntry(name=series_entry["name"], unit=series_entry["unit"], **measurement)
            async for measurement in measurements_cursor
        ]

    pipeline = [
        {
            "$match": {
                "series": series_id,
                "date": {"$gte": start_date, "$lte": end_date},
                "value": {"$type": "number"},
            }
        },
        {"$group": {"_id": _bucket_expression(density), "value": {"$avg": "$value"}}},
        {"$sort": {"_id": 1}},
    ]
    return [
        MeasurementEntry(
            name=series_entry["name"],
            unit=series_entry["unit"],
            date=entry["_id"],
            value=entry["value"],
        )
        async for entry in db["weather"].aggregate(pipeline)
    ]


async def _get_derived_series_measurements(
    db: AsyncIOMotorDatabase,
    metric: DerivedMetric,
    start_date: datetime = None,
    end_date: datetime = None,
    density: str = "raw",
) -> list[MeasurementEntry]:
    """Calculate one derived metric from matching temperature and humidity values."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()

    source_series = (
        await db["series"].find({"shorts": {"$in": ["temp", "hum"]}}).to_list(length=None)
    )
    measurements_by_date: dict[datetime, dict[str, list[float]]] = {}
    for source_series_entry in source_series:
        source_short = next(
            short for short in ("temp", "hum") if short in source_series_entry["shorts"]
        )
        cursor = db["weather"].find(
            {
                "series": source_series_entry["_id"],
                "date": {"$gte": start_date, "$lte": end_date},
            },
            projection={"_id": False},
        )
        async for measurement in cursor:
            bucket = measurement["date"]
            if density != "raw":
                if density == "daily":
                    bucket = bucket.replace(hour=0, minute=0, second=0, microsecond=0)
                elif density == "weekly":
                    bucket = (bucket - timedelta(days=bucket.weekday())).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    bucket = bucket.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            measurements_by_date.setdefault(bucket, {}).setdefault(source_short, []).append(
                float(measurement["value"])
            )

    return [
        MeasurementEntry(
            name=metric.name,
            unit=metric.unit,
            date=date,
            value=metric.calculate(
                sum(values["temp"]) / len(values["temp"]),
                sum(values["hum"]) / len(values["hum"]),
            ),
        )
        for date, values in sorted(measurements_by_date.items())
        if "temp" in values and "hum" in values and values["temp"] and values["hum"]
    ]



async def get_forecast(
    db: AsyncIOMotorDatabase,
    model: torch.nn.Module,
    model_config: dict,
    reference_time: datetime | None = None,
) -> dict[str, list[ForecastEntry]]:
    """Predict the configured horizon from the latest hourly measurements."""
    reference_time = reference_time or datetime.now()
    lookback = int(model_config["lookback"])
    horizon = int(model_config["horizon"])
    channels = model_config["channels"]
    target_variables = model_config["target_variables"]
    time_features = model_config.get("time_features", [])
    derived_features = model_config.get("derived_features", [])

    series_entries = await db["series"].find().to_list(length=None)
    short_to_series: dict[str, dict] = {}
    for series_entry in series_entries:
        for short in series_entry["shorts"]:
            if short not in short_to_series:
                short_to_series[short] = series_entry

    source_series: dict[str, dict] = {}
    for target in target_variables:
        source_short = next(
            (
                short
                for short, channel in WeatherDataset.SERIES_SHORT_MAP.items()
                if channel == target
            ),
            None,
        )
        if source_short is None or source_short not in short_to_series:
            raise ValueError(f"Forecast series for '{target}' is not available in the database.")
        source_series[target] = short_to_series[source_short]

    latest_rows: dict[str, list[dict]] = {}
    for channel, series_entry in source_series.items():
        query = {"series": series_entry["_id"], "date": {"$lte": reference_time}}
        cursor = db["weather"].find(
            query,
            projection={"_id": False, "date": True, "value": True},
        ).sort("date", -1).limit(lookback * 24)
        latest_rows[channel] = [row async for row in cursor]

    frames = {}
    for channel, rows in latest_rows.items():
        if not rows:
            raise ValueError(f"No measurements found for the '{channel}' series.")
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frames[channel] = (
            frame.dropna(subset=["date", "value"])
            .set_index("date")["value"]
            .sort_index()
            .resample("1h")
            .mean()
        )

    measured = pd.concat(
        [frames[channel] for channel in target_variables],
        axis=1,
        keys=target_variables,
    ).dropna()
    if len(measured) < lookback:
        raise ValueError(f"Forecast requires {lookback} complete hourly measurements.")
    measured = measured.iloc[-lookback:]
    measured.index.name = "timestamp"
    sample = WeatherDataset.build_prediction_sample(
        measured.reset_index(),
        time_features=time_features,
        derived_features=derived_features,
    )
    expected_channels = ["temperature", "pressure", "humidity", *derived_features]
    for time_feature in time_features:
        expected_channels.extend((f"{time_feature}_sin", f"{time_feature}_cos"))
    if list(channels) != expected_channels:
        raise ValueError("Forecast model channels do not match the configured input features.")
    inputs = torch.from_numpy(np.asarray(sample, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        means, log_variances = model(inputs)
    means = means[0].cpu().numpy()
    uncertainties = torch.exp(0.5 * log_variances[0]).cpu().numpy()
    forecast_dates = pd.date_range(
        measured.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )

    result: dict[str, list[ForecastEntry]] = {}
    for target_index, target in enumerate(target_variables):
        series_entry = source_series[target]
        result[target] = [
            ForecastEntry(
                name=series_entry["name"],
                unit=series_entry["unit"],
                date=date.to_pydatetime(),
                value=float(means[step, target_index]),
                uncertainty=float(uncertainties[step, target_index]),
            )
            for step, date in enumerate(forecast_dates)
        ]
    return result


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
                new_entry: WeatherEntry = {
                    "series": series_id,
                    "date": current_time,
                    "value": float(weather_entry[short]),
                }
                weather_collection = db["weather"]
                await weather_collection.insert_one(new_entry)
