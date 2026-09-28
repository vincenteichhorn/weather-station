import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "weather-forecast"))

from datasets import WeatherDataset


@pytest.fixture
def hourly_series():
    timestamps = pd.date_range("2024-01-01", periods=120, freq="h")
    return {
        name: pd.DataFrame(
            {"timestamp": timestamps, "value": np.arange(len(timestamps), dtype=float) + offset}
        )
        for name, offset in {
            "temperature": 0.0,
            "pressure": 100.0,
            "humidity": 200.0,
        }.items()
    }


def make_dataset(series, **kwargs):
    return WeatherDataset.from_series({}, series, lookback_hours=3, horizon_hours=2, **kwargs)


def test_compute_additional_features_returns_cyclical_pairs():
    timestamps = pd.date_range("2024-01-01 06:00", periods=2, freq="h")

    features = WeatherDataset._compute_additional_features(timestamps, ["hour", "day_of_year"])

    assert list(features.columns) == [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    np.testing.assert_allclose(features.iloc[0].to_numpy(), [1.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert features.index.equals(timestamps)


def test_build_prediction_sample_accepts_numpy_input():
    measured = np.arange(12, dtype=np.float64).reshape(4, 3)

    result = WeatherDataset.build_prediction_sample(measured, time_features=["hour"])

    assert result.shape == (4, 4)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, :3], measured)
    np.testing.assert_allclose(result[:, 3], np.sin(2 * np.pi * np.arange(4) / 24))


def test_build_prediction_sample_accepts_dataframe_aliases_and_datetime_column():
    measured = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 06:00", periods=2, freq="h"),
            "temp": [10.0, 11.0],
            "bar": [1000.0, 1001.0],
            "hum": [50.0, 51.0],
        }
    )

    result = WeatherDataset.build_prediction_sample(measured, time_features=["hour"])

    assert result.shape == (2, 5)
    np.testing.assert_allclose(result[:, :3], [[10, 1000, 50], [11, 1001, 51]])
    np.testing.assert_allclose(result[0, 3:], [1.0, 0.0], atol=1e-6)


def test_build_prediction_sample_uses_datetime_index_and_validates_inputs():
    measured = pd.DataFrame(
        {"temperature": [1], "pressure": [2], "humidity": [3]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )

    result = WeatherDataset.build_prediction_sample(measured, time_features=[])

    assert result.shape == (1, 3)
    with pytest.raises(ValueError, match="only 'hour'"):
        WeatherDataset.build_prediction_sample(np.ones((2, 3)), time_features=["week"])
    with pytest.raises(ValueError, match="missing the pressure"):
        WeatherDataset.build_prediction_sample(measured.drop(columns="pressure"))
    with pytest.raises(ValueError, match="2D"):
        WeatherDataset.build_prediction_sample(np.ones(3))
    with pytest.raises(TypeError, match="DataFrame or NumPy array"):
        WeatherDataset.build_prediction_sample([[1, 2, 3]])


def test_constructor_validates_window_stride_and_features():
    for kwargs, message in [
        ({"lookback_hours": 0}, "lookback_hours"),
        ({"horizon_hours": 0}, "horizon_hours"),
        ({"stride": 0}, "stride"),
        ({"time_features": ["week"]}, "only 'hour'"),
    ]:
        with pytest.raises(ValueError, match=message):
            WeatherDataset(**kwargs)


def test_from_series_aligns_channels_and_builds_sliding_windows(hourly_series):
    dataset = make_dataset(hourly_series, time_features=[])

    assert len(dataset) == 116
    assert dataset.channels == ["temperature", "pressure", "humidity"]
    assert dataset.timestamps[0] == pd.Timestamp("2024-01-01")
    np.testing.assert_allclose(dataset[0], dataset.aligned_frame[:5].astype(np.float32))
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_stride_and_return_xy_control_samples(hourly_series):
    dataset = make_dataset(hourly_series, stride=3, return_xy=True, time_features=[])

    assert len(dataset) == 39
    lookback, horizon = dataset[1]
    assert lookback.shape == (3, 3)
    assert horizon.shape == (2, 3)
    np.testing.assert_array_equal(lookback, dataset.aligned_frame[3:6])


def test_missing_hours_prevent_windows_from_crossing_gaps(hourly_series):
    for frame in hourly_series.values():
        frame.drop(index=frame.index[10:12], inplace=True)

    dataset = make_dataset(hourly_series, time_features=[])

    assert len(dataset) == 110
    assert not any(6 <= start <= 11 for start in dataset.valid_starts)


def test_normalization_and_denormalization_are_reversible(hourly_series):
    dataset = make_dataset(hourly_series, normalize=True, time_features=["hour"])
    dataset.fit_normalization()

    sample = dataset[0]
    restored = dataset.denormalize(sample)

    assert dataset.mean.shape == (5,)
    assert dataset.std.shape == (5,)
    assert dataset.mean[-2] == 0.0
    assert dataset.std[-2] == 1.0
    np.testing.assert_allclose(restored, dataset.aligned_frame[:5], atol=1e-5)


def test_prepare_is_idempotent_and_empty_dataset_behaves():
    empty = make_dataset({}, time_features=[])
    assert empty.prepare() is empty
    assert len(empty) == 0
    assert empty.aligned_frame.shape == (0, 3)
    with pytest.raises(ValueError, match="empty data"):
        empty.fit_normalization()


def test_summary_reports_series_metadata(hourly_series):
    dataset = make_dataset(hourly_series, time_features=[])

    summary = dataset.summary()

    assert list(summary["series"]) == ["temperature", "pressure", "humidity"]
    assert summary["num_samples"].tolist() == [120, 120, 120]
    assert summary["num_missing"].tolist() == [0, 0, 0]


def test_clean_and_resample_filters_ranges_and_interpolates_short_gaps():
    class BoundedDataset(WeatherDataset):
        VALID_RANGES = {"temperature": (0, 10)}

    rows = [
        {"timestamp": "2024-01-01 00:00", "value": 0},
        {"timestamp": "2024-01-01 02:00", "value": 20},
        {"timestamp": "2024-01-01 03:00", "value": 6},
    ]

    result = BoundedDataset()._clean_and_resample("temperature", rows)

    assert result["value"].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_loading_json_maps_series_and_applies_max_samples(tmp_path):
    (tmp_path / WeatherDataset.SERIES_FILE).write_text(
        json.dumps(
            [
                {"_id": {"$oid": "temp-id"}, "short": "temp"},
                {"_id": {"$oid": "bar-id"}, "short": "bar"},
                {"_id": {"$oid": "hum-id"}, "short": "hum"},
            ]
        ),
        encoding="utf-8",
    )
    entries = []
    for hour in range(3):
        timestamp = f"2024-01-01T{hour:02d}:00:00Z"
        for oid, value in [("temp-id", hour), ("bar-id", hour + 100), ("hum-id", hour + 200)]:
            entries.append({"series": {"$oid": oid}, "date": {"$date": timestamp}, "value": value})
    (tmp_path / WeatherDataset.WEATHER_FILE).write_text(json.dumps(entries), encoding="utf-8")

    dataset = WeatherDataset(tmp_path, max_samples=2, time_features=[])

    assert dataset._oid_to_series_name == {
        "temp-id": "temperature",
        "bar-id": "pressure",
        "hum-id": "humidity",
    }
    assert set(dataset.series) == set(WeatherDataset.SERIES_SUBSET)
    assert len(dataset.series["temperature"]) == 3
    assert len(dataset.series["pressure"]) == 3
    assert len(dataset.series["humidity"]) == 2


def test_split_creates_disjoint_splits_and_propagates_options(hourly_series):
    timestamps = pd.date_range("2024-01-01", periods=24 * 100, freq="h")
    hourly_series = {
        name: pd.DataFrame(
            {"timestamp": timestamps, "value": np.arange(len(timestamps), dtype=float) + offset}
        )
        for name, offset in {
            "temperature": 0.0,
            "pressure": 100.0,
            "humidity": 200.0,
        }.items()
    }
    dataset = make_dataset(
        hourly_series,
        normalize=True,
        return_xy=True,
        stride=2,
        time_features=["hour"],
    )

    train, validation, test = dataset.split(
        test_months=1, val_block_days=1, val_every=2, purge_hours=2
    )

    train_times = set(train.series["temperature"]["timestamp"])
    validation_times = set(validation.series["temperature"]["timestamp"])
    test_times = set(test.series["temperature"]["timestamp"])
    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)
    assert train.normalize and validation.normalize and test.normalize
    assert validation.mean is train.mean
    assert test.std is train.std
    assert validation.return_xy and test.stride == 2

    with pytest.raises(ValueError, match="val_every"):
        dataset.split(val_every=1)
    with pytest.raises(ValueError, match="empty"):
        make_dataset({}, time_features=[]).split()
