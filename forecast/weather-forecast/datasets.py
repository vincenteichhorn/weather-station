import json
from typing import Dict

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class WeatherDataset(Dataset):

    SERIES_FILE = "weather.series.json"
    WEATHER_FILE = "weather.weather.temperature_pressure_humidity.json"
    SERIES_SUBSET = ["temperature", "pressure", "humidity"]
    SERIES_SHORT_MAP = {"temp": "temperature", "bar": "pressure", "hum": "humidity"}

    def __init__(
        self,
        data_dir: str,
        lookback_hours: int = 72,
        horizon_hours: int = 24,
        max_samples=None,
    ):
        """
        Initializes the WeatherDataset.

        Args:
            data_dir (str): Path to the directory containing weather data files.
            lookback_hours (int): Number of hours to look back for the forecast.
            horizon_hours (int): Number of hours to forecast into the future.
            max_samples (int, optional): Maximum number of samples to load. If None, all samples are loaded.
        """
        self.data_dir = data_dir
        self.lookback_hours = lookback_hours
        self.horizon_hours = horizon_hours
        self.max_samples = max_samples

        self._oid_to_series_name: Dict[str, str] = self._load_oid_to_series_name()
        self.series: Dict[str, pd.DataFrame] = self._load_series()

    def _load_oid_to_series_name(self) -> Dict[str, str]:
        """
        Loads the mapping from ObjectId to series names from the series JSON file.

        Returns:
            Dict[str, str]: A dictionary mapping ObjectId strings to series names.
        """

        series_path = f"{self.data_dir}/{self.SERIES_FILE}"
        if not pd.io.common.file_exists(series_path):
            raise FileNotFoundError(f"Series file not found: {series_path}")

        with open(series_path, "r", encoding="utf-8") as f:
            series_data = json.load(f)

        oid_to_series_name = {}
        for series in series_data:
            oid = series.get("_id").get("$oid")
            series_shorts = [series.get("short")]
            series_name = next(
                (
                    self.SERIES_SHORT_MAP.get(short)
                    for short in series_shorts
                    if short in self.SERIES_SHORT_MAP
                ),
                None,
            )
            if series_name:
                oid_to_series_name[oid] = series_name

        return oid_to_series_name

    def _load_series(self) -> Dict[str, pd.DataFrame]:
        """
        Loads weather data series from the specified directory.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are series names and values are corresponding DataFrames.
        """

        weather_path = f"{self.data_dir}/{self.WEATHER_FILE}"
        if not pd.io.common.file_exists(weather_path):
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        with open(weather_path, "r", encoding="utf-8") as f:
            weather_data = json.load(f)

        series_dict = {}

        samples_per_series = {series_name: 0 for series_name in self.SERIES_SUBSET}
        pbar = tqdm(
            desc="Loading weather data",
            unit="entry",
            total=(
                self.max_samples
                if self.max_samples is not None
                else len(weather_data) // len(self.SERIES_SUBSET)
            ),
        )
        last_pbar_update = 0
        for entry in weather_data:
            oid = entry.get("series").get("$oid")
            series_name = self._oid_to_series_name.get(oid)
            timestamp = entry.get("date").get("$date")
            value = entry.get("value")
            if series_name:

                if series_name not in series_dict:
                    series_dict[series_name] = pd.DataFrame(
                        columns=["timestamp", "value"]
                    )
                    series_dict[series_name]["timestamp"] = pd.to_datetime(
                        series_dict[series_name]["timestamp"]
                    )
                    series_dict[series_name]["value"] = pd.to_numeric(
                        series_dict[series_name]["value"], errors="coerce"
                    )

                samples_per_series[series_name] += 1
                if (
                    self.max_samples is not None
                    and min(samples_per_series.values()) > self.max_samples
                ):
                    break
                series_dict[series_name] = pd.concat(
                    [
                        series_dict[series_name],
                        pd.DataFrame({"timestamp": [timestamp], "value": [value]}),
                    ],
                    ignore_index=True,
                )

                if last_pbar_update != min(samples_per_series.values()):
                    pbar.update(1)
                    last_pbar_update = min(samples_per_series.values())

        return series_dict

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        The first sample is the first with a complete lookback period, and the last sample is the last with a complete horizon period.

        Returns:
            int: The number of samples in the dataset.
        """
        min_length = min(
            len(df) for df in self.series.values() if len(df) >= self.lookback_hours
        )
        return max(0, min_length - self.lookback_hours - self.horizon_hours + 1)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        The first sample is the first with a complete lookback period, and the last sample is the last with a complete horizon period.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the lookback and horizon data for each series.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # idx is a *sample* index; the first valid sample's lookback window starts
        # at dataframe position 0, so its horizon window starts at lookback_hours.
        # Shift idx so that "anchor" points at the first horizon timestep.
        anchor = idx + self.lookback_hours

        sample = {}
        for series_name, df in self.series.items():
            if len(df) < self.lookback_hours + self.horizon_hours:
                continue

            lookback_start_idx = anchor - self.lookback_hours
            lookback_end_idx = anchor
            horizon_start_idx = anchor
            horizon_end_idx = anchor + self.horizon_hours

            lookback_data = df.iloc[lookback_start_idx:lookback_end_idx]
            horizon_data = df.iloc[horizon_start_idx:horizon_end_idx]

            sample[series_name] = pd.concat(
                [lookback_data, horizon_data], ignore_index=True
            )

        return sample
