import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class WeatherDataset(Dataset):
    """
    Sliding-window dataset over hourly temperature / pressure / humidity series.

    Public interface is unchanged. New behaviour is opt-in via keyword-only args
    (`stride`, `normalize`, `return_xy`, `dtype`, `time_features`) and the
    `VALID_RANGES` class attribute.

    Notes:
        * Channel order is always SERIES_SUBSET order (temperature, pressure, humidity),
          independent of the order of entries in the JSON file.
        * split() windows each split independently, so no window can span two splits
          (no leakage). Consequently no "gap" is needed between splits.
        * Call `prepare()` (or `len(ds)`) before creating a multi-worker DataLoader so
          the aligned array is built once in the parent process.
    """

    SERIES_FILE = "weather.series.json"
    WEATHER_FILE = "weather.weather.temperature_pressure_humidity.json"
    SERIES_SUBSET = ["temperature", "pressure", "humidity"]
    SERIES_SHORT_MAP = {"temp": "temperature", "bar": "pressure", "hum": "humidity"}
    MAX_INTERPOLATION_GAP = 3
    VALID_RANGES: Dict[str, tuple] = {}

    @staticmethod
    def _compute_additional_features(
        timestamps: pd.DatetimeIndex | Sequence,
        time_features: Sequence[str],
    ) -> pd.DataFrame:
        """Computes the selected cyclical time features for a timestamp sequence."""
        invalid_time_features = set(time_features) - {"hour", "day_of_year"}
        if invalid_time_features:
            raise ValueError("time_features must contain only 'hour' and/or 'day_of_year'")

        index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        features = {}
        if "hour" in time_features:
            hour_phase = 2 * np.pi * index.hour.to_numpy() / 24
            features["hour_sin"] = np.sin(hour_phase)
            features["hour_cos"] = np.cos(hour_phase)
        if "day_of_year" in time_features:
            day_phase = 2 * np.pi * (index.dayofyear.to_numpy() - 1) / 365
            features["day_of_year_sin"] = np.sin(day_phase)
            features["day_of_year_cos"] = np.cos(day_phase)
        return pd.DataFrame(features, index=index)

    @staticmethod
    def build_prediction_sample(
        measured_sequence: pd.DataFrame,
        time_features: Sequence[str] = ("hour", "day_of_year"),
        dtype=np.float32,
    ) -> np.ndarray:
        """Converts a measured temp/bar/hum sequence to a dataset input matrix.

        The sequence must contain a ``datetime`` or ``timestamp`` column (or use a
        ``DatetimeIndex``) and temperature, pressure, and humidity columns. Both
        canonical names and the short names ``temp``, ``bar``, and ``hum`` are
        accepted. The returned columns are ordered as weather channels followed by
        the selected cyclical time features and are not normalized.
        """
        invalid_time_features = set(time_features) - {"hour", "day_of_year"}
        if invalid_time_features:
            raise ValueError("time_features must contain only 'hour' and/or 'day_of_year'")

        if isinstance(measured_sequence, np.ndarray):
            if measured_sequence.ndim != 2:
                raise ValueError("measured_sequence must be a 2D array")

            positions = np.arange(len(measured_sequence))
            additional = []
            if "hour" in time_features:
                additional.append(np.sin(2 * np.pi * positions / 24))
            if "day_of_year" in time_features:
                additional.append(np.sin(2 * np.pi * positions / 365))
            return np.column_stack((measured_sequence, *additional)).astype(dtype, copy=False)

        if not isinstance(measured_sequence, pd.DataFrame):
            raise TypeError("measured_sequence must be a pandas DataFrame or NumPy array")

        timestamp_column = next(
            (column for column in ("timestamp", "datetime") if column in measured_sequence),
            None,
        )
        timestamps = (
            measured_sequence[timestamp_column]
            if timestamp_column is not None
            else measured_sequence.index
        )
        if not isinstance(pd.DatetimeIndex(pd.to_datetime(timestamps)), pd.DatetimeIndex):
            raise ValueError("measured_sequence must have datetime values")

        column_aliases = {
            "temperature": ("temperature", "temp"),
            "pressure": ("pressure", "bar"),
            "humidity": ("humidity", "hum"),
        }
        columns = {}
        for name, aliases in column_aliases.items():
            source = next((column for column in aliases if column in measured_sequence), None)
            if source is None:
                raise ValueError(f"measured_sequence is missing the {name} column")
            columns[name] = measured_sequence[source].to_numpy()

        matrix = pd.DataFrame(columns, index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
        additional = WeatherDataset._compute_additional_features(matrix.index, time_features)
        return pd.concat([matrix, additional], axis=1).to_numpy(dtype=dtype)

    def __init__(
        self,
        data_dir: str | None = None,
        lookback_hours: int = 72,
        horizon_hours: int = 24,
        max_samples=None,
        *,
        stride: int = 1,
        normalize: bool = False,
        return_xy: bool = False,
        dtype=np.float32,
        time_features: Sequence[str] = ("hour", "day_of_year"),
    ):
        """
        Initializes the WeatherDataset.

        Args:
            data_dir (str): Path to the directory containing weather data files.
            lookback_hours (int): Number of hours to look back for the forecast.
            horizon_hours (int): Number of hours to forecast into the future.
            max_samples (int, optional): Maximum number of samples to load. If None, all samples are loaded.
            stride (int): Step between consecutive windows (1 = every valid hour).
            normalize (bool): Standardize channels. Statistics are fitted on the train split
                by split(); on a standalone dataset call fit_normalization().
            return_xy (bool): If True, __getitem__ returns (lookback, horizon) instead of one array.
            dtype: dtype of returned arrays.
            time_features: Optional cyclical features. Supported values are ``"hour"`` and
                ``"day_of_year"``; each adds a sine and cosine channel.
        """
        if lookback_hours < 1 or horizon_hours < 1:
            raise ValueError("lookback_hours and horizon_hours must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        invalid_time_features = set(time_features) - {"hour", "day_of_year"}
        if invalid_time_features:
            raise ValueError("time_features must contain only 'hour' and/or 'day_of_year'")

        self.data_dir = data_dir
        self.lookback_hours = lookback_hours
        self.horizon_hours = horizon_hours
        self.max_samples = max_samples
        self.stride = stride
        self.normalize = normalize
        self.return_xy = return_xy
        self.dtype = dtype
        self.time_features = tuple(dict.fromkeys(time_features))

        self._oid_to_series_name: Dict[str, str] = {}
        self.series: Dict[str, pd.DataFrame] = {}

        if data_dir is not None:
            self._oid_to_series_name = self._load_oid_to_series_name()
            self.series = self._load_series()

        self.aligned_frame: np.ndarray | None = None
        self.valid_starts: np.ndarray | None = None
        self.channels: List[str] = []
        self.timestamps: pd.DatetimeIndex | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset. For each series the summary includes the number of samples,
        the first and last timestamps, and the number of missing values.

        Returns:
            pd.DataFrame: A DataFrame containing the summary information for each series.
        """
        summary_data = []
        for series_name, df in self.series.items():
            if not df.empty:
                summary_data.append(
                    {
                        "series": series_name,
                        "num_samples": len(df),
                        "first_timestamp": df["timestamp"].min(),
                        "last_timestamp": df["timestamp"].max(),
                        "num_missing": int(df.isnull().sum().sum()),
                    }
                )
        return pd.DataFrame(
            summary_data,
            columns=["series", "num_samples", "first_timestamp", "last_timestamp", "num_missing"],
        )

    def _load_oid_to_series_name(self) -> Dict[str, str]:
        """
        Loads the mapping from ObjectId to series names from the series JSON file.

        Returns:
            Dict[str, str]: A dictionary mapping ObjectId strings to series names.
        """
        series_path = Path(self.data_dir) / self.SERIES_FILE
        if not series_path.exists():
            raise FileNotFoundError(f"Series file not found: {series_path}")

        with open(series_path, "r", encoding="utf-8") as f:
            series_data = json.load(f)

        oid_to_series_name: Dict[str, str] = {}
        for series in series_data:
            oid = (series.get("_id") or {}).get("$oid")
            series_name = self.SERIES_SHORT_MAP.get(series.get("short"))
            if oid and series_name:
                if series_name in oid_to_series_name.values():
                    warnings.warn(
                        f"Multiple sensors map to series '{series_name}'; readings will be averaged per hour."
                    )
                oid_to_series_name[oid] = series_name

        return oid_to_series_name

    def _clean_and_resample(self, series_name: str, rows: list) -> pd.DataFrame:
        """Range-check, resample to 1h, and interpolate only short gaps."""
        df = pd.DataFrame(rows)
        ts = pd.to_datetime(df["timestamp"])
        val = pd.to_numeric(df["value"], errors="coerce")

        bounds = self.VALID_RANGES.get(series_name)
        if bounds is not None:
            lo, hi = bounds
            val = val.where((val >= lo) & (val <= hi))

        s = (
            pd.Series(val.to_numpy(), index=pd.DatetimeIndex(ts))
            .rename_axis("timestamp")
            .sort_index()
            .resample("1h")
            .mean()
        )

        na = s.isna()
        run_id = (na != na.shift()).cumsum()
        run_len = na.groupby(run_id).transform("sum")
        too_long = na & (run_len > self.MAX_INTERPOLATION_GAP)
        s = s.interpolate(method="time", limit_area="inside")
        s[too_long] = np.nan

        return s.rename("value").reset_index()

    def _load_series(self) -> Dict[str, pd.DataFrame]:
        """
        Loads weather data series from the specified directory.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are series names and values are corresponding DataFrames.
        """
        weather_path = Path(self.data_dir) / self.WEATHER_FILE
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        with open(weather_path, "r", encoding="utf-8") as f:
            weather_data = json.load(f)

        series_rows: Dict[str, list] = {}
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
            oid = (entry.get("series") or {}).get("$oid")
            series_name = self._oid_to_series_name.get(oid)
            if not series_name:
                continue
            timestamp = (entry.get("date") or {}).get("$date")
            value = entry.get("value")

            series_rows.setdefault(series_name, [])
            samples_per_series[series_name] += 1
            if self.max_samples is not None and min(samples_per_series.values()) > self.max_samples:
                break
            series_rows[series_name].append({"timestamp": timestamp, "value": value})

            current_min = min(samples_per_series.values())
            if last_pbar_update != current_min:
                pbar.update(1)
                last_pbar_update = current_min

        pbar.close()

        missing = [n for n in self.SERIES_SUBSET if not series_rows.get(n)]
        if missing:
            warnings.warn(f"No data found for series: {missing}")

        series_data: Dict[str, pd.DataFrame] = {}
        for series_name in self.SERIES_SUBSET:
            rows = series_rows.get(series_name)
            if rows:
                series_data[series_name] = self._clean_and_resample(series_name, rows)

        return series_data

    @classmethod
    def from_series(
        cls,
        oid_to_series_name: Dict[str, str],
        series_data: Dict[str, pd.DataFrame],
        lookback_hours: int = 72,
        horizon_hours: int = 24,
        **kwargs,
    ):
        """
        Creates a WeatherDataset instance from the provided series data.

        Args:
            oid_to_series_name (Dict[str, str]): A dictionary mapping ObjectId strings to series names.
            series_data (Dict[str, pd.DataFrame]): A dictionary where keys are series names and values are corresponding DataFrames.
            lookback_hours / horizon_hours: window configuration (propagated by split()).
            **kwargs: forwarded to the constructor (stride, normalize, return_xy, dtype,
                time_features).

        Returns:
            WeatherDataset: An instance of WeatherDataset initialized with the provided series data.
        """
        dataset = cls(None, lookback_hours, horizon_hours, **kwargs)
        dataset._oid_to_series_name = oid_to_series_name
        dataset.series = series_data
        return dataset

    def split(
        self,
        test_months: int = 12,
        val_block_days: int = 28,
        val_every: int = 4,
        purge_hours: int | None = None,
    ):
        """
        Season-robust split:
          * test  = the last `test_months` months (default: one full annual cycle).
          * val   = every `val_every`-th block of `val_block_days` days from the remaining
                    pool (default: 28-day blocks, 1 in 4 -> ~25% of the pool). Because the block
                    period (112 days) doesn't divide a year, val blocks drift through all seasons.
          * train = the rest of the pool, minus `purge_hours` (default: lookback + horizon)
                    on both sides of each val block and before the test period.

        Removed rows become NaN holes on the hourly grid, so no window ever spans a
        train/val/test boundary. Use it for model selection; for the final model retrain on
        train + val with the chosen settings and evaluate on test once.

        Args:
            test_months (int): Number of months to reserve for the test split.
            val_block_days (int): Number of days in each validation block.
            val_every (int): Frequency of validation blocks (every `val_every`-th block is used for validation).
            purge_hours (int | None): Number of hours to purge around validation blocks and before the test period.
                If None, defaults to lookback_hours + horizon_hours.

        """
        if val_every < 2:
            raise ValueError("val_every must be >= 2")
        non_empty = {n: df for n, df in self.series.items() if not df.empty}
        if not non_empty:
            raise ValueError("Cannot split an empty dataset")

        purge = pd.Timedelta(
            hours=self.lookback_hours + self.horizon_hours if purge_hours is None else purge_hours
        )
        block = pd.Timedelta(days=val_block_days)
        first = min(df["timestamp"].min() for df in non_empty.values()).floor("h")
        last = min(df["timestamp"].max() for df in non_empty.values())
        test_start = (last - pd.DateOffset(months=test_months)).floor("h")
        pool_end = test_start - purge

        train_series, val_series, test_series = {}, {}, {}
        for name, df in non_empty.items():
            ts = df["timestamp"]
            offset = ts - first
            bi = offset // block
            pos = offset - bi * block
            k = bi % val_every

            in_pool = ts < pool_end
            is_val = in_pool & (k == val_every - 1)
            near_val = ((k == val_every - 2) & (pos > block - purge)) | (
                (k == 0) & (bi > 0) & (pos < purge)
            )

            train_series[name] = df[in_pool & ~is_val & ~near_val]
            val_series[name] = df[is_val]
            test_series[name] = df[ts >= test_start]

        kwargs = dict(
            lookback_hours=self.lookback_hours,
            horizon_hours=self.horizon_hours,
            stride=self.stride,
            normalize=self.normalize,
            return_xy=self.return_xy,
            dtype=self.dtype,
            time_features=self.time_features,
        )
        datasets = [
            WeatherDataset.from_series(self._oid_to_series_name, sr, **kwargs)
            for sr in (train_series, val_series, test_series)
        ]
        train_dataset, val_dataset, test_dataset = datasets

        if self.normalize:
            train_dataset.fit_normalization()
            for ds in (val_dataset, test_dataset):
                ds.mean, ds.std = train_dataset.mean, train_dataset.std

        return train_dataset, val_dataset, test_dataset

    def _align_series(self) -> np.ndarray:
        """
        Aligns the series data on a continuous hourly grid and returns a 2D numpy array
        where each row is one hour and each column is a series (SERIES_SUBSET order).
        Rows are exactly one hour apart; missing hours are NaN.
        """
        names = [n for n in self.SERIES_SUBSET if n in self.series and not self.series[n].empty]
        names += [n for n in self.series if n not in names and not self.series[n].empty]
        self.channels = names

        if not names:
            self.timestamps = pd.DatetimeIndex([])
            return np.empty((0, len(self.SERIES_SUBSET)), dtype=np.float64)

        indexed = {
            n: self.series[n].drop_duplicates("timestamp").set_index("timestamp")["value"]
            for n in names
        }
        start = min(s.index.min() for s in indexed.values())
        end = max(s.index.max() for s in indexed.values())
        grid = pd.date_range(start, end, freq="1h")
        self.timestamps = grid

        frame = pd.DataFrame({n: indexed[n].reindex(grid) for n in names})
        frame = pd.concat(
            [frame, self._compute_additional_features(grid, self.time_features)], axis=1
        )
        self.channels = list(frame.columns)
        return frame.to_numpy(dtype=np.float64)

    def _valid_starts(self):
        if self.aligned_frame is None:
            self.aligned_frame = self._align_series()

        w = self.lookback_hours + self.horizon_hours
        ok = ~np.isnan(self.aligned_frame).any(axis=1)
        if len(ok) < w:
            return np.empty(0, dtype=np.int64)

        c = np.concatenate([[0], np.cumsum(ok, dtype=np.int64)])
        starts = np.flatnonzero(c[w:] - c[:-w] == w)
        return starts[:: self.stride]

    def prepare(self):
        """Builds the aligned array and window index eagerly (call before forking DataLoader workers)."""
        if self.aligned_frame is None:
            self.aligned_frame = self._align_series()
        if self.valid_starts is None:
            self.valid_starts = self._valid_starts()
        return self

    def fit_normalization(self):
        """Fits per-channel mean/std on this dataset's aligned data (use on the train split only)."""
        if self.aligned_frame is None:
            self.aligned_frame = self._align_series()
        if self.aligned_frame.shape[0] == 0:
            raise ValueError("Cannot fit normalization on empty data")
        self.mean = np.nanmean(self.aligned_frame, axis=0)
        std = np.nanstd(self.aligned_frame, axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        time_feature_count = 2 * len(self.time_features)
        if time_feature_count:
            self.mean[-time_feature_count:] = 0.0
            self.std[-time_feature_count:] = 1.0
        return self

    def denormalize(self, x):
        """Inverse of the standardization applied in __getitem__ (works on arrays/tensors, channel-last)."""
        if self.mean is None or self.std is None:
            return x
        return x * self.std + self.mean

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        The first sample is the first with a complete lookback period, and the last sample is the last with a complete horizon period.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.aligned_frame is None:
            self.aligned_frame = self._align_series()
        if self.valid_starts is None:
            self.valid_starts = self._valid_starts()
        return len(self.valid_starts)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        The first sample is the first with a complete lookback period, and the last sample is the last with a complete horizon period.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            np.ndarray of shape (lookback_hours + horizon_hours, n_series), or, if return_xy=True,
            a tuple (lookback, horizon) of arrays.
        """
        if idx < 0 or idx >= len(self):  # len() also builds the cached arrays
            raise IndexError("Index out of range")

        start = int(self.valid_starts[idx])
        L = self.lookback_hours
        window = self.aligned_frame[start : start + L + self.horizon_hours]

        if self.normalize and self.mean is not None:
            window = (window - self.mean) / self.std

        window = window.astype(self.dtype, copy=True)
        if self.return_xy:
            return window[:L], window[L:]

        return window
