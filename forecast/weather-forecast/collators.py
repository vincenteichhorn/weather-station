from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class WeatherCollator:
    def __init__(self, weather_data):
        self.weather_data = weather_data

    def __call__(self, batch: List[Dict[str, pd.DataFrame]]) -> torch.Tensor:
        """
        The WeatherDataset returns a dictionary with keys corresponding to different weather variables (e.g., temperature, humidity, wind speed).
        Each key maps to a pandas DataFrame containing the time series data for that variable of sahpe (lookback_hours + horizon_hours, num_samples).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, lookback_hours + horizon_hours, num_samples) containing the collated weather data for the batch.
        """

        series_names = list(batch[0].keys())

        collated_data = np.empty(
            (len(series_names), len(batch), batch[0][series_names[0]].shape[0]),
            dtype=np.float32,
        )
        for series_name in series_names:
            variable_data = [sample[series_name]["value"].values for sample in batch]

            lengths = {len(v) for v in variable_data}
            if len(lengths) > 1:
                raise ValueError(
                    f"Inconsistent sequence lengths for series '{series_name}' "
                    f"within batch: {lengths}"
                )

            collated_data[series_names.index(series_name)] = np.stack(
                variable_data, axis=0
            )

        # collated_data shape: (num_variables, batch_size, seq_len)
        tensor_data = torch.tensor(collated_data, dtype=torch.float32)

        # -> (batch_size, num_variables, seq_len)
        tensor_data = tensor_data.permute(1, 0, 2).contiguous()

        return tensor_data
