from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class WeatherCollator:
    def __init__(self, weather_data):
        self.weather_data = weather_data

    def __call__(self, batch: List[np.ndarray]) -> torch.Tensor:
        if isinstance(batch[0], tuple):
            inputs, targets = zip(*batch)
            return (
                torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32),
                torch.tensor(np.stack(targets, axis=0), dtype=torch.float32),
            )
        return torch.tensor(np.stack(batch, axis=0), dtype=torch.float32)
