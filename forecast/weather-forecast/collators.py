from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class WeatherCollator:
    def __init__(self, weather_data):
        self.weather_data = weather_data

    def __call__(self, batch: List[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.stack(batch, axis=0), dtype=torch.float32)
