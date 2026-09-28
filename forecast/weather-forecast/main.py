import numpy as np

from datasets import WeatherDataset
from collators import WeatherCollator
from torch.utils.data import DataLoader

if __name__ == "__main__":

    data_dir = "data/weather-data"
    base_dataset = WeatherDataset(data_dir=data_dir)
    print("Base dataset summary:\n", base_dataset.summary())

    train_dataset, val_dataset, test_dataset = base_dataset.split()
    print("Train dataset summary:\n", train_dataset.summary())
    print("Validation dataset summary:\n", val_dataset.summary())
    print("Test dataset summary:\n", test_dataset.summary())

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, collate_fn=WeatherCollator(train_dataset.series)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, collate_fn=WeatherCollator(val_dataset.series)
    )

    for batch in train_dataloader:
        print(batch.shape)  # Should print: (batch_size, lookback_hours + horizon_hours, num_series)
        break
