from datasets import WeatherDataset
from collators import WeatherCollator
from torch.utils.data import DataLoader

if __name__ == "__main__":

    data_dir = "data/weather-data"
    dataset = WeatherDataset(data_dir, max_samples=1_000)

    sample = dataset[0]

    dataloader = DataLoader(
        dataset, batch_size=32, collate_fn=WeatherCollator(dataset.series)
    )

    for batch in dataloader:
        print(
            batch.shape
        )  # Should print: (batch_size, num_variables, lookback_hours + horizon_hours)
        break
