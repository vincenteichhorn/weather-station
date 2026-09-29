import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets import WeatherDataset
from model import WeatherProbabilisticTCN

TARGET_VARIABLES = ("temperature", "pressure", "humidity")


def load_model(model_directory: Path, device: torch.device):
    with (model_directory / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    normalization = config["normalization"]
    model = WeatherProbabilisticTCN(
        lookback=config["lookback"],
        horizon=config["horizon"],
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        levels=config["levels"],
        dropout=config["dropout"],
        min_std=torch.tensor(config["min_std"]),
        means=torch.tensor(normalization["means"]),
        stds=torch.tensor(normalization["stds"]),
        normalize=False,
        denormalize=False,
    ).to(device)
    state_dict = torch.load(model_directory / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def plot_example(
    model,
    sample,
    config,
    output_path: Path,
    device: torch.device,
    example_index: int,
):
    normalization = config["normalization"]
    means = torch.tensor(normalization["means"], device=device)
    stds = torch.tensor(normalization["stds"], device=device)
    lookback = config["lookback"]
    horizon = config["horizon"]

    inputs, targets = sample
    inputs = torch.nan_to_num(torch.from_numpy(inputs).unsqueeze(0).to(device))
    targets = torch.from_numpy(targets).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_means, predicted_log_variances = model(inputs)

    history = inputs[0, :, : len(TARGET_VARIABLES)] * stds[:3] + means[:3]
    targets = targets[0, :, : len(TARGET_VARIABLES)] * stds[:3] + means[:3]
    predicted_means = predicted_means[0, :, :3] * stds[:3] + means[:3]
    predicted_stds = torch.exp(0.5 * predicted_log_variances[0, :, :3]) * stds[:3]

    history = history.cpu()
    targets = targets.cpu()
    predicted_means = predicted_means.cpu()
    predicted_stds = predicted_stds.cpu()
    time_axis = range(lookback + horizon)
    forecast_axis = range(lookback, lookback + horizon)

    figure, axes = plt.subplots(len(TARGET_VARIABLES), 1, figsize=(12, 9), sharex=True)
    for channel, (axis, variable) in enumerate(zip(axes, TARGET_VARIABLES)):
        axis.plot(range(lookback), history[:, channel], label="history", color="0.35")
        axis.plot(forecast_axis, targets[:, channel], label="target", color="black")
        axis.plot(
            forecast_axis,
            predicted_means[:, channel],
            label="prediction",
            color="tab:blue",
        )
        axis.fill_between(
            forecast_axis,
            predicted_means[:, channel] - 1.96 * predicted_stds[:, channel],
            predicted_means[:, channel] + 1.96 * predicted_stds[:, channel],
            color="tab:blue",
            alpha=0.2,
            label="95% uncertainty",
        )
        axis.axvline(lookback - 0.5, color="0.5", linestyle="--")
        axis.set_ylabel(variable)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("hours")
    axes[-1].set_xlim(0, len(time_axis) - 1)
    figure.suptitle(f"Validation example {example_index}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot probabilistic validation forecasts")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("models/weather-probabilistic-tcn"),
    )
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_folder, device)
    data_dir = args.data_dir or config["data_dir"]
    dataset = WeatherDataset(
        data_dir=str(data_dir),
        return_xy=True,
        normalize=config["normalization"]["enabled"],
        time_features=config["time_features"],
        derived_features=config.get("derived_features", ()),
        lookback_hours=config["lookback"],
        horizon_hours=config["horizon"],
    )
    _, validation_dataset, _ = dataset.split()

    args.output.mkdir(parents=True, exist_ok=True)
    example_count = min(args.examples, len(validation_dataset))
    example_indices = random.Random(args.seed).sample(range(len(validation_dataset)), example_count)
    for plot_index, example_index in enumerate(example_indices):
        plot_example(
            model,
            validation_dataset[example_index],
            config,
            args.output / f"validation_{plot_index}.png",
            device,
            example_index,
        )
    print(
        f"Saved {example_count} plots to {args.output} "
        f"(seed={args.seed}, indices={example_indices})"
    )


if __name__ == "__main__":
    main()
