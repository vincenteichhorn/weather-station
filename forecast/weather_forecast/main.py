import json
from pathlib import Path

import torch
from torch.optim import Adam, lr_scheduler
from datasets import WeatherDataset
from collators import WeatherCollator
from criterions import TimeSeriesGaussianNLLLoss
from metrics import (
    TARGET_VARIABLES,
    estimate_min_std,
    forecast_metrics,
    persistence_metrics,
    relative_mae_score,
)
from model import WeatherProbabilisticTCN
from torch.utils.data import DataLoader

EPOCHS = 15
LOOKBACK_HOURS = 96
HORIZON_HOURS = 24
BATCH_SIZE = 32
MODEL_FOLDER = "weather-probabilistic-tcn"
SENSOR_MIN_STD = (0.2, 0.4, 0.8)
DERIVED_FEATURES = (
    "pressure_delta_3h",
    "pressure_delta_24h",
    "dew_point",
    "specific_humidity",
)


def train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs = torch.nan_to_num(inputs.to(device))
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)


def validate(model, validation_dataloader, criterion, normalization_means, normalization_stds):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    totals = {
        variable: {"mae": 0.0, "rmse": 0.0, "nll": 0.0, "coverage_95": 0.0}
        for variable in TARGET_VARIABLES
    }

    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs = torch.nan_to_num(inputs.to(device))
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            batch_metrics = forecast_metrics(
                outputs, targets, normalization_means, normalization_stds
            )
            for variable, metric_values in batch_metrics.items():
                for name, value in metric_values.items():
                    totals[variable][name] += value

    batch_count = len(validation_dataloader)
    return (
        {
            variable: {name: value / batch_count for name, value in metric_values.items()}
            for variable, metric_values in totals.items()
        },
        total_loss / batch_count,
    )


def save_model_artifacts(
    model,
    criterion,
    model_directory,
    train_dataset,
    data_dir,
    epoch,
    selection_metric,
    physical_min_std,
):
    model_directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_directory / "model.pth")
    config = {
        "model_class": "WeatherProbabilisticTCN",
        "lookback": train_dataset.lookback_hours,
        "horizon": train_dataset.horizon_hours,
        "hidden_channels": model.hidden_channels,
        "kernel_size": model.kernel_size,
        "levels": model.levels,
        "dilations": list(model.dilations),
        "receptive_field": model.receptive_field,
        "dropout": model.dropout,
        "min_std": model.min_std.detach().cpu().tolist(),
        "physical_min_std": [float(value) for value in physical_min_std],
        "normalization": {
            "enabled": train_dataset.normalize,
            "means": train_dataset.mean.tolist(),
            "stds": train_dataset.std.tolist(),
        },
        "channels": train_dataset.channels,
        "target_variables": list(TARGET_VARIABLES),
        "task_log_variances": criterion.log_task_variances.detach().cpu().tolist(),
        "task_log_variance_bounds": [
            criterion.task_log_variance_min,
            criterion.task_log_variance_max,
        ],
        "time_features": list(train_dataset.time_features),
        "derived_features": list(train_dataset.derived_features),
        "data_dir": data_dir,
        "selection_metric": "mean_relative_mae_to_24h_persistence",
        "selection_value": selection_metric,
        "best_epoch": epoch,
    }
    with (model_directory / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


if __name__ == "__main__":

    data_dir = "data/weather-data"
    base_dataset = WeatherDataset(
        data_dir=data_dir,
        return_xy=True,
        normalize=True,
        time_features=["hour", "day_of_year"],
        derived_features=DERIVED_FEATURES,
        lookback_hours=LOOKBACK_HOURS,
        horizon_hours=HORIZON_HOURS,
    )
    train_dataset, validation_dataset, test_dataset = base_dataset.split()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=WeatherCollator(train_dataset.series),
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=WeatherCollator(validation_dataset.series),
    )
    baseline_metrics = persistence_metrics(
        validation_dataloader,
        train_dataset.mean,
        train_dataset.std,
    )
    print("24-hour persistence baseline in physical units:")
    for variable, metric_values in baseline_metrics.items():
        print(
            f"  {variable}: "
            f"MAE={metric_values['mae']:.6f}, "
            f"RMSE={metric_values['rmse']:.6f}"
        )
    min_std, physical_min_std = estimate_min_std(
        train_dataloader,
        train_dataset.std,
        residual_fraction=0.01,
        sensor_min_std=SENSOR_MIN_STD,
    )
    print(
        "Minimum physical std: "
        + ", ".join(
            f"{variable}={value:.6f}" for variable, value in zip(TARGET_VARIABLES, physical_min_std)
        )
    )
    model = WeatherProbabilisticTCN(
        lookback=train_dataset.lookback_hours,
        horizon=train_dataset.horizon_hours,
        denormalize=False,
        min_std=min_std,
        means=train_dataset.mean,
        stds=train_dataset.std,
    )
    model.to(device)
    criterion = TimeSeriesGaussianNLLLoss(min_std=min_std).to(device)
    optimizer = Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5,
    )
    model_directory = Path("models") / MODEL_FOLDER
    best_selection_metric = float("inf")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        loss = train_epoch(model, train_dataloader, optimizer, criterion)
        print(f"Training loss: {loss:.6f}")
        validation_metrics, validation_loss = validate(
            model,
            validation_dataloader,
            criterion,
            train_dataset.mean,
            train_dataset.std,
        )
        selection_metric, relative_mae = relative_mae_score(validation_metrics, baseline_metrics)
        # scheduler.step(validation_loss)
        scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Validation NLL: {validation_loss:.6f}; learning rate: {learning_rate:.2e}")
        print(
            "Selection metric (mean relative MAE): "
            f"{selection_metric:.6f} "
            + ", ".join(f"{variable}={value:.6f}" for variable, value in relative_mae.items())
        )
        print(
            "Task log variances: "
            + ", ".join(
                f"{variable}={value:.6f}"
                for variable, value in zip(
                    TARGET_VARIABLES,
                    criterion.log_task_variances.detach()
                    .clamp(
                        criterion.task_log_variance_min,
                        criterion.task_log_variance_max,
                    )
                    .cpu(),
                )
            )
        )
        print("Validation metrics in physical units:")
        for variable, metric_values in validation_metrics.items():
            mae_delta = metric_values["mae"] - baseline_metrics[variable]["mae"]
            print(
                f"  {variable}: "
                f"MAE={metric_values['mae']:.6f}, "
                f"RMSE={metric_values['rmse']:.6f}, "
                f"MAE-vs-persistence={mae_delta:+.6f}, "
                f"NLL={metric_values['nll']:.6f}, "
                f"coverage_95={metric_values['coverage_95']:.3%}"
            )
        if selection_metric < best_selection_metric:
            best_selection_metric = selection_metric
            save_model_artifacts(
                model,
                criterion,
                model_directory,
                train_dataset,
                data_dir,
                epoch + 1,
                selection_metric,
                physical_min_std,
            )
            print(f"Saved new best model to {model_directory}")
