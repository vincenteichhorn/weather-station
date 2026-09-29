import torch

TARGET_VARIABLES = ("temperature", "pressure", "humidity")


def estimate_min_std(
    dataloader,
    normalization_stds: torch.Tensor,
    residual_fraction: float = 0.01,
    sensor_min_std: tuple[float, float, float] = (0.1, 0.1, 0.5),
    fallback_std: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel minimum std from persistence residuals.

    Returns the minimum std in normalized units for the model and in physical
    units for logging. The remaining time-feature channels use ``fallback_std``.
    """
    if residual_fraction <= 0 or fallback_std <= 0:
        raise ValueError("residual_fraction and fallback_std must be > 0")
    sensor_min_std = torch.as_tensor(sensor_min_std, dtype=torch.float32)
    if sensor_min_std.shape != (len(TARGET_VARIABLES),) or torch.any(sensor_min_std <= 0):
        raise ValueError("sensor_min_std must contain three positive values")

    residuals = []
    channel_count = None
    for inputs, targets in dataloader:
        channel_count = targets.shape[-1]
        baseline = inputs[:, -1:, : len(TARGET_VARIABLES)].expand(-1, targets.shape[1], -1)
        residual = targets[:, :, : len(TARGET_VARIABLES)] - baseline
        residuals.append(residual.reshape(-1, len(TARGET_VARIABLES)))

    if not residuals or channel_count is None:
        raise ValueError("cannot estimate minimum std from an empty dataloader")

    residual_std_normalized = torch.cat(residuals).std(dim=0, unbiased=False)
    normalization_stds = torch.as_tensor(normalization_stds, dtype=torch.float32)
    residual_min_std = (residual_std_normalized * normalization_stds[: len(TARGET_VARIABLES)]).mul(
        residual_fraction
    )
    physical_min_std = torch.maximum(residual_min_std, sensor_min_std)
    normalized_min_std = torch.full((channel_count,), fallback_std, dtype=torch.float32)
    normalized_min_std[: len(TARGET_VARIABLES)] = (
        physical_min_std / normalization_stds[: len(TARGET_VARIABLES)]
    ).clamp_min(torch.finfo(torch.float32).eps)
    return normalized_min_std, physical_min_std


def persistence_metrics(
    dataloader,
    normalization_means: torch.Tensor,
    normalization_stds: torch.Tensor,
    lag: int = 24,
) -> dict[str, dict[str, float]]:
    """Evaluate a lagged persistence forecast in physical units."""
    if lag < 1:
        raise ValueError("lag must be >= 1")

    normalization_means = torch.as_tensor(normalization_means, dtype=torch.float32)
    normalization_stds = torch.as_tensor(normalization_stds, dtype=torch.float32)
    sums = torch.zeros(len(TARGET_VARIABLES), 2)
    counts = torch.zeros(len(TARGET_VARIABLES))

    for inputs, targets in dataloader:
        lookback_length = inputs.shape[1]
        horizon_length = targets.shape[1]
        if lookback_length < lag:
            raise ValueError("lookback must be at least as long as lag")
        if horizon_length > lag:
            raise ValueError("horizon must not be longer than lag")
        start = lookback_length - lag
        baseline = inputs[:, start : start + horizon_length, : len(TARGET_VARIABLES)]
        target_values = targets[:, :, : len(TARGET_VARIABLES)]
        means = normalization_means[: len(TARGET_VARIABLES)]
        stds = normalization_stds[: len(TARGET_VARIABLES)]
        baseline = baseline * stds + means
        target_values = target_values * stds + means
        valid = torch.isfinite(baseline) & torch.isfinite(target_values)
        errors = baseline - target_values
        for channel in range(len(TARGET_VARIABLES)):
            channel_errors = errors[..., channel][valid[..., channel]]
            sums[channel, 0] += channel_errors.abs().sum()
            sums[channel, 1] += channel_errors.square().sum()
            counts[channel] += channel_errors.numel()

    if not torch.all(counts > 0):
        raise ValueError("persistence baseline has no finite validation targets")
    return {
        variable: {
            "mae": (sums[channel, 0] / counts[channel]).item(),
            "rmse": (sums[channel, 1] / counts[channel]).sqrt().item(),
        }
        for channel, variable in enumerate(TARGET_VARIABLES)
    }


def relative_mae_score(
    metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float]]:
    """Return the equally weighted MAE ratio against the persistence baseline."""
    ratios = {
        variable: metrics[variable]["mae"] / max(baseline_metrics[variable]["mae"], 1e-8)
        for variable in TARGET_VARIABLES
    }
    return sum(ratios.values()) / len(ratios), ratios


def _valid_values(
    predictions: tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    normalization_means: torch.Tensor,
    normalization_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means, log_variances = predictions
    if means.shape != targets.shape or log_variances.shape != targets.shape:
        raise ValueError("means, log_variances, and targets must have the same shape")
    normalization_means = torch.as_tensor(
        normalization_means, device=targets.device, dtype=targets.dtype
    )
    normalization_stds = torch.as_tensor(
        normalization_stds, device=targets.device, dtype=targets.dtype
    )
    means = means * normalization_stds + normalization_means
    targets = targets * normalization_stds + normalization_means
    log_variances = log_variances + 2 * torch.log(normalization_stds)
    means = means[..., : len(TARGET_VARIABLES)]
    targets = targets[..., : len(TARGET_VARIABLES)]
    log_variances = log_variances[..., : len(TARGET_VARIABLES)]
    valid = torch.isfinite(targets) & torch.isfinite(means) & torch.isfinite(log_variances)
    if not valid.any():
        raise ValueError("targets and predictions must contain finite values")
    return means, log_variances, targets, valid


def forecast_metrics(
    predictions: tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    normalization_means: torch.Tensor,
    normalization_stds: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Return per-variable metrics in the original physical units."""
    means, log_variances, targets, valid = _valid_values(
        predictions, targets, normalization_means, normalization_stds
    )
    metrics = {}
    for channel, variable in enumerate(TARGET_VARIABLES):
        channel_valid = valid[..., channel]
        channel_means = means[..., channel][channel_valid]
        channel_log_variances = log_variances[..., channel][channel_valid].clamp(
            min=-10.0, max=10.0
        )
        channel_targets = targets[..., channel][channel_valid]
        errors = channel_means - channel_targets
        squared_errors = errors.square()
        negative_log_likelihood = 0.5 * (
            channel_log_variances + squared_errors * torch.exp(-channel_log_variances)
        )
        standard_deviations = torch.exp(0.5 * channel_log_variances)
        covered = (channel_targets >= channel_means - 1.96 * standard_deviations) & (
            channel_targets <= channel_means + 1.96 * standard_deviations
        )
        metrics[variable] = {
            "mae": errors.abs().mean().item(),
            "rmse": squared_errors.mean().sqrt().item(),
            "nll": negative_log_likelihood.mean().item(),
            "coverage_95": covered.float().mean().item(),
        }
    return metrics
