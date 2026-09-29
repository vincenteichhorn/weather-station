import torch
from torch import nn


class TimeSeriesGaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood for probabilistic forecasts."""

    def __init__(
        self,
        min_std: float | torch.Tensor = 1e-2,
        task_log_variance_min: float = -2.0,
        task_log_variance_max: float = 3.0,
    ):
        super().__init__()
        if task_log_variance_min > task_log_variance_max:
            raise ValueError("task_log_variance_min must not exceed max")
        min_std = torch.as_tensor(min_std, dtype=torch.float32)
        if torch.any(min_std <= 0):
            raise ValueError("min_std must be > 0")
        self.register_buffer("min_log_variance", 2 * torch.log(min_std))
        self.log_task_variances = nn.Parameter(torch.zeros(3))
        self.task_log_variance_min = task_log_variance_min
        self.task_log_variance_max = task_log_variance_max

    def forward(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        means, log_variances = predictions
        if means.shape != targets.shape or log_variances.shape != targets.shape:
            raise ValueError("means, log_variances, and targets must have the same shape")
        if targets.shape[-1] < len(self.log_task_variances):
            raise ValueError("targets must contain the three weather variables")
        log_variances = torch.maximum(log_variances, self.min_log_variance)
        log_variances = log_variances.clamp(max=10.0)
        squared_error = (targets - means).square()
        negative_log_likelihood = 0.5 * (log_variances + squared_error * torch.exp(-log_variances))
        task_losses = []
        for channel in range(len(self.log_task_variances)):
            valid = torch.isfinite(targets[..., channel])
            if valid.any():
                task_losses.append(negative_log_likelihood[..., channel][valid].mean())
            else:
                task_losses.append(negative_log_likelihood.new_zeros(()))
        task_losses = torch.stack(task_losses)
        task_log_variances = self.log_task_variances.clamp(
            min=self.task_log_variance_min,
            max=self.task_log_variance_max,
        ).to(task_losses.device)
        return 0.5 * (torch.exp(-task_log_variances) * task_losses + task_log_variances).sum()
