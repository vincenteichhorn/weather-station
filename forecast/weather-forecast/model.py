import torch
import torch.nn.functional as F
from torch import nn


class Chomp1d(nn.Module):
    """Remove right-side padding so a temporal block remains causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        if chomp_size < 0:
            raise ValueError("chomp_size must be >= 0")
        self.chomp_size = chomp_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual causal convolutional block for one-dimensional sequences."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.residual(inputs))


class WeatherProbabilisticTCN(nn.Module):
    """Predicts a fixed horizon from a channel-last time-series window.

    The model accepts either a lookback tensor with shape ``(B, L, C)`` or a
    complete dataset window with shape ``(B, L + H, C)``. In the latter case,
    the horizon portion is ignored and only the first ``L`` steps are used.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        levels: int = 4,
        dropout: float = 0.0,
        min_std: float | torch.Tensor = 1e-2,
        means: torch.Tensor = None,
        stds: torch.Tensor = None,
        normalize: bool = False,
        denormalize: bool = False,
    ):
        super().__init__()
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be >= 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.levels = levels
        self.dropout = dropout
        self.task_count = 3
        self.dilations = tuple(2**level for level in range(levels))
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(self.dilations)
        min_std = torch.as_tensor(min_std, dtype=torch.float32)
        if torch.any(min_std <= 0):
            raise ValueError("min_std must be > 0")
        self.register_buffer("min_std", min_std)
        if means is not None:
            means = torch.as_tensor(means, dtype=torch.float32)
        if stds is not None:
            stds = torch.as_tensor(stds, dtype=torch.float32)
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.normalize = normalize
        self.denormalize = denormalize
        self.temporal = nn.Sequential(
            *(
                TemporalBlock(
                    input_channels=1 if level == 0 else hidden_channels,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=self.dilations[level],
                    dropout=dropout,
                )
                for level in range(levels)
            )
        )
        feature_size = hidden_channels * lookback
        self.task_heads = nn.ModuleList(
            [self._create_head(feature_size) for _ in range(self.task_count)]
        )
        self.auxiliary_head = self._create_head(feature_size)

    def _create_head(self, feature_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_size, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, self.horizon * 2),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, L, C) or (B, L + H, C)")
        if inputs.shape[1] not in (self.lookback, self.lookback + self.horizon):
            raise ValueError(
                f"expected sequence length {self.lookback} or "
                f"{self.lookback + self.horizon}, got {inputs.shape[1]}"
            )
        if self.normalize and self.means is not None and self.stds is not None:
            inputs = (inputs - self.means) / self.stds
        lookback = inputs[:, : self.lookback, :]
        batch_size, _, channels = lookback.shape
        if channels < self.task_count:
            raise ValueError(f"expected at least {self.task_count} target channels")
        features = lookback.transpose(1, 2).reshape(batch_size * channels, 1, self.lookback)
        features = self.temporal(features).flatten(start_dim=1)
        features = features.reshape(batch_size, channels, -1)
        predictions = [head(features[:, channel]) for channel, head in enumerate(self.task_heads)]
        if channels > self.task_count:
            auxiliary_features = features[:, self.task_count :].reshape(
                batch_size * (channels - self.task_count), -1
            )
            auxiliary_predictions = self.auxiliary_head(auxiliary_features).reshape(
                batch_size, channels - self.task_count, self.horizon * 2
            )
            predictions.append(auxiliary_predictions)
        predictions = torch.cat(
            [
                prediction.unsqueeze(1) if prediction.ndim == 2 else prediction
                for prediction in predictions
            ],
            dim=1,
        )
        means, raw_scales = predictions.chunk(2, dim=-1)
        means = means.transpose(1, 2)
        raw_scales = raw_scales.transpose(1, 2)
        standard_deviations = F.softplus(raw_scales) + self.min_std
        log_variances = 2 * torch.log(standard_deviations)
        if self.denormalize and self.means is not None and self.stds is not None:
            means = means * self.stds + self.means
            log_variances = log_variances + 2 * torch.log(self.stds)
        return means, log_variances
