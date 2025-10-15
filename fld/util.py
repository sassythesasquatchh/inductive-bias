import torch
from torch import nn


class FLDLoss(nn.Module):
    def __init__(self, forecast=8, alpha=0.9):
        super(FLDLoss, self).__init__()
        self.forecast = forecast
        self.alpha = alpha
        self.register_buffer(
            "weights", torch.tensor([alpha**k for k in range(forecast)])
        )

    def forward(self, pred, target):
        # Pred and target have dimension (batch_size, forecast_length, segment_length, observable_dim)
        return torch.mean((pred - target) ** 2 * self.weights.view(1, -1, 1, 1))
