from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(hidden_layers - 1)
            ],
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class RWMLoss(nn.Module):
    def __init__(self, alpha=0.9, forecast=8):
        super(RWMLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer(
            "weights", torch.tensor([alpha**k for k in range(forecast)])
        )

    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2 * self.weights.view(1, -1, 1))
