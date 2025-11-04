from typing import TypedDict

import torch
from torch import nn

from common.classes import ForwardOutput


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
        # return torch.mean((pred - target) ** 2 * self.weights.view(1, -1, 1))
        return torch.mean((pred - target) ** 2)


class LossConfig(TypedDict):
    supervise_rollout: bool
    supervise_end_to_end: bool
    penalise_latent_magnitude: bool
    penalise_latent_mismatch: bool
    penalise_latent_dynamics: bool


class CombinedLoss(nn.Module):
    def __init__(self, config: LossConfig, rwm_loss: RWMLoss = RWMLoss()):
        super(CombinedLoss, self).__init__()
        self.rwm_loss = rwm_loss
        self.config = config

    def forward(self, x: ForwardOutput, gt_traj: torch.Tensor):
        loss = torch.tensor(0.0, device=gt_traj.device)
        if self.config["supervise_rollout"]:
            loss += self.rwm_loss(x["obs_rollout"], gt_traj)
        if self.config["supervise_end_to_end"]:
            loss += self.rwm_loss(x["obs_end_to_end"], gt_traj)
        if self.config["penalise_latent_magnitude"]:
            loss += torch.mean(x["latent_end_to_end"] ** 2)
            loss += torch.mean(x["latent_rollout"] ** 2)
        if self.config["penalise_latent_mismatch"]:
            loss += torch.mean((x["latent_end_to_end"] - x["latent_rollout"]) ** 2)
        if self.config["penalise_latent_dynamics"]:
            loss += torch.mean(
                (x["latent_rollout"][:, 1:, :] - x["latent_rollout"][:, :-1, :]) ** 2
            )
        return loss
