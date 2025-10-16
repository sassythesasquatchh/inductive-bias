import torch
from jaxtyping import Float
from torch import nn

from util.config import Config

from .util import MLP


class BaseDynamics(nn.Module):
    config: Config

    def __init__(self, config: Config = Config(), **kwargs):
        super().__init__()
        self.config = config

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch 1 latent_dim"]:
        """
        Given a sequence in the latent space, return a rollout of length `rollout_length`
        """
        raise NotImplementedError


class SymplecticPendulumSolver(nn.Module):
    def __init__(self, dt, g, l):
        super().__init__()
        self.dt = dt
        self.g = g
        self.l = l

    def forward(
        self, theta: Float[torch.Tensor, "..."], theta_dot: Float[torch.Tensor, "..."]
    ):
        with torch.enable_grad():
            theta = theta + theta_dot * self.dt
            theta_dot_dot = -(self.g / self.l) * torch.sin(theta)
            theta_dot = theta_dot + self.dt * theta_dot_dot
        return theta, theta_dot


class InformedDynamics(BaseDynamics):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.solver = SymplecticPendulumSolver(config.DT, config.GRAVITY, config.L)
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch 1 latent_dim"]:
        z = z + 0 * self.dummy_param  # Ensure that model output requires grad

        # Latent dimensions are interpreted as [x, y, theta_dot]
        # Only the last frame of context is used
        x = z[:, -1, 0:1]
        y = z[:, -1, 1:2]
        theta_dot = z[:, -1, 2:3]
        # theta = torch.atan2(y, x) + torch.pi / 2
        theta_from_x_axis = torch.atan2(y, x)
        theta_from_equilibrium = theta_from_x_axis + torch.pi / 2
        next_theta_from_equilibrium, next_theta_dot = self.solver(
            theta_from_equilibrium, theta_dot
        )
        next_theta_from_x_axis = next_theta_from_equilibrium - torch.pi / 2
        x = torch.cos(next_theta_from_x_axis)
        y = torch.sin(next_theta_from_x_axis)
        return torch.stack([x, y, next_theta_dot], dim=-1)


class HybridDynamics(BaseDynamics):
    def __init__(self, hidden_dim, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.solver = SymplecticPendulumSolver(config.DT, config.GRAVITY, config.L)
        self.correction = MLP(3, hidden_dim, 3, hidden_layers=2)

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch 1 latent_dim"]:
        # Only the last frame of context is used
        latent_state = z[:, -1, :]
        x = latent_state[:, 0:1]
        y = latent_state[:, 1:2]
        theta_dot = latent_state[:, 2:3]
        theta_from_x_axis = torch.atan2(y, x)
        theta_from_equilibrium = theta_from_x_axis + torch.pi / 2
        next_theta_from_equilibrium, next_theta_dot = self.solver(
            theta_from_equilibrium, theta_dot
        )
        next_theta_from_x_axis = next_theta_from_equilibrium - torch.pi / 2
        x = torch.cos(next_theta_from_x_axis)
        y = torch.sin(next_theta_from_x_axis)
        latent_state = torch.stack([x, y, next_theta_dot], dim=-1) + self.correction(
            latent_state
        )
        return latent_state


class UnstructuredDynamics(BaseDynamics):
    def __init__(self, latent_dim, hidden_dim, context, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.context = context
        self.step = MLP(latent_dim * context, hidden_dim, latent_dim)

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch 1 latent_dim"]:
        B, C, L = z.size()
        return self.step(z.reshape(B, C * L)).view(B, 1, L)
