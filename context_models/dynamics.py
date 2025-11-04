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
        self, z: Float[torch.Tensor, "batch latent_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        """
        Given a sequence in the latent space, return a rollout of length `rollout_length`
        """
        raise NotImplementedError


class PendulumSolver(nn.Module):
    def __init__(
        self,
        dt: float,
        g: float,
        l: float,
        gamma_init: float = 0.0,
        learn_gamma: bool = False,
    ):
        """
        Solver for a (possibly damped) pendulum with optional learnable damping.

        Args:
            dt: Time step
            g: Gravitational acceleration
            l: Pendulum length
            gamma_init: Initial damping coefficient
            learn_gamma: If True, gamma becomes a learnable parameter
        """
        super().__init__()
        self.dt = dt
        self.g = g
        self.l = l

        if learn_gamma:
            # Make gamma a learnable parameter
            self.gamma = nn.Parameter(
                torch.rand_like(torch.tensor(gamma_init, dtype=torch.float32))
            )
        else:
            # Keep gamma fixed
            self.register_buffer("gamma", torch.tensor(gamma_init, dtype=torch.float32))

    def forward(
        self,
        theta: Float[torch.Tensor, "..."],
        theta_dot: Float[torch.Tensor, "..."],
    ):
        # Compute current acceleration
        theta_dot_dot = -(self.g / self.l) * torch.sin(theta) - self.gamma * theta_dot

        theta_new = theta + theta_dot * self.dt + 0.5 * theta_dot_dot * self.dt**2

        # Compute new acceleration
        theta_dot_dot_new = (
            -(self.g / self.l) * torch.sin(theta_new) - self.gamma * theta_dot
        )

        # Velocity update
        theta_dot_new = theta_dot + 0.5 * (theta_dot_dot + theta_dot_dot_new) * self.dt

        return theta_new, theta_dot_new


class InformedDynamics(BaseDynamics):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.solver = PendulumSolver(
            config.DT, config.GRAVITY, config.L, config.DAMPING, learn_gamma=False
        )
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self, z: Float[torch.Tensor, "batch latent_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        z = z + 0 * self.dummy_param  # Ensure that model output requires grad

        x = z[:, 0:1]
        y = z[:, 1:2]
        theta_dot = z[:, 2:3]

        theta_from_x_axis = torch.atan2(y, x)
        theta_from_equilibrium = theta_from_x_axis + torch.pi / 2
        next_theta_from_equilibrium, next_theta_dot = self.solver(
            theta_from_equilibrium, theta_dot
        )
        next_theta_from_x_axis = next_theta_from_equilibrium - torch.pi / 2
        x = torch.cos(next_theta_from_x_axis)
        y = torch.sin(next_theta_from_x_axis)
        return torch.concatenate([x, y, next_theta_dot], dim=-1)


class HybridDynamics(BaseDynamics):
    def __init__(self, hidden_dim, context, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.solver = PendulumSolver(
            config.DT, config.GRAVITY, config.L, 0, learn_gamma=False
        )
        self.context = context
        self.correction = MLP(3, hidden_dim, 3, hidden_layers=2)

    def forward(
        self, z: Float[torch.Tensor, "batch latent_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        latent_state = z
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
        latent_state = torch.cat([x, y, next_theta_dot], dim=-1) + self.correction(
            latent_state
        )
        return latent_state


class HybridDynamics2(BaseDynamics):
    def __init__(self, hidden_dim, context, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.solver = PendulumSolver(
            config.DT, config.GRAVITY, config.L, config.DAMPING, learn_gamma=True
        )
        self.context = context

    def forward(
        self, z: Float[torch.Tensor, "batch latent_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        x = z[:, 0:1]
        y = z[:, 1:2]
        theta_dot = z[:, 2:3]

        theta_from_x_axis = torch.atan2(y, x)
        theta_from_equilibrium = theta_from_x_axis + torch.pi / 2
        next_theta_from_equilibrium, next_theta_dot = self.solver(
            theta_from_equilibrium, theta_dot
        )
        next_theta_from_x_axis = next_theta_from_equilibrium - torch.pi / 2
        x = torch.cos(next_theta_from_x_axis)
        y = torch.sin(next_theta_from_x_axis)
        return torch.cat([x, y, next_theta_dot], dim=-1)


class UnstructuredDynamics(BaseDynamics):
    def __init__(self, latent_dim, hidden_dim, context, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.context = context
        self.step = MLP(latent_dim, hidden_dim, latent_dim)

    def forward(
        self, z: Float[torch.Tensor, "batch latent_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        # B, C, L = z.size()
        return self.step(z)
