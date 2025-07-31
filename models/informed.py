from torch import nn
import torch
from constants import *
from jaxtyping import Float
import ipdb
from math import floor

from .blocks import PointEncoder, PointDecoder


class SymplecticPendulumSolver(nn.Module):
    def __init__(self, dt, g=1, l=1):
        super(SymplecticPendulumSolver, self).__init__()
        self.dt = dt
        self.g = g
        self.l = l

    def forward(self, theta, theta_dot):
        with torch.enable_grad():
            theta = theta + theta_dot * self.dt
            theta_dot_dot = -(self.g / self.l) * torch.sin(theta)
            theta_dot = theta_dot + self.dt * theta_dot_dot
        return theta, theta_dot


class InformedDynamics(nn.Module):
    def __init__(self, dt, g=1, l=1):
        super(InformedDynamics, self).__init__()
        self.solver = SymplecticPendulumSolver(dt, g, l)

    def forward(self, z: Float[torch.Tensor, "batch 3"], k: int = 1)->Float[torch.Tensor, "batch 3"]:
        """
        Args:
            z (torch.Tensor): Current state (batch_size,3)
            k (int): Number of steps to integrate
        """
        # x,y are interpreted as points on the unit circle,
        # they are used to provide a continuous representation of the
        # angular position
        x = z[:, 0]
        y = z[:, 1]
        theta_dot = z[:, 2]
        # Convert to the pendulum coordinate system
        theta = torch.atan2(y, x) + torch.pi / 2
        for _ in range(k):
            theta, theta_dot = self.solver(theta, theta_dot)
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y, theta_dot], dim=1)


class InformedEncoder(nn.Module):
    def __init__(self, observable_dim=12):
        super(InformedEncoder, self).__init__()

    def forward(self, x: torch.Tensor):
        # x is a tensor of shape (batch_size, observable_dim)
        # The first sampling position is at l=1
        X = x[:, 0]
        Y = x[:, 4]
        # At l = 1, the angular velocity is equal to the linear velocity
        vel = x[:, 8]
        return torch.stack([X, Y, vel], dim=1)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        observable_dim,
        hidden_dim,
        latent_channels=2,
        context=33,
    ):
        super(CNNEncoder, self).__init__()
        self.conv_args = {
            "kernel_size": context,  # Kernel is the whole trajectory segment
            "stride": 1,
            "padding": int(
                (context / 2)
            ),  # Padding to ensure the output length is the same as input length
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }

        assert floor(((context + 2*self.conv_args["padding"] - (self.conv_args["kernel_size"]-1)-1)/self.conv_args["stride"])+1)==context, \
            f"Context {context} does not match the expected output length after convolution. Please adjust the context or padding."
        # NOTE in chenhao's implementation, the final batch norm and elu are included
        self.encoder = nn.Sequential(
            nn.Conv1d(
                observable_dim,
                hidden_dim,
                **self.conv_args,
            ),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ELU(),
            nn.Conv1d(
                hidden_dim,
                latent_channels,
                **self.conv_args,
            ),
            # nn.BatchNorm1d(num_features=latent_dim),
            # nn.ELU(),
        )

        self.angle_encoder = nn.Linear(context, 2)
        self.angular_velocity_encoder = nn.Linear(context, 1)

    def forward(self, x: Float[torch.Tensor, "batch observable_dim context"])->Float[torch.Tensor, "batch 3"]:
        # Input x is of shape (batch_size, observable_dim, context)
        # z is of shape (batch_size, latent_channels, context)
        z = self.encoder(x)
        # Interpret the first latent channel as the angle (represented as a 2D vector)
        angle = self.angle_encoder(z[:, 0, :])
        # Interpret the second latent channel as the angular velocity
        angular_velocity = self.angular_velocity_encoder(z[:, 1, :])

        # Output has dimensions (batch_size, 3)
        return torch.cat((angle, angular_velocity), dim=1)


class CNNDecoder(nn.Module):
    def __init__(
        self,
        observable_dim,
        hidden_dim,
        latent_channels=3,
        forecast=9,
    ):
        super(CNNDecoder, self).__init__()
        self.conv_args = {
            "kernel_size": forecast,  # Kernel is the whole trajectory segment
            "stride": 1,
            "padding": int(
                (forecast - 1) / 2
            ),  # Padding to ensure the output length is the same as input length
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }
        self.decoder = nn.Sequential(
            nn.Conv1d(
                latent_channels,
                hidden_dim,
                **self.conv_args,
            ),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ELU(),
            nn.Conv1d(
                hidden_dim,
                observable_dim,
                **self.conv_args,
            ),
        )

    def forward(self, x:Float[torch.Tensor, "batch latent_channels forecast"])->Float[torch.Tensor, "batch observable_dim forecast"]:
        # Input x is of shape (batch_size, 3, forecast)
        z = self.decoder(x)
        # Output z is of shape (batch_size, observable_dim, forecast)
        return z


class InformedDecoder(nn.Module):
    def __init__(self, sampling_positions=SAMPLING_POSITIONS):
        super(InformedDecoder, self).__init__()
        self.sampling_positions = torch.tensor(
            sampling_positions, dtype=torch.float32
        ).view(1, -1)

    def forward(self, x: Float[torch.Tensor, "batch forecast 3"])->Float[torch.Tensor, "batch forecast observable_dim"]:

        # The first sampling position is at l=1

        self.sampling_positions = self.sampling_positions.to(x.device)
        # To align with the coordinate system in the observable space,
        # where the angle is defined relative to the stable equilibrium
        X = x[:, :,1:2] * self.sampling_positions
        Y = x[:,:, 0:1] * -self.sampling_positions
        vel = x[:,:, 2:] * self.sampling_positions
        ret = torch.cat([X, Y, vel], dim=-1)
        return ret


class Informed(nn.Module):
    def __init__(self, dt, g=GRAVITY, l=L, observable_dim=12, hidden_dim=30, **kwargs):
        super(Informed, self).__init__()
        self.encoder = PointEncoder(observable_dim, hidden_dim, 3)
        self.dynamics = InformedDynamics(dt, g, l)
        self.decoder = PointDecoder(3, hidden_dim, observable_dim)

    def forward(self, x: torch.Tensor, k: int = 1):
        z = self.encoder(x)
        z = self.dynamics(z, k)
        return self.decoder(z)


class FullyInformed(nn.Module):
    def __init__(self, dt, g=GRAVITY, l=L, observable_dim=12, hidden_dim=30, **kwargs):
        super(FullyInformed, self).__init__()
        self.encoder = InformedEncoder()
        self.dynamics = InformedDynamics(dt, g, l)
        self.decoder = InformedDecoder()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z = self.dynamics(z)
        return self.decoder(z.unsqueeze(0)).squeeze()  # Add a dimension for forecast length


class InformedEnd2End(nn.Module):
    def __init__(self, dt, g=GRAVITY, l=L, observable_dim=12, hidden_dim=30, **kwargs):
        super(InformedEnd2End, self).__init__()
        self.encoder = PointEncoder(observable_dim, hidden_dim, 3)
        self.dynamics = InformedDynamics(dt, g, l)
        self.decoder = PointDecoder(3, hidden_dim, observable_dim)

    def forward(self, x: torch.Tensor, k: int = 1):
        for i in range(k):
            x = self.encoder(x)
            x = self.dynamics(x)
            x = self.decoder(x)
        return x


class InformedRWM(nn.Module):
    def __init__(
        self,
        dt,
        g=GRAVITY,
        l=L,
        observable_dim=12,
        hidden_dim=30,
        context=32,
        forecast=8,
        **kwargs
    ):
        super(InformedRWM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observable_dim * context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * context),
        )
        self.dynamics = InformedDynamics(dt, g, l)
        self.decoder = nn.Sequential(
            nn.Linear(3 * (forecast + context), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observable_dim * forecast),
        )
        self.context = context
        self.forecast = forecast
        self.latent_dim = 3

    def forward(self, x: torch.Tensor, k: int = 1):
        B = x.size(0)
        z_forecast = torch.zeros(
            (B, self.forecast, self.latent_dim), dtype=x.dtype, device=x.device
        )
        z_context = self.encoder(x)
        z = z_context.view(B, self.context, 3)[:, -1, :].view(B, 3)

        for i in range(self.forecast):
            z = self.dynamics(z)
            z_forecast[:, i, :] = z

        z_total = torch.cat(
            (z_context.view(B, self.context, 3), z_forecast), dim=1
        ).view(B, -1)
        x = self.decoder(z_total)
        x = x.view(B, self.forecast, -1)
        return x


class InformedCNN(nn.Module):
    def __init__(
        self,
        dt,
        g=GRAVITY,
        l=L,
        observable_dim=12,
        hidden_dim=64,
        # latent_channels=3,
        context=33,
        forecast=9,
    ):
        super(InformedCNN, self).__init__()
        self.encoder = CNNEncoder(
            observable_dim, hidden_dim, latent_channels=2, context=context
        )
        self.dynamics = InformedDynamics(dt=dt, g=g, l=l)
        self.decoder = CNNDecoder(observable_dim, hidden_dim=3, forecast=forecast)

        self.forecast = forecast
        self.context = context
        self.latent_dim = 3

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim" ], *args, **kwargs)->Float[torch.Tensor, "batch forecast observable_dim"]:
        B = x.size(0)
        z_forecast = torch.zeros(
            (B, self.forecast, self.latent_dim), dtype=x.dtype, device=x.device
        )
        z = self.encoder(x)

        for i in range(self.forecast):
            z = self.dynamics(z)
            z_forecast[:, i, :] = z
        return self.decoder(z_forecast.transpose(1, 2)).transpose(1, 2)

class PartiallyInformed(nn.Module):
    """
    Generic encoder, canonical dynamics and canonical decoder
    """
    def __init__(self, dt, g=GRAVITY, l=L, observable_dim=12, hidden_dim=64, context=33, forecast=8, **kwargs):
        super(PartiallyInformed, self).__init__()
        self.encoder = CNNEncoder(
            observable_dim, hidden_dim, latent_channels=2, context=context
        )
        self.dynamics = InformedDynamics(dt, g, l)
        self.decoder = InformedDecoder()
        self.forecast = forecast
        self.context = context
        self.latent_dim = 3

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim"])->Float[torch.Tensor, "batch forecast observable_dim"]:
        B = x.size(0)
        z_forecast = torch.zeros(
            (B, self.forecast, self.latent_dim), dtype=x.dtype, device=x.device
        )
        # z has shape (batch_size, 3)
        z = self.encoder(x)

        for i in range(self.forecast):
            z = self.dynamics(z)
            z_forecast[:, i, :] = z

        return self.decoder(z_forecast)

class HybridDynamics(nn.Module):
    def __init__(self, dt, g=1, l=1, hidden_dim=30):
        super(HybridDynamics, self).__init__()
        self.solver = SymplecticPendulumSolver(dt, g, l)
        self.correction = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )

    def forward(self, z: torch.Tensor, k: int = 1):
        """
        Args:
            z (torch.Tensor): Current state (batch_size,3)
            k (int): Number of steps to integrate
        """
        # x,y are interpreted as points on the unit circle,
        # they are used to provide a continuous representation of the
        # angular position
        # The learnable correction is applied to the continuous representation
        for _ in range(k):
            x = z[:, 0]
            y = z[:, 1]
            theta_dot = z[:, 2]
            theta = torch.atan2(y, x)
            theta, theta_dot = self.solver(theta, theta_dot)
            x = torch.cos(theta)
            y = torch.sin(theta)
            z = torch.stack([x, y, theta_dot], dim=1)
            z = z + self.correction(z)
        return z


class HybridPartiallyInformed(nn.Module):
    """
    Generic encoder, canonical dynamics and canonical decoder
    """
    def __init__(self, dt, g=GRAVITY, l=L, observable_dim=12, hidden_dim=64, context=33, forecast=8, **kwargs):
        super(HybridPartiallyInformed, self).__init__()
        self.encoder = CNNEncoder(
            observable_dim, hidden_dim, latent_channels=2, context=context
        )
        self.dynamics = HybridDynamics(dt, g, l, hidden_dim)
        self.decoder = InformedDecoder()
        self.forecast = forecast
        self.context = context
        self.latent_dim = 3

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim"])->Float[torch.Tensor, "batch forecast observable_dim"]:
        B = x.size(0)
        z_forecast = torch.zeros(
            (B, self.forecast, self.latent_dim), dtype=x.dtype, device=x.device
        )
        # z has shape (batch_size, 3)
        z = self.encoder(x)

        for i in range(self.forecast):
            z = self.dynamics(z)
            z_forecast[:, i, :] = z

        return self.decoder(z_forecast)