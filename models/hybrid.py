import torch
from torch import nn

from .informed import SymplecticPendulumSolver, InformedDynamics
from .blocks import PointEncoder, PointDecoder
from constants import GRAVITY, L


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


class Hybrid(nn.Module):
    def __init__(self, dt, g=1, l=1, observable_dim=12, hidden_dim=30, **kwargs):
        super(Hybrid, self).__init__()
        self.encoder = PointEncoder(observable_dim, hidden_dim, 3)
        self.dynamics = HybridDynamics(dt, g, l, hidden_dim)
        self.decoder = PointDecoder(3, hidden_dim, observable_dim)

    def forward(self, x: torch.Tensor, k: int = 1):
        z = self.encoder(x)
        z = self.dynamics(z, k)
        return self.decoder(z)


class HybridEnd2End(nn.Module):
    def __init__(self, dt, g=1, l=1, observable_dim=12, hidden_dim=30, **kwargs):
        super(HybridEnd2End, self).__init__()
        self.encoder = PointEncoder(observable_dim, hidden_dim, 3)
        self.dynamics = HybridDynamics(dt, g, l, hidden_dim)
        self.decoder = PointDecoder(3, hidden_dim, observable_dim)

    def forward(self, x: torch.Tensor, k: int = 1):
        for i in range(k):
            z = self.encoder(x)
            z = self.dynamics(z, k)
            x = self.decoder(z)
        return x


class HybridRWM(nn.Module):
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
        super(HybridRWM, self).__init__()
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
        self.correction = nn.Sequential(
            nn.Linear(3 * (context + forecast), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * (forecast + context)),
        )
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
        z_total = z_total + self.correction(z_total)
        x = self.decoder(z_total)
        x = x.view(B, self.forecast, -1)
        return x
