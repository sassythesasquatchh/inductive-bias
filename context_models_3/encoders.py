import torch
from jaxtyping import Float
from torch import nn

from util.config import Config

from .util import MLP


class BaseEncoder(nn.Module):
    config: Config

    def __init__(self, **kwargs):
        super().__init__()
        self.context = kwargs.get("context")
        self.config = kwargs.get("config")
        assert self.context is not None, (
            "context length must be specified for BaseEncoder"
        )
        assert self.config is not None, "config must be specified for BaseEncoder"

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        """
        Given a sequence in the observable space, return the corresponding sequence in the latent space
        """
        raise NotImplementedError

    def get_context_length(self) -> int:
        return self.context

    def get_latent_dim(self) -> int:
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = kwargs.get("observable_dim")
        assert self.latent_dim is not None, (
            "observable_dim must be specified for IdentityEncoder"
        )

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        return x[:, -1, :]

    def get_latent_dim(self) -> int:
        return self.latent_dim


class UnstructuredEncoder(BaseEncoder):
    def __init__(self, observable_dim, hidden_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.net = MLP(observable_dim * self.context, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        # assert x.size(1) == self.context, (
        #     f"Input sequence length {x.size(1)} does not match model context length {self.context}"
        # )

        B, C, D = x.size()

        return self.net(x.reshape(B, C * D))

    def get_latent_dim(self) -> int:
        return self.latent_dim


class InformedEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.furthest_point_index = self.config.SAMPLING_POSITIONS.index(
            max(self.config.SAMPLING_POSITIONS)
        )
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch 3"]:
        """
        Extracts the x position, y position and angular velocity at the furthest point along the pendulum
        """
        x = x + 0 * self.dummy_param  # Ensure that model output requires grad
        X = x[:, -1, self.furthest_point_index].unsqueeze(-1)
        Y = x[:, -1, self.furthest_point_index + self.config.NUM_POINTS].unsqueeze(-1)
        ang_vel = (
            x[:, -1, self.furthest_point_index + 2 * self.config.NUM_POINTS]
            / self.config.L
        ).unsqueeze(-1)
        return torch.cat([X, Y, ang_vel], dim=-1)

    def get_latent_dim(self) -> int:
        return 3


# class CNNEncoder(BaseEncoder):
#     def __init__(self, observable_dim, hidden_dim, latent_channels=2, **kwargs):
#         super().__init__(**kwargs)
#         self.conv_args = {
#             "kernel_size": self.context,
#             "stride": 1,
#             "padding": int(self.context // 2),
#             "groups": 1,
#             "bias": True,
#             "padding_mode": "zeros",
#         }
#         self.encoder = nn.Sequential(
#             nn.Conv1d(observable_dim, hidden_dim, **self.conv_args),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ELU(),
#             nn.Conv1d(hidden_dim, latent_channels, **self.conv_args),
#             nn.BatchNorm1d(latent_channels),
#             nn.ELU(),
#         )
#         # self.angle_encoder = nn.Linear(self.context, 2 * self.context)
#         self.angle_encoder = nn.Linear(self.context, 2)
#         # self.angle_encoder = MLP(1, hidden_dim, 2, 1)
#         # self.angular_velocity_encoder = nn.Linear(self.context, 1 * self.context)
#         self.angular_velocity_encoder = nn.Linear(self.context, 1)
#         # self.angular_velocity_encoder = MLP(1, hidden_dim, 1, 1)

#     def forward(
#         self, x: Float[torch.Tensor, "batch context observable_dim"]
#     ) -> Float[torch.Tensor, "batch 3"]:
#         assert x.size(1) == self.context, (
#             f"Input sequence length {x.size(1)} does not match model context length {self.context}"
#         )
#         z = self.encoder(x.permute(0, 2, 1))

#         B, C, _ = z.size()
#         # Interpret the first channel as angular position
#         # angle = self.angle_encoder(z[:, 0, :].reshape(B * C, -1)).reshape(B, C, 2)
#         angle = self.angle_encoder(z[:, 0, :])
#         # Interpret the second channel as angular velocity
#         # angular_velocity = self.angular_velocity_encoder(
#         #     z[:, 1, :].reshape(B * C, -1)
#         # ).reshape(B, C, 1)
#         angular_velocity = self.angular_velocity_encoder(z[:, 1, :])
#         return torch.cat((angle, angular_velocity), dim=-1)

#     def get_latent_dim(self) -> int:
#         return 3


class CNNEncoder(BaseEncoder):
    def __init__(self, observable_dim, hidden_dim, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.conv_args = {
            "kernel_size": self.context,
            "stride": 1,
            "padding": int(self.context // 2),
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }
        self.encoder = nn.Sequential(
            nn.Conv1d(observable_dim, hidden_dim, **self.conv_args),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, latent_dim, **self.conv_args),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
        )
        self.latent_dim = latent_dim
        # self.angle_encoder = nn.Linear(self.context, 2 * self.context)
        # self.angle_encoder = nn.Linear(self.context, 2)
        # self.angle_encoder = MLP(1, hidden_dim, 2, 1)
        # self.angular_velocity_encoder = nn.Linear(self.context, 1 * self.context)
        # self.angular_velocity_encoder = nn.Linear(self.context, 1)
        # self.angular_velocity_encoder = MLP(1, hidden_dim, 1, 1)
        self.reducers = nn.ModuleList(
            [nn.Linear(self.context, 1) for _ in range(latent_dim)]
        )

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch latent_dim"]:
        assert x.size(1) == self.context, (
            f"Input sequence length {x.size(1)} does not match model context length {self.context}"
        )
        z = self.encoder(x.permute(0, 2, 1))

        B, D, C = z.size()
        # Interpret the first channel as angular position
        # angle = self.angle_encoder(z[:, 0, :].reshape(B * C, -1)).reshape(B, C, 2)
        # angle = self.angle_encoder(z[:, 0, :])
        # Interpret the second channel as angular velocity
        # angular_velocity = self.angular_velocity_encoder(
        #     z[:, 1, :].reshape(B * C, -1)
        # ).reshape(B, C, 1)
        # angular_velocity = self.angular_velocity_encoder(z[:, 1, :])

        # return torch.cat((angle, angular_velocity), dim=-1)

        reduced_channels = []
        for i in range(D):
            reduced_channel = self.reducers[i](z[:, i, :])
            reduced_channels.append(reduced_channel)
        return torch.cat(reduced_channels, dim=-1)

    def get_latent_dim(self) -> int:
        return self.latent_dim
