import torch
from jaxtyping import Float
from torch import nn

from util.config import Config

from .util import MLP


class BaseDecoder(nn.Module):
    config: Config

    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = config

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch context observable_dim"]:
        """
        Given a sequence in the latent space, return the corresponding sequence in the observable space
        """
        raise NotImplementedError


class IdentityDecoder(BaseDecoder):
    def __init__(self, observable_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = observable_dim

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch context observable_dim"]:
        return z


class InformedDecoder(BaseDecoder):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.sampling_positions = torch.tensor(
            self.config.SAMPLING_POSITIONS, dtype=torch.float32
        ).view(1, -1)
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self, z: Float[torch.Tensor, "batch context 3"]
    ) -> Float[torch.Tensor, "batch context observable_dim"]:
        z = z + 0 * self.dummy_param  # Ensure that model output requires grad
        self.sampling_positions = self.sampling_positions.to(z.device)
        # To align with the coordinate system in the observable space,
        # where the angle is defined relative to the stable equilibrium
        X = z[:, :, 0:1] * self.sampling_positions
        Y = z[:, :, 1:2] * self.sampling_positions
        vel = z[:, :, 2:] * self.sampling_positions
        ret = torch.cat([X, Y, vel], dim=-1)
        return ret


class UnstructuredDecoder(BaseDecoder):
    def __init__(self, latent_dim, observable_dim, hidden_dim, context, **kwargs):
        super().__init__(**kwargs)
        self.net = MLP(latent_dim * context, hidden_dim, observable_dim * context)

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch context observable_dim"]:
        B, F, _ = z.size()
        return self.net(z.view(B, -1)).view(B, F, -1)


class CNNDecoder(BaseDecoder):
    def __init__(self, latent_dim, observable_dim, hidden_dim, context, **kwargs):
        super().__init__(**kwargs)

        self.conv_args = {
            "kernel_size": context,  # Kernel is the whole trajectory segment
            "stride": 1,
            "padding": int(
                (context - 1) / 2
            ),  # Padding to ensure the output length is the same as input length
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }
        self.decoder = nn.Sequential(
            nn.Conv1d(
                latent_dim,
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

    def forward(
        self, z: Float[torch.Tensor, "batch context latent_dim"]
    ) -> Float[torch.Tensor, "batch context observable_dim"]:
        return self.decoder(z.permute(0, 2, 1)).permute(0, 2, 1)
