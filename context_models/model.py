import torch
from jaxtyping import Float

from common.classes import BaseModel, RolloutOutput

from . import encoders


class ContextModel(BaseModel):
    """
    Generic encoder -> dynamics -> decoder pipeline.
    """

    def __init__(
        self, encoder: encoders.BaseEncoder, dynamics, decoder, forecast: int, **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.forecast = forecast

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch forecast observable_dim"]:
        B, N, D = x.size()
        # trajectory_length = output_length or N
        trajectory_length = N + self.forecast
        context_length = self.encoder.get_context_length()

        reconstructed_obs_end_to_end = torch.zeros(
            B, trajectory_length, D, device=x.device, dtype=x.dtype
        )

        latent_end_to_end = torch.zeros(
            B,
            trajectory_length,
            self.encoder.get_latent_dim(),
            device=x.device,
            dtype=x.dtype,
        )

        # Initialise result buffers
        initial_obs_context = x[:, :context_length, :].clone()
        reconstructed_obs_end_to_end[:, :context_length, :] = initial_obs_context
        latent_end_to_end[:, :context_length, :] = self.extract_latent(
            initial_obs_context
        ).clone()

        for i in range(0, trajectory_length - context_length):
            end_to_end_obs_context = reconstructed_obs_end_to_end[
                :, i : i + context_length, :
            ].clone()
            context_latent = self.extract_latent(end_to_end_obs_context).clone()
            next_latent = self.dynamics(context_latent)[:, -1, :]
            latent_end_to_end[:, i + context_length, :] = next_latent

            # Transform back to observable space
            latent_end_to_end_context = latent_end_to_end[
                :, i + 1 : i + context_length + 1
            ].clone()
            next_reconstructed = self.decoder(latent_end_to_end_context)[:, -1, :]
            reconstructed_obs_end_to_end[:, i + context_length, :] = next_reconstructed

        return reconstructed_obs_end_to_end[:, x.size(1) :, :]

    def extract_latent(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch context latent_dim"]:
        return self.encoder(x)

    def rollout(
        self,
        x: Float[torch.Tensor, "batch sequence_length observable_dim"],
        # output_length: Optional[int] = None,
    ) -> RolloutOutput:
        """
        x is a full trajectory, of which only the first `context_length` frames are used as context.
        The model then generates a reconstructed trajectory of the same length as x.

        If output_length is specified, the model will generate a trajectory of that length instead.

        In either case, the first `context_length` frames of the output will be identical to the input.
        """
        B, N, D = x.size()
        # trajectory_length = output_length or N
        trajectory_length = N
        context_length = self.encoder.get_context_length()
        reconstructed_obs_latent_rollout = torch.zeros(
            B, trajectory_length, D, device=x.device, dtype=x.dtype
        )
        reconstructed_obs_end_to_end = torch.zeros(
            B, trajectory_length, D, device=x.device, dtype=x.dtype
        )
        latent_gt = torch.zeros(
            B,
            trajectory_length,
            self.encoder.get_latent_dim(),
            device=x.device,
            dtype=x.dtype,
        )
        latent_rollout = torch.zeros(
            B,
            trajectory_length,
            self.encoder.get_latent_dim(),
            device=x.device,
            dtype=x.dtype,
        )
        latent_end_to_end = torch.zeros(
            B,
            trajectory_length,
            self.encoder.get_latent_dim(),
            device=x.device,
            dtype=x.dtype,
        )

        # Initialise result buffers
        initial_obs_context = x[:, :context_length, :].clone()
        reconstructed_obs_latent_rollout[:, :context_length, :] = initial_obs_context
        reconstructed_obs_end_to_end[:, :context_length, :] = initial_obs_context
        initial_latent_context = self.extract_latent(initial_obs_context).clone()
        latent_gt[:, :context_length, :] = initial_latent_context
        latent_rollout[:, :context_length, :] = initial_latent_context
        latent_end_to_end[:, :context_length, :] = initial_latent_context

        for i in range(0, trajectory_length - context_length):
            # Latent rollout
            obs_context = x[:, i : i + context_length, :].clone()
            latent_gt[:, i + context_length, :] = self.extract_latent(obs_context)[
                :, -1, :
            ]
            latent_rollout_context = latent_rollout[
                :, i : i + context_length, :
            ].clone()
            latent_rollout[:, i + context_length, :] = self.dynamics(
                latent_rollout_context
            )[:, -1, :]
            obs_end_to_end_context = reconstructed_obs_end_to_end[
                :, i : i + context_length, :
            ].clone()
            latent_end_to_end[:, i + context_length, :] = self.dynamics(
                self.extract_latent(obs_end_to_end_context).clone()
            )[:, -1, :]

            # Transform back to observable space
            latent_rollout_context = latent_rollout[
                :, i + 1 : i + context_length + 1, :
            ].clone()
            reconstructed_obs_latent_rollout[:, i + context_length, :] = self.decoder(
                latent_rollout_context
            )[:, -1, :]
            latent_end_to_end_context = latent_end_to_end[
                :, i + 1 : i + context_length + 1, :
            ].clone()
            reconstructed_obs_end_to_end[:, i + context_length, :] = self.decoder(
                latent_end_to_end_context
            )[:, -1, :]

        return RolloutOutput(
            obs_gt=x,
            obs_latent_rollout=reconstructed_obs_latent_rollout,
            obs_end_to_end=reconstructed_obs_end_to_end,
            latent_gt=latent_gt,
            latent_rollout=latent_rollout,
            latent_end_to_end=latent_end_to_end,
        )
