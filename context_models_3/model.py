import torch
from jaxtyping import Float

from common.classes import BaseModel, ForwardOutput, RolloutOutput

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
        self.context = encoder.get_context_length()

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch forecast observable_dim"]:
        B, C, D = x.size()
        latent_dim = self.encoder.get_latent_dim()

        obs_latent_rollout = torch.zeros(
            B, self.forecast, D, device=x.device, dtype=x.dtype
        )
        obs_end_to_end = torch.zeros(
            B, self.forecast, D, device=x.device, dtype=x.dtype
        )

        latent_rollout = torch.zeros(
            B, self.forecast, latent_dim, device=x.device, dtype=x.dtype
        )
        latent_end_to_end = torch.zeros(
            B, self.forecast, latent_dim, device=x.device, dtype=x.dtype
        )

        obs_context = x.clone()
        latent_state = self.extract_latent(obs_context)
        latent_rollout_state = latent_state.clone()
        latent_end_to_end[:, 0, :] = latent_state
        latent_rollout[:, 0, :] = latent_state

        obs_latent_rollout[:, 0, :] = self.decoder(latent_state)
        obs_end_to_end[:, 0, :] = self.decoder(latent_state)
        for i in range(1, self.forecast):
            # Extract current latent state
            end_to_end_latent_state = self.extract_latent(obs_context)

            # Advect latent states from both paths
            latent_rollout_state = self.dynamics(latent_rollout_state)
            end_to_end_latent_state = self.dynamics(end_to_end_latent_state)

            latent_rollout[:, i, :] = latent_rollout_state
            latent_end_to_end[:, i, :] = end_to_end_latent_state

            # Add decoded observations to outputs
            obs_latent_rollout[:, i, :] = self.decoder(latent_rollout_state)
            next_obs_state = self.decoder(end_to_end_latent_state)
            obs_end_to_end[:, i, :] = next_obs_state

            # Prepare next observation context
            obs_context = torch.cat(
                [obs_context[:, 1:, :], next_obs_state.unsqueeze(1)], dim=1
            )

        # obs_output = 0.5 * (obs_latent_rollout + obs_end_to_end)

        return ForwardOutput(
            obs_rollout=obs_latent_rollout,
            obs_end_to_end=obs_end_to_end,
            latent_rollout=latent_rollout,
            latent_end_to_end=latent_end_to_end,
        )

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

        In either case, the first `context_length` frames of the output will be identical to the input.
        """
        B, N, D = x.size()
        trajectory_length = N - self.context + 1

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

        obs_context = x[:, : self.context, :]
        latent_state_rollout = self.extract_latent(obs_context)

        latent_gt[:, 0, :] = latent_state_rollout
        latent_end_to_end[:, 0, :] = latent_state_rollout
        latent_rollout[:, 0, :] = latent_state_rollout
        reconstructed_obs_latent_rollout[:, 0, :] = obs_context[:, -1, :]
        reconstructed_obs_end_to_end[:, 0, :] = obs_context[:, -1, :]

        for i in range(1, trajectory_length):
            latent_state_end_to_end = self.extract_latent(obs_context)

            latent_state_end_to_end = self.dynamics(latent_state_end_to_end)
            latent_state_rollout = self.dynamics(latent_state_rollout)
            latent_state_gt = self.extract_latent(x[:, i : i + self.context, :])

            next_obs_latent_rollout = self.decoder(latent_state_rollout)
            next_obs_end_to_end = self.decoder(latent_state_end_to_end)

            latent_gt[:, i, :] = latent_state_gt
            latent_end_to_end[:, i, :] = latent_state_end_to_end
            latent_rollout[:, i, :] = latent_state_rollout

            reconstructed_obs_latent_rollout[:, i, :] = next_obs_latent_rollout
            reconstructed_obs_end_to_end[:, i, :] = next_obs_end_to_end

            obs_context = torch.cat(
                [obs_context[:, 1:, :], next_obs_end_to_end.unsqueeze(1)], dim=1
            )

        return RolloutOutput(
            obs_gt=x[:, self.context - 1 :, :],
            obs_latent_rollout=reconstructed_obs_latent_rollout,
            obs_end_to_end=reconstructed_obs_end_to_end,
            latent_gt=latent_gt,
            latent_rollout=latent_rollout,
            latent_end_to_end=latent_end_to_end,
        )
