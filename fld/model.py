from typing import Tuple

import torch
from jaxtyping import Float
from torch import nn

from common.classes import BaseModel, RolloutOutput


class FLDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_dim,
                hidden_dim,
                sequence_length,
                stride=1,
                padding=int((sequence_length - 1) / 2),
            ),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                latent_dim,
                sequence_length,
                stride=1,
                padding=int((sequence_length - 1) / 2),
            ),
        )
        self.phase_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sequence_length, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(latent_dim)
            ]
        )
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

    def get_latent_dim(self) -> int:
        return self.latent_dim

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Tuple[
        Float[torch.Tensor, "batch latent_dim"],
        Float[torch.Tensor, "batch latent_dim"],
        Float[torch.Tensor, "batch latent_dim"],
        Float[torch.Tensor, "batch latent_dim"],
    ]:
        # Convolve the input sequence to get a latent trajectory
        z_traj = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Compute the discrete Fourier transform of the latent trajectory
        rfft = torch.fft.rfft(z_traj, dim=1)

        # Exclude constant offset
        amplitude_spectrum = rfft.abs()[:, 1:, :]
        power = amplitude_spectrum**2

        # Get the frequencies corresponding to the FFT bins, excluding constant offset
        freq_bins = torch.fft.rfftfreq(z_traj.size(1))[1:]
        frequency = torch.sum(freq_bins.reshape(1, -1, 1) * power, dim=1) / torch.sum(
            power, dim=1
        )
        amplitude = 2 * torch.sqrt(torch.sum(power, dim=1)) / z_traj.size(1)

        # Compute the constant offset
        offset = rfft.real[:, 0, :] / z_traj.size(1)

        intermediate_angles = torch.zeros(
            (z_traj.size(0), self.latent_dim, 2), device=z_traj.device
        )
        # For each latent channel, encode the phase using a (separate) small neural network
        for i in range(self.latent_dim):
            intermediate_angles[:, i, :] = self.phase_encoders[i](z_traj[:, :, i])
        phase = torch.atan2(intermediate_angles[:, :, 1], intermediate_angles[:, :, 0])
        return amplitude, phase, frequency, offset


class FLDDynamics(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        frequencies: Float[torch.Tensor, "batch latent_dim"],
        phases: Float[torch.Tensor, "batch latent_dim"],
        num_steps: int,
    ) -> Float[torch.Tensor, "batch num_steps latent_dim"]:
        output = torch.zeros(
            phases.size(0),
            num_steps,
            phases.size(1),
            device=phases.device,
            dtype=phases.dtype,
        )
        for t in range(num_steps):
            phases = phases + 2 * torch.pi * frequencies
            output[:, t] = phases
        return output


class FLDDecoder(nn.Module):
    def __init__(
        self, latent_dim, observable_dim, hidden_dim, sequence_length, **kwargs
    ):
        super().__init__()
        self.conv_args = {
            "kernel_size": sequence_length,  # Kernel is the whole trajectory segment
            "stride": 1,
            "padding": int(
                (sequence_length - 1) / 2
            ),  # Padding to ensure the output length is the same as input length
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, **self.conv_args),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, observable_dim, **self.conv_args),
        )
        self.sequence_length = sequence_length

    def forward(
        self,
        amplitudes: Float[torch.Tensor, "batch latent_dim"],
        offsets: Float[torch.Tensor, "batch latent_dim"],
        phases: Float[torch.Tensor, "batch sequence_length latent_dim"],
    ) -> Float[torch.Tensor, "batch sequence_length observable_dim"]:
        z = amplitudes[:, None, :] * torch.sin(phases) + offsets[:, None, :]
        return self.decoder(z.permute(0, 2, 1)).permute(0, 2, 1)


class FLD(BaseModel):
    def __init__(
        self, observable_dim, hidden_dim, latent_dim, segment_length, forecast, **kwargs
    ):
        super(FLD, self).__init__()
        self.encoder = FLDEncoder(
            observable_dim, hidden_dim, latent_dim, segment_length
        )
        self.dynamics = FLDDynamics()
        self.decoder = FLDDecoder(
            latent_dim, observable_dim, hidden_dim, segment_length
        )
        self.forecast = forecast

    def forward(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch forecast observable_dim segment_length"]:
        a, phase, f, b = self.encoder(x)
        output = torch.zeros(
            (x.size(0), self.forecast, x.size(1), x.size(2)), device=x.device
        )

        phase_trajectory = self.dynamics(f, phase, num_steps=x.size(1))

        for i in range(self.forecast):
            last_phase = phase_trajectory[:, -1, :].clone()
            phase_trajectory = torch.concatenate(
                (phase_trajectory[:, 1:, :], self.dynamics(f, last_phase, num_steps=1)),
                dim=1,
            )
            output[:, i, :, :] = self.decoder(a, b, phase_trajectory)
        return output

    def extract_latent(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> Float[torch.Tensor, "batch context latent_dim"]:
        _, phase, f, _ = self.encoder(x)
        phase_traj = torch.concatenate(
            (phase.unsqueeze(1), self.dynamics(f, phase, x.size(1) - 1)), dim=1
        )
        return phase_traj

    def rollout(
        self, x: Float[torch.Tensor, "batch context observable_dim"]
    ) -> RolloutOutput:
        context_length = self.encoder.sequence_length
        B = x.size(0)
        trajectory_length = x.size(1)
        reconstructed_obs_latent_rollout = torch.zeros_like(x)
        reconstructed_obs_end_to_end = torch.zeros_like(x)
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
        reconstructed_obs_latent_rollout[:, :context_length, :] = x[
            :, :context_length, :
        ]
        reconstructed_obs_end_to_end[:, :context_length, :] = x[:, :context_length, :]
        latent_gt[:, :context_length, :] = self.extract_latent(x[:, :context_length, :])
        latent_end_to_end[:, :context_length, :] = latent_gt[:, :context_length, :]

        a, phase, f, b = self.encoder(x[:, :context_length, :])

        latent_rollout = torch.concatenate(
            (phase.unsqueeze(1), self.dynamics(f, phase, trajectory_length - 1)), dim=1
        )

        for i in range(0, trajectory_length - context_length):
            # Latent rollout
            latent_gt[:, i + context_length, :] = self.extract_latent(
                x[:, i + 1 : i + context_length + 1, :]
            )[:, -1, :]

            a, phase, f, b = self.encoder(
                reconstructed_obs_end_to_end[:, i : i + context_length, :]
            )

            # The phase that is returned by the encoder is interpreted as the phase of the first frame in the sequence
            latent_end_to_end[:, i + context_length, :] = self.dynamics(
                f, phase, 1 + context_length
            )[:, -1, :]

            # Transform back to observable space
            reconstructed_obs_latent_rollout[:, i + context_length, :] = self.decoder(
                a, b, latent_rollout[:, i + 1 : i + context_length + 1, :]
            )[:, -1, :]
            reconstructed_obs_end_to_end[:, i + context_length, :] = self.decoder(
                a, b, latent_end_to_end[:, i + 1 : i + context_length + 1, :]
            )[:, -1, :]

        return RolloutOutput(
            obs_gt=x,
            obs_latent_rollout=reconstructed_obs_latent_rollout,
            obs_end_to_end=reconstructed_obs_end_to_end,
            latent_gt=latent_gt,
            latent_rollout=latent_rollout,
            latent_end_to_end=latent_end_to_end,
        )
