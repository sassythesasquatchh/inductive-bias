import torch
from torch import nn
from jaxtyping import Float
from typing import Tuple

T = torch.Tensor


class FLDEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, sequence_length):
        super(FLDEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_dim,
                hidden_dim,
                sequence_length,
                stride=1,
                padding=int((sequence_length - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                embedding_dim,
                sequence_length,
                stride=1,
                padding=int((sequence_length - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self.phase_encoders = [nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ) for _ in range(embedding_dim)]
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

    def forward(self, s: Float[torch.Tensor, "batch context observable_dim"]) -> Tuple[
        Float[torch.Tensor, "batch embedding_dim"],
        Float[torch.Tensor, "batch embedding_dim"],
        Float[torch.Tensor, "batch embedding_dim"],
        Float[torch.Tensor, "batch embedding_dim"]
    ]:
        """
        Encodes the input trajectory into a lower dimensional space,
        then calculates the amplitude, phase, frequency, and offset
        for each channel of each trajectory in the batch.

        Args:
            x (torch.Tensor): Trajectory (batch_size, sequence_length, state_dim)

        Returns:
            amplitude (torch.Tensor): RMS amplitude of the signal (batch_size, embedding_dim)
            phase (torch.Tensor): Phase of the central state (batch_size, embedding_dim)
            frequency (torch.Tensor): Dominant frequency of the signal (batch_size, embedding_dim)
            offset (torch.Tensor): DC offset of the signal (batch_size, embedding_dim)
        """

        # B = batch size
        # N = sequence length
        # d = state dimension
        # p = embedding dimension

        s = s.permute(0, 2, 1)  # (batch, observable_dim, context)

        # (b, observable_dim, context) -> (batch, latent_dim, context)
        z_traj = self.encoder(s)

        z_traj = z_traj.permute(0, 2, 1)  # (batch, context, latent_dim)

        # FFT along the sequence dimension
        rfft = torch.fft.rfft(z_traj, dim=1)

        # Exclude the DC component
        amplitude_spectrum = rfft.abs()[:, :, 1:]
        power = amplitude_spectrum**2

        freq_bins = torch.fft.rfftfreq(z_traj.size(1))[1:]

        # Obtain the signal frequency by summing the contributions of each frequency bin
        # weighted by its power
        frequency = torch.sum(freq_bins * power, dim=1) / torch.sum(power, dim=1)

        amplitude = 2 * torch.sqrt(torch.sum(power, dim=1)) / z_traj.size(1)

        offset = rfft.real[:, :, 0] / z_traj.size(1)

        # (batch, latent_dim, 2)
        intermediate_angles = torch.zeros((z_traj.size(0), self.embedding_dim, 2), device=z_traj.device)

        for i in range(self.embedding_dim):
            intermediate_angles[:,i, :] = self.phase_encoders[i](z_traj[:,:,i])

        phase = torch.atan2(intermediate_angles[:,:, 1], intermediate_angles[:,:, 0])  # Shape: (batch, latent_dim)

        return amplitude, phase, frequency, offset


class FLDDynamics(nn.Module):
    def __init__(self):
        super(FLDDynamics, self).__init__()

    def forward(self, frequencies: Float[torch.Tensor, "batch embedding_dim"], phases: Float[torch.Tensor, "batch embedding_dim"], k: int)-> Float[torch.Tensor, "batch embedding_dim"]:
        """
        Args:
            frequencies (torch.Tensor): (batch_size, embedding_dim)
            phases (torch.Tensor): (batch_size, embedding_dim)
            k (int): Number of steps to integrate
        """

        # Assumes that the input frequencies are in cycles per sample
        for _ in range(k):
            phases = phases + frequencies

        return phases


class FLDDecoder(nn.Module):
    def __init__(self, observable_dim, hidden_dim, embedding_dim, forecast_length):
        super(FLDDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                hidden_dim,
                forecast_length,
                stride=1,
                padding=int((forecast_length - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                observable_dim,
                forecast_length,
                stride=1,
                padding=int((forecast_length - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self.forecast_length = forecast_length

    def forward(self, frequencies: Float[torch.Tensor, "batch embedding_dim"], amplitudes: Float[torch.Tensor, "batch embedding_dim"], offsets: Float[torch.Tensor, "batch embedding_dim"], phases: Float[torch.Tensor, "batch embedding_dim"])-> Float[torch.Tensor, "batch forecast_length observable_dim"]:
        """
        Decodes the amplitude, phase, frequency, and offset
        into the original trajectory space.

        Args:
            f (torch.Tensor): Frequency of the dominant frequency (batch_size, embedding_dim)
            a (torch.Tensor): Amplitude of the dominant frequency (batch_size, embedding_dim)
            b (torch.Tensor): Offset of the trajectory (batch_size, embedding_dim)
            phase (torch.Tensor): Phase of the dominant frequency (batch_size, embedding_dim)

        Returns:
            x (torch.Tensor): Trajectory (batch_size, observable_dim)
        """
        # B = batch size
        # N = sequence length
        # d = state dimension
        # p = embedding dimension

        # Timestamps (units are samples) (1, forecast_length, 1)
        times = torch.linspace(-self.forecast_length / 2, self.forecast_length / 2, self.forecast_length, device=frequencies.device).view(
            1, self.forecast_length, 1
        )

        # Compute the trajectory in the embedding space
        z = (
            amplitudes[:, None, :]
            * torch.sin(2 * torch.pi * (frequencies[:, None, :] * times + phases[:, None, :]))
            + offsets[:, None, :]
        )

        # Decode the trajectory into the original space
        x = self.decoder(z)

        return x


class FLD(nn.Module):
    def __init__(self, observable_dim, hidden_dim, embedding_dim, segment_length):
        super(FLD, self).__init__()
        self.encoder = FLDEncoder(
            observable_dim, hidden_dim, embedding_dim, segment_length
        )
        self.dynamics = FLDDynamics()
        self.decoder = FLDDecoder(
            observable_dim, hidden_dim, embedding_dim, segment_length
        )

    def forward(self, x: T, k: int):
        a, phase, f, b = self.encoder(x)
        phase = self.dynamics(f, phase, k)
        return self.decoder(f, a, b, phase)
