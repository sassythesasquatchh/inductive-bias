import torch
from torch import nn
from jaxtyping import Float

from typing import TypedDict

class RolloutOutput(TypedDict):
    obs_gt: Float[torch.Tensor, "batch output_length observable_dim"]
    obs_latent_rollout: Float[torch.Tensor, "batch output_length observable_dim"]
    obs_end_to_end: Float[torch.Tensor, "batch output_length observable_dim"]
    latent_gt: Float[torch.Tensor, "batch output_length latent_dim"]
    latent_rollout: Float[torch.Tensor, "batch output_length latent_dim"]
    latent_end_to_end: Float[torch.Tensor, "batch output_length latent_dim"]

    def to_torch(self):
        """Converts jax arrays to torch tensors"""
        return RolloutOutput(
            obs_gt=torch.from_numpy(self["obs_gt"].numpy()),
            obs_latent_rollout=torch.from_numpy(self["obs_latent_rollout"].numpy()),
            obs_end_to_end=torch.from_numpy(self["obs_end_to_end"].numpy()),
            latent_gt=torch.from_numpy(self["latent_gt"].numpy()),
            latent_rollout=torch.from_numpy(self["latent_rollout"].numpy()),
            latent_end_to_end=torch.from_numpy(self["latent_end_to_end"].numpy()),
        )

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Float[torch.Tensor, "batch context observable_dim"]):
        """
        Function used for supervised training.
        """
        raise NotImplementedError

    def extract_latent(self, x: Float[torch.Tensor, "batch observable_dim segment_length"])-> Float[torch.Tensor, "batch sequence_length latent_dim"]:
        """
        Given a sequence in the observable space, return the corresponding sequence in the latent space
        """
        raise NotImplementedError

    def rollout(self, x: Float[torch.Tensor, "batch context observable_dim"], output_length:int)-> RolloutOutput:
        """
        Given a sequence in the observable space, return a rollout of length `output_length`
        """
        raise NotImplementedError