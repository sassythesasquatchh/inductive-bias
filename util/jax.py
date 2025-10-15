import jax
import numpy as np
import torch


def to_torch(arr: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.array(arr))
