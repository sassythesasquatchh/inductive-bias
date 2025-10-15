import pickle
from pathlib import Path

import jax.numpy as jnp
import torch
from torch.utils.data import Dataset

from util.config import Config


class BaseTrajectoryDataset(Dataset):
    def __init__(self, data_path: Path):
        data = pickle.load(open(data_path, "rb"))
        self.data = self._process_data(data)
        self.initial_velocities = self._parse_initial_conditions(data)
        self.config = Config(**data["simulation_config"])
        self.traj_names = data.get("names")

    def _process_data(self, data):
        raise NotImplementedError

    def _parse_initial_conditions(self, data):
        raise NotImplementedError


class TorchTrajectoryDataset(BaseTrajectoryDataset, Dataset):
    def __init__(self, data_path: Path, type="observed"):
        self.type = type
        super().__init__(data_path)

    def _process_data(self, data):
        return torch.concatenate(
            [
                torch.tensor(traj[self.type], dtype=torch.float32).unsqueeze(0)
                for traj in data["trajectories"]
            ],
            dim=0,
        )

    def _parse_initial_conditions(self, data):
        return torch.tensor(
            [traj["phase"][0, 1] for traj in data["trajectories"]], dtype=torch.float32
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class JaxTrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, data_path: Path, type="observed"):
        self.type = type
        super().__init__(data_path)

    def _process_data(self, data):
        return jnp.concatenate(
            [
                jnp.array(traj[self.type]).reshape(1, *traj[self.type].shape)
                for traj in data["trajectories"]
            ],
            axis=0,
        )

    def _parse_initial_conditions(self, data):
        return torch.tensor(
            [traj["phase"][0, 1] for traj in data["trajectories"]], dtype=torch.float32
        )

        # return jnp.array([traj["phase"][0, 1] for traj in data["trajectories"]], dtype=jnp.float32)
