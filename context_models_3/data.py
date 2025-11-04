import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.config import Config


class ContextDataset(Dataset):
    def __init__(self, data_path: Path, context=33, forecast=8, noise_level=0.01):
        data = pickle.load(open(data_path, "rb"))
        self.data = data["trajectories"]
        self.config = Config(**data["simulation_config"])
        self.context = context
        self.forecast = forecast
        self.noise_level = noise_level
        self.trajectory_length = self.data[0]["observed"].shape[0]
        self.std_dev = torch.std(
            torch.tensor(self.data[0]["observed"], dtype=torch.float32),
            dim=0,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx % (len(self.data))
        start_idx = random.randint(
            a=0,
            b=self.trajectory_length - self.context - self.forecast,
        )
        X_t = torch.tensor(
            self.data[idx]["observed"][start_idx : start_idx + self.context, :],
            dtype=torch.float32,
        )

        X_tk = torch.tensor(
            self.data[idx]["observed"][
                start_idx + self.context - 1 : start_idx
                + self.context
                + self.forecast
                - 1,
                :,
            ],
            dtype=torch.float32,
        )

        if self.noise_level > 0:
            noise = torch.randn_like(X_t) * self.noise_level * self.std_dev
            X_t = X_t + noise

        return X_t, X_tk

    def get_observable_dim(self) -> int:
        return self.data[0]["observed"].shape[-1]


if __name__ == "__main__":
    data_path = Path("data/normal_training_5.pkl")
    dataset = ContextDataset(data_path)

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
