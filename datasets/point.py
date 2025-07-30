from torch.utils.data import Dataset
import torch
from pathlib import Path
import pickle
import ipdb
import random
import numpy as np


class PointDataset(Dataset):
    def __init__(self, data_path: Path, split="train"):
        self.data = pickle.load(open(data_path, "rb"))
        if split == "train":
            self.data = self.data[: int(0.8 * len(self.data))]
        else:
            self.data = self.data[int(0.8 * len(self.data)) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _idx = random.randint(0, self.data[0]["observed"].shape[1] - 2)
        X_t = torch.tensor(self.data[idx]["observed"][:, _idx], dtype=torch.float32)
        X_t1 = torch.tensor(
            self.data[idx]["observed"][:, _idx + 1], dtype=torch.float32
        )
        return X_t, X_t1


class PointDatasetPrebatched(Dataset):
    def __init__(
        self,
        data_path: Path,
        split="train",
        batch_size=32,
        max_forward_pred=5,
        noise=None,
    ):
        self.data = pickle.load(open(data_path, "rb"))
        self.batch_size = batch_size
        self.max_forward_pred = max_forward_pred
        self.noise = noise
        self.split = split
        # if split == "train":
        #     self.data = self.data[: int(0.8 * len(self.data))]
        # else:
        #     self.data = self.data[int(0.8 * len(self.data)) :]

    def __len__(self):
        return len(self.data) * self.max_forward_pred

    def __getitem__(self, idx):
        # k = random.randint(1, self.max_forward_pred)
        idx = idx % (len(self.data))
        # TODO if the multiplier of the data length doesn't evenly divide max_forward_pred,
        # then certain k's will be seen more often than others (potentially some not at all)
        if self.split != "train":
            k = idx // len(self.data) + 1
        else:
            k = random.randint(1, self.max_forward_pred)
        _idx = np.random.randint(
            low=0,
            # high=self.data[0]["observed"].shape[1] - k - 1,
            high=self.data[0]["observed"].shape[1] - k,
            size=self.batch_size,
        )
        X_t = torch.tensor(self.data[idx]["observed"][:, _idx], dtype=torch.float32)
        if self.noise:
            X_t = X_t.clone() + self.noise * X_t.std() * torch.randn_like(X_t)
        # else:
        #     X_t = X_t.clone()
        X_tk = torch.tensor(
            self.data[idx]["observed"][:, _idx + k], dtype=torch.float32
        )
        return X_t.T, X_tk.T, k


class RWMDataset(Dataset):

    def __init__(
        self, data_path: Path, split="train", context=32, forecast=8, flatten=True
    ):
        self.data = pickle.load(open(data_path, "rb"))
        self.split = split
        self.multiplication_factor = 1
        self.context = context
        self.forecast = forecast
        self.flatten = flatten

    def __len__(self):
        return len(self.data) * self.multiplication_factor

    def __getitem__(self, idx):
        idx = idx % (len(self.data))
        start_idx = random.randint(
            a=0,
            b=self.data[0]["observed"].shape[1] - self.context - self.forecast,
        )
        X_t = torch.tensor(
            self.data[idx]["observed"][:, start_idx : start_idx + self.context],
            dtype=torch.float32,
        )

        X_tk = torch.tensor(
            self.data[idx]["observed"][
                :, start_idx + self.context : start_idx + self.context + self.forecast
            ],
            dtype=torch.float32,
        )
        if self.flatten:
            X_t = X_t.T.flatten()
        return X_t, X_tk.T


if __name__ == "__main__":
    data_path = Path("data/pendulum_trajectories_train.pkl")
    dataset = PointDatasetPrebatched(
        data_path, split="train", batch_size=32, max_forward_pred=1
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][2])

    dataset = RWMDataset(data_path, split="train", context=32, forecast=8)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

    dataset = RWMDataset(
        data_path, split="train", context=33, forecast=9, flatten=False
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
