from torch.utils.data import Dataset
import torch
from pathlib import Path
import pickle
import ipdb
import random


class TrajectoryDataset(Dataset):
    def __init__(self, data_path: Path, split="train", type="observed"):
        self.data = pickle.load(open(data_path, "rb"))
        self.type = type
        if split == "train":
            self.data = self.data[: int(0.8 * len(self.data))]
        elif split == "val":
            self.data = self.data[int(0.8 * len(self.data)) :]
        else:
            self.data = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][self.type], dtype=torch.float32)
